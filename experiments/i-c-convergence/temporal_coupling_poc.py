#!/usr/bin/env python3
"""
実験I-C v0.2：時間相関を持つ逐次観測方策PoC。

v0.1の開ループ方策を保ったまま、A/B信号にAR(1)時間相関を
導入する。目的関数の時刻分離可能性、Hamming-1局所最適、
貪欲改善の吸引域、学習方策のOracle到達率を測る。

方策はまだ現在状態に条件づけられない。NumPyのみ、GPU不要。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    train_size: int
    test_size: int
    time_steps: int
    phis: list[float]
    correlation: float
    observation_noise: float
    lambdas: list[float]
    importance_a: list[float]
    importance_b: list[float]
    policy_iterations: int
    policy_learning_rate: float
    policy_init_scale: float
    ridge: float
    seeds: list[int]


@dataclass
class SufficientStats:
    xtx: np.ndarray
    xty: np.ndarray
    y2: np.ndarray
    variances: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AR(1)時間相関による逐次観測方策の結合を測ります。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="短時間設定（既定）")
    mode.add_argument("--full", action="store_true", help="seedと反復数を増やす")
    parser.add_argument("--output-dir", default="temporal_coupling_results")
    return parser.parse_args()


def make_config(full: bool) -> Config:
    common = dict(
        time_steps=4,
        phis=[0.0, 0.4, 0.8],
        correlation=0.35,
        observation_noise=0.05,
        lambdas=[0.25, 0.5, 0.75],
        importance_a=[1.8, 1.4, 0.6, 0.2],
        importance_b=[0.2, 0.6, 1.4, 1.8],
        policy_learning_rate=0.20,
        policy_init_scale=1.0,
        ridge=1e-7,
    )
    if full:
        return Config(
            train_size=8192,
            test_size=4096,
            policy_iterations=400,
            seeds=list(range(20)),
            **common,
        )
    return Config(
        train_size=2048,
        test_size=1024,
        policy_iterations=250,
        seeds=list(range(5)),
        **common,
    )


def ar1_samples(
    n: int, time_steps: int, phi: float, rng: np.random.Generator
) -> np.ndarray:
    values = np.empty((n, time_steps), dtype=np.float64)
    values[:, 0] = rng.normal(size=n)
    innovation_scale = math.sqrt(max(1.0 - phi * phi, 0.0))
    for t in range(1, time_steps):
        values[:, t] = phi * values[:, t - 1] + innovation_scale * rng.normal(size=n)
    return values


def make_dataset(
    n: int, phi: float, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aと独立潜在源Cを同じAR(1)とし、B=rho*A+sqrt(1-rho^2)*Cとする。"""
    a = ar1_samples(n, cfg.time_steps, phi, rng)
    private = ar1_samples(n, cfg.time_steps, phi, rng)
    b = cfg.correlation * a + math.sqrt(1.0 - cfg.correlation**2) * private
    x = np.stack([a, b], axis=2)
    x += cfg.observation_noise * rng.normal(size=x.shape)
    return x, a, b


def all_schedules(time_steps: int) -> list[tuple[int, ...]]:
    return list(itertools.product([0, 1], repeat=time_steps))


def masked_design(x: np.ndarray, schedule: tuple[int, ...]) -> np.ndarray:
    n, time_steps, _ = x.shape
    flat = np.zeros((n, time_steps * 2), dtype=np.float64)
    rows = np.arange(n)
    for t, action in enumerate(schedule):
        flat[rows, 2 * t + action] = x[:, t, action]
    return np.concatenate([flat, np.ones((n, 1))], axis=1)


def full_design(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(len(x), -1), np.ones((len(x), 1))], axis=1)


def build_empirical_stats(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    schedules: list[tuple[int, ...]],
) -> SufficientStats:
    y = np.concatenate([a, b], axis=1)
    designs = [masked_design(x, schedule) for schedule in schedules]
    n = len(x)
    return SufficientStats(
        xtx=np.stack([design.T @ design / n for design in designs]),
        xty=np.stack([design.T @ y / n for design in designs]),
        y2=np.mean(y * y, axis=0),
        variances=np.var(y, axis=0) + 1e-12,
    )


def build_full_empirical_stats(
    x: np.ndarray, a: np.ndarray, b: np.ndarray
) -> SufficientStats:
    y = np.concatenate([a, b], axis=1)
    design = full_design(x)
    n = len(x)
    return SufficientStats(
        xtx=(design.T @ design / n)[None, :, :],
        xty=(design.T @ y / n)[None, :, :],
        y2=np.mean(y * y, axis=0),
        variances=np.var(y, axis=0) + 1e-12,
    )


def latent_covariance(phi: float, cfg: Config) -> np.ndarray:
    indices = np.arange(cfg.time_steps)
    temporal = phi ** np.abs(indices[:, None] - indices[None, :])
    return np.block(
        [
            [temporal, cfg.correlation * temporal],
            [cfg.correlation * temporal, temporal],
        ]
    )


def build_population_stats(
    phi: float, cfg: Config, schedules: list[tuple[int, ...]]
) -> SufficientStats:
    """定常ガウス過程の母共分散から、全スケジュールの十分統計量を作る。"""
    cov_y = latent_covariance(phi, cfg)
    # yの並び [A1..AT,B1..BT] から、xの並び [A1,B1,A2,B2,...] へ。
    order = []
    for t in range(cfg.time_steps):
        order.extend([t, cfg.time_steps + t])
    cov_xy_full = cov_y[np.asarray(order), :]
    cov_x_full = cov_y[np.ix_(order, order)]
    cov_x_full += cfg.observation_noise**2 * np.eye(cfg.time_steps * 2)
    dim = cfg.time_steps * 2 + 1
    xtx_rows = []
    xty_rows = []
    for schedule in schedules:
        selected = [2 * t + action for t, action in enumerate(schedule)]
        xtx = np.zeros((dim, dim), dtype=np.float64)
        xty = np.zeros((dim, cfg.time_steps * 2), dtype=np.float64)
        xtx[np.ix_(selected, selected)] = cov_x_full[np.ix_(selected, selected)]
        xtx[-1, -1] = 1.0
        xty[selected, :] = cov_xy_full[selected, :]
        xtx_rows.append(xtx)
        xty_rows.append(xty)
    return SufficientStats(
        xtx=np.stack(xtx_rows),
        xty=np.stack(xty_rows),
        y2=np.diag(cov_y),
        variances=np.diag(cov_y),
    )


def build_full_population_stats(phi: float, cfg: Config) -> SufficientStats:
    cov_y = latent_covariance(phi, cfg)
    order = []
    for t in range(cfg.time_steps):
        order.extend([t, cfg.time_steps + t])
    cov_x = cov_y[np.ix_(order, order)]
    cov_x += cfg.observation_noise**2 * np.eye(cfg.time_steps * 2)
    cov_xy = cov_y[np.asarray(order), :]
    dim = cfg.time_steps * 2 + 1
    xtx = np.zeros((dim, dim))
    xty = np.zeros((dim, cfg.time_steps * 2))
    xtx[:-1, :-1] = cov_x
    xtx[-1, -1] = 1.0
    xty[:-1] = cov_xy
    return SufficientStats(
        xtx=xtx[None],
        xty=xty[None],
        y2=np.diag(cov_y),
        variances=np.diag(cov_y),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def schedule_probabilities(
    policy: np.ndarray, schedules: list[tuple[int, ...]]
) -> np.ndarray:
    probs = np.ones(len(schedules), dtype=np.float64)
    for index, schedule in enumerate(schedules):
        for t, action in enumerate(schedule):
            probs[index] *= policy[t, action]
    return probs / probs.sum()


def deterministic_probs(index: int, n_schedules: int) -> np.ndarray:
    probs = np.zeros(n_schedules)
    probs[index] = 1.0
    return probs


def fit_estimator(stats: SufficientStats, probs: np.ndarray, ridge: float) -> np.ndarray:
    xtx = np.tensordot(probs, stats.xtx, axes=(0, 0))
    xty = np.tensordot(probs, stats.xty, axes=(0, 0))
    regularizer = ridge * np.eye(xtx.shape[0])
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(xtx + regularizer, xty)


def losses_by_schedule(
    estimator: np.ndarray, stats: SufficientStats, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    n_schedules = stats.xtx.shape[0]
    output_mse = np.empty((n_schedules, cfg.time_steps * 2), dtype=np.float64)
    for s in range(n_schedules):
        xw = stats.xtx[s] @ estimator
        quadratic = np.sum(estimator * xw, axis=0)
        cross = np.sum(estimator * stats.xty[s], axis=0)
        output_mse[s] = np.maximum(quadratic - 2.0 * cross + stats.y2, 0.0)
    normalized = output_mse / stats.variances[None, :]
    weight_a = np.asarray(cfg.importance_a)
    weight_b = np.asarray(cfg.importance_b)
    d_a = normalized[:, : cfg.time_steps] @ weight_a / weight_a.sum()
    d_b = normalized[:, cfg.time_steps :] @ weight_b / weight_b.sum()
    return d_a, d_b


def objective_by_schedule(
    estimator: np.ndarray,
    stats: SufficientStats,
    cfg: Config,
    lambda_a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_a, d_b = losses_by_schedule(estimator, stats, cfg)
    return lambda_a * d_a + (1.0 - lambda_a) * d_b, d_a, d_b


def evaluate_distribution(
    estimator: np.ndarray,
    stats: SufficientStats,
    probs: np.ndarray,
    cfg: Config,
    lambda_a: float,
) -> tuple[float, float, float]:
    _, d_a, d_b = objective_by_schedule(estimator, stats, cfg, lambda_a)
    mean_a = float(probs @ d_a)
    mean_b = float(probs @ d_b)
    return mean_a, mean_b, lambda_a * mean_a + (1.0 - lambda_a) * mean_b


def format_schedule(schedule: tuple[int, ...]) -> str:
    return "".join("A" if action == 0 else "B" for action in schedule)


def hamming_neighbors(index: int, schedules: list[tuple[int, ...]]) -> list[int]:
    schedule = schedules[index]
    neighbors = []
    for t in range(len(schedule)):
        changed = list(schedule)
        changed[t] = 1 - changed[t]
        neighbors.append(schedules.index(tuple(changed)))
    return neighbors


def landscape_analysis(
    phi: float,
    lambda_a: float,
    cfg: Config,
    schedules: list[tuple[int, ...]],
) -> tuple[list[dict], dict, np.ndarray]:
    stats = build_population_stats(phi, cfg, schedules)
    objectives = np.empty(len(schedules))
    d_as = np.empty(len(schedules))
    d_bs = np.empty(len(schedules))
    for index in range(len(schedules)):
        probs = deterministic_probs(index, len(schedules))
        estimator = fit_estimator(stats, probs, cfg.ridge)
        d_a, d_b, objective = evaluate_distribution(
            estimator, stats, probs, cfg, lambda_a
        )
        objectives[index] = objective
        d_as[index] = d_a
        d_bs[index] = d_b

    additive_x = np.column_stack(
        [np.ones(len(schedules)), np.asarray(schedules, dtype=np.float64)]
    )
    coefficients, *_ = np.linalg.lstsq(additive_x, objectives, rcond=None)
    additive_prediction = additive_x @ coefficients
    residuals = objectives - additive_prediction
    tolerance = 1e-10
    local_indices = [
        index
        for index in range(len(schedules))
        if all(
            objectives[index] <= objectives[neighbor] + tolerance
            for neighbor in hamming_neighbors(index, schedules)
        )
    ]
    global_index = int(np.argmin(objectives))

    def greedy_attractor(start: int) -> int:
        current = start
        while True:
            candidates = [current] + hamming_neighbors(current, schedules)
            best = min(candidates, key=lambda i: (objectives[i], i))
            if objectives[best] >= objectives[current] - tolerance:
                return current
            current = best

    attractors = [greedy_attractor(index) for index in range(len(schedules))]
    basin_counts = Counter(attractors)
    rows = []
    for index, schedule in enumerate(schedules):
        rows.append(
            {
                "phi": phi,
                "lambda_a": lambda_a,
                "schedule": format_schedule(schedule),
                "D_A": float(d_as[index]),
                "D_B": float(d_bs[index]),
                "objective": float(objectives[index]),
                "additive_prediction": float(additive_prediction[index]),
                "additive_residual": float(residuals[index]),
                "is_local_optimum": index in local_indices,
                "is_global_optimum": index == global_index,
                "greedy_attractor": format_schedule(schedules[attractors[index]]),
            }
        )
    summary = {
        "phi": phi,
        "lambda_a": lambda_a,
        "oracle_schedule": format_schedule(schedules[global_index]),
        "oracle_objective": float(objectives[global_index]),
        "additive_max_abs_residual": float(np.max(np.abs(residuals))),
        "additive_rmse": float(np.sqrt(np.mean(residuals**2))),
        "local_optimum_count": len(local_indices),
        "local_optima": ";".join(format_schedule(schedules[i]) for i in local_indices),
        "global_greedy_basin_fraction": basin_counts[global_index] / len(schedules),
        "greedy_basin_counts": ";".join(
            f"{format_schedule(schedules[i])}:{count}"
            for i, count in sorted(basin_counts.items())
        ),
    }
    return rows, summary, objectives


class Adam:
    def __init__(self, shape: tuple[int, ...], learning_rate: float):
        self.lr = learning_rate
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> None:
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient * gradient
        m_hat = self.m / (1.0 - 0.9**self.t)
        v_hat = self.v / (1.0 - 0.999**self.t)
        parameters -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def policy_entropy(policy: np.ndarray) -> float:
    return float(-np.mean(np.sum(policy * np.log(policy + 1e-12), axis=1)))


def optimize_policy(
    train_stats: SufficientStats,
    cfg: Config,
    schedules: list[tuple[int, ...]],
    lambda_a: float,
    phi: float,
    seed: int,
    frozen_estimator: np.ndarray | None,
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(
        71_000 + seed * 1009 + int(lambda_a * 1000) + int(phi * 10_000)
    )
    logits = rng.normal(scale=cfg.policy_init_scale, size=(cfg.time_steps, 2))
    optimizer = Adam(logits.shape, cfg.policy_learning_rate)
    history = []
    indicators = np.empty((cfg.time_steps, 2, len(schedules)))
    for t in range(cfg.time_steps):
        for action in (0, 1):
            indicators[t, action] = [
                1.0 if schedule[t] == action else 0.0 for schedule in schedules
            ]
    for iteration in range(cfg.policy_iterations):
        policy = softmax(logits)
        probs = schedule_probabilities(policy, schedules)
        estimator = (
            frozen_estimator
            if frozen_estimator is not None
            else fit_estimator(train_stats, probs, cfg.ridge)
        )
        losses, _, _ = objective_by_schedule(estimator, train_stats, cfg, lambda_a)
        expected = float(probs @ losses)
        centered = losses - expected
        gradient = np.empty_like(logits)
        for t in range(cfg.time_steps):
            for action in (0, 1):
                gradient[t, action] = float(
                    np.sum(probs * centered * indicators[t, action])
                )
        optimizer.step(logits, gradient)
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == cfg.policy_iterations:
            history.append(
                {
                    "seed": seed,
                    "phi": phi,
                    "lambda_a": lambda_a,
                    "method": method,
                    "iteration": iteration + 1,
                    "train_objective": expected,
                    "entropy": policy_entropy(policy),
                }
            )
    policy = softmax(logits)
    probs = schedule_probabilities(policy, schedules)
    estimator = (
        frozen_estimator
        if frozen_estimator is not None
        else fit_estimator(train_stats, probs, cfg.ridge)
    )
    return policy, estimator, history


def add_policy_result(
    rows: list[dict],
    seed: int,
    phi: float,
    lambda_a: float,
    method: str,
    d_a: float,
    d_b: float,
    objective: float,
    schedule: str,
    oracle_schedule: str,
    entropy: float,
) -> None:
    match = (
        float(np.mean([a == b for a, b in zip(schedule, oracle_schedule)]))
        if len(schedule) == len(oracle_schedule) and set(schedule) <= {"A", "B"}
        else float("nan")
    )
    rows.append(
        {
            "seed": seed,
            "phi": phi,
            "lambda_a": lambda_a,
            "method": method,
            "D_A": d_a,
            "D_B": d_b,
            "objective": objective,
            "schedule": schedule,
            "oracle_schedule": oracle_schedule,
            "schedule_match": match,
            "oracle_reached": schedule == oracle_schedule,
            "policy_entropy": entropy,
        }
    )


def aggregate_policy(rows: list[dict]) -> list[dict]:
    groups = sorted(
        {
            (float(row["phi"]), float(row["lambda_a"]), str(row["method"]))
            for row in rows
        }
    )
    result = []
    for phi, lambda_a, method in groups:
        values = [
            row
            for row in rows
            if float(row["phi"]) == phi
            and float(row["lambda_a"]) == lambda_a
            and str(row["method"]) == method
        ]
        objectives = np.asarray([float(row["objective"]) for row in values])
        d_a = np.asarray([float(row["D_A"]) for row in values])
        d_b = np.asarray([float(row["D_B"]) for row in values])
        matches = np.asarray([float(row["schedule_match"]) for row in values])
        reached = np.asarray([bool(row["oracle_reached"]) for row in values])
        schedules = Counter(str(row["schedule"]) for row in values)
        result.append(
            {
                "phi": phi,
                "lambda_a": lambda_a,
                "method": method,
                "D_A_mean": float(d_a.mean()),
                "D_B_mean": float(d_b.mean()),
                "objective_mean": float(objectives.mean()),
                "objective_sd": float(objectives.std(ddof=0)),
                "schedule_match_mean": (
                    float(np.nanmean(matches)) if not np.all(np.isnan(matches)) else ""
                ),
                "oracle_reach_rate": float(reached.mean()),
                "modal_schedule": schedules.most_common(1)[0][0],
                "schedule_counts": ";".join(
                    f"{schedule}:{count}" for schedule, count in sorted(schedules.items())
                ),
                "policy_entropy_mean": float(
                    np.mean([float(row["policy_entropy"]) for row in values])
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows: list[dict], phi: float, lambda_a: float, method: str) -> dict:
    return next(
        row
        for row in rows
        if float(row["phi"]) == phi
        and float(row["lambda_a"]) == lambda_a
        and str(row["method"]) == method
    )


def find_landscape(rows: list[dict], phi: float, lambda_a: float) -> dict:
    return next(
        row
        for row in rows
        if float(row["phi"]) == phi and float(row["lambda_a"]) == lambda_a
    )


def judge(
    landscape_summary: list[dict], policy_summary: list[dict]
) -> tuple[str, dict]:
    l0 = find_landscape(landscape_summary, 0.0, 0.5)
    l8 = find_landscape(landscape_summary, 0.8, 0.5)
    joint8 = find_row(policy_summary, 0.8, 0.5, "learned_joint")
    random8 = find_row(policy_summary, 0.8, 0.5, "random_fixed")
    frozen8 = find_row(policy_summary, 0.8, 0.5, "learned_frozen")
    full8 = find_row(policy_summary, 0.8, 0.5, "full_observation")
    criteria = {
        "phi0_additive_max_residual": float(l0["additive_max_abs_residual"]),
        "phi0_local_optimum_count": int(l0["local_optimum_count"]),
        "phi08_additive_max_residual": float(l8["additive_max_abs_residual"]),
        "phi08_local_optimum_count": int(l8["local_optimum_count"]),
        "phi08_global_greedy_basin_fraction": float(l8["global_greedy_basin_fraction"]),
        "phi08_joint_oracle_reach_rate": float(joint8["oracle_reach_rate"]),
        "phi08_frozen_oracle_reach_rate": float(frozen8["oracle_reach_rate"]),
        "phi08_joint_gain_over_random": float(
            random8["objective_mean"] - joint8["objective_mean"]
        ),
        "phi08_joint_gain_over_frozen": float(
            frozen8["objective_mean"] - joint8["objective_mean"]
        ),
        "phi08_full_observation_D": float(full8["objective_mean"]),
        "phi0_is_additive_control": bool(l0["additive_max_abs_residual"] < 1e-5),
        "phi0_has_single_local_optimum": bool(int(l0["local_optimum_count"]) == 1),
        "phi08_is_nonadditive": bool(l8["additive_max_abs_residual"] > 0.05),
        "phi08_has_multiple_local_optima": bool(int(l8["local_optimum_count"]) >= 2),
        "phi08_has_subglobal_greedy_basin": bool(
            float(l8["global_greedy_basin_fraction"]) < 1.0
        ),
        "full_observation_reconstructed": bool(full8["objective_mean"] < 0.01),
        "learned_policy_beats_random": bool(
            random8["objective_mean"] - joint8["objective_mean"] > 0.03
        ),
    }
    gate_keys = [
        "phi0_is_additive_control",
        "phi0_has_single_local_optimum",
        "phi08_is_nonadditive",
        "phi08_has_multiple_local_optima",
        "phi08_has_subglobal_greedy_basin",
        "full_observation_reconstructed",
        "learned_policy_beats_random",
    ]
    criteria["passed"] = all(bool(criteria[key]) for key in gate_keys)
    if criteria["passed"]:
        message = (
            "時間結合の校正に成功：phi=0の分離可能な対照に対し、"
            "phi=0.8で非加法性、複数局所最適、非大域的吸引域が現れました。"
        )
    else:
        message = (
            "時間結合の校正は未達：母集団共分散、局所最適判定、"
            "方策最適化またはゲート設定を点検してください。"
        )
    return message, criteria


def write_structure_svg(path: Path, landscape_summary: list[dict]) -> None:
    rows = [find_landscape(landscape_summary, phi, 0.5) for phi in (0.0, 0.4, 0.8)]
    width, height = 820, 470
    left, right, top, bottom = 90, 90, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    max_residual = max(float(row["additive_max_abs_residual"]) for row in rows) * 1.18
    max_residual = max(max_residual, 0.1)

    def sx(index: int) -> float:
        return left + plot_w * index / 2

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value / max_residual)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Temporal coupling breaks additive separability (lambda=0.5)</text>',
    ]
    for tick in range(6):
        value = max_residual * tick / 5
        y = sy(value)
        lines += [
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="13">{value:.3f}</text>',
        ]
    points = []
    for i, row in enumerate(rows):
        x = sx(i)
        value = float(row["additive_max_abs_residual"])
        points.append(f"{x:.1f},{sy(value):.1f}")
        lines += [
            f'<circle cx="{x}" cy="{sy(value)}" r="7" fill="#176B87"/>',
            f'<text x="{x}" y="{top+plot_h+27}" text-anchor="middle" font-size="14">phi={row["phi"]}</text>',
            f'<text x="{x}" y="{sy(value)-13}" text-anchor="middle" font-size="13">local optima: {row["local_optimum_count"]}</text>',
        ]
    lines += [
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#176B87" stroke-width="3"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+plot_h/2})" font-size="15">max absolute additive residual</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_convergence_svg(path: Path, policy_summary: list[dict]) -> None:
    width, height = 820, 470
    left, right, top, bottom = 90, 45, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    methods = ["learned_frozen", "learned_joint"]
    labels = ["Learned / frozen model", "Learned / joint"]
    colors = ["#7E57C2", "#176B87"]
    phis = [0.0, 0.4, 0.8]

    def sx(index: int) -> float:
        return left + plot_w * index / 2

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Policy optimizer reaches the population Oracle (lambda=0.5)</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = sy(value)
        lines += [
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="13">{value:.1f}</text>',
        ]
    for method, label, color in zip(methods, labels, colors):
        values = [float(find_row(policy_summary, phi, 0.5, method)["oracle_reach_rate"]) for phi in phis]
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, value in enumerate(values):
            lines.append(f'<circle cx="{sx(i)}" cy="{sy(value)}" r="6" fill="{color}"/>')
        ly = top + 20 + methods.index(method) * 25
        lines += [
            f'<line x1="{left+plot_w-210}" y1="{ly}" x2="{left+plot_w-175}" y2="{ly}" stroke="{color}" stroke-width="4"/>',
            f'<text x="{left+plot_w-165}" y="{ly+5}" font-size="14">{label}</text>',
        ]
    for i, phi in enumerate(phis):
        lines.append(f'<text x="{sx(i)}" y="{top+plot_h+27}" text-anchor="middle" font-size="14">phi={phi}</text>')
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+plot_h/2})" font-size="15">Oracle reach rate</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = make_config(full=args.full)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schedules = all_schedules(cfg.time_steps)
    random_probs = np.full(len(schedules), 1.0 / len(schedules))
    start = time.time()
    print("実験I-C v0.2：AR(1)時間結合PoCを開始します（GPU不要）")
    print(f"保存先: {output_dir}")

    landscape_rows: list[dict] = []
    landscape_summary: list[dict] = []
    population_objectives: dict[tuple[float, float], np.ndarray] = {}
    for phi in cfg.phis:
        for lambda_a in cfg.lambdas:
            rows, summary, objectives = landscape_analysis(
                phi, lambda_a, cfg, schedules
            )
            landscape_rows.extend(rows)
            landscape_summary.append(summary)
            population_objectives[(phi, lambda_a)] = objectives

    policy_rows: list[dict] = []
    probability_rows: list[dict] = []
    history_rows: list[dict] = []
    total = len(cfg.seeds) * len(cfg.phis) * len(cfg.lambdas)
    count = 0
    for seed in cfg.seeds:
        for phi in cfg.phis:
            rng = np.random.default_rng(100_000 + seed * 97 + int(phi * 1000))
            train = make_dataset(cfg.train_size, phi, cfg, rng)
            test = make_dataset(cfg.test_size, phi, cfg, rng)
            train_stats = build_empirical_stats(*train, schedules)
            test_stats = build_empirical_stats(*test, schedules)
            full_train = build_full_empirical_stats(*train)
            full_test = build_full_empirical_stats(*test)
            full_estimator = fit_estimator(full_train, np.array([1.0]), cfg.ridge)
            for lambda_a in cfg.lambdas:
                landscape = find_landscape(landscape_summary, phi, lambda_a)
                oracle_schedule = str(landscape["oracle_schedule"])
                oracle_index = [format_schedule(s) for s in schedules].index(oracle_schedule)
                oracle_probs = deterministic_probs(oracle_index, len(schedules))
                oracle_estimator = fit_estimator(train_stats, oracle_probs, cfg.ridge)
                d_a, d_b, objective = evaluate_distribution(
                    oracle_estimator, test_stats, oracle_probs, cfg, lambda_a
                )
                add_policy_result(
                    policy_rows, seed, phi, lambda_a, "oracle", d_a, d_b,
                    objective, oracle_schedule, oracle_schedule, 0.0
                )

                random_estimator = fit_estimator(train_stats, random_probs, cfg.ridge)
                d_a, d_b, objective = evaluate_distribution(
                    random_estimator, test_stats, random_probs, cfg, lambda_a
                )
                add_policy_result(
                    policy_rows, seed, phi, lambda_a, "random_fixed", d_a, d_b,
                    objective, "stochastic", oracle_schedule, math.log(2.0)
                )

                frozen_policy, frozen_estimator, frozen_history = optimize_policy(
                    train_stats, cfg, schedules, lambda_a, phi, seed,
                    frozen_estimator=random_estimator, method="learned_frozen"
                )
                frozen_probs = schedule_probabilities(frozen_policy, schedules)
                d_a, d_b, objective = evaluate_distribution(
                    frozen_estimator, test_stats, frozen_probs, cfg, lambda_a
                )
                frozen_schedule_tuple = tuple(np.argmax(frozen_policy, axis=1).tolist())
                add_policy_result(
                    policy_rows, seed, phi, lambda_a, "learned_frozen", d_a, d_b,
                    objective, format_schedule(frozen_schedule_tuple), oracle_schedule,
                    policy_entropy(frozen_policy)
                )
                history_rows.extend(frozen_history)

                joint_policy, joint_estimator, joint_history = optimize_policy(
                    train_stats, cfg, schedules, lambda_a, phi, seed,
                    frozen_estimator=None, method="learned_joint"
                )
                joint_probs = schedule_probabilities(joint_policy, schedules)
                d_a, d_b, objective = evaluate_distribution(
                    joint_estimator, test_stats, joint_probs, cfg, lambda_a
                )
                joint_schedule_tuple = tuple(np.argmax(joint_policy, axis=1).tolist())
                add_policy_result(
                    policy_rows, seed, phi, lambda_a, "learned_joint", d_a, d_b,
                    objective, format_schedule(joint_schedule_tuple), oracle_schedule,
                    policy_entropy(joint_policy)
                )
                history_rows.extend(joint_history)

                full_d_a, full_d_b, full_objective = evaluate_distribution(
                    full_estimator, full_test, np.array([1.0]), cfg, lambda_a
                )
                add_policy_result(
                    policy_rows, seed, phi, lambda_a, "full_observation",
                    full_d_a, full_d_b, full_objective, "both", oracle_schedule, 0.0
                )

                for method, policy in (
                    ("learned_frozen", frozen_policy),
                    ("learned_joint", joint_policy),
                ):
                    for t in range(cfg.time_steps):
                        probability_rows.append(
                            {
                                "seed": seed,
                                "phi": phi,
                                "lambda_a": lambda_a,
                                "method": method,
                                "time": t + 1,
                                "prob_A": float(policy[t, 0]),
                                "prob_B": float(policy[t, 1]),
                            }
                        )
                count += 1
                if count == 1 or count % max(total // 10, 1) == 0 or count == total:
                    print(f"  進捗 {count}/{total}")

    policy_summary = aggregate_policy(policy_rows)
    message, criteria = judge(landscape_summary, policy_summary)
    elapsed = time.time() - start
    metadata = {
        "experiment": "Experiment I-C v0.2: temporal coupling in sequential observation",
        "version": "0.2",
        "config": asdict(cfg),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "criteria": criteria,
        "scope_note": (
            "The policy remains open-loop and time-indexed. AR(1) temporal coupling is isolated "
            "before introducing state-conditioned closed-loop attention."
        ),
    }
    write_csv(output_dir / "landscape_raw.csv", landscape_rows)
    write_csv(output_dir / "landscape_summary.csv", landscape_summary)
    write_csv(output_dir / "policy_raw.csv", policy_rows)
    write_csv(output_dir / "policy_summary.csv", policy_summary)
    write_csv(output_dir / "policy_probabilities.csv", probability_rows)
    write_csv(output_dir / "learning_history.csv", history_rows)
    write_structure_svg(output_dir / "temporal_coupling_structure.svg", landscape_summary)
    write_convergence_svg(output_dir / "policy_oracle_reach.svg", policy_summary)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 実験I-C v0.2・時間結合PoC 実行結果",
        "",
        message,
        "",
        "## lambda=0.5 の目的地形",
        "",
        "| phi | 加法残差の最大絶対値 | 局所最適数 | 局所最適 | 大域的吸引域 | Oracle |",
        "|---:|---:|---:|:---|---:|:---:|",
        *[
            (
                f"| {phi:.1f} "
                f"| {find_landscape(landscape_summary, phi, 0.5)['additive_max_abs_residual']:.6f} "
                f"| {find_landscape(landscape_summary, phi, 0.5)['local_optimum_count']} "
                f"| {find_landscape(landscape_summary, phi, 0.5)['local_optima']} "
                f"| {find_landscape(landscape_summary, phi, 0.5)['global_greedy_basin_fraction']:.3f} "
                f"| {find_landscape(landscape_summary, phi, 0.5)['oracle_schedule']} |"
            )
            for phi in cfg.phis
        ],
        "",
        "## phi=0.8, lambda=0.5 の方策比較",
        "",
        "| 方法 | 平均目的値 | Oracle到達率 | 最頻スケジュール | スケジュール分布 |",
        "|---|---:|---:|:---:|:---|",
        *[
            (
                f"| {method} "
                f"| {find_row(policy_summary, 0.8, 0.5, method)['objective_mean']:.4f} "
                f"| {find_row(policy_summary, 0.8, 0.5, method)['oracle_reach_rate']:.3f} "
                f"| {find_row(policy_summary, 0.8, 0.5, method)['modal_schedule']} "
                f"| {find_row(policy_summary, 0.8, 0.5, method)['schedule_counts']} |"
            )
            for method in (
                "full_observation", "oracle", "random_fixed",
                "learned_frozen", "learned_joint"
            )
        ],
        "",
        "## 自動判定の要点",
        "",
        f"- phi=0 加法残差: {criteria['phi0_additive_max_residual']:.8f}",
        f"- phi=0.8 加法残差: {criteria['phi08_additive_max_residual']:.6f}",
        f"- phi=0.8 局所最適数: {criteria['phi08_local_optimum_count']}",
        f"- phi=0.8 大域的貪欲吸引域: {criteria['phi08_global_greedy_basin_fraction']:.3f}",
        f"- phi=0.8 共同更新のOracle到達率: {criteria['phi08_joint_oracle_reach_rate']:.3f}",
        f"- phi=0.8 共同更新のランダムからの改善: {criteria['phi08_joint_gain_over_random']:.4f}",
        f"- phi=0.8 共同更新の下流固定からの改善: {criteria['phi08_joint_gain_over_frozen']:.4f}",
        "",
        "## 解釈の限界",
        "",
        "phi=0は現行v0.1の分離可能な対照です。phi>0では一つの観測が別時刻の推定に影響し、スケジュール間に相互作用が生じます。",
        "ただし方策は時刻だけに依存し、実際に見た値や信念状態に応じて次の観測先を変えません。",
        "心理志向／客観志向、発達的履歴依存、人間の注意を支持する結果ではありません。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text("\n".join(report), encoding="utf-8")
    print()
    print(message)
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, temporal_coupling_structure.svg, policy_oracle_reach.svg")


if __name__ == "__main__":
    main()
