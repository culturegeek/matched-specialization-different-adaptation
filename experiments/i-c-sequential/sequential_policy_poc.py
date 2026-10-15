#!/usr/bin/env python3
"""
実験I-C v0.1：逐次的で再取得不能な観測機会と、学習される観測方策。

各時刻にA側またはB側の一方だけを観測する。見逃した同時刻の値は
後から取得できない。Oracle、ランダム、A/B特化、学習方策（下流固定）、
学習方策（下流との共同更新）を比較する。依存はNumPyのみ。
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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    train_size: int
    validation_size: int
    test_size: int
    time_steps: int
    correlation: float
    observation_noise: float
    lambdas: list[float]
    importance_a: list[float]
    importance_b: list[float]
    policy_iterations: int
    policy_learning_rate: float
    ridge: float
    seeds: list[int]


@dataclass
class SufficientStats:
    xtx: np.ndarray
    xty: np.ndarray
    y2: np.ndarray
    variances: np.ndarray
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐次観測環境でOracle・固定・学習方策を比較します。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="短時間設定（既定）")
    mode.add_argument("--full", action="store_true", help="標本数とseedを増やす")
    parser.add_argument("--output-dir", default="sequential_policy_results")
    return parser.parse_args()


def make_config(full: bool) -> Config:
    common = dict(
        time_steps=4,
        correlation=0.35,
        observation_noise=0.05,
        lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
        importance_a=[1.8, 1.4, 0.6, 0.2],
        importance_b=[0.2, 0.6, 1.4, 1.8],
        policy_learning_rate=0.30,
        ridge=1e-6,
    )
    if full:
        return Config(
            train_size=8192,
            validation_size=4096,
            test_size=4096,
            policy_iterations=300,
            seeds=list(range(20)),
            **common,
        )
    return Config(
        train_size=2048,
        validation_size=1024,
        test_size=1024,
        policy_iterations=180,
        seeds=list(range(5)),
        **common,
    )


def make_dataset(
    n: int, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """時刻ごとに相関するA/B信号と、各信号の専用観測チャネルを作る。"""
    a = rng.normal(size=(n, cfg.time_steps))
    private_b = rng.normal(size=(n, cfg.time_steps))
    b = cfg.correlation * a + math.sqrt(1.0 - cfg.correlation**2) * private_b
    x = np.stack([a, b], axis=2)
    x += cfg.observation_noise * rng.normal(size=x.shape)
    return x.astype(np.float64), a.astype(np.float64), b.astype(np.float64)


def all_schedules(time_steps: int) -> list[tuple[int, ...]]:
    return list(itertools.product([0, 1], repeat=time_steps))


def masked_design(x: np.ndarray, schedule: tuple[int, ...]) -> np.ndarray:
    """各時刻で選択した一方のチャネルだけを残し、切片を加える。"""
    n, time_steps, _ = x.shape
    flat = np.zeros((n, time_steps * 2), dtype=np.float64)
    rows = np.arange(n)
    for t, action in enumerate(schedule):
        flat[rows, 2 * t + action] = x[:, t, action]
    return np.concatenate([flat, np.ones((n, 1))], axis=1)


def full_design(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(len(x), -1), np.ones((len(x), 1))], axis=1)


def build_stats(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    schedules: list[tuple[int, ...]],
) -> SufficientStats:
    y = np.concatenate([a, b], axis=1)
    designs = [masked_design(x, schedule) for schedule in schedules]
    n = len(x)
    xtx = np.stack([design.T @ design / n for design in designs])
    xty = np.stack([design.T @ y / n for design in designs])
    y2 = np.mean(y * y, axis=0)
    variances = np.var(y, axis=0) + 1e-12
    return SufficientStats(xtx=xtx, xty=xty, y2=y2, variances=variances, n=n)


def build_full_stats(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> SufficientStats:
    y = np.concatenate([a, b], axis=1)
    design = full_design(x)
    n = len(x)
    return SufficientStats(
        xtx=(design.T @ design / n)[None, :, :],
        xty=(design.T @ y / n)[None, :, :],
        y2=np.mean(y * y, axis=0),
        variances=np.var(y, axis=0) + 1e-12,
        n=n,
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
    probs /= probs.sum()
    return probs


def fit_estimator(stats: SufficientStats, probs: np.ndarray, ridge: float) -> np.ndarray:
    """方策分布の下で最適な線形状態推定器を十分統計量から求める。"""
    xtx = np.tensordot(probs, stats.xtx, axes=(0, 0))
    xty = np.tensordot(probs, stats.xty, axes=(0, 0))
    regularizer = ridge * np.eye(xtx.shape[0])
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(xtx + regularizer, xty)


def losses_by_schedule(
    estimator: np.ndarray,
    stats: SufficientStats,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """各観測スケジュールの正規化歪みD_A,D_Bを二次形式で計算する。"""
    n_schedules = stats.xtx.shape[0]
    output_mse = np.empty((n_schedules, cfg.time_steps * 2), dtype=np.float64)
    for s in range(n_schedules):
        xw = stats.xtx[s] @ estimator
        quadratic = np.sum(estimator * xw, axis=0)
        cross = np.sum(estimator * stats.xty[s], axis=0)
        output_mse[s] = np.maximum(quadratic - 2.0 * cross + stats.y2, 0.0)
    normalized = output_mse / stats.variances[None, :]
    weight_a = np.asarray(cfg.importance_a, dtype=np.float64)
    weight_b = np.asarray(cfg.importance_b, dtype=np.float64)
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


class Adam:
    def __init__(self, shape: tuple[int, ...], lr: float):
        self.lr = lr
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
    seed: int,
    frozen_estimator: np.ndarray | None,
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    方策勾配を全スケジュールの厳密な期待値で計算する。

    frozen_estimator=Noneなら、各反復で現在方策に対する最適推定器を
    当て直す。条件付き最適解なので、包絡線定理により方策勾配では
    推定器の微分を取る必要がない。
    """
    rng = np.random.default_rng(30_000 + seed * 101 + int(lambda_a * 1000))
    logits = rng.normal(scale=0.01, size=(cfg.time_steps, 2))
    optimizer = Adam(logits.shape, cfg.policy_learning_rate)
    history = []
    for iteration in range(cfg.policy_iterations):
        policy = softmax(logits)
        probs = schedule_probabilities(policy, schedules)
        estimator = (
            frozen_estimator
            if frozen_estimator is not None
            else fit_estimator(train_stats, probs, cfg.ridge)
        )
        losses, _, _ = objective_by_schedule(
            estimator, train_stats, cfg, lambda_a
        )
        expected = float(probs @ losses)
        centered = losses - expected
        gradient = np.zeros_like(logits)
        for t in range(cfg.time_steps):
            for action in [0, 1]:
                indicator = np.array(
                    [1.0 if schedule[t] == action else 0.0 for schedule in schedules]
                )
                gradient[t, action] = float(np.sum(probs * centered * indicator))
        optimizer.step(logits, gradient)
        if iteration == 0 or (iteration + 1) % 10 == 0 or iteration + 1 == cfg.policy_iterations:
            history.append(
                {
                    "seed": seed,
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


def deterministic_probs(index: int, n_schedules: int) -> np.ndarray:
    probs = np.zeros(n_schedules)
    probs[index] = 1.0
    return probs


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


def select_oracle(
    train_stats: SufficientStats,
    validation_stats: SufficientStats,
    cfg: Config,
    lambda_a: float,
    schedules: list[tuple[int, ...]],
) -> tuple[int, np.ndarray]:
    candidates = []
    for index in range(len(schedules)):
        probs = deterministic_probs(index, len(schedules))
        estimator = fit_estimator(train_stats, probs, cfg.ridge)
        _, _, objective = evaluate_distribution(
            estimator, validation_stats, probs, cfg, lambda_a
        )
        candidates.append((objective, index, estimator))
    _, index, estimator = min(candidates, key=lambda row: (row[0], row[1]))
    return index, estimator


def add_result(
    rows: list[dict],
    seed: int,
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
            "lambda_a": lambda_a,
            "method": method,
            "D_A": d_a,
            "D_B": d_b,
            "objective": objective,
            "schedule": schedule,
            "oracle_schedule": oracle_schedule,
            "schedule_match": match,
            "policy_entropy": entropy,
        }
    )


def aggregate(rows: list[dict]) -> list[dict]:
    groups = sorted({(float(r["lambda_a"]), str(r["method"])) for r in rows})
    result = []
    for lambda_a, method in groups:
        values = [
            row
            for row in rows
            if float(row["lambda_a"]) == lambda_a and str(row["method"]) == method
        ]
        objectives = np.array([float(row["objective"]) for row in values])
        da = np.array([float(row["D_A"]) for row in values])
        db = np.array([float(row["D_B"]) for row in values])
        matches = np.array(
            [float(row["schedule_match"]) for row in values], dtype=np.float64
        )
        entropies = np.array([float(row["policy_entropy"]) for row in values])
        result.append(
            {
                "lambda_a": lambda_a,
                "method": method,
                "D_A_mean": float(da.mean()),
                "D_B_mean": float(db.mean()),
                "objective_mean": float(objectives.mean()),
                "objective_sd": float(objectives.std(ddof=0)),
                "schedule_match_mean": (
                    float(np.nanmean(matches)) if not np.all(np.isnan(matches)) else ""
                ),
                "policy_entropy_mean": float(entropies.mean()),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_value(summary: list[dict], lambda_a: float, method: str, key: str) -> float:
    row = next(
        row
        for row in summary
        if float(row["lambda_a"]) == lambda_a and str(row["method"]) == method
    )
    return float(row[key])


def judge(summary: list[dict], cfg: Config) -> tuple[str, dict]:
    interior = [0.25, 0.5, 0.75]
    joint_gap = np.mean(
        [
            summary_value(summary, value, "learned_joint", "objective_mean")
            - summary_value(summary, value, "oracle", "objective_mean")
            for value in interior
        ]
    )
    joint_gain = np.mean(
        [
            summary_value(summary, value, "random_fixed", "objective_mean")
            - summary_value(summary, value, "learned_joint", "objective_mean")
            for value in interior
        ]
    )
    frozen_gain = np.mean(
        [
            summary_value(summary, value, "random_fixed", "objective_mean")
            - summary_value(summary, value, "learned_frozen", "objective_mean")
            for value in interior
        ]
    )
    joint_gain_over_frozen = np.mean(
        [
            summary_value(summary, value, "learned_frozen", "objective_mean")
            - summary_value(summary, value, "learned_joint", "objective_mean")
            for value in interior
        ]
    )
    match = np.mean(
        [
            summary_value(summary, value, "learned_joint", "schedule_match_mean")
            for value in interior
        ]
    )
    full = summary_value(summary, 0.5, "full_observation", "objective_mean")
    oracle = summary_value(summary, 0.5, "oracle", "objective_mean")
    a_at_high = summary_value(summary, 0.75, "A_fixed", "objective_mean")
    b_at_high = summary_value(summary, 0.75, "B_fixed", "objective_mean")
    a_at_low = summary_value(summary, 0.25, "A_fixed", "objective_mean")
    b_at_low = summary_value(summary, 0.25, "B_fixed", "objective_mean")
    criteria = {
        "full_observation_D": full,
        "oracle_sequential_D_at_half": oracle,
        "sequential_opportunity_cost": oracle - full,
        "joint_gap_to_oracle_interior": float(joint_gap),
        "joint_gain_over_random_interior": float(joint_gain),
        "frozen_gain_over_random_interior": float(frozen_gain),
        "joint_gain_over_frozen_interior": float(joint_gain_over_frozen),
        "joint_schedule_match_interior": float(match),
        "full_observation_reconstructed": bool(full < 0.01),
        "temporal_acquisition_cost_detected": bool(oracle - full > 0.05),
        "joint_policy_near_oracle": bool(joint_gap < 0.04),
        "joint_policy_beats_random": bool(joint_gain > 0.05),
        "frozen_policy_beats_random": bool(frozen_gain > 0.03),
        "joint_schedule_matches_oracle": bool(match > 0.80),
        "specialization_reverses_with_goal": bool(
            a_at_high < b_at_high and b_at_low < a_at_low
        ),
    }
    gate_keys = [
        "full_observation_reconstructed",
        "temporal_acquisition_cost_detected",
        "joint_policy_near_oracle",
        "joint_policy_beats_random",
        "frozen_policy_beats_random",
        "joint_schedule_matches_oracle",
        "specialization_reverses_with_goal",
    ]
    passed = all(bool(criteria[key]) for key in gate_keys)
    criteria["passed"] = passed
    if passed:
        message = (
            "逐次観測方策の最小校正に成功：学習方策はランダム方策を上回り、"
            "Oracleスケジュールへ接近しました。見逃した時刻の情報は下流推定器では回復しません。"
        )
    else:
        message = (
            "逐次観測方策の校正は未達：環境の重要度、方策最適化、固定対照またはOracle選択を点検してください。"
        )
    return message, criteria


def modal_schedule(rows: list[dict], lambda_a: float, method: str) -> str:
    schedules = [
        str(row["schedule"])
        for row in rows
        if float(row["lambda_a"]) == lambda_a and str(row["method"]) == method
    ]
    unique, counts = np.unique(schedules, return_counts=True)
    return str(unique[int(np.argmax(counts))])


def write_performance_svg(path: Path, summary: list[dict], cfg: Config) -> None:
    width, height = 820, 520
    left, right, top, bottom = 85, 45, 50, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    methods = ["oracle", "random_fixed", "learned_frozen", "learned_joint"]
    labels = ["Oracle", "Random fixed", "Learned / frozen model", "Learned / joint"]
    colors = ["#455A64", "#EF8354", "#7E57C2", "#176B87"]
    ymax = 0.65

    def sx(index: int) -> float:
        return left + plot_w * index / (len(cfg.lambdas) - 1)

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value / ymax)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC}.title{font-size:21px;font-weight:700}.label{font-size:16px}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Sequential observation policy performance (lower is better)</text>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = sy(value)
        lines += [
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="13">{value:.2f}</text>',
        ]
    for index, value in enumerate(cfg.lambdas):
        lines.append(
            f'<text x="{sx(index)}" y="{top+plot_h+25}" text-anchor="middle" font-size="13">{value:.2f}</text>'
        )
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
    ]
    for m, label, color in zip(methods, labels, colors):
        values = [summary_value(summary, value, m, "objective_mean") for value in cfg.lambdas]
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        dash = ' stroke-dasharray="8 5"' if m == "oracle" else ""
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
        )
        for i, value in enumerate(values):
            lines.append(f'<circle cx="{sx(i):.1f}" cy="{sy(value):.1f}" r="5" fill="{color}"/>')
        ly = top + 20 + methods.index(m) * 24
        lines += [
            f'<line x1="{left+plot_w-205}" y1="{ly}" x2="{left+plot_w-170}" y2="{ly}" stroke="{color}" stroke-width="4"{dash}/>',
            f'<text x="{left+plot_w-160}" y="{ly+5}" font-size="14">{label}</text>',
        ]
    lines += [
        f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">lambda: importance of target A</text>',
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" class="label">weighted distortion</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_policy_svg(path: Path, policy_rows: list[dict], cfg: Config) -> None:
    means = {}
    for lambda_a in cfg.lambdas:
        for t in range(cfg.time_steps):
            values = [
                float(row["prob_A"])
                for row in policy_rows
                if str(row["method"]) == "learned_joint"
                and float(row["lambda_a"]) == lambda_a
                and int(row["time"]) == t + 1
            ]
            means[(lambda_a, t)] = float(np.mean(values))
    width, height = 760, 500
    left, top, cell_w, cell_h = 145, 75, 125, 65

    def color(value: float) -> str:
        r = int(239 - 120 * value)
        g = int(128 + 65 * (1.0 - abs(value - 0.5) * 2.0))
        b = int(92 + 125 * (1.0 - value))
        return f"#{r:02x}{g:02x}{b:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.label{font-size:16px}.value{font-size:15px;font-weight:700}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Learned joint policy: probability of observing A</text>',
    ]
    for t in range(cfg.time_steps):
        x = left + t * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="62" text-anchor="middle" class="label">time {t+1}</text>')
    for i, lambda_a in enumerate(cfg.lambdas):
        y = top + i * cell_h
        lines.append(
            f'<text x="{left-18}" y="{y+cell_h/2+5}" text-anchor="end" class="label">lambda={lambda_a:.2f}</text>'
        )
        for t in range(cfg.time_steps):
            value = means[(lambda_a, t)]
            x = left + t * cell_w
            lines += [
                f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" rx="6" fill="{color(value)}"/>',
                f'<text x="{x+(cell_w-4)/2}" y="{y+(cell_h-4)/2+5}" text-anchor="middle" class="value">{value:.3f}</text>',
            ]
    lines += [
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle" class="label">0 = observe B, 1 = observe A</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = make_config(full=args.full)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    schedules = all_schedules(cfg.time_steps)
    random_probs = np.full(len(schedules), 1.0 / len(schedules))
    index_a = schedules.index(tuple([0] * cfg.time_steps))
    index_b = schedules.index(tuple([1] * cfg.time_steps))
    print("実験I-C：逐次観測方策PoCを開始します（GPU不要）")
    print(f"保存先: {output_dir}")

    rows: list[dict] = []
    policy_rows: list[dict] = []
    history_rows: list[dict] = []
    total = len(cfg.seeds) * len(cfg.lambdas)
    count = 0
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        train = make_dataset(cfg.train_size, cfg, rng)
        validation = make_dataset(cfg.validation_size, cfg, rng)
        test = make_dataset(cfg.test_size, cfg, rng)
        train_stats = build_stats(*train, schedules)
        validation_stats = build_stats(*validation, schedules)
        test_stats = build_stats(*test, schedules)
        full_train = build_full_stats(*train)
        full_test = build_full_stats(*test)
        full_estimator = fit_estimator(full_train, np.array([1.0]), cfg.ridge)

        for lambda_a in cfg.lambdas:
            oracle_index, oracle_estimator = select_oracle(
                train_stats, validation_stats, cfg, lambda_a, schedules
            )
            oracle_probs = deterministic_probs(oracle_index, len(schedules))
            oracle_da, oracle_db, oracle_obj = evaluate_distribution(
                oracle_estimator, test_stats, oracle_probs, cfg, lambda_a
            )
            oracle_schedule = format_schedule(schedules[oracle_index])
            add_result(
                rows, seed, lambda_a, "oracle", oracle_da, oracle_db, oracle_obj,
                oracle_schedule, oracle_schedule, 0.0
            )

            for method, index in [("A_fixed", index_a), ("B_fixed", index_b)]:
                probs = deterministic_probs(index, len(schedules))
                estimator = fit_estimator(train_stats, probs, cfg.ridge)
                da, db, obj = evaluate_distribution(
                    estimator, test_stats, probs, cfg, lambda_a
                )
                add_result(
                    rows, seed, lambda_a, method, da, db, obj,
                    format_schedule(schedules[index]), oracle_schedule, 0.0
                )

            random_estimator = fit_estimator(train_stats, random_probs, cfg.ridge)
            da, db, obj = evaluate_distribution(
                random_estimator, test_stats, random_probs, cfg, lambda_a
            )
            add_result(
                rows, seed, lambda_a, "random_fixed", da, db, obj,
                "stochastic", oracle_schedule, math.log(2.0)
            )

            frozen_policy, frozen_estimator, frozen_history = optimize_policy(
                train_stats, cfg, schedules, lambda_a, seed,
                frozen_estimator=random_estimator, method="learned_frozen"
            )
            frozen_probs = schedule_probabilities(frozen_policy, schedules)
            da, db, obj = evaluate_distribution(
                frozen_estimator, test_stats, frozen_probs, cfg, lambda_a
            )
            frozen_schedule = tuple(np.argmax(frozen_policy, axis=1).tolist())
            add_result(
                rows, seed, lambda_a, "learned_frozen", da, db, obj,
                format_schedule(frozen_schedule), oracle_schedule,
                policy_entropy(frozen_policy)
            )
            history_rows.extend(frozen_history)

            joint_policy, joint_estimator, joint_history = optimize_policy(
                train_stats, cfg, schedules, lambda_a, seed,
                frozen_estimator=None, method="learned_joint"
            )
            joint_probs = schedule_probabilities(joint_policy, schedules)
            da, db, obj = evaluate_distribution(
                joint_estimator, test_stats, joint_probs, cfg, lambda_a
            )
            joint_schedule = tuple(np.argmax(joint_policy, axis=1).tolist())
            add_result(
                rows, seed, lambda_a, "learned_joint", da, db, obj,
                format_schedule(joint_schedule), oracle_schedule,
                policy_entropy(joint_policy)
            )
            history_rows.extend(joint_history)

            full_da, full_db, full_obj = evaluate_distribution(
                full_estimator, full_test, np.array([1.0]), cfg, lambda_a
            )
            add_result(
                rows, seed, lambda_a, "full_observation", full_da, full_db,
                full_obj, "both", oracle_schedule, 0.0
            )

            for method, policy in [
                ("learned_frozen", frozen_policy), ("learned_joint", joint_policy)
            ]:
                for t in range(cfg.time_steps):
                    policy_rows.append(
                        {
                            "seed": seed,
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

    summary = aggregate(rows)
    message, criteria = judge(summary, cfg)
    elapsed = time.time() - start
    metadata = {
        "experiment": "Experiment I-C: sequential learned observation policy PoC",
        "version": "0.1",
        "config": asdict(cfg),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "criteria": criteria,
        "scope_note": (
            "The policy is an open-loop time-indexed schedule. The experiment does not yet "
            "implement state-conditioned attention, semantic invariance, or developmental persistence."
        ),
    }
    write_csv(output_dir / "raw_results.csv", rows)
    write_csv(output_dir / "summary_results.csv", summary)
    write_csv(output_dir / "policy_probabilities.csv", policy_rows)
    write_csv(output_dir / "learning_history.csv", history_rows)
    write_performance_svg(output_dir / "policy_performance.svg", summary, cfg)
    write_policy_svg(output_dir / "learned_policy.svg", policy_rows, cfg)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 実験I-C・逐次観測方策PoC 実行結果",
        "",
        message,
        "",
        "## 自動判定",
        "",
        f"- 全観測条件（lambda=0.5）: {criteria['full_observation_D']:.4f}",
        f"- Oracle逐次観測（lambda=0.5）: {criteria['oracle_sequential_D_at_half']:.4f}",
        f"- 一時刻一観測による機会費用: {criteria['sequential_opportunity_cost']:.4f}",
        f"- 共同学習方策のOracleとの差（内点平均）: {criteria['joint_gap_to_oracle_interior']:.4f}",
        f"- 共同学習方策のランダムからの改善（内点平均）: {criteria['joint_gain_over_random_interior']:.4f}",
        f"- 表現固定方策のランダムからの改善（内点平均）: {criteria['frozen_gain_over_random_interior']:.4f}",
        f"- 共同更新の下流固定に対する改善（内点平均）: {criteria['joint_gain_over_frozen_interior']:.4f}",
        f"- 共同学習方策のOracleスケジュール一致率: {criteria['joint_schedule_match_interior']:.3f}",
        "",
        "## lambda=0.5 の比較",
        "",
        "| 方法 | 平均目的値 | D_A | D_B |",
        "|---|---:|---:|---:|",
        *[
            f"| {method} "
            f"| {summary_value(summary, 0.5, method, 'objective_mean'):.4f} "
            f"| {summary_value(summary, 0.5, method, 'D_A_mean'):.4f} "
            f"| {summary_value(summary, 0.5, method, 'D_B_mean'):.4f} |"
            for method in [
                "full_observation", "oracle", "A_fixed", "B_fixed",
                "random_fixed", "learned_frozen", "learned_joint"
            ]
        ],
        "",
        "## 最頻スケジュール（A/Bは各時刻の観測先）",
        "",
        "| lambda | Oracle | 表現固定で学習 | 共同更新 |",
        "|---:|:---:|:---:|:---:|",
        *[
            f"| {value:.2f} "
            f"| {modal_schedule(rows, value, 'oracle')} "
            f"| {modal_schedule(rows, value, 'learned_frozen')} "
            f"| {modal_schedule(rows, value, 'learned_joint')} |"
            for value in cfg.lambdas
        ],
        "",
        "## 解釈の限界",
        "",
        "各時刻の方策は現在状態に条件づけられず、時刻だけに依存する開ループ方策です。",
        "共同更新は、方策分布に対する最適線形推定器を反復ごとに当て直す最小循環です。",
        "この線形環境では、ランダム方策下で学習した推定器でも各チャネルを十分に復号でき、共同更新の追加利得はほぼゼロでした。これは循環的学習の必要性を示す結果ではありません。",
        "人間の注意、意味を持つ心理／客観の不変性、発達的な志向性を支持する結果ではありません。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text("\n".join(report), encoding="utf-8")
    print()
    print(message)
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, policy_performance.svg, learned_policy.svg")


if __name__ == "__main__":
    main()
