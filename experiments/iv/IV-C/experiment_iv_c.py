#!/usr/bin/env python3
"""実験IV-C: 形成済み表現を、均衡更新下の取得精度だけで動かせるか。

同一seedで作ったA形成済み・中立・B形成済みcheckpointを、それぞれ
A精密・均衡・B精密取得へ複製する。継続中の更新重みは全条件0.5/0.5に固定する。
主判定は、形成済み状態から反対側を精密に取得したとき、同方向取得よりも
表現方向sが反対側へ大きく移るかである。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


INITIAL_STATES = ("A_formed", "neutral", "B_formed")
ACQUISITION_LEVELS = ("A_precise", "balanced", "B_precise")
FORMATION_WEIGHTS = {"A_formed": 0.85, "B_formed": 0.15}


@dataclass
class Config:
    train_size: int = 3072
    probe_size: int = 768
    test_size: int = 1536
    source_dim: int = 3
    hidden_dim: int = 32
    bottleneck: int = 2
    correlation: float = 0.35
    low_noise: float = 1.20
    high_noise: float = 0.05
    initial_noise: float = 0.15
    batch_size: int = 256
    initial_steps: int = 100
    branch_formation_steps: int = 500
    trajectory_steps: int = 1000
    early_checkpoint_until: int = 100
    early_checkpoint_interval: int = 5
    checkpoint_interval: int = 25
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    probe_ridge: float = 1e-3
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 20260902


@dataclass
class SignalSplit:
    x_a_precise: torch.Tensor
    x_b_precise: torch.Tensor
    x_both_high: torch.Tensor
    x_both_low: torch.Tensor
    x_initial: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirm"], default="smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--trajectory-steps", type=int, default=None)
    parser.add_argument("--preregistration", default=None,
                        help="確認段階の固定判定を記したJSON。confirmでは必須。")
    return parser.parse_args()


def config_for_stage(stage: str) -> Config:
    cfg = Config()
    if stage == "smoke":
        cfg.train_size = 1024
        cfg.probe_size = 384
        cfg.test_size = 512
        cfg.hidden_dim = 24
        cfg.batch_size = 128
        cfg.initial_steps = 40
        cfg.branch_formation_steps = 100
        cfg.trajectory_steps = 150
        cfg.bootstrap_samples = 1000
    elif stage == "pilot":
        cfg.bootstrap_samples = 3000
    return cfg


def choose_device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが見つかりません。")
        return torch.device("cuda")
    if request == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardized_noisy(value: np.ndarray, noise: np.ndarray, sigma: float):
    return (value + sigma * noise) / math.sqrt(1.0 + sigma * sigma)


def make_splits(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
    rotation, _ = np.linalg.qr(raw)

    def sample(n: int) -> SignalSplit:
        a = rng.normal(size=(n, cfg.source_dim))
        private_b = rng.normal(size=(n, cfg.source_dim))
        b = cfg.correlation * a + math.sqrt(1.0 - cfg.correlation**2) * private_b
        low_a = standardized_noisy(a, rng.normal(size=a.shape), cfg.low_noise)
        low_b = standardized_noisy(b, rng.normal(size=b.shape), cfg.low_noise)
        high_a = standardized_noisy(a, rng.normal(size=a.shape), cfg.high_noise)
        high_b = standardized_noisy(b, rng.normal(size=b.shape), cfg.high_noise)
        initial_a = standardized_noisy(a, rng.normal(size=a.shape), cfg.initial_noise)
        initial_b = standardized_noisy(b, rng.normal(size=b.shape), cfg.initial_noise)

        def mixed(first, second):
            joined = np.concatenate([first, second], axis=1) @ rotation
            return torch.from_numpy(joined.astype("float32"))

        return SignalSplit(
            x_a_precise=mixed(high_a, low_b),
            x_b_precise=mixed(low_a, high_b),
            x_both_high=mixed(high_a, high_b),
            x_both_low=mixed(low_a, low_b),
            x_initial=mixed(initial_a, initial_b),
            a=torch.from_numpy(a.astype("float32")),
            b=torch.from_numpy(b.astype("float32")),
        )

    return sample(cfg.train_size), sample(cfg.probe_size), sample(cfg.test_size)


class SharedRepresentation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck), nn.Tanh(),
        )
        self.head_a = nn.Linear(bottleneck, output_dim)
        self.head_b = nn.Linear(bottleneck, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.head_a(z), self.head_b(z)


def make_model(cfg: Config, device):
    return SharedRepresentation(cfg.source_dim * 2, cfg.hidden_dim, cfg.bottleneck,
                                cfg.source_dim).to(device)


def train_weighted(model, optimizer, x, a, b, weight_a: float):
    optimizer.zero_grad(set_to_none=True)
    _, pred_a, pred_b = model(x)
    d_a = torch.mean((pred_a - a) ** 2)
    d_b = torch.mean((pred_b - b) ** 2)
    loss = weight_a * d_a + (1.0 - weight_a) * d_b
    loss.backward()
    optimizer.step()


@torch.no_grad()
def ridge_error(z_train, y_train, z_test, y_test, ridge: float):
    design = torch.cat([z_train, torch.ones((z_train.shape[0], 1), device=z_train.device)], dim=1)
    design_test = torch.cat([z_test, torch.ones((z_test.shape[0], 1), device=z_test.device)], dim=1)
    eye = torch.eye(design.shape[1], device=z_train.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y_train)
    pred = design_test @ weights
    return (torch.mean((pred - y_test) ** 2) /
            (torch.var(y_test, unbiased=False) + 1e-12)).item()


@torch.no_grad()
def representation_metrics(model, probe_x, test_x, probe, test, cfg: Config, device):
    z_probe = model.encoder(probe_x.to(device))
    z_test = model.encoder(test_x.to(device))
    d_a = ridge_error(z_probe, probe.a.to(device), z_test, test.a.to(device), cfg.probe_ridge)
    d_b = ridge_error(z_probe, probe.b.to(device), z_test, test.b.to(device), cfg.probe_ridge)
    s = (d_b - d_a) / (d_a + d_b + 1e-12)
    return d_a, d_b, s


@torch.no_grad()
def raw_information_error(x_probe, y_probe, x_test, y_test, ridge: float):
    return ridge_error(x_probe, y_probe, x_test, y_test, ridge)


def clone_optimizer(model, cfg: Config, state):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(state))
    return optimizer


def balanced_schedule(steps: int, seed: int, stream: int):
    if steps % 2:
        raise ValueError("均衡取得を同数にするため更新回数は偶数にしてください。")
    schedule = np.asarray([0] * (steps // 2) + [1] * (steps // 2))
    np.random.default_rng(stream + seed).shuffle(schedule)
    return schedule


def acquisition_tensor(train: SignalSplit, acquisition: str, balanced_choice: int):
    if acquisition == "A_precise":
        return train.x_a_precise
    if acquisition == "B_precise":
        return train.x_b_precise
    return train.x_a_precise if balanced_choice == 0 else train.x_b_precise


def prepare_initial_states(cfg: Config, seed: int, train: SignalSplit, device):
    """中立checkpointと、そこから更新非対称だけで作るA/B形成済みcheckpoint。"""
    set_seed(seed)
    base = make_model(cfg, device)
    optimizer = torch.optim.Adam(base.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    n = train.a.shape[0]
    rng = np.random.default_rng(100_000 + seed)
    for _ in range(cfg.initial_steps):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(base, optimizer, train.x_initial.to(device)[idx],
                       train.a.to(device)[idx], train.b.to(device)[idx], 0.5)
    states = {"neutral": (copy.deepcopy(base.state_dict()), copy.deepcopy(optimizer.state_dict()))}

    for state_name, weight_a in FORMATION_WEIGHTS.items():
        model = make_model(cfg, device)
        model.load_state_dict(states["neutral"][0])
        branch_optimizer = clone_optimizer(model, cfg, states["neutral"][1])
        branch_rng = np.random.default_rng(250_000 + seed)
        for _ in range(cfg.branch_formation_steps):
            idx = torch.from_numpy(branch_rng.integers(0, n, size=cfg.batch_size)).to(device)
            train_weighted(model, branch_optimizer, train.x_initial.to(device)[idx],
                           train.a.to(device)[idx], train.b.to(device)[idx], weight_a)
        states[state_name] = (copy.deepcopy(model.state_dict()),
                              copy.deepcopy(branch_optimizer.state_dict()))
    return states


def checkpoints(cfg: Config):
    points = set(range(0, min(cfg.early_checkpoint_until, cfg.trajectory_steps) + 1,
                           cfg.early_checkpoint_interval))
    points.update(range(cfg.checkpoint_interval, cfg.trajectory_steps + 1,
                        cfg.checkpoint_interval))
    points.add(cfg.trajectory_steps)
    return points


def run_cell(cfg: Config, seed: int, initial_state: str, acquisition: str,
             model_state, optimizer_state, train, probe, test, device):
    model = make_model(cfg, device)
    model.load_state_dict(model_state)
    optimizer = clone_optimizer(model, cfg, optimizer_state)
    batch_rng = np.random.default_rng(300_000 + seed)
    schedule = balanced_schedule(cfg.trajectory_steps, seed, 400_000)
    n = train.a.shape[0]
    points = checkpoints(cfg)
    rows = []

    def record(step: int):
        h_da, h_db, h_s = representation_metrics(
            model, probe.x_both_high, test.x_both_high, probe, test, cfg, device)
        l_da, l_db, l_s = representation_metrics(
            model, probe.x_both_low, test.x_both_low, probe, test, cfg, device)
        rows.append({
            "seed": seed, "initial_state": initial_state, "acquisition": acquisition,
            "step": step, "high_D_A": h_da, "high_D_B": h_db, "s_high": h_s,
            "high_error_sum": h_da + h_db,
            "low_D_A": l_da, "low_D_B": l_db, "s_low": l_s,
            "low_error_sum": l_da + l_db,
        })

    record(0)
    for step in range(1, cfg.trajectory_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        x = acquisition_tensor(train, acquisition, int(schedule[step - 1]))
        train_weighted(model, optimizer, x.to(device)[idx],
                       train.a.to(device)[idx], train.b.to(device)[idx], 0.5)
        if step in points:
            record(step)
    return rows


def sensor_diagnostics(cfg: Config, probe, test, device):
    pa, pb = probe.a.to(device), probe.b.to(device)
    ta, tb = test.a.to(device), test.b.to(device)
    xa_p, xa_t = probe.x_a_precise.to(device), test.x_a_precise.to(device)
    xb_p, xb_t = probe.x_b_precise.to(device), test.x_b_precise.to(device)
    return {
        "A_error_A_precise": raw_information_error(xa_p, pa, xa_t, ta, cfg.probe_ridge),
        "A_error_B_precise": raw_information_error(xb_p, pa, xb_t, ta, cfg.probe_ridge),
        "B_error_A_precise": raw_information_error(xa_p, pb, xa_t, tb, cfg.probe_ridge),
        "B_error_B_precise": raw_information_error(xb_p, pb, xb_t, tb, cfg.probe_ridge),
    }


def first_zero_crossing(rows, initial_state: str):
    if initial_state == "neutral":
        return None
    for row in sorted(rows, key=lambda item: item["step"]):
        if row["step"] == 0:
            continue
        if initial_state == "A_formed" and row["s_high"] <= 0:
            return row["step"]
        if initial_state == "B_formed" and row["s_high"] >= 0:
            return row["step"]
    return None


def summarize_cell(rows, cfg: Config):
    ordered = sorted(rows, key=lambda row: row["step"])
    initial, final = ordered[0], ordered[-1]
    crossing = first_zero_crossing(ordered, initial["initial_state"])
    summary = {
        "initial_s_high": initial["s_high"], "final_s_high": final["s_high"],
        "change_s_high": final["s_high"] - initial["s_high"],
        "final_abs_s_high": abs(final["s_high"]),
        "final_high_error_sum": final["high_error_sum"],
        "initial_s_low": initial["s_low"], "final_s_low": final["s_low"],
        "change_s_low": final["s_low"] - initial["s_low"],
        "final_abs_s_low": abs(final["s_low"]),
        "final_low_error_sum": final["low_error_sum"],
        "zero_crossed": int(crossing is not None),
        "zero_crossing_step": crossing if crossing is not None else cfg.trajectory_steps + 1,
        "last_interval_change_s_high": final["s_high"] - ordered[-2]["s_high"],
    }
    for target in (100, 500):
        match = [row for row in ordered if row["step"] == target]
        if match:
            summary[f"s_high_at_{target}"] = match[0]["s_high"]
    return summary


def percentile_ci(values, rng, samples: int):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def aggregate(cell_rows, sensor_rows, cfg: Config):
    by_seed = {}
    for row in cell_rows:
        by_seed.setdefault(row["seed"], {})[(row["initial_state"], row["acquisition"])] = row
    seeds = sorted(by_seed)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    outcomes = [key for key in cell_rows[0] if key not in {"seed", "initial_state", "acquisition"}]
    cells = {}
    for state in INITIAL_STATES:
        for acquisition in ACQUISITION_LEVELS:
            label = f"state_{state}__acq_{acquisition}"
            cells[label] = {
                outcome: percentile_ci(
                    [by_seed[seed][(state, acquisition)][outcome] for seed in seeds],
                    rng, cfg.bootstrap_samples)
                for outcome in outcomes
            }

    def contrast(outcome, terms):
        values = []
        for seed in seeds:
            value = sum(coef * by_seed[seed][(state, acq)][outcome]
                        for coef, state, acq in terms)
            values.append(value)
        return percentile_ci(values, rng, cfg.bootstrap_samples)

    effects = {}
    for state in INITIAL_STATES:
        effects[f"acquisition_A_minus_B__{state}"] = {
            outcome: contrast(outcome, [(1, state, "A_precise"), (-1, state, "B_precise")])
            for outcome in outcomes
        }
    effects["formed_mean_acquisition_A_minus_B"] = {
        outcome: contrast(outcome, [
            (0.5, "A_formed", "A_precise"), (-0.5, "A_formed", "B_precise"),
            (0.5, "B_formed", "A_precise"), (-0.5, "B_formed", "B_precise")])
        for outcome in outcomes
    }
    effects["opposite_vs_aligned_reduction_in_initial_direction"] = {
        outcome: contrast(outcome, [
            (0.5, "A_formed", "A_precise"), (-0.5, "A_formed", "B_precise"),
            (-0.5, "B_formed", "B_precise"), (0.5, "B_formed", "A_precise")])
        for outcome in outcomes
    }
    effects["state_dependence_of_acquisition_effect_A_vs_neutral"] = {
        outcome: contrast(outcome, [
            (1, "A_formed", "A_precise"), (-1, "A_formed", "B_precise"),
            (-1, "neutral", "A_precise"), (1, "neutral", "B_precise")])
        for outcome in outcomes
    }
    effects["state_dependence_of_acquisition_effect_B_vs_neutral"] = {
        outcome: contrast(outcome, [
            (1, "B_formed", "A_precise"), (-1, "B_formed", "B_precise"),
            (-1, "neutral", "A_precise"), (1, "neutral", "B_precise")])
        for outcome in outcomes
    }

    sensor = {}
    for key in sensor_rows[0]:
        if key != "seed":
            sensor[key] = percentile_ci([row[key] for row in sensor_rows], rng, cfg.bootstrap_samples)
    sensor["A_precision_gain"] = percentile_ci(
        [row["A_error_B_precise"] - row["A_error_A_precise"] for row in sensor_rows],
        rng, cfg.bootstrap_samples)
    sensor["B_precision_gain"] = percentile_ci(
        [row["B_error_A_precise"] - row["B_error_B_precise"] for row in sensor_rows],
        rng, cfg.bootstrap_samples)
    return {"n_seeds": len(seeds), "cells": cells, "effects": effects, "sensor": sensor}


def evaluate_confirmatory_gates(result, prereg):
    formed = result["effects"]["formed_mean_acquisition_A_minus_B"]["final_s_high"]
    a_formed = result["effects"]["acquisition_A_minus_B__A_formed"]["final_s_high"]
    b_formed = result["effects"]["acquisition_A_minus_B__B_formed"]["final_s_high"]
    neutral = result["effects"]["acquisition_A_minus_B__neutral"]["final_s_high"]
    formed_low = result["effects"]["formed_mean_acquisition_A_minus_B"]["final_s_low"]
    a_start = result["cells"]["state_A_formed__acq_balanced"]["initial_s_high"]
    b_start = result["cells"]["state_B_formed__acq_balanced"]["initial_s_high"]
    a_gain = result["sensor"]["A_precision_gain"]
    b_gain = result["sensor"]["B_precision_gain"]
    a_to_b = result["cells"]["state_A_formed__acq_B_precise"]
    b_to_a = result["cells"]["state_B_formed__acq_A_precise"]
    return {
        "sensor_A": {"pass": a_gain["low"] > 0, "value": a_gain},
        "sensor_B": {"pass": b_gain["low"] > 0, "value": b_gain},
        "A_formed_start": {"pass": a_start["low"] >= prereg["formed_start_minimum"], "value": a_start},
        "B_formed_start": {"pass": b_start["high"] <= -prereg["formed_start_minimum"], "value": b_start},
        "primary_formed_acquisition_effect": {
            "pass": formed["low"] > 0 and formed["mean"] >= prereg["formed_acquisition_minimum"],
            "value": formed,
        },
        "A_formed_acquisition_effect": {"pass": a_formed["low"] > 0, "value": a_formed},
        "B_formed_acquisition_effect": {"pass": b_formed["low"] > 0, "value": b_formed},
        "A_formed_reaches_B_direction": {
            "pass": a_to_b["final_s_high"]["high"] <= -prereg["opposite_destination_minimum"],
            "value": a_to_b["final_s_high"],
        },
        "B_formed_reaches_A_direction": {
            "pass": b_to_a["final_s_high"]["low"] >= prereg["opposite_destination_minimum"],
            "value": b_to_a["final_s_high"],
        },
        "A_formed_crosses_zero": {
            "pass": a_to_b["zero_crossed"]["low"] >= prereg["minimum_crossing_rate"],
            "value": a_to_b["zero_crossed"],
        },
        "B_formed_crosses_zero": {
            "pass": b_to_a["zero_crossed"]["low"] >= prereg["minimum_crossing_rate"],
            "value": b_to_a["zero_crossed"],
        },
        "neutral_positive_control": {
            "pass": neutral["low"] > 0 and neutral["mean"] >= prereg["neutral_control_minimum"],
            "value": neutral,
        },
        "low_resolution_same_direction": {"pass": formed_low["low"] > 0, "value": formed_low},
    }


def validate_preregistration(prereg, cfg: Config, seeds):
    observed = {
        "seeds": seeds,
        "initial_steps": cfg.initial_steps,
        "branch_formation_steps": cfg.branch_formation_steps,
        "trajectory_steps": cfg.trajectory_steps,
        "low_noise": cfg.low_noise,
        "high_noise": cfg.high_noise,
        "initial_noise": cfg.initial_noise,
        "formation_weights": FORMATION_WEIGHTS,
        "branch_formation_acquisition": "both A and B observed with noise SD 0.15",
        "trajectory_update_weight_A": 0.5,
        "initial_states": list(INITIAL_STATES),
        "acquisition_levels": list(ACQUISITION_LEVELS),
    }
    if observed != prereg["fixed_design"]:
        raise ValueError("確認設定が事前登録と一致しません。\n" +
                         json.dumps({"expected": prereg["fixed_design"], "observed": observed},
                                    ensure_ascii=False, indent=2))


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_text(stage, result, runtime, gates):
    lines = [
        f"# 実験IV-C {stage} 結果", "", "## この実験が答える問い", "",
        "形成済み表現を出発点にしても、更新をA/B均衡に保ったまま、"
        "どちらを精密に取得するかだけで表現方向を動かせるか。", "",
        "## 共通高精度測定", "",
        "| 出発状態 | A精密 | 均衡取得 | B精密 |", "|---|---:|---:|---:|",
    ]
    for state in INITIAL_STATES:
        values = [result["cells"][f"state_{state}__acq_{acq}"]["final_s_high"]["mean"]
                  for acq in ACQUISITION_LEVELS]
        lines.append(f"| {state} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    formed = result["effects"]["formed_mean_acquisition_A_minus_B"]["final_s_high"]
    neutral = result["effects"]["acquisition_A_minus_B__neutral"]["final_s_high"]
    lines += ["", "## 中心比較", "",
              f"- 形成済み2状態で平均したA精密－B精密: {formed['mean']:.4f} "
              f"(95%区間 [{formed['low']:.4f}, {formed['high']:.4f}])",
              f"- 中立状態のA精密－B精密: {neutral['mean']:.4f} "
              f"(95%区間 [{neutral['low']:.4f}, {neutral['high']:.4f}])"]
    if gates is not None:
        lines += ["", "## 事前判定", ""]
        lines.extend(f"- {'PASS' if item['pass'] else 'FAIL'}: {name}" for name, item in gates.items())
    lines += ["", "## 適用範囲", "",
              "この結果が直接示す範囲は、相関する二つの三次元信号を二次元の共有表現へ"
              "圧縮して学ぶ人工系である。形成済み状態は更新重みの非対称で作り、継続中の"
              "更新重みは全条件で0.5/0.5に固定した。", "", f"- 実行時間: {runtime:.1f}秒"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    if args.trajectory_steps is not None:
        cfg.trajectory_steps = args.trajectory_steps
    defaults = {"smoke": (7300, 4), "pilot": (7600, 16), "confirm": (7700, 40)}
    seed_start, seed_count = defaults[args.stage]
    seed_start = args.seed_start if args.seed_start is not None else seed_start
    seed_count = args.seed_count if args.seed_count is not None else seed_count
    seeds = list(range(seed_start, seed_start + seed_count))
    output_dir = Path(args.output_dir or f"results_{args.stage}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    prereg = None
    if args.stage == "confirm":
        if not args.preregistration:
            raise ValueError("confirmには--preregistrationが必要です。")
        prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
        validate_preregistration(prereg, cfg, seeds)

    started = time.time()
    timecourse_rows, cell_rows, sensor_rows = [], [], []
    for index, seed in enumerate(seeds, 1):
        train, probe, test = make_splits(cfg, seed)
        states = prepare_initial_states(cfg, seed, train, device)
        sensor_rows.append({"seed": seed, **sensor_diagnostics(cfg, probe, test, device)})
        for initial_state in INITIAL_STATES:
            model_state, optimizer_state = states[initial_state]
            for acquisition in ACQUISITION_LEVELS:
                rows = run_cell(cfg, seed, initial_state, acquisition,
                                model_state, optimizer_state, train, probe, test, device)
                timecourse_rows.extend(rows)
                cell_rows.append({"seed": seed, "initial_state": initial_state,
                                  "acquisition": acquisition, **summarize_cell(rows, cfg)})
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    expected_cells = len(seeds) * len(INITIAL_STATES) * len(ACQUISITION_LEVELS)
    expected_timecourse = expected_cells * len(checkpoints(cfg))
    if len(cell_rows) != expected_cells or len(timecourse_rows) != expected_timecourse:
        raise RuntimeError(f"不完全な結果: cells={len(cell_rows)}/{expected_cells}, "
                           f"timecourse={len(timecourse_rows)}/{expected_timecourse}")
    aggregate_result = aggregate(cell_rows, sensor_rows, cfg)
    gates = evaluate_confirmatory_gates(aggregate_result, prereg) if prereg else None
    runtime = time.time() - started
    result = {
        "experiment": "IV-C", "stage": args.stage,
        "question": "Can acquisition precision redirect a formed representation under balanced updates?",
        "design": {
            "seeds": seeds, "same_checkpoint_cloned_across_acquisition_within_state": True,
            "same_batch_stream_within_seed": True,
            "formed_states_created_by_update_weight_asymmetry": FORMATION_WEIGHTS,
            "formed_state_acquisition": "both A and B at initial_noise 0.15, matching IV-A formation",
            "trajectory_update_weight_A": 0.5,
            "initial_states": list(INITIAL_STATES),
            "acquisition_levels": list(ACQUISITION_LEVELS),
            "balanced_acquisition": "exactly half A-precise and half B-precise in randomized order",
            "primary_probe": "common both-high-precision probe",
            "secondary_probe": "common both-low-precision probe",
            "statistical_unit": "seed",
            "completeness": {"cells": len(cell_rows), "expected_cells": expected_cells,
                             "timecourse_rows": len(timecourse_rows),
                             "expected_timecourse_rows": expected_timecourse, "pass": True},
        },
        "config": asdict(cfg), "aggregate": aggregate_result,
        "confirmatory_gates": gates, "runtime_seconds": runtime,
    }
    write_csv(output_dir / "timecourse.csv", timecourse_rows)
    write_csv(output_dir / "cell_summary.csv", cell_rows)
    write_csv(output_dir / "sensor_diagnostics.csv", sensor_rows)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else None
    provenance = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(prereg_path) if prereg_path else None,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "device": str(device),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    report = report_text(args.stage, aggregate_result, runtime, gates)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
