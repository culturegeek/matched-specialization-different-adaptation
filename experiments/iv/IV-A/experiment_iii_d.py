#!/usr/bin/env python3
"""実験III-D: 情報取得×更新配分が形成済み表現の維持性と再配分可能性へ与える効果。

同一seed・同一形成済みcheckpointから、取得先（基準／代替）と更新先
（基準／代替）の2×2へ分岐する。追加センサなし・無作為センサを補助対照とする。
要求変更は学習系へ入力せず、同じ軌道を元の要求と代替要求の双方から評価する。
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


PRIMARY_ACQUISITION = ("reference", "alternative")
UPDATE_SIDES = ("reference", "alternative")
AUXILIARY_ACQUISITION = ("none", "random")
FORMED_BRANCHES = ("A", "B")


@dataclass
class Config:
    train_size: int = 3072
    probe_size: int = 768
    test_size: int = 1536
    source_dim: int = 3
    hidden_dim: int = 32
    bottleneck: int = 2
    correlation: float = 0.35
    low_noise: float = 0.70
    high_noise: float = 0.08
    formation_noise: float = 0.15
    batch_size: int = 256
    balanced_steps: int = 100
    formation_steps: int = 600
    trajectory_steps: int = 1000
    checkpoint_interval: int = 25
    early_checkpoint_until: int = 100
    early_checkpoint_interval: int = 5
    formation_weight: float = 0.80
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    probe_ridge: float = 1e-3
    branch_threshold: float = 0.10
    old_requirement_weight: float = 0.80
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 20260827


@dataclass
class SignalSplit:
    x_none: torch.Tensor
    x_a: torch.Tensor
    x_b: torch.Tensor
    x_both: torch.Tensor
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
    parser.add_argument("--formation-steps", type=int, default=None)
    parser.add_argument("--low-noise", type=float, default=None)
    parser.add_argument("--high-noise", type=float, default=None)
    parser.add_argument("--formation-weight", type=float, default=None)
    parser.add_argument("--branch-threshold", type=float, default=None)
    return parser.parse_args()


def config_for_stage(stage: str) -> Config:
    cfg = Config()
    if stage == "smoke":
        cfg.train_size = 1024
        cfg.probe_size = 384
        cfg.test_size = 512
        cfg.hidden_dim = 24
        cfg.batch_size = 128
        cfg.balanced_steps = 50
        cfg.formation_steps = 250
        cfg.trajectory_steps = 350
        cfg.checkpoint_interval = 25
        cfg.bootstrap_samples = 1000
    elif stage == "pilot":
        cfg.formation_steps = 500
        cfg.trajectory_steps = 800
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


def _standardized_noisy(value: np.ndarray, noise: np.ndarray, sigma: float):
    return (value + sigma * noise) / math.sqrt(1.0 + sigma * sigma)


def make_splits(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
    rotation, _ = np.linalg.qr(raw)

    def sample(n: int) -> SignalSplit:
        a = rng.normal(size=(n, cfg.source_dim))
        private_b = rng.normal(size=(n, cfg.source_dim))
        b = cfg.correlation * a + math.sqrt(1.0 - cfg.correlation**2) * private_b
        low_a = _standardized_noisy(a, rng.normal(size=a.shape), cfg.low_noise)
        low_b = _standardized_noisy(b, rng.normal(size=b.shape), cfg.low_noise)
        high_a = _standardized_noisy(a, rng.normal(size=a.shape), cfg.high_noise)
        high_b = _standardized_noisy(b, rng.normal(size=b.shape), cfg.high_noise)
        form_a = _standardized_noisy(a, rng.normal(size=a.shape), cfg.formation_noise)
        form_b = _standardized_noisy(b, rng.normal(size=b.shape), cfg.formation_noise)

        def mixed(first, second):
            return torch.from_numpy((np.concatenate([first, second], axis=1) @ rotation).astype("float32"))

        return SignalSplit(
            x_none=mixed(low_a, low_b),
            x_a=mixed(high_a, low_b),
            x_b=mixed(low_a, high_b),
            x_both=mixed(form_a, form_b),
            a=torch.from_numpy(a.astype("float32")),
            b=torch.from_numpy(b.astype("float32")),
        )

    return sample(cfg.train_size), sample(cfg.probe_size), sample(cfg.test_size)


class SharedRepresentation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck),
            nn.Tanh(),
        )
        self.head_a = nn.Linear(bottleneck, output_dim)
        self.head_b = nn.Linear(bottleneck, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.head_a(z), self.head_b(z)


def train_weighted(model, optimizer, x, a, b, weight_a: float):
    optimizer.zero_grad(set_to_none=True)
    _, pred_a, pred_b = model(x)
    d_a = torch.mean((pred_a - a) ** 2)
    d_b = torch.mean((pred_b - b) ** 2)
    loss = weight_a * d_a + (1.0 - weight_a) * d_b
    loss.backward()
    optimizer.step()


def train_selected(model, optimizer, x, a, b, select_a: bool):
    optimizer.zero_grad(set_to_none=True)
    _, pred_a, pred_b = model(x)
    loss = torch.mean((pred_a - a) ** 2) if select_a else torch.mean((pred_b - b) ** 2)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def ridge_error(z_train, y_train, z_test, y_test, ridge: float):
    one_train = torch.ones((z_train.shape[0], 1), device=z_train.device)
    one_test = torch.ones((z_test.shape[0], 1), device=z_test.device)
    design = torch.cat([z_train, one_train], dim=1)
    design_test = torch.cat([z_test, one_test], dim=1)
    eye = torch.eye(design.shape[1], device=z_train.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y_train)
    pred = design_test @ weights
    return (torch.mean((pred - y_test) ** 2) / (torch.var(y_test, unbiased=False) + 1e-12)).item()


@torch.no_grad()
def representation_metrics(model, probe: SignalSplit, test: SignalSplit, cfg: Config, device):
    z_probe = model.encoder(probe.x_none.to(device))
    z_test = model.encoder(test.x_none.to(device))
    a_probe, b_probe = probe.a.to(device), probe.b.to(device)
    a_test, b_test = test.a.to(device), test.b.to(device)
    d_a = ridge_error(z_probe, a_probe, z_test, a_test, cfg.probe_ridge)
    d_b = ridge_error(z_probe, b_probe, z_test, b_test, cfg.probe_ridge)
    s_ab = (d_b - d_a) / (d_a + d_b + 1e-12)
    return d_a, d_b, s_ab


@torch.no_grad()
def decoder_error(model, x, target, target_name: str):
    _, pred_a, pred_b = model(x)
    pred = pred_a if target_name == "A" else pred_b
    return (torch.mean((pred - target) ** 2) / (torch.var(target, unbiased=False) + 1e-12)).item()


def clone_optimizer(model, cfg: Config, optimizer_state):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    return optimizer


def make_model(cfg: Config, device):
    return SharedRepresentation(cfg.source_dim * 2, cfg.hidden_dim, cfg.bottleneck, cfg.source_dim).to(device)


def prepare_branches(cfg: Config, seed: int, train: SignalSplit, device):
    set_seed(seed)
    base = make_model(cfg, device)
    optimizer = torch.optim.Adam(base.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(100_000 + seed)
    n = train.a.shape[0]
    for _ in range(cfg.balanced_steps):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(base, optimizer, train.x_both.to(device)[idx], train.a.to(device)[idx], train.b.to(device)[idx], 0.5)
    initial_model = copy.deepcopy(base.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    branches = {}
    for reference in FORMED_BRANCHES:
        model = make_model(cfg, device)
        model.load_state_dict(initial_model)
        branch_optimizer = clone_optimizer(model, cfg, initial_optimizer)
        branch_rng = np.random.default_rng(200_000 + seed)
        weight_a = cfg.formation_weight if reference == "A" else 1.0 - cfg.formation_weight
        for _ in range(cfg.formation_steps):
            idx = torch.from_numpy(branch_rng.integers(0, n, size=cfg.batch_size)).to(device)
            train_weighted(
                model,
                branch_optimizer,
                train.x_both.to(device)[idx],
                train.a.to(device)[idx],
                train.b.to(device)[idx],
                weight_a,
            )
        branches[reference] = (copy.deepcopy(model.state_dict()), copy.deepcopy(branch_optimizer.state_dict()))
    return branches


def resolved_side(relative_side: str, reference: str):
    if relative_side == "reference":
        return reference
    if relative_side == "alternative":
        return "B" if reference == "A" else "A"
    return relative_side


def acquisition_tensor(train: SignalSplit, acquisition: str, reference: str, random_choose_a: bool):
    if acquisition == "none":
        return train.x_none
    if acquisition == "random":
        return train.x_a if random_choose_a else train.x_b
    absolute = resolved_side(acquisition, reference)
    return train.x_a if absolute == "A" else train.x_b


def run_cell(
    cfg: Config,
    seed: int,
    reference: str,
    acquisition: str,
    update: str,
    formed_model_state,
    formed_optimizer_state,
    train: SignalSplit,
    probe: SignalSplit,
    test: SignalSplit,
    device,
):
    model = make_model(cfg, device)
    model.load_state_dict(formed_model_state)
    optimizer = clone_optimizer(model, cfg, formed_optimizer_state)
    batch_rng = np.random.default_rng(300_000 + seed)
    sensor_rng = np.random.default_rng(400_000 + seed)
    n = train.a.shape[0]
    update_absolute = resolved_side(update, reference)
    checkpoints = set(range(0, min(cfg.early_checkpoint_until, cfg.trajectory_steps) + 1, cfg.early_checkpoint_interval))
    checkpoints.update(range(cfg.checkpoint_interval, cfg.trajectory_steps + 1, cfg.checkpoint_interval))
    checkpoints.add(cfg.trajectory_steps)
    rows = []

    def record(step: int):
        d_a, d_b, s_ab = representation_metrics(model, probe, test, cfg, device)
        if reference == "A":
            d_ref, d_alt, s_ref = d_a, d_b, s_ab
        else:
            d_ref, d_alt, s_ref = d_b, d_a, -s_ab
        old_loss = cfg.old_requirement_weight * d_ref + (1.0 - cfg.old_requirement_weight) * d_alt
        alt_loss = (1.0 - cfg.old_requirement_weight) * d_ref + cfg.old_requirement_weight * d_alt
        rows.append({
            "seed": seed,
            "reference": reference,
            "acquisition": acquisition,
            "update": update,
            "step": step,
            "probe_D_A": d_a,
            "probe_D_B": d_b,
            "s_ref": s_ref,
            "old_requirement_loss": old_loss,
            "alternative_requirement_loss": alt_loss,
        })

    record(0)
    for step in range(1, cfg.trajectory_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        random_choose_a = bool(sensor_rng.integers(0, 2))
        x_source = acquisition_tensor(train, acquisition, reference, random_choose_a).to(device)
        train_selected(
            model,
            optimizer,
            x_source[idx],
            train.a.to(device)[idx],
            train.b.to(device)[idx],
            update_absolute == "A",
        )
        if step in checkpoints:
            record(step)
    return rows


def sensor_diagnostics(model_state, cfg: Config, test: SignalSplit, device):
    model = make_model(cfg, device)
    model.load_state_dict(model_state)
    target_a, target_b = test.a.to(device), test.b.to(device)
    x_none = test.x_none.to(device)
    x_a = test.x_a.to(device)
    x_b = test.x_b.to(device)
    return {
        "A_gain": decoder_error(model, x_none, target_a, "A") - decoder_error(model, x_a, target_a, "A"),
        "B_gain": decoder_error(model, x_none, target_b, "B") - decoder_error(model, x_b, target_b, "B"),
    }


def first_transition(rows, cfg: Config):
    for row in sorted(rows, key=lambda item: item["step"]):
        if row["step"] > 0 and row["s_ref"] <= -cfg.branch_threshold:
            return row["step"], True
    return cfg.trajectory_steps, False


def summarize_cell(rows, cfg: Config):
    ordered = sorted(rows, key=lambda item: item["step"])
    initial, final = ordered[0], ordered[-1]
    transition_time, event = first_transition(ordered, cfg)
    steps = np.asarray([row["step"] for row in ordered], dtype=float)
    alt_losses = np.asarray([row["alternative_requirement_loss"] for row in ordered], dtype=float)
    old_losses = np.asarray([row["old_requirement_loss"] for row in ordered], dtype=float)
    alt_auc = float(np.trapezoid(alt_losses, steps) / cfg.trajectory_steps)
    old_auc = float(np.trapezoid(old_losses, steps) / cfg.trajectory_steps)
    return {
        "initial_s_ref": initial["s_ref"],
        "final_s_ref": final["s_ref"],
        "s_change": final["s_ref"] - initial["s_ref"],
        "retained": float(final["s_ref"] >= cfg.branch_threshold),
        "transition_event": float(event),
        "transition_time_restricted": float(transition_time),
        "old_loss_auc": old_auc,
        "alternative_loss_auc": alt_auc,
        "final_old_loss": final["old_requirement_loss"],
        "final_alternative_loss": final["alternative_requirement_loss"],
    }


def percentile_ci(values, rng, samples: int):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def interaction(cell_map, outcome: str):
    rr = cell_map[("reference", "reference")][outcome]
    ra = cell_map[("reference", "alternative")][outcome]
    ar = cell_map[("alternative", "reference")][outcome]
    aa = cell_map[("alternative", "alternative")][outcome]
    return rr - ra - ar + aa


def main_effect(cell_map, outcome: str, factor: str):
    if factor == "acquisition":
        ref = 0.5 * (cell_map[("reference", "reference")][outcome] + cell_map[("reference", "alternative")][outcome])
        alt = 0.5 * (cell_map[("alternative", "reference")][outcome] + cell_map[("alternative", "alternative")][outcome])
    else:
        ref = 0.5 * (cell_map[("reference", "reference")][outcome] + cell_map[("alternative", "reference")][outcome])
        alt = 0.5 * (cell_map[("reference", "alternative")][outcome] + cell_map[("alternative", "alternative")][outcome])
    return ref - alt


def simple_effect(cell_map, outcome: str, factor: str, level: str):
    if factor == "acquisition":
        return cell_map[("reference", level)][outcome] - cell_map[("alternative", level)][outcome]
    return cell_map[(level, "reference")][outcome] - cell_map[(level, "alternative")][outcome]


def aggregate(seed_cell_rows, sensor_rows, cfg: Config):
    # A/Bの形成枝をseed内で平均し、seedを統計単位とする。
    per_seed = {}
    for row in seed_cell_rows:
        key = (row["acquisition"], row["update"])
        per_seed.setdefault(row["seed"], {}).setdefault(key, []).append(row)
    seed_cells = {}
    for seed, cells in per_seed.items():
        seed_cells[seed] = {}
        for key, branch_rows in cells.items():
            outcomes = [name for name in branch_rows[0] if name not in {"seed", "reference", "acquisition", "update"}]
            seed_cells[seed][key] = {name: float(np.mean([row[name] for row in branch_rows])) for name in outcomes}

    rng = np.random.default_rng(cfg.bootstrap_seed)
    outcomes = [
        "final_s_ref",
        "retained",
        "transition_event",
        "transition_time_restricted",
        "old_loss_auc",
        "alternative_loss_auc",
    ]
    effects = {}
    for outcome in outcomes:
        interaction_values = [interaction(seed_cells[seed], outcome) for seed in sorted(seed_cells)]
        acquisition_values = [main_effect(seed_cells[seed], outcome, "acquisition") for seed in sorted(seed_cells)]
        update_values = [main_effect(seed_cells[seed], outcome, "update") for seed in sorted(seed_cells)]
        effects[outcome] = {
            "interaction": percentile_ci(interaction_values, rng, cfg.bootstrap_samples),
            "acquisition_reference_minus_alternative": percentile_ci(acquisition_values, rng, cfg.bootstrap_samples),
            "update_reference_minus_alternative": percentile_ci(update_values, rng, cfg.bootstrap_samples),
            "acquisition_effect_given_reference_update": percentile_ci(
                [simple_effect(seed_cells[seed], outcome, "acquisition", "reference") for seed in sorted(seed_cells)],
                rng,
                cfg.bootstrap_samples,
            ),
            "acquisition_effect_given_alternative_update": percentile_ci(
                [simple_effect(seed_cells[seed], outcome, "acquisition", "alternative") for seed in sorted(seed_cells)],
                rng,
                cfg.bootstrap_samples,
            ),
            "update_effect_given_reference_acquisition": percentile_ci(
                [simple_effect(seed_cells[seed], outcome, "update", "reference") for seed in sorted(seed_cells)],
                rng,
                cfg.bootstrap_samples,
            ),
            "update_effect_given_alternative_acquisition": percentile_ci(
                [simple_effect(seed_cells[seed], outcome, "update", "alternative") for seed in sorted(seed_cells)],
                rng,
                cfg.bootstrap_samples,
            ),
        }

    cells_summary = {}
    all_keys = sorted(next(iter(seed_cells.values())).keys())
    for key in all_keys:
        label = f"acq_{key[0]}__upd_{key[1]}"
        cells_summary[label] = {}
        for outcome in outcomes:
            values = [seed_cells[seed][key][outcome] for seed in sorted(seed_cells)]
            cells_summary[label][outcome] = percentile_ci(values, rng, cfg.bootstrap_samples)

    formation = {}
    for reference in FORMED_BRANCHES:
        values = [
            row["initial_s_ref"]
            for row in seed_cell_rows
            if row["reference"] == reference and row["acquisition"] == "reference" and row["update"] == "reference"
        ]
        formation[reference] = percentile_ci(values, rng, cfg.bootstrap_samples)

    sensor_by_seed = {}
    for row in sensor_rows:
        sensor_by_seed.setdefault(row["seed"], []).append(row)
    sensor = {}
    for name in ("A_gain", "B_gain"):
        values = [float(np.mean([row[name] for row in sensor_by_seed[seed]])) for seed in sorted(sensor_by_seed)]
        sensor[name] = percentile_ci(values, rng, cfg.bootstrap_samples)
    return {
        "n_seeds": len(seed_cells),
        "formation_s_ref": formation,
        "sensor_gain": sensor,
        "primary_cells": cells_summary,
        "factorial_effects": effects,
    }


def confirmatory_gates(aggregate_result):
    cells = aggregate_result["primary_cells"]
    retention_reference_update = 0.5 * (
        cells["acq_reference__upd_reference"]["retained"]["mean"]
        + cells["acq_alternative__upd_reference"]["retained"]["mean"]
    )
    transition_alternative_update = 0.5 * (
        cells["acq_reference__upd_alternative"]["transition_event"]["mean"]
        + cells["acq_alternative__upd_alternative"]["transition_event"]["mean"]
    )
    acquisition_time_effect = aggregate_result["factorial_effects"]["transition_time_restricted"][
        "acquisition_effect_given_alternative_update"
    ]
    return {
        "C1_A_sensor_gain": {
            "pass": aggregate_result["sensor_gain"]["A_gain"]["low"] > 0.10,
            "value": aggregate_result["sensor_gain"]["A_gain"],
        },
        "C1_B_sensor_gain": {
            "pass": aggregate_result["sensor_gain"]["B_gain"]["low"] > 0.10,
            "value": aggregate_result["sensor_gain"]["B_gain"],
        },
        "C1_A_formed_branch": {
            "pass": aggregate_result["formation_s_ref"]["A"]["low"] > 0.10,
            "value": aggregate_result["formation_s_ref"]["A"],
        },
        "C1_B_formed_branch": {
            "pass": aggregate_result["formation_s_ref"]["B"]["low"] > 0.10,
            "value": aggregate_result["formation_s_ref"]["B"],
        },
        "C2_reference_update_retention": {
            "pass": retention_reference_update >= 0.90,
            "value": retention_reference_update,
        },
        "C2_alternative_update_transition": {
            "pass": transition_alternative_update >= 0.90,
            "value": transition_alternative_update,
        },
        "C3_alternative_acquisition_accelerates_transition": {
            "pass": acquisition_time_effect["mean"] >= 10.0 and acquisition_time_effect["low"] > 0.0,
            "value": acquisition_time_effect,
        },
    }


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


def report_text(stage, aggregate_result, runtime, gates=None):
    lines = [
        f"# 実験III-D {stage} 結果",
        "",
        "## 操作確認",
        "",
        f"- A追加センサによるA復号誤差改善: {aggregate_result['sensor_gain']['A_gain']['mean']:.4f}",
        f"- B追加センサによるB復号誤差改善: {aggregate_result['sensor_gain']['B_gain']['mean']:.4f}",
        f"- A形成枝の開始時 s_ref: {aggregate_result['formation_s_ref']['A']['mean']:.4f}",
        f"- B形成枝の開始時 s_ref: {aggregate_result['formation_s_ref']['B']['mean']:.4f}",
        "",
        "## 主2×2",
        "",
        "| 条件 | 最終 s_ref | 枝保持率 | 代替枝到達率 | 制限付き到達時間 | 代替要求損失AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in aggregate_result["primary_cells"].items():
        lines.append(
            f"| {label} | {values['final_s_ref']['mean']:.4f} | {values['retained']['mean']:.3f} | "
            f"{values['transition_event']['mean']:.3f} | {values['transition_time_restricted']['mean']:.1f} | "
            f"{values['alternative_loss_auc']['mean']:.4f} |"
        )
    lines += ["", "## 取得×更新の交互作用", ""]
    for outcome, values in aggregate_result["factorial_effects"].items():
        item = values["interaction"]
        lines.append(f"- {outcome}: {item['mean']:.4f} (95%区間 [{item['low']:.4f}, {item['high']:.4f}])")
    if gates is not None:
        lines += ["", "## 事前判定", ""]
        for name, item in gates.items():
            lines.append(f"- {'PASS' if item['pass'] else 'FAIL'}: {name}")
    lines += [
        "",
        "## 読み方",
        "",
        "同じ軌道を、形成済み枝を保つ維持性と、代替枝へ移る再配分可能性の双方から評価した。",
        "代替要求は学習系へ入力しておらず、要求変更を自律的に検知する能力は測っていない。",
        "",
        f"- 実行時間: {runtime:.1f}秒",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    for field in (
        "trajectory_steps",
        "formation_steps",
        "low_noise",
        "high_noise",
        "formation_weight",
        "branch_threshold",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)
    defaults = {
        "smoke": (6000, 4),
        "pilot": (6100, 12),
        "confirm": (6200, 40),
    }
    seed_start, seed_count = defaults[args.stage]
    if args.seed_start is not None:
        seed_start = args.seed_start
    if args.seed_count is not None:
        seed_count = args.seed_count
    seeds = list(range(seed_start, seed_start + seed_count))
    output_dir = Path(args.output_dir or f"results_{args.stage}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    started = time.time()
    timecourse_rows = []
    cell_rows = []
    sensor_rows = []
    for index, seed in enumerate(seeds, 1):
        train, probe, test = make_splits(cfg, seed)
        branches = prepare_branches(cfg, seed, train, device)
        for reference in FORMED_BRANCHES:
            model_state, optimizer_state = branches[reference]
            diagnostic = sensor_diagnostics(model_state, cfg, test, device)
            sensor_rows.append({"seed": seed, "reference": reference, **diagnostic})
            for acquisition in PRIMARY_ACQUISITION + AUXILIARY_ACQUISITION:
                for update in UPDATE_SIDES:
                    rows = run_cell(
                        cfg,
                        seed,
                        reference,
                        acquisition,
                        update,
                        model_state,
                        optimizer_state,
                        train,
                        probe,
                        test,
                        device,
                    )
                    timecourse_rows.extend(rows)
                    cell_rows.append({
                        "seed": seed,
                        "reference": reference,
                        "acquisition": acquisition,
                        "update": update,
                        **summarize_cell(rows, cfg),
                    })
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    checkpoint_count = len(
        set(range(0, min(cfg.early_checkpoint_until, cfg.trajectory_steps) + 1, cfg.early_checkpoint_interval))
        | set(range(cfg.checkpoint_interval, cfg.trajectory_steps + 1, cfg.checkpoint_interval))
        | {cfg.trajectory_steps}
    )
    expected_timecourse = len(seeds) * len(FORMED_BRANCHES) * (len(PRIMARY_ACQUISITION) + len(AUXILIARY_ACQUISITION)) * len(UPDATE_SIDES) * checkpoint_count
    expected_cells = len(seeds) * len(FORMED_BRANCHES) * (len(PRIMARY_ACQUISITION) + len(AUXILIARY_ACQUISITION)) * len(UPDATE_SIDES)
    if len(timecourse_rows) != expected_timecourse or len(cell_rows) != expected_cells:
        raise RuntimeError(
            f"不完全な結果です: timecourse={len(timecourse_rows)}/{expected_timecourse}, "
            f"cells={len(cell_rows)}/{expected_cells}"
        )
    aggregate_result = aggregate(
        [row for row in cell_rows if row["acquisition"] in PRIMARY_ACQUISITION],
        sensor_rows,
        cfg,
    )
    runtime = time.time() - started
    gates = confirmatory_gates(aggregate_result) if args.stage == "confirm" else None
    result = {
        "experiment": "III-D",
        "stage": args.stage,
        "design": {
            "seeds": seeds,
            "same_formed_checkpoint_cloned": True,
            "same_batch_and_sensor_random_stream_within_seed": True,
            "primary_acquisition": list(PRIMARY_ACQUISITION),
            "update_sides": list(UPDATE_SIDES),
            "auxiliary_acquisition": list(AUXILIARY_ACQUISITION),
            "statistical_unit": "seed; A/B formed branches averaged within seed",
            "alternative_requirement_is_evaluation_only": True,
            "completeness": {
                "timecourse_rows": len(timecourse_rows),
                "expected_timecourse_rows": expected_timecourse,
                "cell_rows": len(cell_rows),
                "expected_cell_rows": expected_cells,
                "pass": True,
            },
        },
        "config": asdict(cfg),
        "aggregate": aggregate_result,
        "confirmatory_gates": gates,
        "runtime_seconds": runtime,
    }
    write_csv(output_dir / "timecourse.csv", timecourse_rows)
    write_csv(output_dir / "cell_summary.csv", cell_rows)
    write_csv(output_dir / "sensor_diagnostics.csv", sensor_rows)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    script_path = Path(__file__).resolve()
    preregistration_path = Path(__file__).resolve().parent / "preregistration_confirmatory.md"
    provenance = {
        "script_sha256": sha256(script_path),
        "preregistration_sha256": sha256(preregistration_path) if preregistration_path.exists() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_text(args.stage, aggregate_result, runtime, gates)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
