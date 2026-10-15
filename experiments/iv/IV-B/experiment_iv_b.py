#!/usr/bin/env python3
"""実験IV-B: 形成段階の取得情報と更新配分は、表現の形成方向をどう決めるか。

同一の未形成checkpointから、取得方向3水準×更新方向3水準へ分岐する。
主判定は、更新をA/B均衡に保ったまま、取得情報の精度だけを左右反転すると、
形成後の共有表現の方向も反転するかである。
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


ACQUISITION_LEVELS = ("A_precise", "balanced", "B_precise")
UPDATE_LEVELS = ("A_heavy", "balanced", "B_heavy")
UPDATE_WEIGHTS = {"A_heavy": 0.85, "balanced": 0.50, "B_heavy": 0.15}


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
    formation_steps: int = 500
    early_checkpoint_until: int = 100
    early_checkpoint_interval: int = 5
    checkpoint_interval: int = 25
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    probe_ridge: float = 1e-3
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 20260901


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
    parser.add_argument("--formation-steps", type=int, default=None)
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
        cfg.formation_steps = 100
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
        initial_a = _standardized_noisy(a, rng.normal(size=a.shape), cfg.initial_noise)
        initial_b = _standardized_noisy(b, rng.normal(size=b.shape), cfg.initial_noise)

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


def make_model(cfg: Config, device):
    return SharedRepresentation(cfg.source_dim * 2, cfg.hidden_dim, cfg.bottleneck, cfg.source_dim).to(device)


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
    one_train = torch.ones((z_train.shape[0], 1), device=z_train.device)
    one_test = torch.ones((z_test.shape[0], 1), device=z_test.device)
    design = torch.cat([z_train, one_train], dim=1)
    design_test = torch.cat([z_test, one_test], dim=1)
    eye = torch.eye(design.shape[1], device=z_train.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y_train)
    pred = design_test @ weights
    variance = torch.var(y_test, unbiased=False) + 1e-12
    return (torch.mean((pred - y_test) ** 2) / variance).item()


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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(state))
    return optimizer


def prepare_initial_state(cfg: Config, seed: int, train: SignalSplit, device):
    set_seed(seed)
    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(100_000 + seed)
    n = train.a.shape[0]
    for _ in range(cfg.initial_steps):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(
            model, optimizer, train.x_initial.to(device)[idx],
            train.a.to(device)[idx], train.b.to(device)[idx], 0.5,
        )
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def acquisition_schedule(cfg: Config, seed: int):
    if cfg.formation_steps % 2:
        raise ValueError("balanced取得を同数にするためformation_stepsは偶数にしてください。")
    schedule = np.asarray([0] * (cfg.formation_steps // 2) + [1] * (cfg.formation_steps // 2))
    np.random.default_rng(200_000 + seed).shuffle(schedule)
    return schedule


def checkpoints(cfg: Config):
    points = set(range(0, min(cfg.early_checkpoint_until, cfg.formation_steps) + 1,
                       cfg.early_checkpoint_interval))
    points.update(range(cfg.checkpoint_interval, cfg.formation_steps + 1, cfg.checkpoint_interval))
    points.add(cfg.formation_steps)
    return points


def run_cell(cfg: Config, seed: int, acquisition: str, update: str,
             initial_model, initial_optimizer, train, probe, test, device):
    model = make_model(cfg, device)
    model.load_state_dict(initial_model)
    optimizer = clone_optimizer(model, cfg, initial_optimizer)
    batch_rng = np.random.default_rng(300_000 + seed)
    schedule = acquisition_schedule(cfg, seed)
    n = train.a.shape[0]
    points = checkpoints(cfg)
    rows = []

    def record(step: int):
        h_da, h_db, h_s = representation_metrics(
            model, probe.x_both_high, test.x_both_high, probe, test, cfg, device)
        l_da, l_db, l_s = representation_metrics(
            model, probe.x_both_low, test.x_both_low, probe, test, cfg, device)
        rows.append({
            "seed": seed, "acquisition": acquisition, "update": update, "step": step,
            "high_D_A": h_da, "high_D_B": h_db, "s_high": h_s,
            "low_D_A": l_da, "low_D_B": l_db, "s_low": l_s,
        })

    record(0)
    for step in range(1, cfg.formation_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        if acquisition == "A_precise":
            x = train.x_a_precise
        elif acquisition == "B_precise":
            x = train.x_b_precise
        else:
            x = train.x_a_precise if schedule[step - 1] == 0 else train.x_b_precise
        train_weighted(
            model, optimizer, x.to(device)[idx], train.a.to(device)[idx], train.b.to(device)[idx],
            UPDATE_WEIGHTS[update],
        )
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


def summarize_cell(rows):
    ordered = sorted(rows, key=lambda row: row["step"])
    initial, final = ordered[0], ordered[-1]
    return {
        "initial_s_high": initial["s_high"], "final_s_high": final["s_high"],
        "change_s_high": final["s_high"] - initial["s_high"],
        "final_abs_s_high": abs(final["s_high"]),
        "initial_s_low": initial["s_low"], "final_s_low": final["s_low"],
        "change_s_low": final["s_low"] - initial["s_low"],
        "final_abs_s_low": abs(final["s_low"]),
    }


def percentile_ci(values, rng, samples: int):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def aggregate(cell_rows, sensor_rows, cfg: Config):
    by_seed = {}
    for row in cell_rows:
        by_seed.setdefault(row["seed"], {})[(row["acquisition"], row["update"])] = row
    seeds = sorted(by_seed)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    outcomes = ["final_s_high", "change_s_high", "final_abs_s_high",
                "final_s_low", "change_s_low", "final_abs_s_low"]

    cells = {}
    for acquisition in ACQUISITION_LEVELS:
        for update in UPDATE_LEVELS:
            label = f"acq_{acquisition}__upd_{update}"
            cells[label] = {
                outcome: percentile_ci(
                    [by_seed[seed][(acquisition, update)][outcome] for seed in seeds],
                    rng, cfg.bootstrap_samples)
                for outcome in outcomes
            }

    def contrast(outcome, terms):
        values = []
        for seed in seeds:
            value = 0.0
            for coefficient, acquisition, update in terms:
                value += coefficient * by_seed[seed][(acquisition, update)][outcome]
            values.append(value)
        return percentile_ci(values, rng, cfg.bootstrap_samples)

    effects = {
        "acquisition_only_A_minus_B": {
            outcome: contrast(outcome, [(1, "A_precise", "balanced"), (-1, "B_precise", "balanced")])
            for outcome in outcomes
        },
        "update_only_A_minus_B": {
            outcome: contrast(outcome, [(1, "balanced", "A_heavy"), (-1, "balanced", "B_heavy")])
            for outcome in outcomes
        },
        "congruent_A_minus_crossed_A": {
            outcome: contrast(outcome, [(1, "A_precise", "A_heavy"), (-1, "B_precise", "A_heavy")])
            for outcome in outcomes
        },
        "congruent_B_minus_crossed_B_in_B_direction": {
            outcome: contrast(outcome, [(-1, "B_precise", "B_heavy"), (1, "A_precise", "B_heavy")])
            for outcome in outcomes
        },
        "difference_in_acquisition_effect_by_update": {
            outcome: contrast(outcome, [
                (1, "A_precise", "A_heavy"), (-1, "B_precise", "A_heavy"),
                (-1, "A_precise", "B_heavy"), (1, "B_precise", "B_heavy")])
            for outcome in outcomes
        },
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


def evaluate_confirmatory_gates(aggregate_result, prereg):
    primary = aggregate_result["effects"]["acquisition_only_A_minus_B"]["final_s_high"]
    secondary = aggregate_result["effects"]["acquisition_only_A_minus_B"]["final_s_low"]
    update_control = aggregate_result["effects"]["update_only_A_minus_B"]["final_s_high"]
    a_gain = aggregate_result["sensor"]["A_precision_gain"]
    b_gain = aggregate_result["sensor"]["B_precision_gain"]
    threshold = float(prereg["primary_minimum_effect"])
    cell_minimum = float(prereg["directional_cell_minimum"])
    neutral_maximum = float(prereg["neutral_cell_maximum_absolute_mean"])
    a_cell = aggregate_result["cells"]["acq_A_precise__upd_balanced"]["final_s_high"]
    b_cell = aggregate_result["cells"]["acq_B_precise__upd_balanced"]["final_s_high"]
    neutral_cell = aggregate_result["cells"]["acq_balanced__upd_balanced"]["final_s_high"]
    return {
        "sensor_A": {"pass": a_gain["low"] > 0, "value": a_gain},
        "sensor_B": {"pass": b_gain["low"] > 0, "value": b_gain},
        "positive_control_update": {
            "pass": update_control["low"] > 0 and update_control["mean"] >= prereg["update_control_minimum"],
            "value": update_control,
        },
        "primary_acquisition_selects_direction": {
            "pass": primary["low"] > 0 and primary["mean"] >= threshold,
            "value": primary,
        },
        "A_precise_forms_A_direction": {
            "pass": a_cell["low"] >= cell_minimum,
            "value": a_cell,
        },
        "B_precise_forms_B_direction": {
            "pass": b_cell["high"] <= -cell_minimum,
            "value": b_cell,
        },
        "balanced_condition_remains_near_center": {
            "pass": abs(neutral_cell["mean"]) <= neutral_maximum,
            "value": neutral_cell,
        },
        "secondary_low_resolution_same_direction": {
            "pass": secondary["low"] > 0,
            "value": secondary,
        },
    }


def validate_preregistration(prereg, cfg: Config, seeds):
    expected = prereg["fixed_design"]
    observed = {
        "seeds": seeds,
        "formation_steps": cfg.formation_steps,
        "initial_steps": cfg.initial_steps,
        "low_noise": cfg.low_noise,
        "high_noise": cfg.high_noise,
        "initial_noise": cfg.initial_noise,
        "formation_update_weights": UPDATE_WEIGHTS,
        "acquisition_levels": list(ACQUISITION_LEVELS),
        "update_levels": list(UPDATE_LEVELS),
    }
    if observed != expected:
        raise ValueError(
            "確認実験の設定が事前登録と一致しません。\n"
            + json.dumps({"expected": expected, "observed": observed}, ensure_ascii=False, indent=2)
        )


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
        f"# 実験IV-B {stage} 結果", "",
        "## この実験が答える問い", "",
        "学習前の同じモデルに対し、どちらの情報を精密に取得するかだけを変えると、"
        "A/Bを同じ重みで学習しても、形成される共有表現の方向は変わるか。", "",
        "## 操作確認", "",
        f"- Aを精密にしたときのA情報利得: {result['sensor']['A_precision_gain']['mean']:.4f}",
        f"- Bを精密にしたときのB情報利得: {result['sensor']['B_precision_gain']['mean']:.4f}", "",
        "## 形成後の共通高精度測定", "",
        "| 取得 | 更新 | 最終方向 s | 特化の深さ |", "|---|---|---:|---:|",
    ]
    for acquisition in ACQUISITION_LEVELS:
        for update in UPDATE_LEVELS:
            values = result["cells"][f"acq_{acquisition}__upd_{update}"]
            lines.append(f"| {acquisition} | {update} | {values['final_s_high']['mean']:.4f} | "
                         f"{values['final_abs_s_high']['mean']:.4f} |")
    primary = result["effects"]["acquisition_only_A_minus_B"]["final_s_high"]
    update = result["effects"]["update_only_A_minus_B"]["final_s_high"]
    lines += ["", "## 中心比較", "",
              f"- 更新を均衡させた取得A精密－B精密: {primary['mean']:.4f} "
              f"(95%区間 [{primary['low']:.4f}, {primary['high']:.4f}])",
              f"- 取得を均衡させた更新A重視－B重視: {update['mean']:.4f} "
              f"(95%区間 [{update['low']:.4f}, {update['high']:.4f}])"]
    if gates is not None:
        lines += ["", "## 事前判定", ""]
        lines.extend(f"- {'PASS' if item['pass'] else 'FAIL'}: {name}" for name, item in gates.items())
    lines += ["", "## 適用範囲", "",
              "この結果が直接示す範囲は、二つの相関信号を二次元共有表現へ圧縮して学ぶ人工系である。"
              "取得と更新の役割が、他のモデルや自然な課題でも同じかは次の実験で確かめる。", "",
              f"- 実行時間: {runtime:.1f}秒"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    if args.formation_steps is not None:
        cfg.formation_steps = args.formation_steps
    defaults = {"smoke": (7000, 4), "pilot": (7100, 16), "confirm": (7200, 40)}
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
        initial_model, initial_optimizer = prepare_initial_state(cfg, seed, train, device)
        sensor_rows.append({"seed": seed, **sensor_diagnostics(cfg, probe, test, device)})
        for acquisition in ACQUISITION_LEVELS:
            for update in UPDATE_LEVELS:
                rows = run_cell(cfg, seed, acquisition, update, initial_model, initial_optimizer,
                                train, probe, test, device)
                timecourse_rows.extend(rows)
                cell_rows.append({"seed": seed, "acquisition": acquisition, "update": update,
                                  **summarize_cell(rows)})
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    expected_cells = len(seeds) * len(ACQUISITION_LEVELS) * len(UPDATE_LEVELS)
    expected_timecourse = expected_cells * len(checkpoints(cfg))
    if len(cell_rows) != expected_cells or len(timecourse_rows) != expected_timecourse:
        raise RuntimeError(f"不完全な結果: cells={len(cell_rows)}/{expected_cells}, "
                           f"timecourse={len(timecourse_rows)}/{expected_timecourse}")
    aggregate_result = aggregate(cell_rows, sensor_rows, cfg)
    gates = evaluate_confirmatory_gates(aggregate_result, prereg) if prereg else None
    runtime = time.time() - started
    result = {
        "experiment": "IV-B", "stage": args.stage,
        "question": "Can acquisition precision select formation direction under balanced updates?",
        "design": {
            "seeds": seeds, "same_initial_checkpoint_cloned": True,
            "same_batch_stream_within_seed": True,
            "acquisition_levels": list(ACQUISITION_LEVELS), "update_levels": list(UPDATE_LEVELS),
            "balanced_acquisition": "exactly half A-precise and half B-precise in randomized order",
            "primary_probe": "common both-high-precision probe",
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
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else None
    provenance = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(prereg_path) if prereg_path else None,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "device": str(device),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_text(args.stage, aggregate_result, runtime, gates)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
