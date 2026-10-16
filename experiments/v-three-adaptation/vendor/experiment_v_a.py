#!/usr/bin/env python3
"""実験V-A: 三つの対称な推定対象への表現配分を調べる。

同一の中立checkpointから、取得精度だけをA/B/Cのいずれかへ偏らせる
三条件と均衡取得条件へ分岐する。中心比較では三対象の更新重みを
1/3ずつに固定する。陽性対照では取得を均衡にして更新配分だけを偏らせる。
容量2と容量3を同時に調べ、対象ラベルを入れ替えても結果が再現するかを判定する。
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


TARGETS = ("A", "B", "C")
ACQUISITION_LEVELS = ("A_precise", "B_precise", "C_precise", "balanced")
UPDATE_LEVELS = ("balanced", "A_heavy", "B_heavy", "C_heavy")
CAPACITIES = (2, 3)
HEAVY_WEIGHT = 0.80


@dataclass
class Config:
    train_size: int = 3072
    probe_size: int = 768
    test_size: int = 1536
    source_dim: int = 3
    hidden_dim: int = 32
    correlation: float = 0.35
    low_noise: float = 1.20
    high_noise: float = 0.05
    initial_noise: float = 0.15
    batch_size: int = 256
    initial_steps: int = 100
    formation_steps: int = 600
    early_checkpoint_until: int = 120
    early_checkpoint_interval: int = 10
    checkpoint_interval: int = 50
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    probe_ridge: float = 1e-3
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 20260902


@dataclass
class SignalSplit:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirm"], default="smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--formation-steps", type=int, default=None)
    parser.add_argument("--preregistration", default=None,
                        help="確認段階の固定条件と判定を記したJSON。confirmでは必須。")
    return parser.parse_args()


def config_for_stage(stage: str) -> Config:
    cfg = Config()
    if stage == "smoke":
        cfg.train_size = 768
        cfg.probe_size = 256
        cfg.test_size = 384
        cfg.hidden_dim = 24
        cfg.batch_size = 128
        cfg.initial_steps = 30
        cfg.formation_steps = 90
        cfg.early_checkpoint_until = 30
        cfg.early_checkpoint_interval = 10
        cfg.checkpoint_interval = 30
        cfg.bootstrap_samples = 500
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
    """交換可能な三信号を作り、seed固有の直交変換で9入力を混合する。"""
    rng = np.random.default_rng(seed)
    input_dim = cfg.source_dim * len(TARGETS)
    raw_rotation = rng.normal(size=(input_dim, input_dim))
    rotation, _ = np.linalg.qr(raw_rotation)

    def sample(n: int) -> SignalSplit:
        shared = rng.normal(size=(n, cfg.source_dim))
        target_values = {}
        high = {}
        low = {}
        initial = {}
        for target in TARGETS:
            private = rng.normal(size=(n, cfg.source_dim))
            value = math.sqrt(cfg.correlation) * shared + math.sqrt(1.0 - cfg.correlation) * private
            target_values[target] = value
            high[target] = _standardized_noisy(value, rng.normal(size=value.shape), cfg.high_noise)
            low[target] = _standardized_noisy(value, rng.normal(size=value.shape), cfg.low_noise)
            initial[target] = _standardized_noisy(value, rng.normal(size=value.shape), cfg.initial_noise)

        def mixed(parts: dict[str, np.ndarray]):
            joined = np.concatenate([parts[target] for target in TARGETS], axis=1) @ rotation
            return torch.from_numpy(joined.astype("float32"))

        inputs = {
            f"{target}_precise": mixed({
                other: high[other] if other == target else low[other] for other in TARGETS
            })
            for target in TARGETS
        }
        inputs["all_high"] = mixed(high)
        inputs["all_low"] = mixed(low)
        inputs["initial"] = mixed(initial)
        return SignalSplit(
            inputs=inputs,
            targets={target: torch.from_numpy(value.astype("float32"))
                     for target, value in target_values.items()},
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
        self.heads = nn.ModuleDict({target: nn.Linear(bottleneck, output_dim) for target in TARGETS})

    def forward(self, x):
        z = self.encoder(x)
        return z, {target: self.heads[target](z) for target in TARGETS}


def make_model(cfg: Config, bottleneck: int, device):
    return SharedRepresentation(
        cfg.source_dim * len(TARGETS), cfg.hidden_dim, bottleneck, cfg.source_dim
    ).to(device)


def update_weights(level: str):
    if level == "balanced":
        return {target: 1.0 / len(TARGETS) for target in TARGETS}
    favored = level.split("_")[0]
    remainder = (1.0 - HEAVY_WEIGHT) / (len(TARGETS) - 1)
    return {target: HEAVY_WEIGHT if target == favored else remainder for target in TARGETS}


def train_weighted(model, optimizer, x, targets, weights):
    optimizer.zero_grad(set_to_none=True)
    _, predictions = model(x)
    losses = {
        target: torch.mean((predictions[target] - targets[target]) ** 2) for target in TARGETS
    }
    loss = sum(weights[target] * losses[target] for target in TARGETS)
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
    prediction = design_test @ weights
    variance = torch.var(y_test, unbiased=False) + 1e-12
    return (torch.mean((prediction - y_test) ** 2) / variance).item()


def allocation_scores(errors: dict[str, float]):
    logs = np.asarray([math.log(errors[target] + 1e-12) for target in TARGETS])
    centered = logs.mean() - logs
    return {target: float(centered[index]) for index, target in enumerate(TARGETS)}


@torch.no_grad()
def representation_metrics(model, probe_x, test_x, probe, test, cfg: Config, device):
    z_probe = model.encoder(probe_x.to(device))
    z_test = model.encoder(test_x.to(device))
    errors = {
        target: ridge_error(
            z_probe, probe.targets[target].to(device), z_test, test.targets[target].to(device),
            cfg.probe_ridge,
        )
        for target in TARGETS
    }
    scores = allocation_scores(errors)
    return errors, scores


def clone_optimizer(model, cfg: Config, state):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(state))
    return optimizer


def prepare_initial_state(cfg: Config, bottleneck: int, seed: int, train: SignalSplit, device):
    set_seed(seed + 10_000 * bottleneck)
    model = make_model(cfg, bottleneck, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(100_000 + seed)
    n = train.targets["A"].shape[0]
    weights = update_weights("balanced")
    for _ in range(cfg.initial_steps):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(
            model, optimizer, train.inputs["initial"].to(device)[idx],
            {target: train.targets[target].to(device)[idx] for target in TARGETS}, weights,
        )
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def balanced_acquisition_schedule(cfg: Config, seed: int):
    if cfg.formation_steps % len(TARGETS):
        raise ValueError("均衡取得を同数にするためformation_stepsは3の倍数にしてください。")
    each = cfg.formation_steps // len(TARGETS)
    schedule = np.asarray(list(range(len(TARGETS))) * each)
    np.random.default_rng(200_000 + seed).shuffle(schedule)
    return schedule


def checkpoints(cfg: Config):
    points = set(range(0, min(cfg.early_checkpoint_until, cfg.formation_steps) + 1,
                       cfg.early_checkpoint_interval))
    points.update(range(cfg.checkpoint_interval, cfg.formation_steps + 1, cfg.checkpoint_interval))
    points.add(cfg.formation_steps)
    return points


def input_for_step(train: SignalSplit, acquisition: str, schedule, step: int):
    if acquisition == "balanced":
        target = TARGETS[int(schedule[step - 1])]
    else:
        target = acquisition.split("_")[0]
    return train.inputs[f"{target}_precise"]


def condition_list():
    conditions = [(acquisition, "balanced") for acquisition in ACQUISITION_LEVELS]
    conditions.extend(("balanced", f"{target}_heavy") for target in TARGETS)
    return conditions


def run_cell(cfg, bottleneck, seed, acquisition, update, initial_model, initial_optimizer,
             train, probe, test, device):
    model = make_model(cfg, bottleneck, device)
    model.load_state_dict(initial_model)
    optimizer = clone_optimizer(model, cfg, initial_optimizer)
    batch_rng = np.random.default_rng(300_000 + seed)
    schedule = balanced_acquisition_schedule(cfg, seed)
    n = train.targets["A"].shape[0]
    points = checkpoints(cfg)
    rows = []

    def record(step: int):
        high_errors, high_scores = representation_metrics(
            model, probe.inputs["all_high"], test.inputs["all_high"], probe, test, cfg, device)
        low_errors, low_scores = representation_metrics(
            model, probe.inputs["all_low"], test.inputs["all_low"], probe, test, cfg, device)
        row = {"seed": seed, "bottleneck": bottleneck, "acquisition": acquisition,
               "update": update, "step": step}
        for target in TARGETS:
            row[f"high_D_{target}"] = high_errors[target]
            row[f"high_q_{target}"] = high_scores[target]
            row[f"low_D_{target}"] = low_errors[target]
            row[f"low_q_{target}"] = low_scores[target]
        row["high_error_sum"] = sum(high_errors.values())
        row["low_error_sum"] = sum(low_errors.values())
        rows.append(row)

    record(0)
    weights = update_weights(update)
    for step in range(1, cfg.formation_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        x = input_for_step(train, acquisition, schedule, step)
        train_weighted(
            model, optimizer, x.to(device)[idx],
            {target: train.targets[target].to(device)[idx] for target in TARGETS}, weights,
        )
        if step in points:
            record(step)
    return rows


@torch.no_grad()
def sensor_diagnostics(cfg, probe, test, device):
    rows = {}
    for measured in TARGETS:
        for precise in TARGETS:
            rows[f"D_{measured}_when_{precise}_precise"] = ridge_error(
                probe.inputs[f"{precise}_precise"].to(device), probe.targets[measured].to(device),
                test.inputs[f"{precise}_precise"].to(device), test.targets[measured].to(device),
                cfg.probe_ridge,
            )
    return rows


def summarize_cell(rows):
    ordered = sorted(rows, key=lambda row: row["step"])
    initial, final = ordered[0], ordered[-1]
    summary = {"initial_high_error_sum": initial["high_error_sum"],
               "final_high_error_sum": final["high_error_sum"],
               "final_low_error_sum": final["low_error_sum"]}
    for resolution in ("high", "low"):
        for target in TARGETS:
            summary[f"initial_{resolution}_D_{target}"] = initial[f"{resolution}_D_{target}"]
            summary[f"final_{resolution}_D_{target}"] = final[f"{resolution}_D_{target}"]
            summary[f"initial_{resolution}_q_{target}"] = initial[f"{resolution}_q_{target}"]
            summary[f"final_{resolution}_q_{target}"] = final[f"{resolution}_q_{target}"]
    return summary


def percentile_ci(values, rng, samples: int):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def allocation_plane_metrics(by_seed, bottleneck, seeds, rng, samples):
    """三つの取得効果が配分平面内で二方向へ開くかをpaired bootstrapする。"""
    ratios = []
    second_singular = []
    seed_array = np.asarray(seeds)
    for _ in range(samples):
        sampled = seed_array[rng.integers(0, len(seed_array), size=len(seed_array))]
        effects = []
        for target in TARGETS:
            vectors = []
            for seed in sampled:
                precise = by_seed[(bottleneck, int(seed))][(f"{target}_precise", "balanced")]
                neutral = by_seed[(bottleneck, int(seed))][("balanced", "balanced")]
                vectors.append([
                    precise[f"final_high_q_{coordinate}"]
                    - neutral[f"final_high_q_{coordinate}"] for coordinate in TARGETS
                ])
            effects.append(np.mean(vectors, axis=0))
        matrix = np.asarray(effects)
        matrix -= matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, compute_uv=False)
        ratios.append(float(singular[1] / (singular[0] + 1e-12)))
        second_singular.append(float(singular[1]))
    return {
        "second_to_first_ratio": {
            "mean": float(np.mean(ratios)), "low": float(np.quantile(ratios, 0.025)),
            "high": float(np.quantile(ratios, 0.975)),
        },
        "second_singular_value": {
            "mean": float(np.mean(second_singular)),
            "low": float(np.quantile(second_singular, 0.025)),
            "high": float(np.quantile(second_singular, 0.975)),
        },
    }


def aggregate(cell_rows, sensor_rows, cfg: Config):
    by_seed = {}
    for row in cell_rows:
        by_seed.setdefault((row["bottleneck"], row["seed"]), {})[
            (row["acquisition"], row["update"])
        ] = row
    seeds = sorted({row["seed"] for row in cell_rows})
    rng = np.random.default_rng(cfg.bootstrap_seed)
    result = {"n_seeds": len(seeds), "by_capacity": {}, "sensor": {}}

    for bottleneck in CAPACITIES:
        cap = {"cells": {}, "acquisition_target_effects": {}, "update_target_effects": {}}
        for acquisition, update in condition_list():
            label = f"acq_{acquisition}__upd_{update}"
            rows = [by_seed[(bottleneck, seed)][(acquisition, update)] for seed in seeds]
            cap["cells"][label] = {
                metric: percentile_ci([row[metric] for row in rows], rng, cfg.bootstrap_samples)
                for metric in rows[0]
                if metric.startswith("final_")
            }

        for target in TARGETS:
            other = tuple(item for item in TARGETS if item != target)
            acq_values = []
            acq_other_asymmetry = []
            acq_cost = []
            acq_winner = []
            upd_values = []
            upd_winner = []
            for seed in seeds:
                precise = by_seed[(bottleneck, seed)][(f"{target}_precise", "balanced")]
                neutral = by_seed[(bottleneck, seed)][("balanced", "balanced")]
                heavy = by_seed[(bottleneck, seed)][("balanced", f"{target}_heavy")]
                acq_values.append(precise[f"final_high_q_{target}"] - neutral[f"final_high_q_{target}"])
                acq_other_asymmetry.append(abs(
                    precise[f"final_high_q_{other[0]}"] - precise[f"final_high_q_{other[1]}"]
                ))
                acq_cost.append(precise["final_high_error_sum"] - neutral["final_high_error_sum"])
                acq_winner.append(float(
                    precise[f"final_high_D_{target}"] < min(
                        precise[f"final_high_D_{other[0]}"], precise[f"final_high_D_{other[1]}"]
                    )
                ))
                upd_values.append(heavy[f"final_high_q_{target}"] - neutral[f"final_high_q_{target}"])
                upd_winner.append(float(
                    heavy[f"final_high_D_{target}"] < min(
                        heavy[f"final_high_D_{other[0]}"], heavy[f"final_high_D_{other[1]}"]
                    )
                ))
            cap["acquisition_target_effects"][target] = {
                "target_q_gain": percentile_ci(acq_values, rng, cfg.bootstrap_samples),
                "other_target_absolute_asymmetry": percentile_ci(
                    acq_other_asymmetry, rng, cfg.bootstrap_samples),
                "error_sum_cost": percentile_ci(acq_cost, rng, cfg.bootstrap_samples),
                "favored_fraction": float(np.mean(acq_winner)),
            }
            cap["update_target_effects"][target] = {
                "target_q_gain": percentile_ci(upd_values, rng, cfg.bootstrap_samples),
                "favored_fraction": float(np.mean(upd_winner)),
            }

        acq_means = [cap["acquisition_target_effects"][target]["target_q_gain"]["mean"]
                     for target in TARGETS]
        upd_means = [cap["update_target_effects"][target]["target_q_gain"]["mean"]
                     for target in TARGETS]
        neutral_rows = [by_seed[(bottleneck, seed)][("balanced", "balanced")] for seed in seeds]
        neutral_spread = [
            max(row[f"final_high_q_{target}"] for target in TARGETS)
            - min(row[f"final_high_q_{target}"] for target in TARGETS)
            for row in neutral_rows
        ]
        neutral_group_means = {
            target: float(np.mean([row[f"final_high_q_{target}"] for row in neutral_rows]))
            for target in TARGETS
        }
        cap["symmetry"] = {
            "acquisition_effect_mean_range": float(max(acq_means) - min(acq_means)),
            "update_effect_mean_range": float(max(upd_means) - min(upd_means)),
            "balanced_q_spread": percentile_ci(neutral_spread, rng, cfg.bootstrap_samples),
            "balanced_group_means": neutral_group_means,
            "balanced_group_mean_range": float(
                max(neutral_group_means.values()) - min(neutral_group_means.values())),
        }
        cap["allocation_plane"] = allocation_plane_metrics(
            by_seed, bottleneck, seeds, rng, cfg.bootstrap_samples)
        result["by_capacity"][str(bottleneck)] = cap

    for measured in TARGETS:
        gains = []
        for row in sensor_rows:
            own = row[f"D_{measured}_when_{measured}_precise"]
            others = [row[f"D_{measured}_when_{other}_precise"]
                      for other in TARGETS if other != measured]
            gains.append(float(np.mean(others) - own))
        result["sensor"][f"{measured}_precision_gain"] = percentile_ci(
            gains, rng, cfg.bootstrap_samples)
    return result


def evaluate_confirmatory_gates(result, prereg):
    thresholds = prereg["thresholds"]
    gates = {}
    for target in TARGETS:
        sensor = result["sensor"][f"{target}_precision_gain"]
        gates[f"sensor_{target}"] = {"pass": sensor["low"] > 0, "value": sensor}
    for bottleneck in CAPACITIES:
        cap = result["by_capacity"][str(bottleneck)]
        minimum = thresholds[f"capacity_{bottleneck}_minimum_q_gain"]
        for target in TARGETS:
            acq = cap["acquisition_target_effects"][target]
            upd = cap["update_target_effects"][target]
            gates[f"capacity_{bottleneck}_acquisition_{target}"] = {
                "pass": acq["target_q_gain"]["low"] > 0
                and acq["target_q_gain"]["mean"] >= minimum
                and acq["favored_fraction"] >= thresholds["minimum_favored_fraction"],
                "value": acq,
            }
            gates[f"capacity_{bottleneck}_update_{target}"] = {
                "pass": upd["target_q_gain"]["low"] > 0
                and upd["favored_fraction"] >= thresholds["minimum_favored_fraction"],
                "value": upd,
            }
            gates[f"capacity_{bottleneck}_other_symmetry_{target}"] = {
                "pass": acq["other_target_absolute_asymmetry"]["high"]
                <= thresholds["maximum_other_target_asymmetry"],
                "value": acq["other_target_absolute_asymmetry"],
            }
        gates[f"capacity_{bottleneck}_balanced_label_symmetry"] = {
            "pass": cap["symmetry"]["balanced_group_mean_range"]
            <= thresholds["maximum_balanced_group_mean_range"],
            "value": cap["symmetry"]["balanced_group_mean_range"],
        }
        gates[f"capacity_{bottleneck}_label_exchangeability"] = {
            "pass": cap["symmetry"]["acquisition_effect_mean_range"]
            <= thresholds["maximum_acquisition_effect_range"],
            "value": cap["symmetry"]["acquisition_effect_mean_range"],
        }
        gates[f"capacity_{bottleneck}_two_dimensional_allocation_span"] = {
            "pass": cap["allocation_plane"]["second_to_first_ratio"]["low"]
            >= thresholds["minimum_plane_singular_ratio"],
            "value": cap["allocation_plane"],
        }
    return gates


def validate_preregistration(prereg, cfg: Config, seeds):
    expected = prereg["fixed_design"]
    observed = {
        "seeds": seeds,
        "capacities": list(CAPACITIES),
        "formation_steps": cfg.formation_steps,
        "initial_steps": cfg.initial_steps,
        "correlation": cfg.correlation,
        "low_noise": cfg.low_noise,
        "high_noise": cfg.high_noise,
        "initial_noise": cfg.initial_noise,
        "heavy_weight": HEAVY_WEIGHT,
        "conditions": [list(item) for item in condition_list()],
    }
    if observed != expected:
        raise ValueError("確認設定が事前登録と一致しません。\n" + json.dumps(
            {"expected": expected, "observed": observed}, ensure_ascii=False, indent=2))


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
    aggregate_result = result["aggregate"]
    lines = [
        f"# 実験V-A {stage} 結果", "", "## この実験が答える問い", "",
        "三つの推定対象を一つの有限な共有表現で学ぶとき、更新を三対象で均衡させても、"
        "どの対象を精密に取得するかによって、形成後に優位となる対象は変わるか。", "",
        "## 結果", "",
    ]
    for bottleneck in CAPACITIES:
        lines += [f"### 表現容量 {bottleneck}", "",
                  "| 精密に取得した対象 | その対象の配分利得 | 優位となったseed割合 | 他二対象間の非対称 |",
                  "|---|---:|---:|---:|"]
        cap = aggregate_result["by_capacity"][str(bottleneck)]
        for target in TARGETS:
            item = cap["acquisition_target_effects"][target]
            lines.append(
                f"| {target} | {item['target_q_gain']['mean']:.4f} "
                f"[{item['target_q_gain']['low']:.4f}, {item['target_q_gain']['high']:.4f}] | "
                f"{item['favored_fraction']:.3f} | "
                f"{item['other_target_absolute_asymmetry']['mean']:.4f} |"
            )
        ratio = cap["allocation_plane"]["second_to_first_ratio"]
        lines += ["", f"均衡条件におけるラベル別平均値の範囲: "
                  f"{cap['symmetry']['balanced_group_mean_range']:.4f}。",
                  f"ラベル別取得効果の平均値の範囲: "
                  f"{cap['symmetry']['acquisition_effect_mean_range']:.4f}。",
                  f"配分平面の第2/第1特異値比: {ratio['mean']:.4f} "
                  f"[{ratio['low']:.4f}, {ratio['high']:.4f}]。", ""]
    if gates is not None:
        passed = sum(bool(item["pass"]) for item in gates.values())
        lines += ["## 事前判定", "", f"{passed}/{len(gates)}判定を通過。", ""]
        lines.extend(f"- {'PASS' if item['pass'] else 'FAIL'}: {name}" for name, item in gates.items())
    lines += ["", "## 適用範囲", "",
              "この結果が直接示す範囲は、相関0.35の三つの三次元信号を、二次元または三次元の"
              "共有表現へ圧縮して学ぶ人工系である。取得条件と更新重みは研究者が外部から割り当て、"
              f"形成を{result['config']['formation_steps']}更新まで観測した。", "",
              f"- 実行時間: {runtime:.1f}秒"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    if args.formation_steps is not None:
        cfg.formation_steps = args.formation_steps
    defaults = {"smoke": (7800, 2), "pilot": (8000, 12), "confirm": (8200, 40)}
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
        sensor_rows.append({"seed": seed, **sensor_diagnostics(cfg, probe, test, device)})
        for bottleneck in CAPACITIES:
            initial_model, initial_optimizer = prepare_initial_state(cfg, bottleneck, seed, train, device)
            for acquisition, update in condition_list():
                rows = run_cell(
                    cfg, bottleneck, seed, acquisition, update, initial_model, initial_optimizer,
                    train, probe, test, device,
                )
                timecourse_rows.extend(rows)
                cell_rows.append({"seed": seed, "bottleneck": bottleneck,
                                  "acquisition": acquisition, "update": update,
                                  **summarize_cell(rows)})
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    expected_cells = len(seeds) * len(CAPACITIES) * len(condition_list())
    expected_timecourse = expected_cells * len(checkpoints(cfg))
    if len(cell_rows) != expected_cells or len(timecourse_rows) != expected_timecourse:
        raise RuntimeError(f"不完全な結果: cells={len(cell_rows)}/{expected_cells}, "
                           f"timecourse={len(timecourse_rows)}/{expected_timecourse}")
    aggregate_result = aggregate(cell_rows, sensor_rows, cfg)
    gates = evaluate_confirmatory_gates(aggregate_result, prereg) if prereg else None
    runtime = time.time() - started
    result = {
        "experiment": "V-A", "stage": args.stage,
        "question": "Can acquisition precision select one of three representational targets under balanced updating?",
        "design": {
            "seeds": seeds, "same_initial_checkpoint_cloned_within_capacity": True,
            "same_batch_stream_within_seed": True,
            "exchangeable_three_target_generator": True,
            "balanced_acquisition": "exactly one third A/B/C-precise in seed-specific randomized order",
            "primary_probe": "common all-high-precision held-out probe",
            "statistical_unit": "seed",
            "conditions": [list(item) for item in condition_list()],
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
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else None
    provenance = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(prereg_path) if prereg_path else None,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "device": str(device),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_text(args.stage, result, runtime, gates)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
