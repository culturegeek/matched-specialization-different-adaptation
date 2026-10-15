#!/usr/bin/env python3
"""実験II v0.4: 初期選択系列を操作し、形成経路の方向への因果効果を測る。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from history_closed_loop_poc import (
    SharedRepresentation,
    choose_device,
    ci95,
    full_metrics,
    make_config,
    make_dataset,
    representation_selector_index,
    selector_probability,
    set_seed,
    train_selected,
    train_weighted,
)


PRETRAIN_STEPS = 25
FORCED_STEPS = 100
FORCED_A_COUNT = 55
CONTINUATIONS = ["balanced_sampled", "fixed_propensity", "stabilization", "exploration"]
MANIPULATIONS = ["A55_B45", "A45_B55"]


def mirrored_schedules(seed: int):
    rng = np.random.default_rng(710_000 + seed)
    schedule = np.zeros(FORCED_STEPS, dtype=bool)
    schedule[:FORCED_A_COUNT] = True
    rng.shuffle(schedule)
    return {"A55_B45": schedule, "A45_B55": ~schedule}


def train_precondition(cfg, seed, train, device):
    set_seed(seed)
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(700_000 + seed)
    n = train[0].shape[0]
    for _ in range(PRETRAIN_STEPS):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(value[idx] for value in train)
        train_weighted(model, optimizer, *batch, 0.5)
    return copy.deepcopy(model.state_dict())


def apply_manipulation(cfg, seed, initial_state, train, schedule, device):
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(720_000 + seed)
    n = train[0].shape[0]
    for choose_a in schedule:
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(value[idx] for value in train)
        train_selected(model, optimizer, *batch, bool(choose_a))
    return copy.deepcopy(model.state_dict())


def continue_learning(cfg, seed, manipulation, condition, release_state, train, train_cpu, selector_cpu, test_cpu, device):
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model.load_state_dict(release_state)
    # 操作期のAdam状態を持ち越さず、表現重みを介した効果を測る。
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_rng = np.random.default_rng(730_000 + seed)
    selection_rng = np.random.default_rng(740_000 + seed)
    n = train[0].shape[0]
    release_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
    frozen_probability = selector_probability(release_index, "stabilization", cfg)
    probability = selector_probability(release_index, condition, cfg, frozen_probability)
    choices_a = 0
    rows = []

    def record(step):
        metrics, _ = full_metrics(model, train_cpu, test_cpu, cfg, device)
        selector_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
        rows.append({
            "seed": seed,
            "manipulation": manipulation,
            "condition": condition,
            "common_step": step,
            "selector_index": selector_index,
            "test_probe_index": metrics["probe_index"],
            "decoder_index": metrics["allocation_index"],
            "selector_p_A": probability,
            "realized_A_fraction": choices_a / max(step, 1),
        })

    checkpoints = set(cfg.checkpoints)
    record(0)
    for step in range(1, cfg.common_steps + 1):
        if condition in ("stabilization", "exploration") and (step == 1 or (step - 1) % cfg.selector_interval == 0):
            current_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
            probability = selector_probability(current_index, condition, cfg)
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(value[idx] for value in train)
        choose_a = selection_rng.random() < probability
        choices_a += int(choose_a)
        train_selected(model, optimizer, *batch, choose_a)
        if step in checkpoints:
            record(step)
    return rows


def run_seed(cfg, seed, device):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(cfg, seed)
    train = tuple(value.to(device) for value in train_cpu)
    initial_state = train_precondition(cfg, seed, train, device)
    schedules = mirrored_schedules(seed)
    release_states = {
        name: apply_manipulation(cfg, seed, initial_state, train, schedule, device)
        for name, schedule in schedules.items()
    }
    rows = []
    for condition in CONTINUATIONS:
        for manipulation in MANIPULATIONS:
            rows.extend(continue_learning(
                cfg, seed, manipulation, condition, release_states[manipulation],
                train, train_cpu, selector_cpu, test_cpu, device,
            ))
    return rows


def paired_effects(rows, step, condition):
    selected = [row for row in rows if row["common_step"] == step and row["condition"] == condition]
    by_seed = {}
    for row in selected:
        by_seed.setdefault(row["seed"], {})[row["manipulation"]] = row
    effects = []
    midpoints = []
    for seed in sorted(by_seed):
        plus = by_seed[seed]["A55_B45"]["test_probe_index"]
        minus = by_seed[seed]["A45_B55"]["test_probe_index"]
        effects.append(0.5 * (plus - minus))
        midpoints.append(0.5 * (plus + minus))
    return np.asarray(effects), np.asarray(midpoints)


def summarize(rows, cfg):
    release, release_midpoint = paired_effects(rows, 0, "balanced_sampled")
    release_mean, release_low, release_high = ci95(release)
    result = {
        "release_directional_effect": {
            "mean": release_mean,
            "ci_low": release_low,
            "ci_high": release_high,
            "midpoint_mean": float(release_midpoint.mean()),
        },
        "conditions": {},
    }
    per_condition = {}
    for condition in CONTINUATIONS:
        final, midpoint = paired_effects(rows, cfg.common_steps, condition)
        final_mean, final_low, final_high = ci95(final)
        change = final - release
        change_mean, change_low, change_high = ci95(change)
        sign_tracking = []
        selected = [row for row in rows if row["common_step"] == cfg.common_steps and row["condition"] == condition]
        for row in selected:
            desired = 1.0 if row["manipulation"] == "A55_B45" else -1.0
            sign_tracking.append(np.sign(row["test_probe_index"]) == desired)
        result["conditions"][condition] = {
            "final_directional_effect_mean": final_mean,
            "final_directional_effect_ci_low": final_low,
            "final_directional_effect_ci_high": final_high,
            "gain_over_release": final_mean / (release_mean + 1e-12),
            "change_from_release_mean": change_mean,
            "change_from_release_ci_low": change_low,
            "change_from_release_ci_high": change_high,
            "final_midpoint_mean": float(midpoint.mean()),
            "direction_tracking_rate": float(np.mean(sign_tracking)),
        }
        per_condition[condition] = final

    dynamic_minus_fixed = per_condition["stabilization"] - per_condition["fixed_propensity"]
    diff_mean, diff_low, diff_high = ci95(dynamic_minus_fixed)
    stabilization = result["conditions"]["stabilization"]["final_directional_effect_mean"]
    fixed = result["conditions"]["fixed_propensity"]["final_directional_effect_mean"]
    exploration = result["conditions"]["exploration"]["final_directional_effect_mean"]
    result["stabilization_minus_fixed"] = {
        "mean": diff_mean,
        "ci_low": diff_low,
        "ci_high": diff_high,
        "ratio": stabilization / (fixed + 1e-12),
    }
    # 実行前に固定した判定。操作確認と主要仮説を分けて記録する。
    result["gates"] = {
        "forced_sequence_creates_weak_directional_effect": release_low > 0 and release_mean < 0.30,
        "stabilization_tracks_forced_direction": result["conditions"]["stabilization"]["final_directional_effect_ci_low"] > 0,
        "stabilization_amplifies_release_by_50_percent": stabilization > 1.50 * release_mean,
        "stabilization_exceeds_fixed_by_20_percent": stabilization > 1.20 * fixed,
        "paired_stabilization_minus_fixed_ci_excludes_zero": diff_low > 0,
        "exploration_below_half_stabilization": abs(exploration) < 0.50 * abs(stabilization),
        "forced_direction_tracking_exceeds_80_percent": result["conditions"]["stabilization"]["direction_tracking_rate"] > 0.80,
    }
    return result


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="初期の誤差選択系列を操作する因果PoCを実行します。")
    parser.add_argument("--output-dir", default="forced_selection_causal_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    cfg = make_config(full=False)
    cfg.history_steps = PRETRAIN_STEPS
    cfg.seeds = seeds
    cfg.conditions = CONTINUATIONS
    rows = []
    for index, seed in enumerate(seeds, 1):
        rows.extend(run_seed(cfg, seed, device))
        print(f"seed {index}/{len(seeds)} を完了")
    result = summarize(rows, cfg)
    result["preregistered_design"] = {
        "seeds": seeds,
        "balanced_pretrain_steps": PRETRAIN_STEPS,
        "forced_steps": FORCED_STEPS,
        "forced_schedules": {"A55_B45": "55 A / 45 B", "A45_B55": "45 A / 55 B, mirrored order"},
        "optimizer_reset_after_forced_period": True,
        "common_steps": cfg.common_steps,
        "conditions": CONTINUATIONS,
    }
    write_csv(output_dir / "raw_timecourse.csv", rows)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({"config": asdict(cfg), "device": str(device)}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 初期選択系列操作PoC 結果", "",
        f"- 操作直後の方向効果: {result['release_directional_effect']['mean']:.4f} "
        f"(95% CI [{result['release_directional_effect']['ci_low']:.4f}, {result['release_directional_effect']['ci_high']:.4f}])",
    ]
    for condition in CONTINUATIONS:
        item = result["conditions"][condition]
        lines.append(
            f"- {condition}: 最終方向効果 {item['final_directional_effect_mean']:.4f}, "
            f"gain {item['gain_over_release']:.3f}, 方向追随率 {item['direction_tracking_rate']:.3f}"
        )
    contrast = result["stabilization_minus_fixed"]
    lines.extend([
        f"- stabilization / fixed: {contrast['ratio']:.3f}",
        f"- stabilization - fixed 95% CI: [{contrast['ci_low']:.4f}, {contrast['ci_high']:.4f}]", "",
        "## 事前判定", "",
    ])
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in result["gates"].items())
    report = "\n".join(lines)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
