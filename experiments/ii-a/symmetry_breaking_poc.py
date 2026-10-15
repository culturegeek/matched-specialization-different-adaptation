#!/usr/bin/env python3
"""対称な学習条件から偶発的な微小差が増幅されるかを測るPoC。"""

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


CONDITIONS = ["balanced_sampled", "fixed_propensity", "stabilization", "exploration"]
PRETRAIN_STEPS = 25


def run_seed(cfg, seed, device):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(cfg, seed)
    train = tuple(value.to(device) for value in train_cpu)
    n = train[0].shape[0]
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pretrain_rng = np.random.default_rng(400_000 + seed)
    for _ in range(PRETRAIN_STEPS):
        idx = torch.from_numpy(pretrain_rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(value[idx] for value in train)
        train_weighted(model, optimizer, *batch, 0.5)

    common_state = copy.deepcopy(model.state_dict())
    initial_selector_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
    initial_test_metrics, _ = full_metrics(model, train_cpu, test_cpu, cfg, device)
    frozen_probability = selector_probability(initial_selector_index, "stabilization", cfg)
    rows = []

    for condition in CONDITIONS:
        current = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
        current.load_state_dict(common_state)
        optimizer = torch.optim.Adam(current.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        batch_rng = np.random.default_rng(500_000 + seed)
        selection_rng = np.random.default_rng(600_000 + seed)
        probability = 0.5 if condition == "balanced_sampled" else selector_probability(initial_selector_index, condition, cfg, frozen_probability)
        choice_history = []

        def early_choice_index(horizon):
            values = choice_history[:horizon]
            if not values:
                return float("nan")
            return float(np.mean([1.0 if choose_a else -1.0 for choose_a in values]))

        def record(step):
            metrics, _ = full_metrics(current, train_cpu, test_cpu, cfg, device)
            current_selector_index = representation_selector_index(current, selector_cpu, cfg.probe_ridge, device)
            rows.append({
                "seed": seed,
                "condition": condition,
                "common_step": step,
                "initial_selector_index": initial_selector_index,
                "initial_test_probe_index": initial_test_metrics["probe_index"],
                "selector_index": current_selector_index,
                "test_probe_index": metrics["probe_index"],
                "decoder_index": metrics["allocation_index"],
                "selector_p_A": probability,
                "choice_index_first_25": early_choice_index(25),
                "choice_index_first_50": early_choice_index(50),
                "choice_index_first_100": early_choice_index(100),
            })

        record(0)
        checkpoints = set(cfg.checkpoints)
        for step in range(1, cfg.common_steps + 1):
            if condition in ("stabilization", "exploration") and (step == 1 or (step - 1) % cfg.selector_interval == 0):
                index = representation_selector_index(current, selector_cpu, cfg.probe_ridge, device)
                probability = selector_probability(index, condition, cfg)
            idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
            batch = tuple(value[idx] for value in train)
            choose_a = selection_rng.random() < probability
            choice_history.append(choose_a)
            train_selected(current, optimizer, *batch, choose_a)
            if step in checkpoints:
                record(step)
    return rows


def main():
    parser = argparse.ArgumentParser(description="対称条件からの自発的分岐を測定します。")
    parser.add_argument("--output-dir", default="symmetry_breaking_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    cfg = make_config(full=False)
    cfg.seeds = seeds
    cfg.conditions = CONDITIONS
    rows = []
    for index, seed in enumerate(seeds, 1):
        rows.extend(run_seed(cfg, seed, device))
        print(f"seed {index}/{len(seeds)} を完了")

    finals = [row for row in rows if row["common_step"] == cfg.common_steps]
    initial_by_seed = {row["seed"]: row["initial_selector_index"] for row in rows if row["condition"] == "stabilization" and row["common_step"] == 0}
    initial_values = np.asarray(list(initial_by_seed.values()))
    summaries = {}
    for condition in CONDITIONS:
        values = [abs(row["test_probe_index"]) for row in finals if row["condition"] == condition]
        mean, low, high = ci95(values)
        summaries[condition] = {"final_abs_probe_mean": mean, "ci_low": low, "ci_high": high}

    stabilization_rows = sorted([row for row in finals if row["condition"] == "stabilization"], key=lambda row: row["seed"])
    fixed_rows = sorted([row for row in finals if row["condition"] == "fixed_propensity"], key=lambda row: row["seed"])
    exploration_rows = sorted([row for row in finals if row["condition"] == "exploration"], key=lambda row: row["seed"])
    dynamic_minus_fixed = [abs(stab["test_probe_index"]) - abs(fixed["test_probe_index"]) for stab, fixed in zip(stabilization_rows, fixed_rows)]
    difference_mean, difference_low, difference_high = ci95(dynamic_minus_fixed)
    sign_alignment = float(np.mean([
        np.sign(initial_by_seed[row["seed"]]) == np.sign(row["test_probe_index"])
        for row in stabilization_rows if abs(initial_by_seed[row["seed"]]) > 0.01
    ]))
    early_choice_alignment = {}
    for horizon in (25, 50, 100):
        key = f"choice_index_first_{horizon}"
        eligible = [row for row in stabilization_rows if abs(row[key]) > 1e-12]
        early_choice_alignment[str(horizon)] = float(np.mean([
            np.sign(row[key]) == np.sign(row["test_probe_index"])
            for row in eligible
        ])) if eligible else float("nan")
    initial_mean, initial_low, initial_high = ci95(initial_values)
    positive_fraction = float(np.mean(initial_values > 0))
    stabilization_mean = summaries["stabilization"]["final_abs_probe_mean"]
    fixed_mean = summaries["fixed_propensity"]["final_abs_probe_mean"]
    exploration_mean = summaries["exploration"]["final_abs_probe_mean"]

    # 新規仮説に対し、実行前に固定した探索判定。
    gates = {
        "initial_signed_mean_near_zero": abs(initial_mean) < 0.10,
        "both_initial_signs_present": 0.25 < positive_fraction < 0.75,
        "stabilization_exceeds_fixed_by_20_percent": stabilization_mean > 1.20 * fixed_mean,
        "paired_stabilization_minus_fixed_ci_excludes_zero": difference_low > 0,
        "exploration_below_half_stabilization": exploration_mean < 0.50 * stabilization_mean,
        "initial_sign_predicts_stabilized_sign": sign_alignment > 0.75,
    }
    result = {
        "design": {
            "seeds": seeds,
            "balanced_pretrain_steps": PRETRAIN_STEPS,
            "history_lambda": 0.5,
            "conditions": CONDITIONS,
        },
        "initial_selector_index": {
            "mean": initial_mean,
            "ci_low": initial_low,
            "ci_high": initial_high,
            "positive_fraction": positive_fraction,
        },
        "final_abs_probe_index": summaries,
        "stabilization_minus_fixed": {
            "mean": difference_mean,
            "ci_low": difference_low,
            "ci_high": difference_high,
            "ratio": stabilization_mean / (fixed_mean + 1e-12),
        },
        "stabilization_sign_alignment": sign_alignment,
        "stabilization_early_choice_sign_alignment": early_choice_alignment,
        "gates": gates,
    }

    with open(output_dir / "raw_timecourse.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({"config": asdict(cfg), "device": str(device)}, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# 対称条件からの分岐PoC 結果", "",
        f"- 初期selector index平均: {initial_mean:.4f}（95% CI [{initial_low:.4f}, {initial_high:.4f}]）",
        f"- 初期正符号率: {positive_fraction:.3f}",
        f"- balanced最終|probe index|: {summaries['balanced_sampled']['final_abs_probe_mean']:.4f}",
        f"- fixed最終|probe index|: {fixed_mean:.4f}",
        f"- stabilization最終|probe index|: {stabilization_mean:.4f}",
        f"- exploration最終|probe index|: {exploration_mean:.4f}",
        f"- stabilization / fixed: {stabilization_mean / (fixed_mean + 1e-12):.3f}",
        f"- 対応差95% CI: [{difference_low:.4f}, {difference_high:.4f}]",
        f"- 初期符号と安定化後符号の一致率: {sign_alignment:.3f}", "",
        f"- 最初の25選択と最終符号の一致率: {early_choice_alignment['25']:.3f}",
        f"- 最初の50選択と最終符号の一致率: {early_choice_alignment['50']:.3f}",
        f"- 最初の100選択と最終符号の一致率: {early_choice_alignment['100']:.3f}", "",
        "## 探索判定", "",
    ]
    report.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items())
    report_text = "\n".join(report)
    (output_dir / "result_report.md").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
