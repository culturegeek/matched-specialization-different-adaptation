#!/usr/bin/env python3
"""強い初期履歴による選択確率の飽和を診断する補助走査。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from history_closed_loop_poc import choose_device, ci95, make_config, run_seed, summarize, trapezoid_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="history_strength_sweep_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["all_targets", "fixed_propensity", "stabilization", "exploration"]
    history_steps_list = [25, 50, 100, 200, 400]
    results = []
    contrasts = []

    for history_steps in history_steps_list:
        cfg = make_config(full=False)
        cfg.history_steps = history_steps
        cfg.seeds = list(range(10))
        cfg.conditions = conditions
        rows = []
        for condition in conditions:
            for seed in cfg.seeds:
                seed_rows, _ = run_seed(cfg, seed, condition, device)
                rows.extend(seed_rows)
        summaries = summarize(rows, cfg)
        for condition in conditions:
            items = [row for row in summaries if row["condition"] == condition]
            start = min(items, key=lambda row: row["common_step"])
            final = max(items, key=lambda row: row["common_step"])
            results.append({
                "history_steps": history_steps,
                "condition": condition,
                "initial_gap": start["paired_gap_mean"],
                "final_gap": final["paired_gap_mean"],
                "retention": abs(final["paired_gap_mean"]) / (abs(start["paired_gap_mean"]) + 1e-12),
                "gap_auc": trapezoid_auc(items, "paired_gap_mean"),
                "initial_selector_gap": start["A_history_p_A_mean"] - start["B_history_p_A_mean"],
                "final_selector_gap": final["A_history_p_A_mean"] - final["B_history_p_A_mean"],
            })
        auc_by_condition = {}
        for condition in conditions:
            auc_by_condition[condition] = []
            for seed in cfg.seeds:
                arows = sorted([row for row in rows if row["condition"] == condition and row["seed"] == seed and row["history"] == "A_history"], key=lambda row: row["common_step"])
                brows = sorted([row for row in rows if row["condition"] == condition and row["seed"] == seed and row["history"] == "B_history"], key=lambda row: row["common_step"])
                xs = np.asarray([row["common_step"] for row in arows], dtype=float)
                gaps = np.asarray([abs(a["allocation_index"] - b["allocation_index"]) for a, b in zip(arows, brows)], dtype=float)
                auc_by_condition[condition].append(float(np.trapezoid(gaps, xs) / (xs[-1] - xs[0])))
        auc_difference = np.asarray(auc_by_condition["stabilization"]) - np.asarray(auc_by_condition["fixed_propensity"])
        difference_mean, difference_low, difference_high = ci95(auc_difference)
        contrasts.append({
            "history_steps": history_steps,
            "stabilization_minus_fixed_auc_mean": difference_mean,
            "ci_low": difference_low,
            "ci_high": difference_high,
            "ci_excludes_zero": difference_low > 0 or difference_high < 0,
        })
        print(f"履歴 {history_steps} steps を完了")

    with open(output_dir / "history_strength_sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(output_dir / "history_strength_contrasts.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0].keys()))
        writer.writeheader()
        writer.writerows(contrasts)
    (output_dir / "metadata.json").write_text(json.dumps({
        "base_config": asdict(make_config(False)),
        "effective_seeds": list(range(10)),
        "effective_conditions": conditions,
        "history_steps": history_steps_list,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n履歴強度別の安定化／固定配分AUC比")
    for history_steps in history_steps_list:
        by_condition = {row["condition"]: row for row in results if row["history_steps"] == history_steps}
        ratio = by_condition["stabilization"]["gap_auc"] / (by_condition["fixed_propensity"]["gap_auc"] + 1e-12)
        contrast = next(row for row in contrasts if row["history_steps"] == history_steps)
        print(f"  {history_steps:>3} steps: ratio={ratio:.3f}  initial_gap={by_condition['stabilization']['initial_gap']:.3f}  AUC差95%CI=[{contrast['ci_low']:.3f}, {contrast['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
