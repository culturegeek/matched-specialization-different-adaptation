#!/usr/bin/env python3
"""弱い初期履歴に対する閉ループ上乗せ効果の独立seed確認。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from history_closed_loop_poc import choose_device, ci95, make_config, run_seed


CONFIRMATORY_SEEDS = list(range(100, 120))
CONDITIONS = ["fixed_propensity", "stabilization", "exploration"]
HISTORY_STEPS = 25


def seed_auc(rows, condition, seed):
    arows = sorted([row for row in rows if row["condition"] == condition and row["seed"] == seed and row["history"] == "A_history"], key=lambda row: row["common_step"])
    brows = sorted([row for row in rows if row["condition"] == condition and row["seed"] == seed and row["history"] == "B_history"], key=lambda row: row["common_step"])
    xs = np.asarray([row["common_step"] for row in arows], dtype=float)
    gaps = np.asarray([abs(a["allocation_index"] - b["allocation_index"]) for a, b in zip(arows, brows)], dtype=float)
    return float(np.trapezoid(gaps, xs) / (xs[-1] - xs[0]))


def main():
    parser = argparse.ArgumentParser(description="探索に使っていないseedで弱履歴効果を確認します。")
    parser.add_argument("--output-dir", default="history_strength_confirmatory_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    cfg = make_config(full=False)
    cfg.history_steps = HISTORY_STEPS
    cfg.seeds = CONFIRMATORY_SEEDS
    cfg.conditions = CONDITIONS
    rows = []
    for condition in CONDITIONS:
        for seed in CONFIRMATORY_SEEDS:
            seed_rows, _ = run_seed(cfg, seed, condition, device)
            rows.extend(seed_rows)
        print(f"{condition} を完了")

    seed_rows = []
    for seed in CONFIRMATORY_SEEDS:
        initial_a = next(row for row in rows if row["condition"] == "stabilization" and row["seed"] == seed and row["history"] == "A_history" and row["common_step"] == 0)
        initial_b = next(row for row in rows if row["condition"] == "stabilization" and row["seed"] == seed and row["history"] == "B_history" and row["common_step"] == 0)
        values = {condition: seed_auc(rows, condition, seed) for condition in CONDITIONS}
        seed_rows.append({
            "seed": seed,
            "initial_gap": initial_a["allocation_index"] - initial_b["allocation_index"],
            "fixed_auc": values["fixed_propensity"],
            "stabilization_auc": values["stabilization"],
            "exploration_auc": values["exploration"],
            "stabilization_minus_fixed": values["stabilization"] - values["fixed_propensity"],
        })

    initial_mean, initial_low, initial_high = ci95([row["initial_gap"] for row in seed_rows])
    difference_mean, difference_low, difference_high = ci95([row["stabilization_minus_fixed"] for row in seed_rows])
    fixed_mean = float(np.mean([row["fixed_auc"] for row in seed_rows]))
    stabilization_mean = float(np.mean([row["stabilization_auc"] for row in seed_rows]))
    exploration_mean = float(np.mean([row["exploration_auc"] for row in seed_rows]))
    auc_ratio = stabilization_mean / (fixed_mean + 1e-12)

    # すべて実行前に固定した確認条件。
    gates = {
        "initial_history_direction_is_reproduced": initial_low > 0,
        "stabilization_to_fixed_auc_ratio_exceeds_1_20": auc_ratio > 1.20,
        "paired_auc_difference_ci_excludes_zero": difference_low > 0,
        "exploration_auc_below_half_stabilization": exploration_mean < 0.50 * stabilization_mean,
    }
    result = {
        "preregistered": {
            "seeds": CONFIRMATORY_SEEDS,
            "history_steps": HISTORY_STEPS,
            "gates": {
                "auc_ratio": "> 1.20",
                "paired_difference_ci": "lower bound > 0",
                "exploration_relative_auc": "< 0.50 x stabilization",
            },
        },
        "initial_gap": {"mean": initial_mean, "ci_low": initial_low, "ci_high": initial_high},
        "auc": {
            "fixed_mean": fixed_mean,
            "stabilization_mean": stabilization_mean,
            "exploration_mean": exploration_mean,
            "stabilization_to_fixed_ratio": auc_ratio,
            "paired_difference_mean": difference_mean,
            "paired_difference_ci_low": difference_low,
            "paired_difference_ci_high": difference_high,
        },
        "gates": gates,
    }

    with open(output_dir / "seed_level_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({"config": asdict(cfg), "device": str(device)}, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# 弱履歴・独立seed確認結果", "",
        f"- 初期履歴差: {initial_mean:.4f}（95% CI [{initial_low:.4f}, {initial_high:.4f}]）",
        f"- fixed AUC: {fixed_mean:.4f}",
        f"- stabilization AUC: {stabilization_mean:.4f}",
        f"- stabilization / fixed: {auc_ratio:.3f}",
        f"- 対応AUC差: {difference_mean:.4f}（95% CI [{difference_low:.4f}, {difference_high:.4f}]）",
        f"- exploration AUC: {exploration_mean:.4f}", "",
        "## 事前判定", "",
    ]
    report.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items())
    report_text = "\n".join(report)
    (output_dir / "result_report.md").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
