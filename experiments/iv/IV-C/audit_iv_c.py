#!/usr/bin/env python3
"""実験IV-C確認CSVを、実験コードとは独立に再集計する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STATES = ("A_formed", "neutral", "B_formed")
ACQUISITIONS = ("A_precise", "balanced", "B_precise")


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci(values, rng, samples=10000):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_confirm")
    parser.add_argument("--script", default="experiment_iv_c.py")
    parser.add_argument("--preregistration", default="preregistration_confirm.json")
    args = parser.parse_args()
    root = Path(args.results)
    cells = read_csv(root / "cell_summary.csv")
    timecourse = read_csv(root / "timecourse.csv")
    sensors = read_csv(root / "sensor_diagnostics.csv")
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260903)

    seeds = sorted({int(row["seed"]) for row in cells})
    by = {(int(row["seed"]), row["initial_state"], row["acquisition"]): row for row in cells}
    complete_keys = {(seed, state, acq) for seed in seeds for state in STATES for acq in ACQUISITIONS}
    observed_keys = set(by)
    clone_max_difference = 0.0
    for seed in seeds:
        for state in STATES:
            starts = [float(by[(seed, state, acq)]["initial_s_high"]) for acq in ACQUISITIONS]
            clone_max_difference = max(clone_max_difference, max(starts) - min(starts))

    def paired_effect(state, field="final_s_high"):
        return [float(by[(seed, state, "A_precise")][field]) -
                float(by[(seed, state, "B_precise")][field]) for seed in seeds]

    formed_effect = [0.5 * (paired_effect("A_formed")[i] + paired_effect("B_formed")[i])
                     for i in range(len(seeds))]
    low_formed_effect = [0.5 * (paired_effect("A_formed", "final_s_low")[i] +
                                paired_effect("B_formed", "final_s_low")[i])
                         for i in range(len(seeds))]
    neutral_effect = paired_effect("neutral")
    a_to_b = [float(by[(seed, "A_formed", "B_precise")]["final_s_high"]) for seed in seeds]
    b_to_a = [float(by[(seed, "B_formed", "A_precise")]["final_s_high"]) for seed in seeds]
    a_cross = [float(by[(seed, "A_formed", "B_precise")]["zero_crossed"]) for seed in seeds]
    b_cross = [float(by[(seed, "B_formed", "A_precise")]["zero_crossed"]) for seed in seeds]
    a_cross_step = [float(by[(seed, "A_formed", "B_precise")]["zero_crossing_step"]) for seed in seeds]
    b_cross_step = [float(by[(seed, "B_formed", "A_precise")]["zero_crossing_step"]) for seed in seeds]
    state_diff_a = [paired_effect("A_formed")[i] - neutral_effect[i] for i in range(len(seeds))]
    state_diff_b = [paired_effect("B_formed")[i] - neutral_effect[i] for i in range(len(seeds))]

    metrics = {
        "formed_acquisition_effect_high": ci(formed_effect, rng),
        "formed_acquisition_effect_low": ci(low_formed_effect, rng),
        "neutral_acquisition_effect_high": ci(neutral_effect, rng),
        "A_formed_to_B_final_s": ci(a_to_b, rng),
        "B_formed_to_A_final_s": ci(b_to_a, rng),
        "A_formed_to_B_crossing_rate": ci(a_cross, rng),
        "B_formed_to_A_crossing_rate": ci(b_cross, rng),
        "A_formed_to_B_crossing_step": ci(a_cross_step, rng),
        "B_formed_to_A_crossing_step": ci(b_cross_step, rng),
        "A_formed_vs_neutral_acquisition_effect": ci(state_diff_a, rng),
        "B_formed_vs_neutral_acquisition_effect": ci(state_diff_b, rng),
    }
    checks = {
        "40_seeds": len(seeds) == 40,
        "360_complete_cells": len(cells) == 360 and observed_keys == complete_keys,
        "20520_complete_timecourse_rows": len(timecourse) == 20520,
        "40_sensor_rows": len(sensors) == 40,
        "same_checkpoint_clones_exact": clone_max_difference == 0.0,
        "script_hash_matches": sha256(Path(args.script)) == provenance["script_sha256"],
        "preregistration_hash_matches": sha256(Path(args.preregistration)) == provenance["preregistration_sha256"],
        "primary_threshold": metrics["formed_acquisition_effect_high"]["mean"] >= prereg["formed_acquisition_minimum"] and metrics["formed_acquisition_effect_high"]["low"] > 0,
        "A_reaches_B": metrics["A_formed_to_B_final_s"]["high"] <= -prereg["opposite_destination_minimum"],
        "B_reaches_A": metrics["B_formed_to_A_final_s"]["low"] >= prereg["opposite_destination_minimum"],
        "crossing_rates": metrics["A_formed_to_B_crossing_rate"]["low"] >= prereg["minimum_crossing_rate"] and metrics["B_formed_to_A_crossing_rate"]["low"] >= prereg["minimum_crossing_rate"],
        "low_resolution_direction": metrics["formed_acquisition_effect_low"]["low"] > 0,
    }
    gate_values = result["confirmatory_gates"]
    checks["reported_gates_all_pass"] = all(item["pass"] for item in gate_values.values())
    audit = {
        "independent_bootstrap_seed": 20260903,
        "counts": {"seeds": len(seeds), "cells": len(cells), "timecourse_rows": len(timecourse),
                   "sensor_rows": len(sensors)},
        "clone_start_max_difference": clone_max_difference,
        "metrics": metrics,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    (root / "independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    lines = ["# 実験IV-C 独立再集計", "",
             f"- 完全性: {len(seeds)} seed / {len(cells)} cell / {len(timecourse)}時点行",
             f"- clone開始値の最大差: {clone_max_difference:.16f}",
             f"- 形成済み状態の取得効果: {metrics['formed_acquisition_effect_high']['mean']:.4f} "
             f"[{metrics['formed_acquisition_effect_high']['low']:.4f}, "
             f"{metrics['formed_acquisition_effect_high']['high']:.4f}]",
             f"- A形成→B方向のゼロ交差: {metrics['A_formed_to_B_crossing_rate']['mean']:.3f}, "
             f"平均{metrics['A_formed_to_B_crossing_step']['mean']:.1f}更新",
             f"- B形成→A方向のゼロ交差: {metrics['B_formed_to_A_crossing_rate']['mean']:.3f}, "
             f"平均{metrics['B_formed_to_A_crossing_step']['mean']:.1f}更新", "",
             "## 監査判定", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in checks.items())
    (root / "independent_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not audit["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
