#!/usr/bin/env python3
"""実験IV-E追加対照を保存CSVだけから独立再集計する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROUTES = ("constant", "variable")
DIRECTIONS = ("A", "B")
PROTOCOLS = ("acquisition_pulse", "update_pulse", "combined_switch")
COMPONENTS = ("own", "m_other", "v_other", "both_other", "reset")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ci(values, seed, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def in_margin(interval, margin):
    return interval["low"] >= -margin and interval["high"] <= margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_confirm")
    parser.add_argument("--preregistration", default="preregistration_confirmatory.json")
    args = parser.parse_args()
    root = Path(args.results)
    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    formation = read_csv(root / "formation_endpoints.csv")
    moments = read_csv(root / "optimizer_moment_endpoints.csv")
    protocols = read_csv(root / "protocol_summary.csv")
    timecourse = read_csv(root / "timecourse.csv")
    reported = json.loads((root / "result.json").read_text(encoding="utf-8"))

    expected_seeds = prereg["fixed_design"]["seeds"]
    seeds = sorted({int(row["seed"]) for row in formation})
    counts = {
        "formation": len(formation),
        "moments": len(moments),
        "protocols": len(protocols),
        "timecourse": len(timecourse),
    }
    expected_counts = {
        "formation": len(seeds) * 2 * 2,
        "moments": len(seeds) * 2 * 2,
        "protocols": len(seeds) * 2 * 2 * 5 * 3,
        "timecourse": len(seeds) * 2 * 2 * 5 * 3 * 11,
    }
    keys_form = {(int(r["seed"]), r["route"], r["direction"]): r for r in formation}
    keys_mom = {(int(r["seed"]), r["route"], r["direction"]): r for r in moments}
    keys_pro = {(int(r["seed"]), r["route"], r["direction"],
                 r["optimizer_component"], r["protocol"]): r for r in protocols}

    match = {}
    gates = {}
    for index, (metric, margin) in enumerate(
            prereg["formation_match_tolerances"].items(), 1):
        values = [0.5 * sum(float(keys_form[(seed, "constant", direction)][metric]) -
                            float(keys_form[(seed, "variable", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        interval = ci(values, 20261000 + index)
        match[metric] = interval
        gates["formation_" + metric] = in_margin(interval, margin)

    optimizer = {}
    for index, metric in enumerate(("v_sum", "effective_lr_mean"), 1):
        values = [0.5 * sum(float(keys_mom[(seed, "constant", direction)][metric]) -
                            float(keys_mom[(seed, "variable", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        optimizer[metric] = ci(values, 20261100 + index)
    v_ratios = [0.5 * sum(float(keys_mom[(seed, "constant", direction)]["v_sum"]) /
                          float(keys_mom[(seed, "variable", direction)]["v_sum"])
                          for direction in DIRECTIONS) for seed in seeds]
    optimizer["v_sum_ratio"] = ci(v_ratios, 20261103)

    effects = {}
    for p_index, protocol in enumerate(PROTOCOLS):
        effects[protocol] = {}
        for c_index, component in enumerate(COMPONENTS):
            values = [0.5 * sum(
                float(keys_pro[(seed, "constant", direction, component, protocol)]["pulse_drop"]) -
                float(keys_pro[(seed, "variable", direction, component, protocol)]["pulse_drop"])
                for direction in DIRECTIONS) for seed in seeds]
            effects[protocol][component] = ci(
                values, 20261200 + 10 * p_index + c_index)

    own = effects["combined_switch"]["own"]
    m_other = effects["combined_switch"]["m_other"]
    v_other = effects["combined_switch"]["v_other"]
    both_other = effects["combined_switch"]["both_other"]
    reset = effects["combined_switch"]["reset"]
    gates.update({
        "variable_schedule_increases_v": optimizer["v_sum"]["high"] <= -0.0025,
        "own_state_produces_response_difference": own["low"] >= 0.01,
        "m_swap_preserves_positive_difference": m_other["low"] >= 0.015,
        "v_swap_reverses_difference": v_other["high"] <= -0.005,
        "v_swap_change_is_large": (own["mean"] - v_other["mean"]) >= 0.025,
        "both_swap_reverses_difference": both_other["high"] <= -0.003,
        "reset_removes_difference": in_margin(reset, 0.01),
    })

    start_groups = {}
    for row in timecourse:
        if int(row["step"]) != 0:
            continue
        key = (int(row["seed"]), row["route"], row["direction"])
        start_groups.setdefault(key, []).append(float(row["aligned_s"]))
    max_start_difference = max(max(values) - min(values) for values in start_groups.values())

    reported_aggregate = reported["aggregate"]
    mean_checks = {
        "aligned_s": abs(match["aligned_s"]["mean"] -
                         reported_aggregate["formation_match"]["aligned_s"]["mean"]),
        "v_sum": abs(optimizer["v_sum"]["mean"] -
                     reported_aggregate["optimizer_constant_minus_variable"]["v_sum"]["mean"]),
        "combined_own": abs(own["mean"] -
                            reported_aggregate["route_effects_constant_minus_variable"]
                            ["combined_switch"]["own"]["mean"]),
        "combined_v_other": abs(v_other["mean"] -
                                reported_aggregate["route_effects_constant_minus_variable"]
                                ["combined_switch"]["v_other"]["mean"]),
    }
    audit = {
        "independent_from_experiment_module": True,
        "seeds_match_preregistration": seeds == expected_seeds,
        "counts": counts,
        "expected_counts": expected_counts,
        "counts_match": counts == expected_counts,
        "max_step0_difference_across_clones": max_start_difference,
        "formation_match_independent_bootstrap": match,
        "optimizer_independent_bootstrap": optimizer,
        "route_effects_independent_bootstrap": effects,
        "gates_independent": gates,
        "all_gates_pass": all(gates.values()),
        "reported_mean_absolute_differences": mean_checks,
        "reported_means_reproduced": max(mean_checks.values()) < 1e-12,
        "files_sha256": {name: sha256(root / name) for name in (
            "formation_endpoints.csv", "optimizer_moment_endpoints.csv",
            "protocol_summary.csv", "timecourse.csv", "result.json")},
    }
    (root / "independent_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験IV-E追加対照 独立監査", "",
             f"- seed: {'一致' if audit['seeds_match_preregistration'] else '不一致'}",
             f"- 行数: {'一致' if audit['counts_match'] else '不一致'}",
             f"- clone開始値の最大差: {max_start_difference:.12g}",
             f"- 報告平均の再現: {'PASS' if audit['reported_means_reproduced'] else 'FAIL'}",
             f"- 独立bootstrapによる全判定: {'PASS' if audit['all_gates_pass'] else 'FAIL'}", ""]
    for name, passed in gates.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    (root / "independent_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
