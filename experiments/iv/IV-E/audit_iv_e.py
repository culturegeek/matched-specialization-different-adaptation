#!/usr/bin/env python3
"""実験IV-Eの独立監査。実験スクリプトをimportせず、保存CSVから再集計する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROUTES = ("acquisition", "update")
DIRECTIONS = ("A", "B")
PROTOCOLS = ("acquisition_pulse", "update_pulse", "combined_switch")
COMPONENTS = ("own", "m_other", "v_other", "both_other", "reset")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--script", required=True)
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ci(values, rng, n=10000):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def main():
    args = parse_args()
    results = Path(args.results)
    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    reported = json.loads((results / "result.json").read_text(encoding="utf-8"))["aggregate"]
    formation = read_csv(results / "formation_endpoints.csv")
    moments = read_csv(results / "optimizer_moment_endpoints.csv")
    protocols = read_csv(results / "protocol_summary.csv")
    timecourse = read_csv(results / "timecourse.csv")
    seeds = prereg["fixed_design"]["seeds"]

    expected = {
        "formation": len(seeds) * len(ROUTES) * len(DIRECTIONS),
        "moments": len(seeds) * len(ROUTES) * len(DIRECTIONS),
        "protocols": len(seeds) * len(ROUTES) * len(DIRECTIONS) * len(COMPONENTS) * len(PROTOCOLS),
        "timecourse": len(seeds) * len(ROUTES) * len(DIRECTIONS) * len(COMPONENTS) * len(PROTOCOLS) * 11,
    }
    observed = {"formation": len(formation), "moments": len(moments),
                "protocols": len(protocols), "timecourse": len(timecourse)}
    row_counts_pass = observed == expected

    protocol_keys = {(int(r["seed"]), r["route"], r["direction"],
                      r["optimizer_component"], r["protocol"]) for r in protocols}
    expected_keys = {(seed, route, direction, component, protocol)
                     for seed in seeds for route in ROUTES for direction in DIRECTIONS
                     for component in COMPONENTS for protocol in PROTOCOLS}
    combinations_pass = protocol_keys == expected_keys

    time_keys = {(int(r["seed"]), r["route"], r["direction"],
                  r["optimizer_component"], r["protocol"], int(r["step"]))
                 for r in timecourse}
    expected_time_keys = {(seed, route, direction, component, protocol, step)
                          for seed in seeds for route in ROUTES for direction in DIRECTIONS
                          for component in COMPONENTS for protocol in PROTOCOLS
                          for step in range(11)}
    timecourse_pass = time_keys == expected_time_keys

    starts = {}
    for row in protocols:
        key = (int(row["seed"]), row["route"], row["direction"])
        starts.setdefault(key, []).append(float(row["start_aligned_s"]))
    max_start_spread = max(max(values) - min(values) for values in starts.values())

    by_protocol = {(int(r["seed"]), r["route"], r["direction"],
                    r["optimizer_component"], r["protocol"]): r for r in protocols}
    rng = np.random.default_rng(20260904)
    independent = {}
    mean_match = {}
    gate_results = {}
    for protocol in PROTOCOLS:
        independent[protocol] = {}
        for component in COMPONENTS:
            values = [0.5 * sum(
                float(by_protocol[(seed, "acquisition", direction, component, protocol)]["pulse_drop"]) -
                float(by_protocol[(seed, "update", direction, component, protocol)]["pulse_drop"])
                for direction in DIRECTIONS) for seed in seeds]
            interval = ci(values, rng)
            independent[protocol][component] = interval
            reported_mean = reported["route_effects_acquisition_minus_update"][protocol][component]["mean"]
            mean_match[f"{protocol}:{component}"] = abs(interval["mean"] - reported_mean) < 1e-12

    for name, spec in prereg["primary_gates"].items():
        if spec["source"] == "route_effects_acquisition_minus_update":
            interval = independent[spec["protocol"]][spec["component"]]
        else:
            own = independent[spec["protocol"]]["own"]
            component = independent[spec["protocol"]][spec["component"]]
            # 区間は対応差を再計算する。
            values = [0.5 * sum(
                (float(by_protocol[(seed, "acquisition", direction, "own", spec["protocol"])]["pulse_drop"]) -
                 float(by_protocol[(seed, "update", direction, "own", spec["protocol"])]["pulse_drop"])) -
                (float(by_protocol[(seed, "acquisition", direction, spec["component"], spec["protocol"])]["pulse_drop"]) -
                 float(by_protocol[(seed, "update", direction, spec["component"], spec["protocol"])]["pulse_drop"]))
                for direction in DIRECTIONS) for seed in seeds]
            interval = ci(values, rng)
        threshold = spec["threshold"]
        if spec["rule"] == "lower_bound_at_least":
            passed = interval["low"] >= threshold
        elif spec["rule"] == "upper_bound_at_most":
            passed = interval["high"] <= threshold
        elif spec["rule"] == "equivalent_within":
            passed = interval["low"] >= -threshold and interval["high"] <= threshold
        else:
            raise ValueError(spec["rule"])
        gate_results[name] = {"pass": passed, "interval": interval}

    moment_by = {(int(r["seed"]), r["route"], r["direction"]): r for r in moments}
    moment_ratios = {}
    for metric in ("m_l2", "v_sum", "v_l2", "effective_lr_mean"):
        values = [0.5 * sum(float(moment_by[(seed, "acquisition", direction)][metric]) /
                            float(moment_by[(seed, "update", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        moment_ratios[metric] = ci(values, rng)

    script_hash = sha256(args.script)
    prereg_hash = sha256(args.preregistration)
    audit = {
        "row_counts": {"expected": expected, "observed": observed, "pass": row_counts_pass},
        "complete_combinations": combinations_pass,
        "complete_timecourse": timecourse_pass,
        "max_start_spread_across_interventions": max_start_spread,
        "reported_means_reproduced": all(mean_match.values()),
        "reported_mean_checks": mean_match,
        "independent_intervals": independent,
        "independent_primary_gates": gate_results,
        "independent_primary_gates_all_pass": all(item["pass"] for item in gate_results.values()),
        "moment_ratios_acquisition_over_update": moment_ratios,
        "script_sha256": script_hash,
        "script_hash_matches_preregistration": script_hash == prereg["script_sha256"],
        "preregistration_sha256": prereg_hash,
    }
    (results / "independent_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験IV-E 独立監査", "",
             f"- 行数: {'PASS' if row_counts_pass else 'FAIL'}",
             f"- 条件組合せ: {'PASS' if combinations_pass else 'FAIL'}",
             f"- 時系列step: {'PASS' if timecourse_pass else 'FAIL'}",
             f"- 介入間の開始値最大差: {max_start_spread:.12g}",
             f"- 報告平均の再現: {'PASS' if all(mean_match.values()) else 'FAIL'}",
             f"- 独立bootstrapによる主判定: {'PASS' if audit['independent_primary_gates_all_pass'] else 'FAIL'}",
             f"- 実験コードhash: {'PASS' if audit['script_hash_matches_preregistration'] else 'FAIL'}",
             "", "## 複合摂動の経路差"]
    for component in COMPONENTS:
        value = independent["combined_switch"][component]
        lines.append(f"- {component}: {value['mean']:.6f} "
                     f"[{value['low']:.6f}, {value['high']:.6f}]")
    lines.extend(["", "## Adam形成終点の比（取得主導／更新主導）"])
    for metric, value in moment_ratios.items():
        lines.append(f"- {metric}: {value['mean']:.6f} "
                     f"[{value['low']:.6f}, {value['high']:.6f}]")
    (results / "independent_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
