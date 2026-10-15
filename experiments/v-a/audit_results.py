#!/usr/bin/env python3
"""実験V-Aの確認CSVを、実験本体をimportせずに独立再集計する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TARGETS = ("A", "B", "C")
CAPACITIES = (2, 3)
CONDITIONS = (
    ("A_precise", "balanced"), ("B_precise", "balanced"),
    ("C_precise", "balanced"), ("balanced", "balanced"),
    ("balanced", "A_heavy"), ("balanced", "B_heavy"),
    ("balanced", "C_heavy"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_confirm")
    parser.add_argument("--preregistration", default="preregistration_confirmatory.json")
    parser.add_argument("--script", default="experiment_v_a.py")
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def ci(values, rng, samples=5000):
    values = np.asarray(values, dtype=float)
    bootstrap = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(bootstrap, 0.025)),
            "high": float(np.quantile(bootstrap, 0.975))}


def plane_ratio(by_seed, capacity, seeds, rng, samples=5000):
    ratios = []
    seed_array = np.asarray(seeds)
    for _ in range(samples):
        sampled = seed_array[rng.integers(0, len(seeds), size=len(seeds))]
        effects = []
        for target in TARGETS:
            vectors = []
            for seed in sampled:
                precise = by_seed[(capacity, int(seed), f"{target}_precise", "balanced")]
                neutral = by_seed[(capacity, int(seed), "balanced", "balanced")]
                vectors.append([number(precise, f"final_high_q_{coordinate}")
                                - number(neutral, f"final_high_q_{coordinate}")
                                for coordinate in TARGETS])
            effects.append(np.mean(vectors, axis=0))
        matrix = np.asarray(effects)
        matrix -= matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, compute_uv=False)
        ratios.append(float(singular[1] / (singular[0] + 1e-12)))
    return ci(ratios, rng, samples=5000)


def main():
    args = parse_args()
    results = Path(args.results)
    prereg_path = Path(args.preregistration)
    script_path = Path(args.script)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    reported = json.loads((results / "result.json").read_text(encoding="utf-8"))
    provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8"))
    cells = read_csv(results / "cell_summary.csv")
    timecourse = read_csv(results / "timecourse.csv")
    sensors = read_csv(results / "sensor_diagnostics.csv")
    seeds = prereg["fixed_design"]["seeds"]
    thresholds = prereg["thresholds"]

    expected_cells = len(seeds) * len(CAPACITIES) * len(CONDITIONS)
    checkpoint_count = len(timecourse) // expected_cells
    completeness = (
        len(cells) == expected_cells
        and len({(int(r["seed"]), int(r["bottleneck"]), r["acquisition"], r["update"])
                 for r in cells}) == expected_cells
        and checkpoint_count * expected_cells == len(timecourse)
        and len(sensors) == len(seeds)
    )
    by_seed = {(int(r["bottleneck"]), int(r["seed"]), r["acquisition"], r["update"]): r
               for r in cells}
    rng = np.random.default_rng(20260903)
    gates = {"completeness": completeness}
    details = {"row_counts": {"cells": len(cells), "timecourse": len(timecourse),
                              "sensors": len(sensors), "checkpoints_per_cell": checkpoint_count}}

    initial_max_difference = 0.0
    initial_columns = [f"initial_high_D_{target}" for target in TARGETS]
    initial_columns += [f"initial_high_q_{target}" for target in TARGETS]
    for capacity in CAPACITIES:
        for seed in seeds:
            group = [by_seed[(capacity, seed, acquisition, update)]
                     for acquisition, update in CONDITIONS]
            for column in initial_columns:
                values = [number(row, column) for row in group]
                initial_max_difference = max(initial_max_difference, max(values) - min(values))
    gates["identical_initial_state_within_seed"] = initial_max_difference == 0.0
    details["initial_max_difference"] = initial_max_difference

    for target in TARGETS:
        gains = []
        for row in sensors:
            own = number(row, f"D_{target}_when_{target}_precise")
            other = [number(row, f"D_{target}_when_{item}_precise")
                     for item in TARGETS if item != target]
            gains.append(float(np.mean(other) - own))
        estimate = ci(gains, rng)
        gates[f"sensor_{target}"] = estimate["low"] > 0
        details[f"sensor_{target}"] = estimate

    for capacity in CAPACITIES:
        acquisition_means = []
        for target in TARGETS:
            others = [item for item in TARGETS if item != target]
            acq_gain, acq_asym, acq_win, upd_gain, upd_win = [], [], [], [], []
            for seed in seeds:
                precise = by_seed[(capacity, seed, f"{target}_precise", "balanced")]
                neutral = by_seed[(capacity, seed, "balanced", "balanced")]
                heavy = by_seed[(capacity, seed, "balanced", f"{target}_heavy")]
                acq_gain.append(number(precise, f"final_high_q_{target}")
                                - number(neutral, f"final_high_q_{target}"))
                acq_asym.append(abs(number(precise, f"final_high_q_{others[0]}")
                                     - number(precise, f"final_high_q_{others[1]}")))
                acq_win.append(number(precise, f"final_high_D_{target}")
                               < min(number(precise, f"final_high_D_{item}") for item in others))
                upd_gain.append(number(heavy, f"final_high_q_{target}")
                                - number(neutral, f"final_high_q_{target}"))
                upd_win.append(number(heavy, f"final_high_D_{target}")
                               < min(number(heavy, f"final_high_D_{item}") for item in others))
            acq_estimate, asym_estimate, upd_estimate = ci(acq_gain, rng), ci(acq_asym, rng), ci(upd_gain, rng)
            acquisition_means.append(acq_estimate["mean"])
            minimum = thresholds[f"capacity_{capacity}_minimum_q_gain"]
            gates[f"capacity_{capacity}_acquisition_{target}"] = (
                acq_estimate["low"] > 0 and acq_estimate["mean"] >= minimum
                and np.mean(acq_win) >= thresholds["minimum_favored_fraction"])
            gates[f"capacity_{capacity}_update_{target}"] = (
                upd_estimate["low"] > 0
                and np.mean(upd_win) >= thresholds["minimum_favored_fraction"])
            gates[f"capacity_{capacity}_other_symmetry_{target}"] = (
                asym_estimate["high"] <= thresholds["maximum_other_target_asymmetry"])
            details[f"capacity_{capacity}_{target}"] = {
                "acquisition_q_gain": acq_estimate,
                "acquisition_favored_fraction": float(np.mean(acq_win)),
                "other_absolute_asymmetry": asym_estimate,
                "update_q_gain": upd_estimate,
                "update_favored_fraction": float(np.mean(upd_win)),
            }

        neutral_means = {}
        for target in TARGETS:
            neutral_means[target] = float(np.mean([
                number(by_seed[(capacity, seed, "balanced", "balanced")],
                       f"final_high_q_{target}") for seed in seeds]))
        neutral_range = max(neutral_means.values()) - min(neutral_means.values())
        effect_range = max(acquisition_means) - min(acquisition_means)
        ratio = plane_ratio(by_seed, capacity, seeds, rng)
        gates[f"capacity_{capacity}_balanced_label_symmetry"] = (
            neutral_range <= thresholds["maximum_balanced_group_mean_range"])
        gates[f"capacity_{capacity}_label_exchangeability"] = (
            effect_range <= thresholds["maximum_acquisition_effect_range"])
        gates[f"capacity_{capacity}_two_dimensional_allocation_span"] = (
            ratio["low"] >= thresholds["minimum_plane_singular_ratio"])
        details[f"capacity_{capacity}_symmetry_and_span"] = {
            "balanced_group_means": neutral_means,
            "balanced_group_mean_range": neutral_range,
            "acquisition_effect_mean_range": effect_range,
            "plane_second_to_first_ratio": ratio,
        }

    reported_gates = {name: bool(item["pass"])
                      for name, item in reported["confirmatory_gates"].items()}
    recomputed_registered = {name: value for name, value in gates.items()
                             if name not in {"completeness", "identical_initial_state_within_seed"}}
    gates["reported_gate_vector_matches"] = reported_gates == recomputed_registered
    gates["script_hash_matches_registration"] = sha256(script_path) == prereg["script_sha256"]
    gates["script_hash_matches_provenance"] = sha256(script_path) == provenance["script_sha256"]
    gates["preregistration_hash_matches_provenance"] = (
        sha256(prereg_path) == provenance["preregistration_sha256"])

    gates = {name: bool(value) for name, value in gates.items()}
    audit = {"experiment": "V-A", "audit_seed": 20260903, "gates": gates,
             "all_pass": all(gates.values()), "details": details}
    (results / "independent_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験V-A 独立再集計", "",
             f"生CSVから{sum(gates.values())}/{len(gates)}監査項目を再現。", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items())
    (results / "independent_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not audit["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
