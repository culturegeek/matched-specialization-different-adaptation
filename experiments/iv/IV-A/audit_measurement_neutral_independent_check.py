#!/usr/bin/env python3
"""Independent aggregation check for the two Experiment III-D audits."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_measurement_neutral_audit"
PRIMARY = RESULTS / "primary_dual_probe_timecourse.csv"
NEUTRAL = RESULTS / "neutral_timecourse.csv"
OUT = RESULTS / "independent_check.json"
SEEDS = tuple(range(6200, 6240))
STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
         80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325,
         350, 375, 400)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_ci(values: list[float], seed: int = 20260830) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def first_nonpositive(curve: dict[int, float]) -> int:
    for step in STEPS[1:]:
        if curve[step] <= 0.0:
            return step
    return 425


def main() -> None:
    primary = read_rows(PRIMARY)
    neutral = read_rows(NEUTRAL)

    condition_parts = {
        "acq_reference__upd_reference": ("ref", "ref"),
        "acq_reference__upd_alternative": ("ref", "alt"),
        "acq_alternative__upd_reference": ("alt", "ref"),
        "acq_alternative__upd_alternative": ("alt", "alt"),
    }
    primary_keys = {
        (int(r["seed"]), r["reference"], *condition_parts[r["condition"]],
         int(r["step"]), r["measurement"])
        for r in primary
    }
    neutral_keys = {
        (int(r["seed"]), r["reference"], int(r["step"]), r["measurement"])
        for r in neutral
    }
    expected_primary = {
        (seed, branch, acq, upd, step, measurement)
        for seed in SEEDS
        for branch in ("A", "B")
        for acq in ("ref", "alt")
        for upd in ("ref", "alt")
        for step in STEPS
        for measurement in ("x_none", "x_both")
    }
    expected_neutral = {
        (seed, branch, step, measurement)
        for seed in SEEDS
        for branch in ("A", "B")
        for step in STEPS
        for measurement in ("x_none", "x_both")
    }

    curves: dict[tuple[int, str, str, str], dict[int, float]] = defaultdict(dict)
    d_values: dict[str, list[float]] = defaultdict(list)
    s_values: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        measurement = row["measurement"]
        d_values[measurement].extend((float(row["D_ref"]), float(row["D_alt"])))
        s_values[measurement].append(float(row["s_ref"]))
        if measurement == "x_both":
            acq, upd = condition_parts[row["condition"]]
            curves[(int(row["seed"]), row["reference"], acq, upd)][int(row["step"])] = float(row["s_ref"])

    crossing_effects = []
    auc_effects = []
    for seed in SEEDS:
        branch_cross = []
        branch_auc = []
        for branch in ("A", "B"):
            ref_curve = curves[(seed, branch, "ref", "alt")]
            alt_curve = curves[(seed, branch, "alt", "alt")]
            branch_cross.append(first_nonpositive(ref_curve) - first_nonpositive(alt_curve))
            early = [step for step in STEPS if step <= 100]
            differences = [ref_curve[s] - alt_curve[s] for s in early]
            branch_auc.append(float(np.trapezoid(differences, early) / 100.0))
        crossing_effects.append(float(np.mean(branch_cross)))
        auc_effects.append(float(np.mean(branch_auc)))

    neutral_curves: dict[tuple[int, str, str], dict[int, float]] = defaultdict(dict)
    for row in neutral:
        neutral_curves[(int(row["seed"]), row["reference"], row["measurement"])][int(row["step"])] = float(row["s_ref"])

    neutral_summary = {}
    for measurement in ("x_none", "x_both"):
        initials = []
        finals = []
        ratios = []
        retained = []
        for seed in SEEDS:
            branch_initial = []
            branch_final = []
            branch_retained = []
            for branch in ("A", "B"):
                curve = neutral_curves[(seed, branch, measurement)]
                branch_initial.append(curve[0])
                branch_final.append(curve[400])
                branch_retained.append(float(curve[400] >= 0.08))
            initial = float(np.mean(branch_initial))
            final = float(np.mean(branch_final))
            initials.append(initial)
            finals.append(final)
            ratios.append(float(np.mean([
                branch_final[index] / branch_initial[index]
                for index in range(2)
            ])))
            retained.append(float(np.mean(branch_retained)))
        neutral_summary[measurement] = {
            "initial_s": mean_ci(initials),
            "final_s": mean_ci(finals),
            "final_over_initial": mean_ci(ratios),
            "retention_at_0_08": mean_ci(retained),
        }

    result = {
        "completeness": {
            "primary_exact": primary_keys == expected_primary,
            "neutral_exact": neutral_keys == expected_neutral,
            "primary_missing": len(expected_primary - primary_keys),
            "primary_extra": len(primary_keys - expected_primary),
            "neutral_missing": len(expected_neutral - neutral_keys),
            "neutral_extra": len(neutral_keys - expected_neutral),
        },
        "common_low_noise": {
            "D_range": [min(d_values["x_both"]), max(d_values["x_both"])],
            "s_range": [min(s_values["x_both"]), max(s_values["x_both"])],
            "zero_crossing_time_effect": mean_ci(crossing_effects),
            "first_100_s_mean_difference": mean_ci(auc_effects),
        },
        "neutral": neutral_summary,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
