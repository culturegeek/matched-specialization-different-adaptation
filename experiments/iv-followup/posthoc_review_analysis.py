"""Post-hoc analyses requested after the IV-E follow-up review.

This script reads only frozen CSV outputs.  It does not import the training or
registered analysis modules.  All comparisons below are exploratory because
their exact combinations were not specified in protocol.md.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_confirm"
OUT = ROOT / "posthoc_review_results.json"
RNG_SEED = 20260904
N_BOOT = 20_000


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def interval(values: list[float] | np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "n_seed": int(len(values)),
    }


def seed_direction_means(rows: list[dict[str, str]], field: str) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["route"])].append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def main() -> None:
    moments = read_csv("optimizer_moments.csv")
    route_effects = read_csv("seed_route_effects.csv")
    endpoints = read_csv("formation_endpoints.csv")

    # Participation ratio: (sum v)^2 / sum(v^2); v_l2 stores sqrt(sum(v^2)).
    pr_rows = []
    for row in moments:
        pr_rows.append({
            **row,
            "participation_ratio": float(row["v_sum"]) ** 2 / float(row["v_l2"]) ** 2,
        })
    result: dict[str, object] = {"status": "post_hoc_exploratory", "bootstrap_seed": RNG_SEED}
    result["participation_ratio"] = {}
    for route in ("acquisition", "update"):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in pr_rows:
            if row["route"] == route:
                grouped[int(row["seed"])].append(float(row["participation_ratio"]))
        result["participation_ratio"][route] = interval([np.mean(grouped[s]) for s in sorted(grouped)])

    moment_lookup = {
        (int(row["seed"]), row["route"], row["direction"]): row
        for row in moments
    }
    seeds = sorted({seed for seed, _, _ in moment_lookup})
    # Match the registered analysis: calculate the route ratio separately for
    # each mirrored direction, then average the two ratios within seed.
    v_ratio = np.array([
        np.mean([
            float(moment_lookup[(s, "acquisition", d)]["v_sum"])
            / float(moment_lookup[(s, "update", d)]["v_sum"])
            for d in ("A", "B")
        ])
        for s in seeds
    ])
    q_ratio = np.array([
        np.mean([
            float(moment_lookup[(s, "acquisition", d)]["effective_lr_mean"])
            / float(moment_lookup[(s, "update", d)]["effective_lr_mean"])
            for d in ("A", "B")
        ])
        for s in seeds
    ])
    uniform_q_ratio = np.array([
        np.mean([
            1.0 / np.sqrt(
                float(moment_lookup[(s, "acquisition", d)]["v_sum"])
                / float(moment_lookup[(s, "update", d)]["v_sum"])
            )
            for d in ("A", "B")
        ])
        for s in seeds
    ])
    result["route_ratios"] = {
        "v_sum_acquisition_over_update": interval(v_ratio),
        "q_acquisition_over_update": interval(q_ratio),
        "q_ratio_predicted_from_v_sum_under_uniform_v": interval(uniform_q_ratio),
    }

    by_mode_step: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in route_effects:
        by_mode_step[(row["mode"], row["trajectory"], int(row["step"]))][int(row["seed"])] = float(row["route_drop_difference"])

    def values(mode: str, trajectory: str, step: int) -> np.ndarray:
        data = by_mode_step[(mode, trajectory, step)]
        return np.array([data[s] for s in sorted(data)])

    own = values("own", "continued_opposition", 10)
    scalar_own = values("scalar_v_own", "continued_opposition", 10)
    equal = values("v_equal_scale", "continued_opposition", 10)
    scalar_equal = values("scalar_v_equal", "continued_opposition", 10)
    interaction = (own - scalar_own) - (equal - scalar_equal)
    result["posthoc_two_by_two"] = {
        "own": interval(own),
        "scalar_v_own": interval(scalar_own),
        "v_equal_scale": interval(equal),
        "scalar_v_equal": interval(scalar_equal),
        "scale_effect_route_pattern": interval(own - equal),
        "scale_effect_scalar_pattern": interval(scalar_own - scalar_equal),
        "pattern_effect_route_scale": interval(own - scalar_own),
        "pattern_effect_common_scale": interval(equal - scalar_equal),
        "interaction": interval(interaction),
        "additive_prediction_fraction_of_observed_mean": float((own.mean() - interaction.mean()) / own.mean()),
    }

    result["route_drop_difference_over_time"] = {}
    for mode in ("own", "reset", "sgd_calibrated"):
        result["route_drop_difference_over_time"][mode] = {}
        for step in (10, 20, 50, 100, 200, 500):
            result["route_drop_difference_over_time"][mode][str(step)] = interval(
                values(mode, "continued_opposition", step)
            )

    # Endpoint index differences and response differences must not be mixed.
    endpoint_grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in endpoints:
        endpoint_grouped[(int(row["seed"]), row["route"])].append(float(row["aligned_s"]))
    start_diff = {
        seed: float(np.mean(endpoint_grouped[(seed, "acquisition")]) - np.mean(endpoint_grouped[(seed, "update")]))
        for seed in seeds
    }
    result["endpoint_vs_drop_difference"] = {}
    for trajectory, steps in (("continued_opposition", (500,)), ("pulse_then_neutral", (50, 510))):
        result["endpoint_vs_drop_difference"][trajectory] = {}
        for step in steps:
            drop = values("own", trajectory, step)
            end = np.array([start_diff[s] for s in seeds]) - drop
            result["endpoint_vs_drop_difference"][trajectory][str(step)] = {
                "drop_difference": interval(drop),
                "endpoint_allocation_difference": interval(end),
            }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
