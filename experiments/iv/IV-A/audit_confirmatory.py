#!/usr/bin/env python3
"""実験III-D確認結果の独立集計。主要判定は変更せず、対称性と補助対照を監査する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


AUDIT_SEED = 20260828
BOOTSTRAP_SAMPLES = 10000


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_confirmatory")
    return parser.parse_args()


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def main():
    root = Path(args().results_dir).resolve()
    with (root / "cell_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "initial_s_ref",
            "final_s_ref",
            "retained",
            "transition_event",
            "transition_time_restricted",
            "old_loss_auc",
            "alternative_loss_auc",
        ):
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])

    seeds = sorted({row["seed"] for row in rows})
    branches = sorted({row["reference"] for row in rows})
    expected = len(seeds) * len(branches) * 4 * 2
    if len(rows) != expected:
        raise RuntimeError(f"cell rows {len(rows)} != {expected}")

    lookup = {
        (row["seed"], row["reference"], row["acquisition"], row["update"]): row
        for row in rows
    }
    rng = np.random.default_rng(AUDIT_SEED)

    # 代替更新時の取得効果を、A/B形成枝ごとに再集計する。
    branch_time_effects = {}
    per_branch_values = {}
    for branch in branches:
        values = [
            lookup[(seed, branch, "reference", "alternative")]["transition_time_restricted"]
            - lookup[(seed, branch, "alternative", "alternative")]["transition_time_restricted"]
            for seed in seeds
        ]
        per_branch_values[branch] = values
        branch_time_effects[branch] = ci(values, rng)
    symmetry_difference = ci(
        [a - b for a, b in zip(per_branch_values["A"], per_branch_values["B"])],
        rng,
    )

    # 取得先と更新先が一致したとき、特化の絶対値がどれだけ強まるか。
    congruence_values = []
    for seed in seeds:
        branch_values = []
        for branch in branches:
            rr = abs(lookup[(seed, branch, "reference", "reference")]["final_s_ref"])
            ra = abs(lookup[(seed, branch, "reference", "alternative")]["final_s_ref"])
            ar = abs(lookup[(seed, branch, "alternative", "reference")]["final_s_ref"])
            aa = abs(lookup[(seed, branch, "alternative", "alternative")]["final_s_ref"])
            branch_values.append(0.5 * ((rr + aa) - (ra + ar)))
        congruence_values.append(float(np.mean(branch_values)))

    auxiliary = {}
    for acquisition in ("none", "random"):
        for update in ("reference", "alternative"):
            selected = [
                row
                for row in rows
                if row["acquisition"] == acquisition and row["update"] == update
            ]
            label = f"acq_{acquisition}__upd_{update}"
            auxiliary[label] = {
                "final_s_ref": float(np.mean([row["final_s_ref"] for row in selected])),
                "transition_rate": float(np.mean([row["transition_event"] for row in selected])),
                "restricted_transition_time": float(
                    np.mean([row["transition_time_restricted"] for row in selected])
                ),
            }

    failed_retention = [
        {
            "seed": row["seed"],
            "reference": row["reference"],
            "acquisition": row["acquisition"],
            "final_s_ref": row["final_s_ref"],
        }
        for row in rows
        if row["acquisition"] in ("reference", "alternative")
        and row["update"] == "reference"
        and row["retained"] < 0.5
    ]

    result = {
        "audit_seed": AUDIT_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "completeness": {
            "seeds": len(seeds),
            "branches": branches,
            "rows": len(rows),
            "expected_rows": expected,
            "unique_keys": len(lookup),
            "pass": len(rows) == expected == len(lookup),
        },
        "transition_time_acquisition_effect_by_formed_branch": branch_time_effects,
        "A_minus_B_time_effect": symmetry_difference,
        "exploratory_matching_acquisition_update_effect_on_abs_s": ci(congruence_values, rng),
        "auxiliary_controls": auxiliary,
        "failed_reference_update_retention_cases": failed_retention,
    }
    (root / "audit_confirmatory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 実験III-D 確認結果監査",
        "",
        f"- 完全性: {'PASS' if result['completeness']['pass'] else 'FAIL'} "
        f"({len(rows)}/{expected} cell rows)",
        "",
        "## A/B形成枝の対称性",
        "",
    ]
    for branch, value in branch_time_effects.items():
        lines.append(
            f"- {branch}形成枝の取得効果: {value['mean']:.3f} "
            f"(95%区間 [{value['low']:.3f}, {value['high']:.3f}])"
        )
    lines.append(
        f"- A－B差: {symmetry_difference['mean']:.3f} "
        f"(95%区間 [{symmetry_difference['low']:.3f}, {symmetry_difference['high']:.3f}])"
    )
    congruence = result["exploratory_matching_acquisition_update_effect_on_abs_s"]
    lines += [
        "",
        "## 探索的な一致効果",
        "",
        f"- 取得先と更新先が一致したときの |s| 増分: {congruence['mean']:.4f} "
        f"(95%区間 [{congruence['low']:.4f}, {congruence['high']:.4f}])",
        "- この量は確認実験前の主要判定に含めておらず、探索的結果として扱う。",
        "",
        "## 補助対照",
        "",
    ]
    for name, value in auxiliary.items():
        lines.append(
            f"- {name}: final s_ref={value['final_s_ref']:.4f}, "
            f"transition={value['transition_rate']:.3f}, "
            f"restricted time={value['restricted_transition_time']:.1f}"
        )
    lines += [
        "",
        f"- 基準側更新で枝保持閾値を下回ったcell: {len(failed_retention)}",
    ]
    (root / "audit_confirmatory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
