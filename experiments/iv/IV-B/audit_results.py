#!/usr/bin/env python3
"""result.jsonを使わず、cell_summary.csvから中心比較と完全性を再集計する。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_CELLS = {
    (acq, upd)
    for acq in ("A_precise", "balanced", "B_precise")
    for upd in ("A_heavy", "balanced", "B_heavy")
}


def ci(values, seed=20260901, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "low": float(np.quantile(means, .025)), "high": float(np.quantile(means, .975)),
            "min": float(values.min()), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    with (result_dir / "cell_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_seed = defaultdict(dict)
    initial_spreads = []
    for row in rows:
        seed = int(row["seed"])
        key = (row["acquisition"], row["update"])
        if key in by_seed[seed]:
            raise RuntimeError(f"重複セル: seed={seed}, cell={key}")
        by_seed[seed][key] = row
    for seed, cells in by_seed.items():
        if set(cells) != EXPECTED_CELLS:
            raise RuntimeError(f"欠損セル: seed={seed}, cells={sorted(cells)}")
        initials = [float(row["initial_s_high"]) for row in cells.values()]
        initial_spreads.append(max(initials) - min(initials))
    primary = []
    secondary = []
    update_control = []
    a_cells, b_cells, neutral_cells = [], [], []
    for cells in by_seed.values():
        a = cells[("A_precise", "balanced")]
        b = cells[("B_precise", "balanced")]
        neutral = cells[("balanced", "balanced")]
        primary.append(float(a["final_s_high"]) - float(b["final_s_high"]))
        secondary.append(float(a["final_s_low"]) - float(b["final_s_low"]))
        update_control.append(
            float(cells[("balanced", "A_heavy")]["final_s_high"])
            - float(cells[("balanced", "B_heavy")]["final_s_high"])
        )
        a_cells.append(float(a["final_s_high"]))
        b_cells.append(float(b["final_s_high"]))
        neutral_cells.append(float(neutral["final_s_high"]))
    audit = {
        "n_seeds": len(by_seed), "row_count": len(rows),
        "all_9_cells_present": True,
        "maximum_initial_difference_within_seed": max(initial_spreads),
        "primary_A_precise_minus_B_precise": ci(primary),
        "secondary_low_resolution": ci(secondary),
        "positive_control_update_A_minus_B": ci(update_control),
        "A_precise_balanced_update": ci(a_cells),
        "B_precise_balanced_update": ci(b_cells),
        "balanced_balanced": ci(neutral_cells),
    }
    output = result_dir / "independent_audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
