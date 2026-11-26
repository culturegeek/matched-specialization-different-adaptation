"""Audit the exploratory common-weight control from mirrored-direction records."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def records(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check(root):
    study = root / "experiments/iv-g/equal-weight-080"
    results = study / "results"
    data = json.loads((results / "result.json").read_text())
    plan = json.loads((study / "plan.json").read_text())
    provenance = json.loads((results / "provenance.json").read_text())
    for paths, base in ((provenance["source_hashes"], root), (provenance["output_hashes"], results),
                        ({"run.py": provenance["runner_sha256"], "plan.json": provenance["plan_sha256"]}, study)):
        for name, expected in paths.items():
            assert sha256((base / name).read_bytes()).hexdigest() == expected, name
    directions = records(results / "directional.csv")
    summaries = records(results / "per_seed.csv")
    original = records(root / "experiments/iv-g/results/saved_decomposition_seed_summary.csv")
    seeds = list(range(20000, 20040))
    assert len(directions) == 80 and [int(r["seed"]) for r in summaries] == seeds
    assert [int(r["seed"]) for r in original] == seeds
    by_seed = defaultdict(dict)
    for raw in directions:
        seed, direction = int(raw["seed"]), raw["direction"]
        assert direction not in by_seed[seed]
        vals = {key: float(value) for key, value in raw.items() if key not in ("seed", "direction")}
        for pair, first in (("acq_minus_mix", "acq"), ("alt_minus_mix", "alt")):
            for key in plan["endpoint_margins"]:
                np.testing.assert_allclose(vals[f"{pair}_{key}"], vals[f"{first}_start_{key}"] - vals[f"mix_start_{key}"], rtol=0, atol=1e-14)
        for route in ("alt", "mix"):
            for mode in plan["pulse_modes"]:
                calc = vals[f"{route}_{mode}_acq_response"] - vals[f"{route}_{mode}_upd_response"]
                np.testing.assert_allclose(vals[f"{route}_{mode}"], calc, rtol=0, atol=1e-14)
            for rec, key in (("acq","acq"),("upd",route)):
                expected = vals[f"{key}_start_aligned_s"] - vals[f"{route}_{rec}_end_s"]
                np.testing.assert_allclose(vals[f"{route}_own_{rec}_response"], expected, atol=1e-14,rtol=0)
            np.testing.assert_allclose(vals[f"{route}_pattern"], vals[f"{route}_v_equal_scale"] - vals[f"{route}_v_other_equal_scale"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(vals["alt_minus_mix_pattern"], vals["alt_pattern"] - vals["mix_pattern"], atol=1e-14, rtol=0)
        np.testing.assert_allclose(vals["alt_minus_mix_own_contrast"], vals["alt_own"] - vals["mix_own"], atol=1e-14, rtol=0)
        by_seed[seed][direction] = vals
    assert sorted(by_seed) == seeds and all(set(by_seed[s]) == {"A", "B"} for s in seeds)
    keys = list(data["estimates"])
    assert set(keys) == set(summaries[0]) - {"seed"} == set(by_seed[seeds[0]]["A"])
    combined = np.array([[(by_seed[s]["A"][k] + by_seed[s]["B"][k]) / 2 for k in keys] for s in seeds])
    np.testing.assert_allclose(combined, [[float(r[k]) for k in keys] for r in summaries], rtol=0, atol=1e-14)
    draws = np.random.default_rng(plan["bootstrap_seed"]).integers(len(seeds), size=(plan["bootstrap_draws"], len(seeds)))
    for j, key in enumerate(keys):
        values = combined[:, j]
        low, high = np.percentile(values[draws].mean(axis=1), [2.5,97.5])
        est = data["estimates"][key]
        np.testing.assert_allclose([values.mean(), low, high],[est["mean"], est["low"], est["high"]], rtol=0, atol=1e-12, err_msg=key)
    for pair, gates in data["endpoint_pass"].items():
        for key, margin in plan["endpoint_margins"].items():
            est = data["estimates"][f"{pair}_{key}"]
            assert gates[key] == (-margin <= est["low"] <= est["high"] <= margin)
    for condition, count in data["switches_at_update_10"].items():
        assert count == sum(float(r[f"{condition}_end_s"]) <= 0 for r in directions)
    for col, old in (("alt_v_equal_scale","global_common_scale_own"),("alt_v_other_equal_scale","global_common_scale_swap"),("alt_pattern","global_pattern_effect")):
        np.testing.assert_allclose(combined[:,keys.index(col)], [float(r[old]) for r in original], atol=1e-6,rtol=0)
    assert len(list((results / "checkpoints").glob("seed_*.pt"))) == 40
    print(f"PASS: {len(keys)} bootstrap estimates from 80 directional records, all endpoint decisions, four counts, 40 checkpoint pairs and alternating reference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    check(parser.parse_args().archive_root.resolve())

