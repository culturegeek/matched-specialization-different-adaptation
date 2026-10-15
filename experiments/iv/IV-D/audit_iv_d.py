#!/usr/bin/env python3
"""実験IV-D確認CSVの独立再集計。実験スクリプトやresult.jsonを読み込まない。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_confirm"
PREREG = HERE / "preregistration_confirmatory.json"
SCRIPT = HERE / "experiment_iv_d.py"


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "")
    return None if value in (None, "") else float(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci(values, rng, samples=10000):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def classify(interval, margin):
    if interval["low"] > margin or interval["high"] < -margin:
        return "different"
    if interval["low"] >= -margin and interval["high"] <= margin:
        return "equivalent_within_margin"
    return "unresolved"


def main():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    formation = read_csv(RESULTS / "formation_endpoints.csv")
    summaries = read_csv(RESULTS / "protocol_summary.csv")
    timecourse = read_csv(RESULTS / "timecourse.csv")
    geometry = read_csv(RESULTS / "geometry.csv")
    seeds = sorted({int(row["seed"]) for row in formation})
    assert seeds == prereg["fixed_design"]["seeds"]
    assert len(formation) == 40 * 2 * 2
    assert len(summaries) == 40 * 2 * 2 * 2 * 4
    assert len(timecourse) == 218880
    assert len(geometry) == 40 * 2

    form = {(int(r["seed"]), r["route"], r["direction"]): r for r in formation}
    protocol = {(int(r["seed"]), r["route"], r["direction"],
                 r["optimizer_mode"], r["protocol"]): r for r in summaries}
    rng = np.random.default_rng(20260903)

    match = {}
    for metric, margin in prereg["formation_match_tolerances"].items():
        values = [0.5 * sum(number(form[(seed, "acquisition", direction)], metric) -
                            number(form[(seed, "update", direction)], metric)
                            for direction in ("A", "B")) for seed in seeds]
        interval = ci(values, rng)
        match[metric] = {"value": interval,
                         "pass": interval["low"] >= -margin and interval["high"] <= margin,
                         "margin": margin}

    route_effects = defaultdict(dict)
    for name, spec in prereg["outcome_equivalence_margins"].items():
        mode, protocol_name, outcome = spec["optimizer_mode"], spec["protocol"], spec["outcome"]
        values = [0.5 * sum(
            number(protocol[(seed, "acquisition", direction, mode, protocol_name)], outcome) -
            number(protocol[(seed, "update", direction, mode, protocol_name)], outcome)
            for direction in ("A", "B")) for seed in seeds]
        interval = ci(values, rng)
        route_effects[name] = {"value": interval,
                               "classification": classify(interval, spec["margin"]),
                               "margin": spec["margin"]}

    interactions = {}
    for name, spec in prereg["optimizer_state_interaction_minima"].items():
        protocol_name, outcome = spec["protocol"], spec["outcome"]
        values = []
        for seed in seeds:
            carried = 0.5 * sum(
                number(protocol[(seed, "acquisition", direction, "carried", protocol_name)], outcome) -
                number(protocol[(seed, "update", direction, "carried", protocol_name)], outcome)
                for direction in ("A", "B"))
            reset = 0.5 * sum(
                number(protocol[(seed, "acquisition", direction, "reset", protocol_name)], outcome) -
                number(protocol[(seed, "update", direction, "reset", protocol_name)], outcome)
                for direction in ("A", "B"))
            values.append(carried - reset)
        interval = ci(values, rng)
        interactions[name] = {"value": interval, "minimum": spec["minimum"],
                              "pass": interval["low"] >= spec["minimum"]}

    starts = defaultdict(list)
    for row in timecourse:
        if int(row["step"]) == 0:
            key = (int(row["seed"]), row["route"], row["direction"])
            starts[key].append(float(row["aligned_s"]))
    max_start_difference = max(max(values) - min(values) for values in starts.values())

    audit = {
        "source": "raw CSV only; result.json and experiment module not imported",
        "completeness": {"seeds": len(seeds), "formation_rows": len(formation),
                         "protocol_rows": len(summaries), "timecourse_rows": len(timecourse),
                         "geometry_rows": len(geometry)},
        "same_checkpoint_max_start_difference": max_start_difference,
        "formation_match": match,
        "formation_match_all_pass": all(item["pass"] for item in match.values()),
        "outcome_classifications": dict(route_effects),
        "optimizer_state_interaction": interactions,
        "optimizer_state_interaction_all_pass": all(item["pass"] for item in interactions.values()),
        "hashes": {"script": sha256(SCRIPT), "preregistration": sha256(PREREG),
                   "audit": sha256(Path(__file__).resolve())},
    }
    (RESULTS / "independent_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験IV-D 独立再集計", "",
             "実験スクリプトとresult.jsonを使わず、確認CSVから再計算した。", "",
             f"- seed: {len(seeds)}", f"- 形成セル: {len(formation)}",
             f"- 後続条件セル: {len(summaries)}", f"- 時系列行: {len(timecourse)}",
             f"- 同一checkpoint開始値の最大差: {max_start_difference:.12f}",
             f"- 形成終点の一致: {'PASS' if audit['formation_match_all_pass'] else 'FAIL'}",
             f"- optimizer状態の交互作用: {'PASS' if audit['optimizer_state_interaction_all_pass'] else 'FAIL'}",
             "", "## 後続応答の分類", ""]
    lines.extend(f"- {name}: {item['classification']} "
                 f"({item['value']['mean']:.4f}, 95%区間 "
                 f"[{item['value']['low']:.4f}, {item['value']['high']:.4f}])"
                 for name, item in route_effects.items())
    (RESULTS / "independent_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
