#!/usr/bin/env python3
"""実験IV-E追加対照のパイロット校正。

取得条件、更新回数、平均更新配分を揃えたまま、更新重みの時間的な
偏り方だけを変える。一定の0.80/0.20更新と、同じ平均をもつ変動更新の
形成終点を比較し、確認実験に用いる候補を固定する。
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "iv_e_base", HERE.parent / "IV-E" / "experiment_iv_e.py")
IVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = IVE
SPEC.loader.exec_module(IVE)
IVD = IVE.IVD
IVB = IVE.IVB

DIRECTIONS = ("A", "B")
CANDIDATES = {
    "constant_080": (0.80,),
    "shuffle_090_070": (0.90, 0.70),
    "shuffle_095_065": (0.95, 0.65),
    "shuffle_100_060": (1.00, 0.60),
    "alternate_090_070": (0.90, 0.70),
    "alternate_095_065": (0.95, 0.65),
    "alternate_100_060": (1.00, 0.60),
    "block_090_070": (0.90, 0.70),
    "block_095_065": (0.95, 0.65),
    "block_100_060": (1.00, 0.60),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_calibration")
    parser.add_argument("--seed-start", type=int, default=9200)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def weight_schedule(name: str, steps: int, seed: int, direction: str):
    values = CANDIDATES[name]
    if len(values) == 1:
        a_weights = np.full(steps, values[0], dtype=float)
    else:
        high, low = values
        if name.startswith("shuffle"):
            a_weights = np.asarray([high] * (steps // 2) + [low] * (steps - steps // 2),
                                   dtype=float)
            np.random.default_rng(920_000 + seed).shuffle(a_weights)
        elif name.startswith("alternate"):
            a_weights = np.resize(np.asarray([high, low], dtype=float), steps)
        elif name.startswith("block"):
            a_weights = np.asarray([high] * (steps // 2) + [low] * (steps - steps // 2),
                                   dtype=float)
        else:
            raise ValueError(name)
    return a_weights if direction == "A" else 1.0 - a_weights


def form_update_only(cfg, seed, direction, schedule_name, neutral_model,
                     neutral_optimizer, train, device):
    model = IVB.make_model(cfg, device)
    model.load_state_dict(neutral_model)
    optimizer = IVB.clone_optimizer(model, cfg, neutral_optimizer)
    batch_rng = np.random.default_rng(600_000 + seed)
    acquisition_choices = IVD.balanced_schedule(cfg.formation_steps, seed, 610_000)
    weights = weight_schedule(schedule_name, cfg.formation_steps, seed, direction)
    n = train.a.shape[0]
    for step in range(cfg.formation_steps):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        x = IVD.acquisition_tensor(train, "balanced", int(acquisition_choices[step]))
        IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                           train.b.to(device)[idx], float(weights[step]))
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cfg = IVD.config_for_stage("pilot")
    cfg.bootstrap_samples = 3000
    device = IVB.choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows = []
    for index, seed in enumerate(seeds, 1):
        train, probe, test = IVB.make_splits(cfg, seed)
        neutral_model, neutral_optimizer = IVD.prepare_neutral(cfg, seed, train, device)
        for direction in DIRECTIONS:
            for schedule_name in CANDIDATES:
                model_state, optimizer_state = form_update_only(
                    cfg, seed, direction, schedule_name, neutral_model,
                    neutral_optimizer, train, device)
                model = IVB.make_model(cfg, device)
                model.load_state_dict(model_state)
                metrics = IVD.aligned_metrics(model, probe, test, cfg, device, direction)
                layers = IVE.optimizer_moment_rows(
                    seed, schedule_name, direction, model_state, optimizer_state, cfg, device)
                moments = IVE.aggregate_moment_endpoint(layers)
                rows.append({
                    "seed": seed,
                    "direction": direction,
                    "schedule": schedule_name,
                    "mean_weight_toward_target": float(np.mean(
                        weight_schedule(schedule_name, cfg.formation_steps, seed, "A"))),
                    **metrics,
                    **moments,
                })
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    baseline = "constant_080"
    metrics = ("aligned_s", "target_D", "other_D", "error_sum", "aligned_low_s")
    summary = {"seeds": seeds, "baseline": baseline, "candidates": {}}
    for candidate in CANDIDATES:
        if candidate == baseline:
            continue
        paired = [row for row in rows if row["schedule"] == candidate]
        base = {(int(row["seed"]), row["direction"]): row for row in rows
                if row["schedule"] == baseline}
        entry = {}
        for metric in metrics + ("v_sum", "effective_lr_mean"):
            diffs = np.asarray([
                float(row[metric]) - float(base[(int(row["seed"]), row["direction"])][metric])
                for row in paired
            ])
            entry[metric + "_difference"] = {
                "mean": float(diffs.mean()),
                "abs_mean": float(np.abs(diffs).mean()),
            }
        entry["match_score"] = float(np.sqrt(sum(
            (entry[metric + "_difference"]["mean"] / scale) ** 2
            for metric, scale in (("aligned_s", 0.02), ("target_D", 0.02),
                                  ("other_D", 0.02), ("error_sum", 0.02),
                                  ("aligned_low_s", 0.01))
        )))
        summary["candidates"][candidate] = entry
    summary["selected_by_match"] = min(
        summary["candidates"], key=lambda name: summary["candidates"][name]["match_score"])

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "calibration_cells.csv", rows)
    (out / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
