#!/usr/bin/env python3
"""実験IV-Dパイロット: 取得主導と更新主導の形成終点を事前に揃える。

IV-Bと同じ未形成checkpointから、取得非対称・均衡更新で形成する経路と、
均衡取得・更新非対称で形成する経路を分岐する。取得主導経路はIV-Bの
固定条件を維持し、更新主導経路の重みだけを候補グリッドで変える。
確認実験では、ここで固定した重みを結果に応じて変更しない。
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
SPEC = importlib.util.spec_from_file_location("iv_b", HERE.parent / "IV-B" / "experiment_iv_b.py")
IVB = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = IVB
SPEC.loader.exec_module(IVB)


def balanced_schedule(steps: int, seed: int):
    schedule = np.asarray([0] * (steps // 2) + [1] * (steps // 2))
    np.random.default_rng(610_000 + seed).shuffle(schedule)
    return schedule


def metrics(model, probe, test, cfg, device):
    return IVB.representation_metrics(
        model, probe.x_both_high, test.x_both_high, probe, test, cfg, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_calibration")
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    cfg = IVB.Config()
    cfg.bootstrap_samples = 3000
    device = IVB.choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    weights_a = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
    rows = []
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.seed_count), 1):
        train, probe, test = IVB.make_splits(cfg, seed)
        initial_model, initial_optimizer = IVB.prepare_initial_state(cfg, seed, train, device)
        conditions = [("acquisition", "A", "A_precise", 0.50),
                      ("acquisition", "B", "B_precise", 0.50)]
        for weight in weights_a:
            conditions.extend([
                ("update", "A", "balanced", weight),
                ("update", "B", "balanced", 1.0 - weight),
            ])
        for route, direction, acquisition, weight_a in conditions:
            model = IVB.make_model(cfg, device)
            model.load_state_dict(initial_model)
            optimizer = IVB.clone_optimizer(model, cfg, initial_optimizer)
            batch_rng = np.random.default_rng(600_000 + seed)
            schedule = balanced_schedule(cfg.formation_steps, seed)
            n = train.a.shape[0]
            for step in range(cfg.formation_steps):
                idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
                if acquisition == "A_precise":
                    x = train.x_a_precise
                elif acquisition == "B_precise":
                    x = train.x_b_precise
                else:
                    x = train.x_a_precise if schedule[step] == 0 else train.x_b_precise
                IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                                   train.b.to(device)[idx], weight_a)
            d_a, d_b, s = metrics(model, probe, test, cfg, device)
            rows.append({"seed": seed, "route": route, "direction": direction,
                         "weight_a": weight_a, "D_A": d_a, "D_B": d_b, "s": s,
                         "error_sum": d_a + d_b})
        print(f"seed {index}/{args.seed_count} 完了", flush=True)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "calibration_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    def mean(items, key):
        return float(np.mean([row[key] for row in items]))

    summary = {"seeds": list(range(args.seed_start, args.seed_start + args.seed_count)),
               "acquisition": {}, "candidates": {}}
    for direction in ("A", "B"):
        items = [r for r in rows if r["route"] == "acquisition" and r["direction"] == direction]
        summary["acquisition"][direction] = {k: mean(items, k) for k in ("D_A", "D_B", "s", "error_sum")}
    for weight in weights_a:
        entry = {}
        distances = []
        for direction, actual_weight in (("A", weight), ("B", 1.0 - weight)):
            items = [r for r in rows if r["route"] == "update" and r["direction"] == direction
                     and abs(r["weight_a"] - actual_weight) < 1e-9]
            values = {k: mean(items, k) for k in ("D_A", "D_B", "s", "error_sum")}
            target = summary["acquisition"][direction]
            values["distance"] = float(np.sqrt(
                ((values["D_A"] - target["D_A"]) / 0.05) ** 2 +
                ((values["D_B"] - target["D_B"]) / 0.05) ** 2 +
                ((values["s"] - target["s"]) / 0.05) ** 2))
            distances.append(values["distance"])
            entry[direction] = values
        entry["mean_distance"] = float(np.mean(distances))
        summary["candidates"][f"{weight:.2f}"] = entry
    summary["selected_weight_a"] = float(min(weights_a,
        key=lambda w: summary["candidates"][f"{w:.2f}"]["mean_distance"]))
    (out / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
