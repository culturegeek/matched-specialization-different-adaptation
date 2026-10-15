#!/usr/bin/env python3
"""実験IV-E追加対照: 更新重みの履歴だけで潜在的な反応差が生じるか。

二経路とも取得、更新回数、平均更新配分を同一にする。一方は各更新で
一定の0.80/0.20、他方は1.00/0.60を同数・無作為順に用い、平均を
0.80/0.20に揃える。形成終点を比較した後、Adamのm/vを経路間で交換し、
同じ10更新の摂動に対する応答を測る。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
from dataclasses import asdict
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

ROUTES = ("constant", "variable")
DIRECTIONS = ("A", "B")
PROTOCOLS = IVE.PROTOCOLS
OPTIMIZER_COMPONENTS = IVE.OPTIMIZER_COMPONENTS
PULSE_STEPS = IVE.PULSE_STEPS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirm"], default="smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--preregistration", default=None)
    return parser.parse_args()


def config_for_stage(stage: str):
    cfg = IVD.config_for_stage("smoke" if stage == "smoke" else "confirm")
    cfg.bootstrap_seed = 20260904
    if stage == "pilot":
        cfg.bootstrap_samples = 3000
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_weight_schedule(route: str, steps: int, seed: int):
    if route == "constant":
        return np.full(steps, 0.80, dtype=float)
    if route == "variable":
        schedule = np.asarray([1.00] * (steps // 2) + [0.60] * (steps - steps // 2),
                              dtype=float)
        np.random.default_rng(920_000 + seed).shuffle(schedule)
        return schedule
    raise ValueError(route)


def form_state(cfg, seed, route, direction, neutral_model, neutral_optimizer,
               train, device):
    model = IVB.make_model(cfg, device)
    model.load_state_dict(neutral_model)
    optimizer = IVB.clone_optimizer(model, cfg, neutral_optimizer)
    batch_rng = np.random.default_rng(600_000 + seed)
    acquisition_choices = IVD.balanced_schedule(cfg.formation_steps, seed, 610_000)
    target_weights = target_weight_schedule(route, cfg.formation_steps, seed)
    n = train.a.shape[0]
    for step in range(cfg.formation_steps):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        x = IVD.acquisition_tensor(train, "balanced", int(acquisition_choices[step]))
        weight_a = float(target_weights[step] if direction == "A"
                         else 1.0 - target_weights[step])
        IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                           train.b.to(device)[idx], weight_a)
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def percentile_ci(values, rng, samples):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def aggregate(formation_rows, moment_rows, protocol_rows, cfg):
    rng = np.random.default_rng(cfg.bootstrap_seed)
    seeds = sorted(set(int(row["seed"]) for row in formation_rows))
    form = {(int(r["seed"]), r["route"], r["direction"]): r for r in formation_rows}
    moment = {(int(r["seed"]), r["route"], r["direction"]): r for r in moment_rows}
    protocol = {(int(r["seed"]), r["route"], r["direction"],
                 r["optimizer_component"], r["protocol"]): r for r in protocol_rows}

    formation_match = {}
    for metric in ("aligned_s", "target_D", "other_D", "error_sum", "aligned_low_s"):
        values = [0.5 * sum(float(form[(seed, "constant", direction)][metric]) -
                            float(form[(seed, "variable", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        formation_match[metric] = percentile_ci(values, rng, cfg.bootstrap_samples)

    optimizer_differences = {}
    for metric in ("m_l2", "m_hat_l2", "v_sum", "v_l2", "effective_lr_mean"):
        values = [0.5 * sum(float(moment[(seed, "constant", direction)][metric]) -
                            float(moment[(seed, "variable", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        optimizer_differences[metric] = percentile_ci(values, rng, cfg.bootstrap_samples)
        ratios = [0.5 * sum(float(moment[(seed, "constant", direction)][metric]) /
                            float(moment[(seed, "variable", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        optimizer_differences[metric + "_ratio"] = percentile_ci(
            ratios, rng, cfg.bootstrap_samples)

    route_effects = {}
    changes_from_own = {}
    condition_means = {}
    for pulse in PROTOCOLS:
        route_effects[pulse] = {}
        changes_from_own[pulse] = {}
        condition_means[pulse] = {}
        for component in OPTIMIZER_COMPONENTS:
            condition_means[pulse][component] = {}
            for route in ROUTES:
                values = [float(protocol[(seed, route, direction, component, pulse)]["pulse_drop"])
                          for seed in seeds for direction in DIRECTIONS]
                condition_means[pulse][component][route] = percentile_ci(
                    values, rng, cfg.bootstrap_samples)
            values = [0.5 * sum(
                float(protocol[(seed, "constant", direction, component, pulse)]["pulse_drop"]) -
                float(protocol[(seed, "variable", direction, component, pulse)]["pulse_drop"])
                for direction in DIRECTIONS) for seed in seeds]
            route_effects[pulse][component] = percentile_ci(values, rng, cfg.bootstrap_samples)
            if component != "own":
                values = [0.5 * sum(
                    (float(protocol[(seed, "constant", direction, "own", pulse)]["pulse_drop"]) -
                     float(protocol[(seed, "variable", direction, "own", pulse)]["pulse_drop"])) -
                    (float(protocol[(seed, "constant", direction, component, pulse)]["pulse_drop"]) -
                     float(protocol[(seed, "variable", direction, component, pulse)]["pulse_drop"]))
                    for direction in DIRECTIONS) for seed in seeds]
                changes_from_own[pulse][component] = percentile_ci(
                    values, rng, cfg.bootstrap_samples)

    return {"n_seeds": len(seeds), "formation_match": formation_match,
            "optimizer_constant_minus_variable": optimizer_differences,
            "condition_means": condition_means,
            "route_effects_constant_minus_variable": route_effects,
            "route_effect_changes_from_own": changes_from_own}


def validate_preregistration(prereg, cfg, seeds):
    fixed = prereg["fixed_design"]
    observed = {
        "seeds": seeds,
        "initial_steps": cfg.initial_steps,
        "formation_steps": cfg.formation_steps,
        "routes": {"constant": [0.80], "variable": [1.00, 0.60]},
        "mean_target_weight": 0.80,
        "acquisition": "balanced half target-precise/non-target-precise",
        "protocols": list(PROTOCOLS),
        "optimizer_components": list(OPTIMIZER_COMPONENTS),
        "pulse_steps": PULSE_STEPS,
    }
    if observed != fixed:
        raise ValueError("確認設定が事前登録と一致しません。\n" + json.dumps(
            {"expected": fixed, "observed": observed}, ensure_ascii=False, indent=2))
    expected_hash = prereg.get("script_sha256")
    if expected_hash and expected_hash != sha256_file(Path(__file__)):
        raise ValueError("事前登録後に実験スクリプトが変更されています。")


def interval_in_margin(interval, margin):
    return interval["low"] >= -margin and interval["high"] <= margin


def evaluate(result, prereg):
    formation = {}
    for metric, margin in prereg["formation_match_tolerances"].items():
        interval = result["formation_match"][metric]
        formation[metric] = {"pass": interval_in_margin(interval, margin),
                             "margin": margin, "interval": interval}
    gates = {}
    for name, spec in prereg["primary_gates"].items():
        if spec["source"] == "optimizer":
            interval = result["optimizer_constant_minus_variable"][spec["metric"]]
        elif spec["source"] == "route_effect":
            interval = result["route_effects_constant_minus_variable"][
                spec["protocol"]][spec["component"]]
        elif spec["source"] == "change_from_own":
            interval = result["route_effect_changes_from_own"][
                spec["protocol"]][spec["component"]]
        else:
            raise ValueError(spec["source"])
        rule, threshold = spec["rule"], spec["threshold"]
        if rule == "lower_bound_at_least":
            passed = interval["low"] >= threshold
        elif rule == "upper_bound_at_most":
            passed = interval["high"] <= threshold
        elif rule == "equivalent_within":
            passed = interval_in_margin(interval, threshold)
        else:
            raise ValueError(rule)
        gates[name] = {"pass": passed, "rule": rule, "threshold": threshold,
                       "interval": interval}
    return {"formation_match": formation,
            "formation_match_all_pass": all(item["pass"] for item in formation.values()),
            "primary_gates": gates,
            "primary_gates_all_pass": all(item["pass"] for item in gates.values())}


def report_text(stage, result, evaluation, runtime):
    lines = [f"# 実験IV-E追加対照 {stage} 結果", "",
             "## この実験が答える問い", "",
             "取得と平均更新配分を揃えても、更新重みの時間的な偏り方だけでAdamの二次モーメントと短期応答は変わるか。",
             "", "## 形成終点の差（一定更新－変動更新）", ""]
    for metric, interval in result["formation_match"].items():
        lines.append(f"- {metric}: {interval['mean']:.6f} "
                     f"[{interval['low']:.6f}, {interval['high']:.6f}]")
    lines += ["", "## optimizer状態の差（一定更新－変動更新）", ""]
    for metric, interval in result["optimizer_constant_minus_variable"].items():
        lines.append(f"- {metric}: {interval['mean']:.6f} "
                     f"[{interval['low']:.6f}, {interval['high']:.6f}]")
    lines += ["", "## 10更新後の応答差（一定更新－変動更新）", ""]
    for pulse in PROTOCOLS:
        lines.append(f"- {pulse}")
        for component in OPTIMIZER_COMPONENTS:
            interval = result["route_effects_constant_minus_variable"][pulse][component]
            lines.append(f"  - {component}: {interval['mean']:.6f} "
                         f"[{interval['low']:.6f}, {interval['high']:.6f}]")
    if evaluation is not None:
        lines += ["", "## 事前判定", "",
                  f"- 形成終点の一致: {'PASS' if evaluation['formation_match_all_pass'] else 'FAIL'}"]
        for name, gate in evaluation["primary_gates"].items():
            lines.append(f"- {name}: {'PASS' if gate['pass'] else 'FAIL'}")
    lines += ["", "## 適用範囲", "",
              "この対照は、二対象を二次元共有表現へ圧縮し、Adamで500回更新する人工学習系を対象とする。",
              f"", f"- 実行時間: {runtime:.1f}秒"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    defaults = {"smoke": (9250, 2), "pilot": (9200, 16), "confirm": (9300, 40)}
    seed_start, seed_count = defaults[args.stage]
    if args.seed_start is not None:
        seed_start = args.seed_start
    if args.seed_count is not None:
        seed_count = args.seed_count
    seeds = list(range(seed_start, seed_start + seed_count))
    prereg = None
    if args.stage == "confirm":
        if not args.preregistration:
            raise ValueError("confirmには--preregistrationが必要です。")
        prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
        validate_preregistration(prereg, cfg, seeds)

    device = IVB.choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = Path(args.output_dir or f"results_{args.stage}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    formation_rows, moment_layer_rows, moment_rows = [], [], []
    protocol_rows, timecourse_rows = [], []

    for index, seed in enumerate(seeds, 1):
        train, probe, test = IVB.make_splits(cfg, seed)
        neutral_model, neutral_optimizer = IVD.prepare_neutral(cfg, seed, train, device)
        endpoints = {}
        for direction in DIRECTIONS:
            for route in ROUTES:
                model_state, optimizer_state = form_state(
                    cfg, seed, route, direction, neutral_model, neutral_optimizer,
                    train, device)
                endpoints[(route, direction)] = (model_state, optimizer_state)
                model = IVB.make_model(cfg, device)
                model.load_state_dict(model_state)
                formation_rows.append({"seed": seed, "route": route,
                                       "direction": direction,
                                       **IVD.aligned_metrics(model, probe, test, cfg,
                                                             device, direction)})
                layers = IVE.optimizer_moment_rows(
                    seed, route, direction, model_state, optimizer_state, cfg, device)
                moment_layer_rows.extend(layers)
                moment_rows.append({"seed": seed, "route": route, "direction": direction,
                                    **IVE.aggregate_moment_endpoint(layers)})

        for direction in DIRECTIONS:
            for route in ROUTES:
                other = "variable" if route == "constant" else "constant"
                model_state, own_optimizer = endpoints[(route, direction)]
                other_optimizer = endpoints[(other, direction)][1]
                for component in OPTIMIZER_COMPONENTS:
                    for pulse in PROTOCOLS:
                        rows = IVE.run_pulse(
                            cfg, seed, route, direction, component, pulse,
                            model_state, own_optimizer, other_optimizer,
                            train, probe, test, device)
                        timecourse_rows.extend(rows)
                        protocol_rows.append({
                            "seed": seed, "route": route, "direction": direction,
                            "optimizer_component": component, "protocol": pulse,
                            **IVE.summarize_pulse(rows),
                        })
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    result = aggregate(formation_rows, moment_rows, protocol_rows, cfg)
    evaluation = evaluate(result, prereg) if prereg else None
    runtime = time.time() - started
    output = {"experiment": "IV-E update-bias control", "stage": args.stage,
              "design": {"seeds": seeds, "routes": list(ROUTES),
                         "directions": list(DIRECTIONS),
                         "mean_target_weight": 0.80,
                         "same_acquisition": True, "same_updates": True,
                         "same_mean_update_allocation": True,
                         "statistical_unit": "seed"},
              "config": asdict(cfg), "aggregate": result,
              "confirmatory_evaluation": evaluation, "runtime_seconds": runtime}
    write_csv(out / "formation_endpoints.csv", formation_rows)
    write_csv(out / "optimizer_moment_layers.csv", moment_layer_rows)
    write_csv(out / "optimizer_moment_endpoints.csv", moment_rows)
    write_csv(out / "protocol_summary.csv", protocol_rows)
    write_csv(out / "timecourse.csv", timecourse_rows)
    (out / "result.json").write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else None
    provenance = {"script_sha256": sha256_file(Path(__file__).resolve()),
                  "base_iv_e_sha256": sha256_file(HERE.parent / "IV-E" / "experiment_iv_e.py"),
                  "preregistration_sha256": sha256_file(prereg_path) if prereg_path else None,
                  "python": sys.version, "platform": platform.platform(),
                  "numpy": np.__version__, "torch": torch.__version__, "device": str(device)}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    report = report_text(args.stage, result, evaluation, runtime)
    (out / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
