#!/usr/bin/env python3
"""実験IV-E: 形成経路の記憶はAdamのどの成分に保持されるか。

実験IV-Dと同じ二つの形成経路を用い、表面指標を事前許容幅内へ揃える。
形成終点のmodel parameterは固定したまま、Adamの一次モーメント m と
二次モーメント v を経路間で独立に交換し、同じ10更新の摂動への応答を測る。
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
    "iv_d_base", HERE.parent / "IV-D" / "experiment_iv_d.py")
IVD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = IVD
SPEC.loader.exec_module(IVD)
IVB = IVD.IVB

ROUTES = IVD.ROUTES
DIRECTIONS = IVD.DIRECTIONS
PROTOCOLS = ("acquisition_pulse", "update_pulse", "combined_switch")
OPTIMIZER_COMPONENTS = ("own", "m_other", "v_other", "both_other", "reset")
PULSE_STEPS = 10


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
    cfg.bootstrap_seed = 20260903
    if stage == "pilot":
        cfg.bootstrap_samples = 3000
    return cfg


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"空のCSVは作成しません: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state_keys(state_dict):
    return list(state_dict["state"].keys())


def compose_optimizer_state(own_state, other_state, component_mode: str):
    """ownを基準に、m/vだけをotherから移す。stepと設定はownを保つ。"""
    if component_mode == "reset":
        return None
    out = copy.deepcopy(own_state)
    own_keys = state_keys(own_state)
    other_keys = state_keys(other_state)
    if len(own_keys) != len(other_keys):
        raise ValueError("optimizer stateのparameter数が一致しません。")
    for own_key, other_key in zip(own_keys, other_keys):
        target = out["state"][own_key]
        source = other_state["state"][other_key]
        if component_mode in ("m_other", "both_other"):
            target["exp_avg"] = source["exp_avg"].detach().clone()
        if component_mode in ("v_other", "both_other"):
            target["exp_avg_sq"] = source["exp_avg_sq"].detach().clone()
            if "max_exp_avg_sq" in target and "max_exp_avg_sq" in source:
                target["max_exp_avg_sq"] = source["max_exp_avg_sq"].detach().clone()
    return out


def make_optimizer(model, cfg, own_state, other_state, component_mode):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    state = compose_optimizer_state(own_state, other_state, component_mode)
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def scalar_step(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def optimizer_moment_rows(seed, route, direction, model_state, optimizer_state, cfg, device):
    """形成終点のAdam状態をparameterごとに記録する。追加更新は行わない。"""
    model = IVB.make_model(cfg, device)
    model.load_state_dict(model_state)
    names = [name for name, _ in model.named_parameters()]
    keys = state_keys(optimizer_state)
    if len(names) != len(keys):
        raise ValueError("model parameterとoptimizer stateの対応が取れません。")
    group = optimizer_state["param_groups"][0]
    beta1, beta2 = group.get("betas", (0.9, 0.999))
    eps = group.get("eps", 1e-8)
    lr = group.get("lr", cfg.learning_rate)
    rows = []
    for name, key in zip(names, keys):
        state = optimizer_state["state"][key]
        m = state["exp_avg"].detach().double().cpu().reshape(-1)
        v = state["exp_avg_sq"].detach().double().cpu().reshape(-1)
        step = scalar_step(state["step"])
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        m_hat = m / bc1
        v_hat = v / bc2
        effective = lr / (torch.sqrt(v_hat) + eps)
        rows.append({
            "seed": seed, "route": route, "direction": direction,
            "parameter": name, "n_elements": int(m.numel()), "step": step,
            "m_l2": float(torch.linalg.vector_norm(m)),
            "m_hat_l2": float(torch.linalg.vector_norm(m_hat)),
            "v_sum": float(v.sum()),
            "v_l2": float(torch.linalg.vector_norm(v)),
            "sqrt_v_hat_l2": float(torch.linalg.vector_norm(torch.sqrt(v_hat))),
            "effective_lr_mean": float(effective.mean()),
            "effective_lr_median": float(effective.median()),
        })
    return rows


def aggregate_moment_endpoint(rows):
    m_l2 = float(np.sqrt(sum(float(row["m_l2"]) ** 2 for row in rows)))
    m_hat_l2 = float(np.sqrt(sum(float(row["m_hat_l2"]) ** 2 for row in rows)))
    v_sum = float(sum(float(row["v_sum"]) for row in rows))
    v_l2 = float(np.sqrt(sum(float(row["v_l2"]) ** 2 for row in rows)))
    n = sum(int(row["n_elements"]) for row in rows)
    eff_mean = sum(float(row["effective_lr_mean"]) * int(row["n_elements"])
                   for row in rows) / n
    return {"m_l2": m_l2, "m_hat_l2": m_hat_l2, "v_sum": v_sum,
            "v_l2": v_l2, "effective_lr_mean": float(eff_mean)}


def pulse_condition(protocol, direction):
    opposite_acq = "B_precise" if direction == "A" else "A_precise"
    opposite_mild = 0.20 if direction == "A" else 0.80
    opposite_strong = 0.15 if direction == "A" else 0.85
    if protocol == "acquisition_pulse":
        return opposite_acq, 0.5
    if protocol == "update_pulse":
        return "balanced", opposite_mild
    if protocol == "combined_switch":
        return opposite_acq, opposite_strong
    raise ValueError(protocol)


def run_pulse(cfg, seed, route, direction, component_mode, protocol,
              model_state, own_optimizer_state, other_optimizer_state,
              train, probe, test, device):
    model = IVB.make_model(cfg, device)
    model.load_state_dict(model_state)
    optimizer = make_optimizer(model, cfg, own_optimizer_state,
                               other_optimizer_state, component_mode)
    batch_rng = np.random.default_rng(700_000 + seed)
    schedule = IVD.balanced_schedule(PULSE_STEPS, seed, 710_000)
    n = train.a.shape[0]
    rows = []

    def record(step):
        values = IVD.aligned_metrics(model, probe, test, cfg, device, direction)
        rows.append({"seed": seed, "route": route, "direction": direction,
                     "optimizer_component": component_mode, "protocol": protocol,
                     "step": step, **values})

    record(0)
    for step in range(1, PULSE_STEPS + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        acquisition, weight_a = pulse_condition(protocol, direction)
        x = IVD.acquisition_tensor(train, acquisition, int(schedule[step - 1]))
        IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                           train.b.to(device)[idx], weight_a)
        record(step)
    return rows


def summarize_pulse(rows):
    ordered = sorted(rows, key=lambda row: row["step"])
    start = float(ordered[0]["aligned_s"])
    end = float(ordered[-1]["aligned_s"])
    minimum = min(float(row["aligned_s"]) for row in ordered[1:])
    return {
        "start_aligned_s": start,
        "pulse_end_aligned_s": end,
        "pulse_drop": start - end,
        "max_pulse_drop": start - minimum,
        "switched_during_pulse": int(any(float(row["aligned_s"]) <= 0 for row in ordered[1:])),
    }


def percentile_ci(values, rng, samples):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def aggregate(formation_rows, moment_endpoint_rows, protocol_rows, cfg):
    rng = np.random.default_rng(cfg.bootstrap_seed)
    seeds = sorted(set(int(row["seed"]) for row in formation_rows))
    by_form = {(int(r["seed"]), r["route"], r["direction"]): r for r in formation_rows}
    by_moment = {(int(r["seed"]), r["route"], r["direction"]): r
                 for r in moment_endpoint_rows}
    by_protocol = {(int(r["seed"]), r["route"], r["direction"],
                    r["optimizer_component"], r["protocol"]): r for r in protocol_rows}

    formation_metrics = ["aligned_s", "target_D", "other_D", "error_sum", "aligned_low_s"]
    formation_match = {}
    for metric in formation_metrics:
        values = [0.5 * sum(float(by_form[(seed, "acquisition", direction)][metric]) -
                            float(by_form[(seed, "update", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        formation_match[metric] = percentile_ci(values, rng, cfg.bootstrap_samples)

    moment_metrics = ["m_l2", "m_hat_l2", "v_sum", "v_l2", "effective_lr_mean"]
    moments = {}
    for metric in moment_metrics:
        values = [0.5 * sum(float(by_moment[(seed, "acquisition", direction)][metric]) -
                            float(by_moment[(seed, "update", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        moments[f"acquisition_minus_update_{metric}"] = percentile_ci(
            values, rng, cfg.bootstrap_samples)
        ratios = [0.5 * sum(float(by_moment[(seed, "acquisition", direction)][metric]) /
                            float(by_moment[(seed, "update", direction)][metric])
                            for direction in DIRECTIONS) for seed in seeds]
        moments[f"acquisition_over_update_{metric}"] = percentile_ci(
            ratios, rng, cfg.bootstrap_samples)

    conditions = {}
    route_effects = {}
    changes_from_own = {}
    for protocol in PROTOCOLS:
        conditions[protocol] = {}
        route_effects[protocol] = {}
        changes_from_own[protocol] = {}
        for mode in OPTIMIZER_COMPONENTS:
            conditions[protocol][mode] = {}
            for route in ROUTES:
                values = [float(by_protocol[(seed, route, direction, mode, protocol)]["pulse_drop"])
                          for seed in seeds for direction in DIRECTIONS]
                conditions[protocol][mode][route] = percentile_ci(
                    values, rng, cfg.bootstrap_samples)
            differences = [0.5 * sum(
                float(by_protocol[(seed, "acquisition", direction, mode, protocol)]["pulse_drop"]) -
                float(by_protocol[(seed, "update", direction, mode, protocol)]["pulse_drop"])
                for direction in DIRECTIONS) for seed in seeds]
            route_effects[protocol][mode] = percentile_ci(
                differences, rng, cfg.bootstrap_samples)
            if mode != "own":
                attenuation = [0.5 * sum(
                    (float(by_protocol[(seed, "acquisition", direction, "own", protocol)]["pulse_drop"]) -
                     float(by_protocol[(seed, "update", direction, "own", protocol)]["pulse_drop"])) -
                    (float(by_protocol[(seed, "acquisition", direction, mode, protocol)]["pulse_drop"]) -
                     float(by_protocol[(seed, "update", direction, mode, protocol)]["pulse_drop"]))
                    for direction in DIRECTIONS) for seed in seeds]
                changes_from_own[protocol][mode] = percentile_ci(
                    attenuation, rng, cfg.bootstrap_samples)

    return {"n_seeds": len(seeds), "formation_match": formation_match,
            "optimizer_moment_diagnostics": moments, "conditions": conditions,
            "route_effects_acquisition_minus_update": route_effects,
            "route_effect_attenuation_from_own": changes_from_own}


def classify_interval(interval, margin):
    if interval["low"] >= -margin and interval["high"] <= margin:
        return "equivalent_within_margin"
    if interval["low"] > margin:
        return "positive_beyond_margin"
    if interval["high"] < -margin:
        return "negative_beyond_margin"
    return "unresolved"


def validate_preregistration(prereg, cfg, seeds, update_weight_a):
    fixed = prereg["fixed_design"]
    checks = {
        "seeds": list(seeds), "initial_steps": cfg.initial_steps,
        "formation_steps": cfg.formation_steps, "update_weight_A": update_weight_a,
        "protocols": list(PROTOCOLS), "optimizer_components": list(OPTIMIZER_COMPONENTS),
        "pulse_steps": PULSE_STEPS,
    }
    for key, actual in checks.items():
        if fixed[key] != actual:
            raise ValueError(f"事前登録不一致 {key}: {fixed[key]} != {actual}")
    expected_hash = prereg.get("script_sha256")
    if expected_hash and expected_hash != sha256_file(Path(__file__)):
        raise ValueError("事前登録後に実験スクリプトが変更されています。")


def evaluate(result, prereg):
    formation = {}
    for metric, margin in prereg["formation_match_tolerances"].items():
        interval = result["formation_match"][metric]
        formation[metric] = {"pass": interval["low"] >= -margin and interval["high"] <= margin,
                             "margin": margin, "interval": interval}
    gates = {}
    for name, spec in prereg["primary_gates"].items():
        interval = result[spec["source"]][spec["protocol"]][spec["component"]]
        rule = spec["rule"]
        threshold = spec["threshold"]
        if rule == "lower_bound_at_least":
            passed = interval["low"] >= threshold
        elif rule == "upper_bound_at_most":
            passed = interval["high"] <= threshold
        elif rule == "equivalent_within":
            passed = interval["low"] >= -threshold and interval["high"] <= threshold
        else:
            raise ValueError(rule)
        gates[name] = {"pass": passed, "rule": rule, "threshold": threshold,
                       "interval": interval}
    return {"formation_match": formation,
            "formation_match_all_pass": all(item["pass"] for item in formation.values()),
            "primary_gates": gates,
            "primary_gates_all_pass": all(item["pass"] for item in gates.values())}


def report_text(result, evaluation=None):
    lines = ["# 実験IV-E 結果", "", f"- seed数: {result['n_seeds']}", ""]
    lines.append("## Adam形成終点診断")
    for name, interval in result["optimizer_moment_diagnostics"].items():
        lines.append(f"- {name}: {interval['mean']:.6f} "
                     f"[{interval['low']:.6f}, {interval['high']:.6f}]")
    lines.extend(["", "## 10更新後の経路差（取得主導－更新主導）"])
    for protocol in PROTOCOLS:
        lines.append(f"- {protocol}")
        for mode in OPTIMIZER_COMPONENTS:
            interval = result["route_effects_acquisition_minus_update"][protocol][mode]
            lines.append(f"  - {mode}: {interval['mean']:.6f} "
                         f"[{interval['low']:.6f}, {interval['high']:.6f}]")
    if evaluation is not None:
        lines.extend(["", "## 事前判定"])
        lines.append(f"- 形成終点の一致: {'PASS' if evaluation['formation_match_all_pass'] else 'FAIL'}")
        for name, gate in evaluation["primary_gates"].items():
            lines.append(f"- {name}: {'PASS' if gate['pass'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    if args.seed_start is None:
        seed_start = {"smoke": 8900, "pilot": 9000, "confirm": 9100}[args.stage]
    else:
        seed_start = args.seed_start
    if args.seed_count is None:
        seed_count = {"smoke": 2, "pilot": 16, "confirm": 40}[args.stage]
    else:
        seed_count = args.seed_count
    seeds = list(range(seed_start, seed_start + seed_count))
    update_weight_a = 0.80
    prereg = None
    if args.stage == "confirm":
        if not args.preregistration:
            raise ValueError("confirmには--preregistrationが必要です。")
        prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
        validate_preregistration(prereg, cfg, seeds, update_weight_a)

    device = IVB.choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = Path(args.output_dir or f"results_{args.stage}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    formation_rows, moment_layer_rows, moment_endpoint_rows = [], [], []
    protocol_rows, timecourse_rows = [], []

    for index, seed in enumerate(seeds, 1):
        train, probe, test = IVB.make_splits(cfg, seed)
        neutral_model, neutral_optimizer = IVD.prepare_neutral(cfg, seed, train, device)
        endpoints = {}
        for direction in DIRECTIONS:
            for route in ROUTES:
                model_state, optimizer_state = IVD.form_state(
                    cfg, seed, route, direction, update_weight_a,
                    neutral_model, neutral_optimizer, train, device)
                endpoints[(route, direction)] = (model_state, optimizer_state)
                model = IVB.make_model(cfg, device)
                model.load_state_dict(model_state)
                formation_rows.append({"seed": seed, "route": route, "direction": direction,
                                       **IVD.aligned_metrics(model, probe, test, cfg, device, direction)})
                layers = optimizer_moment_rows(seed, route, direction, model_state,
                                               optimizer_state, cfg, device)
                moment_layer_rows.extend(layers)
                moment_endpoint_rows.append({"seed": seed, "route": route,
                                             "direction": direction,
                                             **aggregate_moment_endpoint(layers)})

        for direction in DIRECTIONS:
            for route in ROUTES:
                other_route = "update" if route == "acquisition" else "acquisition"
                model_state, own_optimizer = endpoints[(route, direction)]
                other_optimizer = endpoints[(other_route, direction)][1]
                for component_mode in OPTIMIZER_COMPONENTS:
                    for protocol in PROTOCOLS:
                        rows = run_pulse(cfg, seed, route, direction, component_mode, protocol,
                                         model_state, own_optimizer, other_optimizer,
                                         train, probe, test, device)
                        timecourse_rows.extend(rows)
                        protocol_rows.append({"seed": seed, "route": route,
                                              "direction": direction,
                                              "optimizer_component": component_mode,
                                              "protocol": protocol,
                                              **summarize_pulse(rows)})
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    result = aggregate(formation_rows, moment_endpoint_rows, protocol_rows, cfg)
    evaluation = evaluate(result, prereg) if prereg else None
    provenance = {
        "experiment": "IV-E", "stage": args.stage, "seeds": seeds,
        "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "config": asdict(cfg), "update_weight_A": update_weight_a,
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    (out / "result.json").write_text(json.dumps(
        {"aggregate": result, "evaluation": evaluation}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(
        provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "result_report.md").write_text(report_text(result, evaluation), encoding="utf-8")
    write_csv(out / "formation_endpoints.csv", formation_rows)
    write_csv(out / "optimizer_moment_layers.csv", moment_layer_rows)
    write_csv(out / "optimizer_moment_endpoints.csv", moment_endpoint_rows)
    write_csv(out / "protocol_summary.csv", protocol_rows)
    write_csv(out / "timecourse.csv", timecourse_rows)


if __name__ == "__main__":
    main()
