#!/usr/bin/env python3
"""実験IV-D: 表面状態を揃えた二つの形成経路は、後続応答も同じか。

同じ未形成checkpointから、(1)取得精度の非対称と均衡更新、(2)均衡取得と
更新配分の非対称、という二経路でA/B方向の表現を形成する。パイロットで
固定した介入強度を用い、形成終点の方向指標と二つの復号誤差が近いことを
確認してから、同じ均衡継続、取得摂動、更新摂動、複合摂動への応答を比較する。
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
SPEC = importlib.util.spec_from_file_location("iv_b_base", HERE.parent / "IV-B" / "experiment_iv_b.py")
IVB = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = IVB
SPEC.loader.exec_module(IVB)

ROUTES = ("acquisition", "update")
DIRECTIONS = ("A", "B")
PROTOCOLS = ("neutral", "acquisition_pulse", "update_pulse", "combined_switch")
OPTIMIZER_MODES = ("carried", "reset")
PROTOCOL_STEPS = {"neutral": 400, "acquisition_pulse": 110,
                  "update_pulse": 110, "combined_switch": 60}


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
    cfg = IVB.Config()
    cfg.bootstrap_seed = 20260902
    if stage == "smoke":
        cfg.train_size = 1024
        cfg.probe_size = 384
        cfg.test_size = 512
        cfg.hidden_dim = 24
        cfg.batch_size = 128
        cfg.initial_steps = 40
        cfg.formation_steps = 100
        cfg.bootstrap_samples = 1000
    elif stage == "pilot":
        cfg.bootstrap_samples = 3000
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_schedule(steps: int, seed: int, stream: int):
    first = steps // 2
    schedule = np.asarray([0] * first + [1] * (steps - first))
    np.random.default_rng(stream + seed).shuffle(schedule)
    return schedule


def acquisition_tensor(train, acquisition: str, balanced_choice: int):
    if acquisition == "A_precise":
        return train.x_a_precise
    if acquisition == "B_precise":
        return train.x_b_precise
    return train.x_a_precise if balanced_choice == 0 else train.x_b_precise


def prepare_neutral(cfg, seed, train, device):
    set_seed(seed)
    model = IVB.make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(100_000 + seed)
    n = train.a.shape[0]
    for _ in range(cfg.initial_steps):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        IVB.train_weighted(model, optimizer, train.x_initial.to(device)[idx],
                           train.a.to(device)[idx], train.b.to(device)[idx], 0.5)
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def formation_condition(route: str, direction: str, update_weight_a: float):
    if route == "acquisition":
        return ("A_precise" if direction == "A" else "B_precise"), 0.5
    return "balanced", (update_weight_a if direction == "A" else 1.0 - update_weight_a)


def form_state(cfg, seed, route, direction, update_weight_a, neutral_model,
               neutral_optimizer, train, device):
    model = IVB.make_model(cfg, device)
    model.load_state_dict(neutral_model)
    optimizer = IVB.clone_optimizer(model, cfg, neutral_optimizer)
    acquisition, weight_a = formation_condition(route, direction, update_weight_a)
    batch_rng = np.random.default_rng(600_000 + seed)
    schedule = balanced_schedule(cfg.formation_steps, seed, 610_000)
    n = train.a.shape[0]
    for step in range(cfg.formation_steps):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        x = acquisition_tensor(train, acquisition, int(schedule[step]))
        IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                           train.b.to(device)[idx], weight_a)
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def signed(value: float, direction: str):
    return value if direction == "A" else -value


def aligned_metrics(model, probe, test, cfg, device, direction):
    h_da, h_db, h_s = IVB.representation_metrics(
        model, probe.x_both_high, test.x_both_high, probe, test, cfg, device)
    l_da, l_db, l_s = IVB.representation_metrics(
        model, probe.x_both_low, test.x_both_low, probe, test, cfg, device)
    if direction == "A":
        target, other = h_da, h_db
        target_low, other_low = l_da, l_db
    else:
        target, other = h_db, h_da
        target_low, other_low = l_db, l_da
    return {
        "D_A": h_da, "D_B": h_db, "s": h_s, "aligned_s": signed(h_s, direction),
        "target_D": target, "other_D": other, "error_sum": h_da + h_db,
        "low_D_A": l_da, "low_D_B": l_db, "low_s": l_s,
        "aligned_low_s": signed(l_s, direction), "target_low_D": target_low,
        "other_low_D": other_low,
    }


@torch.no_grad()
def endpoint_cka(state_one, state_two, cfg, probe, device):
    one = IVB.make_model(cfg, device); one.load_state_dict(state_one)
    two = IVB.make_model(cfg, device); two.load_state_dict(state_two)
    x = probe.x_both_high.to(device)
    a = one.encoder(x); b = two.encoder(x)
    a = a - a.mean(dim=0, keepdim=True); b = b - b.mean(dim=0, keepdim=True)
    numerator = torch.sum((a.T @ b) ** 2)
    denominator = torch.sqrt(torch.sum((a.T @ a) ** 2) * torch.sum((b.T @ b) ** 2))
    return float((numerator / (denominator + 1e-12)).item())


def protocol_condition(protocol: str, direction: str, step: int, balanced_choice: int):
    own_acq = "A_precise" if direction == "A" else "B_precise"
    opposite_acq = "B_precise" if direction == "A" else "A_precise"
    own_mild = 0.80 if direction == "A" else 0.20
    opposite_mild = 0.20 if direction == "A" else 0.80
    own_strong = 0.85 if direction == "A" else 0.15
    opposite_strong = 0.15 if direction == "A" else 0.85
    if protocol == "neutral":
        return "balanced", 0.5
    if protocol == "acquisition_pulse":
        return (opposite_acq, 0.5) if step <= 10 else (own_acq, 0.5)
    if protocol == "update_pulse":
        return ("balanced", opposite_mild) if step <= 10 else ("balanced", own_mild)
    if protocol == "combined_switch":
        return (opposite_acq, opposite_strong) if step <= 10 else (own_acq, own_strong)
    raise ValueError(protocol)


def run_protocol(cfg, seed, route, direction, optimizer_mode, protocol, model_state,
                 optimizer_state, train, probe, test, device):
    model = IVB.make_model(cfg, device); model.load_state_dict(model_state)
    if optimizer_mode == "carried":
        optimizer = IVB.clone_optimizer(model, cfg, optimizer_state)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
    steps = PROTOCOL_STEPS[protocol]
    batch_rng = np.random.default_rng(700_000 + seed)
    schedule = balanced_schedule(steps, seed, 710_000)
    n = train.a.shape[0]
    rows = []

    def record(step, phase):
        values = aligned_metrics(model, probe, test, cfg, device, direction)
        rows.append({"seed": seed, "route": route, "direction": direction,
                     "optimizer_mode": optimizer_mode, "protocol": protocol,
                     "step": step, "phase": phase, **values})

    record(0, "start")
    for step in range(1, steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        acquisition, weight_a = protocol_condition(protocol, direction, step, int(schedule[step - 1]))
        x = acquisition_tensor(train, acquisition, int(schedule[step - 1]))
        IVB.train_weighted(model, optimizer, x.to(device)[idx], train.a.to(device)[idx],
                           train.b.to(device)[idx], weight_a)
        if protocol == "neutral":
            phase = "neutral"
        elif step <= 10:
            phase = "perturbation"
        else:
            phase = "recovery"
        record(step, phase)
    return rows


def trapezoid_average(rows, last_step: int):
    selected = sorted((r for r in rows if r["step"] <= last_step), key=lambda r: r["step"])
    x = np.asarray([r["step"] for r in selected], dtype=float)
    y = np.asarray([r["aligned_s"] for r in selected], dtype=float)
    return float(np.trapezoid(y, x) / last_step)


def first_step(rows, predicate, minimum_step=1):
    for row in sorted(rows, key=lambda item: item["step"]):
        if row["step"] >= minimum_step and predicate(row):
            return int(row["step"])
    return None


def summarize_protocol(rows):
    ordered = sorted(rows, key=lambda row: row["step"])
    protocol = ordered[0]["protocol"]
    start = ordered[0]["aligned_s"]
    final = ordered[-1]["aligned_s"]
    out = {"start_aligned_s": start, "final_aligned_s": final,
           "change_aligned_s": final - start}
    if protocol == "neutral":
        out["auc_aligned_s_0_200"] = trapezoid_average(ordered, 200)
        out["auc_aligned_s_0_400"] = trapezoid_average(ordered, 400)
        half = first_step(ordered, lambda r: r["aligned_s"] <= 0.5 * start)
        out["half_decay_reached"] = int(half is not None)
        out["half_decay_step"] = half if half is not None else 401
    else:
        pulse = [r for r in ordered if 1 <= r["step"] <= 10]
        recovery = [r for r in ordered if r["step"] > 10]
        pulse_end = [r for r in ordered if r["step"] == 10][0]["aligned_s"]
        out["pulse_end_aligned_s"] = pulse_end
        out["pulse_drop"] = start - pulse_end
        out["switched_during_pulse"] = int(any(r["aligned_s"] <= 0 for r in pulse))
        switch = first_step(pulse, lambda r: r["aligned_s"] <= 0)
        out["switch_step"] = switch if switch is not None else 11
        recross = first_step(recovery, lambda r: r["aligned_s"] >= 0, minimum_step=11)
        out["recrossed_during_recovery"] = int(recross is not None)
        out["recross_step_after_recovery_start"] = recross - 10 if recross is not None else len(recovery) + 1
        target = 0.9 * start
        recovered = first_step(recovery, lambda r: r["aligned_s"] >= target, minimum_step=11)
        out["recovered_90pct"] = int(recovered is not None)
        out["recovery_step_90pct"] = recovered - 10 if recovered is not None else len(recovery) + 1
    return out


def percentile_ci(values, rng, samples):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975))}


def aggregate(formation_rows, protocol_rows, geometry_rows, cfg):
    rng = np.random.default_rng(cfg.bootstrap_seed)
    by_form = {(r["seed"], r["route"], r["direction"]): r for r in formation_rows}
    by_protocol = {(r["seed"], r["route"], r["direction"], r["optimizer_mode"], r["protocol"]): r
                   for r in protocol_rows}
    seeds = sorted(set(r["seed"] for r in formation_rows))
    formation_metrics = ["aligned_s", "target_D", "other_D", "error_sum", "aligned_low_s"]
    formation = {}
    for route in ROUTES:
        formation[route] = {}
        for metric in formation_metrics:
            values = [by_form[(seed, route, direction)][metric]
                      for seed in seeds for direction in DIRECTIONS]
            formation[route][metric] = percentile_ci(values, rng, cfg.bootstrap_samples)
    match = {}
    for metric in formation_metrics:
        values = [0.5 * sum(by_form[(seed, "acquisition", direction)][metric] -
                            by_form[(seed, "update", direction)][metric]
                            for direction in DIRECTIONS) for seed in seeds]
        match[metric] = percentile_ci(values, rng, cfg.bootstrap_samples)

    all_outcomes = sorted(set().union(*(r.keys() for r in protocol_rows)) -
                          {"seed", "route", "direction", "optimizer_mode", "protocol"})
    protocols = {}
    effects = {}
    for mode in OPTIMIZER_MODES:
        protocols[mode] = {}; effects[mode] = {}
        for protocol in PROTOCOLS:
            protocols[mode][protocol] = {}
            for route in ROUTES:
                protocols[mode][protocol][route] = {}
                for outcome in all_outcomes:
                    available = [by_protocol[(seed, route, direction, mode, protocol)][outcome]
                                 for seed in seeds for direction in DIRECTIONS
                                 if outcome in by_protocol[(seed, route, direction, mode, protocol)]]
                    if available:
                        protocols[mode][protocol][route][outcome] = percentile_ci(
                            available, rng, cfg.bootstrap_samples)
            effects[mode][protocol] = {}
            common = (set(protocols[mode][protocol]["acquisition"]) &
                      set(protocols[mode][protocol]["update"]))
            for outcome in common:
                values = [0.5 * sum(
                    by_protocol[(seed, "acquisition", direction, mode, protocol)][outcome] -
                    by_protocol[(seed, "update", direction, mode, protocol)][outcome]
                    for direction in DIRECTIONS) for seed in seeds]
                effects[mode][protocol][outcome] = percentile_ci(values, rng, cfg.bootstrap_samples)
    optimizer_interaction = {}
    for protocol in PROTOCOLS:
        optimizer_interaction[protocol] = {}
        common = set(effects["carried"][protocol]) & set(effects["reset"][protocol])
        for outcome in common:
            values = []
            for seed in seeds:
                carried = 0.5 * sum(
                    by_protocol[(seed, "acquisition", direction, "carried", protocol)][outcome] -
                    by_protocol[(seed, "update", direction, "carried", protocol)][outcome]
                    for direction in DIRECTIONS)
                reset = 0.5 * sum(
                    by_protocol[(seed, "acquisition", direction, "reset", protocol)][outcome] -
                    by_protocol[(seed, "update", direction, "reset", protocol)][outcome]
                    for direction in DIRECTIONS)
                values.append(carried - reset)
            optimizer_interaction[protocol][outcome] = percentile_ci(
                values, rng, cfg.bootstrap_samples)
    geometry = {"linear_cka": percentile_ci([r["linear_cka"] for r in geometry_rows],
                                               rng, cfg.bootstrap_samples)}
    return {"n_seeds": len(seeds), "formation": formation, "formation_match": match,
            "protocols": protocols, "route_effects_acquisition_minus_update": effects,
            "optimizer_state_interaction": optimizer_interaction, "geometry": geometry}


def classify_interval(interval, threshold):
    if interval["low"] > threshold or interval["high"] < -threshold:
        return "different"
    if interval["low"] >= -threshold and interval["high"] <= threshold:
        return "equivalent_within_margin"
    return "unresolved"


def evaluate_confirmatory(result, prereg):
    tolerances = prereg["formation_match_tolerances"]
    match = result["formation_match"]
    match_gates = {}
    for metric, margin in tolerances.items():
        interval = match[metric]
        match_gates[metric] = {
            "pass": interval["low"] >= -margin and interval["high"] <= margin,
            "value": interval, "margin": margin,
        }
    outcomes = prereg["outcome_equivalence_margins"]
    classifications = {}
    for name, spec in outcomes.items():
        interval = result["route_effects_acquisition_minus_update"][spec["optimizer_mode"]][spec["protocol"]][spec["outcome"]]
        classifications[name] = {"classification": classify_interval(interval, spec["margin"]),
                                 "value": interval, "margin": spec["margin"]}
    optimizer_gates = {}
    for name, spec in prereg["optimizer_state_interaction_minima"].items():
        interval = result["optimizer_state_interaction"][spec["protocol"]][spec["outcome"]]
        optimizer_gates[name] = {"pass": interval["low"] >= spec["minimum"],
                                 "value": interval, "minimum": spec["minimum"]}
    return {"formation_match": match_gates,
            "formation_match_all_pass": all(item["pass"] for item in match_gates.values()),
            "outcome_classifications": classifications,
            "optimizer_state_interaction": optimizer_gates,
            "optimizer_state_interaction_all_pass": all(item["pass"] for item in optimizer_gates.values())}


def validate_preregistration(prereg, cfg, seeds, update_weight_a):
    observed = {
        "seeds": seeds, "initial_steps": cfg.initial_steps, "formation_steps": cfg.formation_steps,
        "acquisition_route": {"acquisition": "direction-precise", "update_weight_A": 0.5},
        "update_route": {"acquisition": "balanced half A-precise/B-precise",
                         "A_direction_weight_A": update_weight_a,
                         "B_direction_weight_A": 1.0 - update_weight_a},
        "post_protocols": list(PROTOCOLS), "optimizer_modes": list(OPTIMIZER_MODES),
        "protocol_steps": PROTOCOL_STEPS,
        "low_noise": cfg.low_noise, "high_noise": cfg.high_noise,
        "initial_noise": cfg.initial_noise,
    }
    if observed != prereg["fixed_design"]:
        raise ValueError("確認設定が事前登録と一致しません。\n" +
                         json.dumps({"expected": prereg["fixed_design"], "observed": observed},
                                    ensure_ascii=False, indent=2))


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_text(stage, result, evaluation, runtime):
    lines = [f"# 実験IV-D {stage} 結果", "", "## この実験が答える問い", "",
             "取得主導と更新主導で表面上ほぼ同じ特化を形成したとき、共通条件と同じ摂動への後続応答も同じになるか。",
             "", "## 形成終点", "", "| 形成経路 | 方向の深さ | 対象側誤差 | 反対側誤差 |",
             "|---|---:|---:|---:|"]
    for route in ROUTES:
        values = result["formation"][route]
        lines.append(f"| {route} | {values['aligned_s']['mean']:.4f} | "
                     f"{values['target_D']['mean']:.4f} | {values['other_D']['mean']:.4f} |")
    lines += ["", "## 経路差（取得主導－更新主導）", ""]
    for mode in OPTIMIZER_MODES:
        lines.append(f"- optimizer={mode}")
        for protocol, outcome in (("neutral", "auc_aligned_s_0_200"),
                                  ("acquisition_pulse", "pulse_drop"),
                                  ("update_pulse", "pulse_drop"),
                                  ("combined_switch", "pulse_drop")):
            value = result["route_effects_acquisition_minus_update"][mode][protocol][outcome]
            lines.append(f"  - {protocol} / {outcome}: {value['mean']:.4f} "
                         f"(95%区間 [{value['low']:.4f}, {value['high']:.4f}])")
    if evaluation is not None:
        lines += ["", "## 事前判定", "",
                  f"- 形成終点の一致: {'PASS' if evaluation['formation_match_all_pass'] else 'FAIL'}"]
        for name, item in evaluation["outcome_classifications"].items():
            lines.append(f"- {name}: {item['classification']}")
        lines.append(f"- optimizer状態による経路差の増幅: "
                     f"{'PASS' if evaluation['optimizer_state_interaction_all_pass'] else 'FAIL'}")
    lines += ["", "## 適用範囲", "",
              "この結果が直接届く範囲は、相関する二つの三次元信号を二次元共有表現へ圧縮する人工学習系である。"
              "形成経路は研究者が固定し、後続応答は最大400更新まで観測した。", "",
              f"- 実行時間: {runtime:.1f}秒"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = config_for_stage(args.stage)
    defaults = {"smoke": (8040, 4), "pilot": (8020, 16), "confirm": (8100, 40)}
    seed_start, seed_count = defaults[args.stage]
    seed_start = args.seed_start if args.seed_start is not None else seed_start
    seed_count = args.seed_count if args.seed_count is not None else seed_count
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
    out = Path(args.output_dir or f"results_{args.stage}").resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    formation_rows, timecourse_rows, protocol_rows, geometry_rows = [], [], [], []
    for index, seed in enumerate(seeds, 1):
        train, probe, test = IVB.make_splits(cfg, seed)
        neutral_model, neutral_optimizer = prepare_neutral(cfg, seed, train, device)
        endpoints = {}
        for direction in DIRECTIONS:
            for route in ROUTES:
                model_state, optimizer_state = form_state(
                    cfg, seed, route, direction, update_weight_a, neutral_model,
                    neutral_optimizer, train, device)
                endpoints[(route, direction)] = (model_state, optimizer_state)
                model = IVB.make_model(cfg, device); model.load_state_dict(model_state)
                formation_rows.append({"seed": seed, "route": route, "direction": direction,
                                       **aligned_metrics(model, probe, test, cfg, device, direction)})
            geometry_rows.append({"seed": seed, "direction": direction,
                                  "linear_cka": endpoint_cka(
                                      endpoints[("acquisition", direction)][0],
                                      endpoints[("update", direction)][0], cfg, probe, device)})
        for direction in DIRECTIONS:
            for route in ROUTES:
                model_state, optimizer_state = endpoints[(route, direction)]
                for optimizer_mode in OPTIMIZER_MODES:
                    for protocol in PROTOCOLS:
                        rows = run_protocol(cfg, seed, route, direction, optimizer_mode,
                                            protocol, model_state, optimizer_state, train,
                                            probe, test, device)
                        timecourse_rows.extend(rows)
                        protocol_rows.append({"seed": seed, "route": route,
                                              "direction": direction,
                                              "optimizer_mode": optimizer_mode,
                                              "protocol": protocol,
                                              **summarize_protocol(rows)})
        print(f"seed {index}/{len(seeds)} 完了", flush=True)

    expected_formation = len(seeds) * len(ROUTES) * len(DIRECTIONS)
    expected_protocols = expected_formation * len(OPTIMIZER_MODES) * len(PROTOCOLS)
    if len(formation_rows) != expected_formation or len(protocol_rows) != expected_protocols:
        raise RuntimeError("不完全な結果です。")
    aggregate_result = aggregate(formation_rows, protocol_rows, geometry_rows, cfg)
    evaluation = evaluate_confirmatory(aggregate_result, prereg) if prereg else None
    runtime = time.time() - started
    result = {
        "experiment": "IV-D", "stage": args.stage,
        "question": "Do acquisition-formed and update-formed matched endpoints have the same response dynamics?",
        "design": {"seeds": seeds, "same_neutral_checkpoint": True,
                   "update_route_weight_A": update_weight_a,
                   "directions": list(DIRECTIONS), "routes": list(ROUTES),
                   "protocols": list(PROTOCOLS), "optimizer_modes": list(OPTIMIZER_MODES),
                   "protocol_steps": PROTOCOL_STEPS,
                   "same_batch_stream_within_seed": True,
                   "primary_probe": "common both-high-precision held-out probe",
                   "statistical_unit": "seed"},
        "config": asdict(cfg), "aggregate": aggregate_result,
        "confirmatory_evaluation": evaluation, "runtime_seconds": runtime,
    }
    write_csv(out / "formation_endpoints.csv", formation_rows)
    write_csv(out / "timecourse.csv", timecourse_rows)
    write_csv(out / "protocol_summary.csv", protocol_rows)
    write_csv(out / "geometry.csv", geometry_rows)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else None
    provenance = {"script_sha256": sha256(Path(__file__).resolve()),
                  "base_script_sha256": sha256(HERE.parent / "IV-B" / "experiment_iv_b.py"),
                  "preregistration_sha256": sha256(prereg_path) if prereg_path else None,
                  "python": sys.version, "platform": platform.platform(),
                  "numpy": np.__version__, "torch": torch.__version__, "device": str(device)}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_text(args.stage, aggregate_result, evaluation, runtime)
    (out / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
