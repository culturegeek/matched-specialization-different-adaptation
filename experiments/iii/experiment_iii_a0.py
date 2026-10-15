#!/usr/bin/env python3
"""実験III-A0: 資源管理環境・Oracle・観測・資源水準の事前校正。

有限状態・有限地平の生成過程を完全に列挙する。評価関数を変えて得たOracle方策を
帰結ベクトル上で比較し、現在不足と将来障害リスクの競合を確認する。別に、需要状態と
設備状態の観測チャネル、観測予算、表現スロットをBayes誤りで校正する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PREREGISTERED_GATES = r"""
G1 environment_integrity:
  transition probabilities are nonnegative; maximum row-sum error <= 1e-12;
  all costs are finite and nonnegative.
G2 oracle_effectiveness:
  at shortage weights 0.25, 0.50, 0.75, scalar Oracle cost is at least 10%
  lower than the uniform-random-policy cost in the competing environment.
G3 outcome_competition:
  competing environment has >=4 nondominated Oracle outcome points across 101 weights;
  shortage range >=0.05 and risk range >=0.02;
  null environment has a common policy attaining both component minima within 1e-10.
G4 sensor_selectivity:
  matched-sensor Bayes-accuracy gain >=0.20; off-target absolute gain <=0.03.
G5 resource_levels:
  budget=2 and slots=2 equal full-channel Bayes error within 1e-12;
  budget=3 or slots=3 improves by <=1e-12;
  budget=0 and slots=1 is at least 0.10 worse than the nonbinding level.
"""


@dataclass(frozen=True)
class Config:
    horizon: int = 10
    stock_max: int = 4
    demand_persist: float = 0.82
    health_persist_no_use: float = 0.92
    wear_base: float = 0.06
    wear_per_delivery: float = 0.18
    repair_success: float = 0.82
    recharge_low_demand: float = 0.78
    recharge_high_demand: float = 0.32
    initial_stock: int = 3
    initial_high_demand: float = 0.45
    initial_worn: float = 0.20
    standard_accuracy: float = 0.62
    sensor_accuracy: float = 0.94
    effort_tiebreak: float = 1e-7
    weight_count: int = 101


STATES = [(x, d, h) for x in range(5) for d in (0, 1) for h in (0, 1)]
ACTIONS_COMPETING = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
ACTIONS_NULL = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_a0")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def state_index():
    return {s: i for i, s in enumerate(STATES)}


def initial_distribution(cfg: Config):
    p = np.zeros(len(STATES), dtype=float)
    idx = state_index()
    for d in (0, 1):
        for h in (0, 1):
            pd = cfg.initial_high_demand if d else 1 - cfg.initial_high_demand
            ph = cfg.initial_worn if h else 1 - cfg.initial_worn
            p[idx[(cfg.initial_stock, d, h)]] = pd * ph
    return p


def transition_and_cost(cfg: Config, null_environment: bool):
    """Return P[action,state,next_state] and vector cost [shortage,risk,effort]."""
    actions = ACTIONS_NULL if null_environment else ACTIONS_COMPETING
    idx = state_index()
    p = np.zeros((len(actions), len(STATES), len(STATES)), dtype=float)
    c = np.zeros((len(actions), len(STATES), 3), dtype=float)
    for ai, (requested, maintain) in enumerate(actions):
        for si, (stock, demand_regime, worn) in enumerate(STATES):
            demand = 1 + demand_regime
            if null_environment:
                delivered = min(requested, demand)
            else:
                delivered = min(requested, demand, stock)
            shortage = (demand - delivered) / 2.0
            if null_environment:
                failure_risk = 0.02 * worn * (1 - maintain)
            else:
                failure_risk = (0.05 + 0.20 * delivered) * worn
                failure_risk += 0.08 * max(0, 2 - stock) / 2.0
                failure_risk *= 0.18 if maintain else 1.0
            effort = 0.03 * requested + 0.04 * maintain
            c[ai, si] = (shortage, failure_risk, effort)

            p_d1 = cfg.demand_persist if demand_regime else 1 - cfg.demand_persist
            for next_d in (0, 1):
                pd = p_d1 if next_d else 1 - p_d1
                if null_environment:
                    next_stock_probs = [(cfg.stock_max, 1.0)]
                else:
                    recharge_p = cfg.recharge_high_demand if demand_regime else cfg.recharge_low_demand
                    base_stock = max(0, stock - delivered)
                    next_stock_probs = [
                        (min(cfg.stock_max, base_stock), 1 - recharge_p),
                        (min(cfg.stock_max, base_stock + 1), recharge_p),
                    ]

                if maintain:
                    p_worn1 = (1 - cfg.repair_success) if worn else 0.02
                elif null_environment:
                    p_worn1 = 0.02 if not worn else cfg.health_persist_no_use
                elif worn:
                    p_worn1 = min(0.99, cfg.health_persist_no_use + 0.02 * delivered)
                else:
                    p_worn1 = min(0.95, cfg.wear_base + cfg.wear_per_delivery * delivered)
                for next_h in (0, 1):
                    ph = p_worn1 if next_h else 1 - p_worn1
                    for next_stock, ps in next_stock_probs:
                        p[ai, si, idx[(next_stock, next_d, next_h)]] += pd * ph * ps
    return actions, p, c


def solve_finite_horizon(cfg: Config, p, c, shortage_weight: float):
    """Exact finite-horizon dynamic programming for one scalarization."""
    scalar = shortage_weight * c[:, :, 0] + (1 - shortage_weight) * c[:, :, 1]
    scalar = scalar + cfg.effort_tiebreak * c[:, :, 2]
    value = np.zeros((cfg.horizon + 1, len(STATES)), dtype=float)
    policy = np.zeros((cfg.horizon, len(STATES)), dtype=np.int16)
    for t in range(cfg.horizon - 1, -1, -1):
        q = scalar + np.einsum("ask,k->as", p, value[t + 1])
        policy[t] = np.argmin(q, axis=0)
        value[t] = q[policy[t], np.arange(len(STATES))]
    return policy


def evaluate_policy(cfg: Config, p, c, policy, initial):
    occupancy = initial.copy()
    total = np.zeros(3, dtype=float)
    for t in range(cfg.horizon):
        acts = policy[t]
        total += np.sum(occupancy[:, None] * c[acts, np.arange(len(STATES))], axis=0)
        next_occ = np.zeros_like(occupancy)
        for si in range(len(STATES)):
            next_occ += occupancy[si] * p[acts[si], si]
        occupancy = next_occ
    return total / cfg.horizon


def evaluate_random_policy(cfg: Config, p, c, initial):
    occupancy = initial.copy()
    total = np.zeros(3, dtype=float)
    mean_p = p.mean(axis=0)
    mean_c = c.mean(axis=0)
    for _ in range(cfg.horizon):
        total += np.sum(occupancy[:, None] * mean_c, axis=0)
        occupancy = occupancy @ mean_p
    return total / cfg.horizon


def nondominated(rows, tol=1e-12):
    unique = []
    for row in rows:
        point = np.array([row["shortage"], row["risk"]])
        if not any(np.max(np.abs(point - np.array([u["shortage"], u["risk"]]))) <= tol for u in unique):
            unique.append(row)
    keep = []
    for i, row in enumerate(unique):
        p = np.array([row["shortage"], row["risk"]])
        dominated = False
        for j, other in enumerate(unique):
            if i == j:
                continue
            q = np.array([other["shortage"], other["risk"]])
            if np.all(q <= p + tol) and np.any(q < p - tol):
                dominated = True
                break
        if not dominated:
            keep.append(row)
    return sorted(keep, key=lambda r: r["shortage"])


def binary_observation_distribution(cfg: Config, sensors: tuple[str, ...]):
    """Joint P(d,h,obs), with independent uniform latent prior."""
    outcomes = []
    for d in (0, 1):
        for h in (0, 1):
            for od in (0, 1):
                for oh in (0, 1):
                    for sd in ((0, 1) if "demand" in sensors else (-1,)):
                        for sh in ((0, 1) if "health" in sensors else (-1,)):
                            prob = 0.25
                            prob *= cfg.standard_accuracy if od == d else 1 - cfg.standard_accuracy
                            prob *= cfg.standard_accuracy if oh == h else 1 - cfg.standard_accuracy
                            if sd != -1:
                                prob *= cfg.sensor_accuracy if sd == d else 1 - cfg.sensor_accuracy
                            if sh != -1:
                                prob *= cfg.sensor_accuracy if sh == h else 1 - cfg.sensor_accuracy
                            outcomes.append((d, h, (od, oh, sd, sh), prob))
    return outcomes


def bayes_accuracy(cfg: Config, target: str, sensors: tuple[str, ...]):
    totals = {}
    for d, h, obs, prob in binary_observation_distribution(cfg, sensors):
        y = d if target == "demand" else h
        totals.setdefault(obs, [0.0, 0.0])[y] += prob
    return sum(max(values) for values in totals.values())


def resource_grid(cfg: Config):
    rows = []
    sensor_sets = {
        0: [()],
        1: [("demand",), ("health",)],
        2: [("demand", "health")],
        3: [("demand", "health")],
    }
    for budget in range(4):
        for slots in range(1, 4):
            best = None
            best_set = None
            for sensors in sensor_sets[budget]:
                acc_d = bayes_accuracy(cfg, "demand", sensors)
                acc_h = bayes_accuracy(cfg, "health", sensors)
                if slots == 1:
                    # One retained coordinate: preserve the better decoded target; the other is at chance.
                    loss = min(((1 - acc_d) + 0.5) / 2, ((1 - acc_h) + 0.5) / 2)
                else:
                    loss = ((1 - acc_d) + (1 - acc_h)) / 2
                if best is None or loss < best:
                    best, best_set = loss, sensors
            rows.append({"budget": budget, "slots": slots, "bayes_error": best,
                         "sensors": "+".join(best_set) if best_set else "none"})
    return rows


def compute_all(cfg: Config):
    initial = initial_distribution(cfg)
    all_env = {}
    frontier_rows = []
    selected = {}
    integrity = {}
    for name, null_flag in (("competing", False), ("null", True)):
        actions, p, c = transition_and_cost(cfg, null_flag)
        integrity[name] = {
            "min_probability": float(p.min()),
            "max_row_sum_error": float(np.max(np.abs(p.sum(axis=2) - 1.0))),
            "all_costs_finite": bool(np.all(np.isfinite(c))),
            "min_cost": float(c.min()),
        }
        random_vec = evaluate_random_policy(cfg, p, c, initial)
        rows = []
        for w in np.linspace(0.0, 1.0, cfg.weight_count):
            policy = solve_finite_horizon(cfg, p, c, float(w))
            vec = evaluate_policy(cfg, p, c, policy, initial)
            row = {"environment": name, "shortage_weight": float(w),
                   "shortage": float(vec[0]), "risk": float(vec[1]), "effort": float(vec[2]),
                   "policy_hash": hashlib.sha256(policy.tobytes()).hexdigest()[:16]}
            rows.append(row)
            frontier_rows.append(row)
            if name == "competing" and any(abs(w - target) < 1e-12 for target in (0.25, 0.5, 0.75)):
                scalar_oracle = w * vec[0] + (1 - w) * vec[1]
                scalar_random = w * random_vec[0] + (1 - w) * random_vec[1]
                selected[f"{w:.2f}"] = {
                    "oracle": float(scalar_oracle), "random": float(scalar_random),
                    "relative_improvement": float((scalar_random - scalar_oracle) / scalar_random),
                }
        nd = nondominated(rows)
        all_env[name] = {
            "actions": len(actions), "random": random_vec.tolist(), "nondominated": nd,
            "shortage_min": min(r["shortage"] for r in rows),
            "risk_min": min(r["risk"] for r in rows),
        }

    sensor_rows = []
    for sensors in [(), ("demand",), ("health",), ("demand", "health")]:
        sensor_rows.append({
            "sensors": "+".join(sensors) if sensors else "none",
            "demand_accuracy": bayes_accuracy(cfg, "demand", sensors),
            "health_accuracy": bayes_accuracy(cfg, "health", sensors),
        })
    resources = resource_grid(cfg)
    return integrity, selected, all_env, frontier_rows, sensor_rows, resources


def assess_gates(cfg, integrity, selected, env, sensor_rows, resources):
    gates = {}
    g1 = all(v["min_probability"] >= -1e-15 and v["max_row_sum_error"] <= 1e-12
             and v["all_costs_finite"] and v["min_cost"] >= 0 for v in integrity.values())
    gates["G1_environment_integrity"] = {"pass": g1, "details": integrity}

    improvements = [selected[f"{w:.2f}"]["relative_improvement"] for w in (0.25, 0.5, 0.75)]
    gates["G2_oracle_effectiveness"] = {"pass": min(improvements) >= 0.10,
                                         "relative_improvements": improvements}

    nd = env["competing"]["nondominated"]
    shortage_range = max(r["shortage"] for r in nd) - min(r["shortage"] for r in nd)
    risk_range = max(r["risk"] for r in nd) - min(r["risk"] for r in nd)
    null_rows = env["null"]["nondominated"]
    null_short_min = env["null"]["shortage_min"]
    null_risk_min = env["null"]["risk_min"]
    common_null = any(abs(r["shortage"] - null_short_min) <= 1e-10 and
                      abs(r["risk"] - null_risk_min) <= 1e-10 for r in null_rows)
    gates["G3_outcome_competition"] = {
        "pass": len(nd) >= 4 and shortage_range >= 0.05 and risk_range >= 0.02 and common_null,
        "competing_nondominated_count": len(nd), "shortage_range": shortage_range,
        "risk_range": risk_range, "null_common_minimizer": common_null,
    }

    sm = {r["sensors"]: r for r in sensor_rows}
    d_gain = sm["demand"]["demand_accuracy"] - sm["none"]["demand_accuracy"]
    h_gain = sm["health"]["health_accuracy"] - sm["none"]["health_accuracy"]
    d_off = sm["demand"]["health_accuracy"] - sm["none"]["health_accuracy"]
    h_off = sm["health"]["demand_accuracy"] - sm["none"]["demand_accuracy"]
    gates["G4_sensor_selectivity"] = {
        "pass": d_gain >= 0.20 and h_gain >= 0.20 and abs(d_off) <= 0.03 and abs(h_off) <= 0.03,
        "demand_gain": d_gain, "health_gain": h_gain,
        "demand_sensor_off_target": d_off, "health_sensor_off_target": h_off,
    }

    rm = {(r["budget"], r["slots"]): r["bayes_error"] for r in resources}
    nonbinding = rm[(2, 2)]
    g5 = (abs(nonbinding - rm[(3, 2)]) <= 1e-12 and
          abs(nonbinding - rm[(2, 3)]) <= 1e-12 and
          rm[(0, 1)] - nonbinding >= 0.10)
    gates["G5_resource_levels"] = {
        "pass": g5, "nonbinding_error": nonbinding,
        "budget3_error": rm[(3, 2)], "slots3_error": rm[(2, 3)],
        "binding_error": rm[(0, 1)], "binding_gap": rm[(0, 1)] - nonbinding,
    }
    return gates


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_report(cfg, gates, selected, env, sensor_rows, resources):
    lines = [
        "# 実験III-A0 結果：環境・Oracle・観測・資源水準の校正", "",
        "## 総括", "",
    ]
    for name, gate in gates.items():
        lines.append(f"- **{name}**: {'PASS' if gate['pass'] else 'FAIL'}")
    lines += ["", "## Oracle と帰結競合", "",
              "|不足重み|Oracle|一様ランダム|相対改善|", "|---:|---:|---:|---:|"]
    for key, row in selected.items():
        lines.append(f"|{key}|{row['oracle']:.6f}|{row['random']:.6f}|{row['relative_improvement']:.3f}|")
    g3 = gates["G3_outcome_competition"]
    lines += ["", f"競合環境の非劣点は **{g3['competing_nondominated_count']}点**、"
              f"現在不足の範囲は {g3['shortage_range']:.6f}、"
              f"将来障害リスクの範囲は {g3['risk_range']:.6f}。",
              f"帰無環境の共通最小化方策: **{g3['null_common_minimizer']}**。", "",
              "## 観測チャネル", "", "|センサ|需要正答率|設備正答率|", "|---|---:|---:|"]
    for row in sensor_rows:
        lines.append(f"|{row['sensors']}|{row['demand_accuracy']:.4f}|{row['health_accuracy']:.4f}|")
    lines += ["", "## 資源水準", "", "|観測予算|表現スロット|最良センサ|平均Bayes誤り|",
              "|---:|---:|---|---:|"]
    for row in resources:
        lines.append(f"|{row['budget']}|{row['slots']}|{row['sensors']}|{row['bayes_error']:.4f}|")
    lines += ["", "## 進行判断", ""]
    if all(g["pass"] for g in gates.values()):
        lines.append("全仮説群が成立した。III-A1の一要因介入へ進む。")
    else:
        failed = [k for k, g in gates.items() if not g["pass"]]
        lines.append("不成立の仮説群があるためIII-A1へ進まない: " + ", ".join(failed))
    lines += ["", "## 主張の範囲", "",
              "この結果は有限状態の人工環境とBayes観測モデルの校正である。人間の課題形成、"
              "価値判断、志向性を実証したものではない。"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    script = Path(__file__).resolve()
    if args.summarize_only:
        raise SystemExit("この版の再集計はresult.jsonを直接参照してください。")

    integrity, selected, env, frontier_rows, sensor_rows, resources = compute_all(cfg)
    gates = assess_gates(cfg, integrity, selected, env, sensor_rows, resources)
    complete = {
        "expected_weights_per_environment": cfg.weight_count,
        "actual_rows": len(frontier_rows),
        "expected_rows": cfg.weight_count * 2,
        "resource_rows": len(resources), "expected_resource_rows": 12,
        "sensor_rows": len(sensor_rows), "expected_sensor_rows": 4,
    }
    complete["pass"] = (complete["actual_rows"] == complete["expected_rows"] and
                         complete["resource_rows"] == complete["expected_resource_rows"] and
                         complete["sensor_rows"] == complete["expected_sensor_rows"])
    if not complete["pass"]:
        raise RuntimeError(f"完全性検査に失敗: {complete}")

    write_csv(out / "oracle_weight_sweep.csv", frontier_rows)
    write_csv(out / "sensor_selectivity.csv", sensor_rows)
    write_csv(out / "resource_grid.csv", resources)
    result = {
        "experiment": "III-A0", "config": asdict(cfg),
        "preregistered_gates": PREREGISTERED_GATES.strip(),
        "integrity": integrity, "selected_oracle_comparisons": selected,
        "environment_summary": env, "gates": gates, "completeness": complete,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "result_a0.md").write_text(make_report(cfg, gates, selected, env, sensor_rows, resources), encoding="utf-8")
    provenance = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "script": str(script), "script_sha256": sha256(script),
        "preregistration_sha256": sha256(script.parent / "preregistration_a0.md"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gates": {k: v["pass"] for k, v in gates.items()},
                      "completeness": complete, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
