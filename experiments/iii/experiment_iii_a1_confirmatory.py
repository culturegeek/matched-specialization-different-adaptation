#!/usr/bin/env python3
"""実験III-A1.1: 未使用の40環境集合による五要因分離の確認実験。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from experiment_iii_a0 import Config, bayes_accuracy, sha256, transition_and_cost
from experiment_iii_a1 import receding_policy, evaluate_detailed, disagreement


ENV_SEEDS = tuple(range(3200, 3240))
BOOT_SEED = 20260824
BOOT_N = 10000

PREREGISTERED_GATES = r"""
Unit: environment seed 3200--3239. 95% percentile bootstrap CI, seed=20260824, B=10000.
K: all sensor choices matched; mean target accuracy >=0.90; gain CI lower >0.20.
M: disagreement CI lower >0.05; misspecified prediction-L1 CI lower >0.01;
   correct prediction error max <=1e-10.
Hpred: disagreement CI lower >0.05; long-horizon objective-gain CI lower >0.005.
Heval: disagreement CI lower >0.05; early-shortage-gain CI lower >0;
       long-evaluation risk-gain CI lower >0.005.
Risk: disagreement CI lower >0.05; absolute dangerous-occupancy reduction CI lower >0.005;
      relative reduction CI lower >0.40; shortage-change CI lower >=0.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results_a1_confirmatory")
    return p.parse_args()


def environment_config(seed):
    rng = np.random.default_rng(seed)
    return Config(
        demand_persist=float(rng.uniform(0.74, 0.90)),
        health_persist_no_use=float(rng.uniform(0.88, 0.96)),
        wear_base=float(rng.uniform(0.04, 0.08)),
        wear_per_delivery=float(rng.uniform(0.14, 0.22)),
        repair_success=float(rng.uniform(0.76, 0.88)),
        recharge_low_demand=float(rng.uniform(0.70, 0.85)),
        recharge_high_demand=float(rng.uniform(0.24, 0.40)),
        initial_high_demand=float(rng.uniform(0.35, 0.55)),
        initial_worn=float(rng.uniform(0.12, 0.28)),
        standard_accuracy=float(rng.uniform(0.58, 0.66)),
        sensor_accuracy=float(rng.uniform(0.91, 0.97)),
    )


def bootstrap_mean(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975))}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    args = parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in ENV_SEEDS:
        cfg = environment_config(seed)
        _, p, c = transition_and_cost(cfg, False)
        base_pol = receding_policy(cfg, p, c, 10, 1.0, 0.0)
        base = evaluate_detailed(cfg, p, c, base_pol)

        demand_acc = bayes_accuracy(cfg, "demand", ("demand",))
        demand_mismatch = bayes_accuracy(cfg, "demand", ("health",))
        health_acc = bayes_accuracy(cfg, "health", ("health",))
        health_mismatch = bayes_accuracy(cfg, "health", ("demand",))

        wrong_cfg = replace(cfg, wear_per_delivery=0.0,
                            recharge_low_demand=0.90, recharge_high_demand=0.90)
        _, pw, cw = transition_and_cost(wrong_cfg, False)
        m_pol = receding_policy(cfg, pw, cw, 10, 1.0, 0.0)
        m_pred = evaluate_detailed(cfg, pw, cw, m_pol)
        m_true = evaluate_detailed(cfg, p, c, m_pol)
        m_l1 = abs(m_pred["shortage"] - m_true["shortage"]) + abs(m_pred["risk"] - m_true["risk"])
        correct_pred = evaluate_detailed(cfg, p, c, base_pol)
        correct_l1 = abs(correct_pred["shortage"] - base["shortage"]) + abs(correct_pred["risk"] - base["risk"])

        hp_pol = receding_policy(cfg, p, c, 1, 1.0, 0.0)
        hp = evaluate_detailed(cfg, p, c, hp_pol)
        he_pol = receding_policy(cfg, p, c, 10, 0.10, 0.0)
        he = evaluate_detailed(cfg, p, c, he_pol)
        risk_pol = receding_policy(cfg, p, c, 10, 1.0, 1.0)
        risk = evaluate_detailed(cfg, p, c, risk_pol)
        rel_reduction = ((base["dangerous_occupancy"] - risk["dangerous_occupancy"]) /
                         max(base["dangerous_occupancy"], 1e-12))

        rows.append({
            "seed": seed,
            "k_choices_matched": int(demand_acc > demand_mismatch and health_acc > health_mismatch),
            "k_target_accuracy": (demand_acc + health_acc) / 2,
            "k_accuracy_gain": ((demand_acc - demand_mismatch) + (health_acc - health_mismatch)) / 2,
            "m_disagreement": disagreement(base_pol, m_pol), "m_prediction_l1": m_l1,
            "m_correct_l1": correct_l1,
            "hpred_disagreement": disagreement(base_pol, hp_pol),
            "hpred_objective_gain": hp["base_objective"] - base["base_objective"],
            "heval_disagreement": disagreement(base_pol, he_pol),
            "heval_early_shortage_gain": base["early_shortage"] - he["early_shortage"],
            "heval_risk_gain": he["risk"] - base["risk"],
            "risk_disagreement": disagreement(base_pol, risk_pol),
            "risk_danger_abs_reduction": base["dangerous_occupancy"] - risk["dangerous_occupancy"],
            "risk_danger_rel_reduction": rel_reduction,
            "risk_shortage_change": risk["shortage"] - base["shortage"],
        })

    stats = {key: bootstrap_mean([r[key] for r in rows]) for key in rows[0] if key not in ("seed",)}
    gates = {
        "K_task_frame": {
            "pass": sum(r["k_choices_matched"] for r in rows) == len(rows) and
                    stats["k_target_accuracy"]["mean"] >= 0.90 and stats["k_accuracy_gain"]["low"] > 0.20,
            "matched_count": sum(r["k_choices_matched"] for r in rows),
            "target_accuracy": stats["k_target_accuracy"], "accuracy_gain": stats["k_accuracy_gain"],
        },
        "M_causal_model": {
            "pass": stats["m_disagreement"]["low"] > 0.05 and stats["m_prediction_l1"]["low"] > 0.01 and
                    max(r["m_correct_l1"] for r in rows) <= 1e-10,
            "disagreement": stats["m_disagreement"], "prediction_l1": stats["m_prediction_l1"],
            "correct_l1_max": max(r["m_correct_l1"] for r in rows),
        },
        "Hpred_prediction_horizon": {
            "pass": stats["hpred_disagreement"]["low"] > 0.05 and stats["hpred_objective_gain"]["low"] > 0.005,
            "disagreement": stats["hpred_disagreement"], "objective_gain": stats["hpred_objective_gain"],
        },
        "Heval_evaluation_horizon": {
            "pass": stats["heval_disagreement"]["low"] > 0.05 and
                    stats["heval_early_shortage_gain"]["low"] > 0 and stats["heval_risk_gain"]["low"] > 0.005,
            "disagreement": stats["heval_disagreement"],
            "early_shortage_gain": stats["heval_early_shortage_gain"], "risk_gain": stats["heval_risk_gain"],
        },
        "Risk_sensitivity": {
            "pass": stats["risk_disagreement"]["low"] > 0.05 and
                    stats["risk_danger_abs_reduction"]["low"] > 0.005 and
                    stats["risk_danger_rel_reduction"]["low"] > 0.40 and
                    stats["risk_shortage_change"]["low"] >= 0,
            "disagreement": stats["risk_disagreement"],
            "absolute_reduction": stats["risk_danger_abs_reduction"],
            "relative_reduction": stats["risk_danger_rel_reduction"],
            "shortage_change": stats["risk_shortage_change"],
        },
    }
    completeness = {"environment_rows": len(rows), "expected": len(ENV_SEEDS),
                    "unique_seeds": len({r["seed"] for r in rows})}
    completeness["pass"] = completeness["environment_rows"] == completeness["expected"] == completeness["unique_seeds"]
    if not completeness["pass"]: raise RuntimeError(f"完全性検査に失敗: {completeness}")
    write_csv(out / "environment_results.csv", rows)
    result = {"experiment": "III-A1.1", "environment_seeds": list(ENV_SEEDS),
              "preregistered_gates": PREREGISTERED_GATES.strip(), "gates": gates,
              "statistics": stats, "completeness": completeness}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験III-A1.1 結果：独立環境集合による確認", "", "## 仮説群", ""]
    for k, v in gates.items(): lines.append(f"- **{k}**: {'PASS' if v['pass'] else 'FAIL'}")
    lines += ["", "## 区間推定", "", "|測定量|平均|95%区間|", "|---|---:|---:|"]
    for k, v in stats.items():
        if k == "k_choices_matched": continue
        lines.append(f"|{k}|{v['mean']:.6f}|[{v['low']:.6f}, {v['high']:.6f}]|")
    lines += ["", "## 進行判断", ""]
    failed = [k for k, v in gates.items() if not v["pass"]]
    lines.append("全仮説群が成立した。III-A2へ進む。" if not failed else
                 "不成立要因が残るためIII-A2へ進まない: " + ", ".join(failed))
    lines += ["", "区間は事前に定めた人工環境集合内の頑健性を表し、現実環境への母集団推論ではない。"]
    (out / "result_a1_confirmatory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "script_sha256": sha256(Path(__file__).resolve()),
            "preregistration_sha256": sha256(Path(__file__).parent / "preregistration_a1_confirmatory.md")}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gates": {k: v["pass"] for k, v in gates.items()},
                      "completeness": completeness, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
