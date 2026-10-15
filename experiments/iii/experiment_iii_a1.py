#!/usr/bin/env python3
"""実験III-A1: 課題枠組み・因果モデル・予測地平・評価地平・リスク感度の分離。"""

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

from experiment_iii_a0 import (
    Config, STATES, bayes_accuracy, initial_distribution, sha256,
    transition_and_cost,
)


PREREGISTERED_GATES = r"""
K task_frame:
  demand frame chooses demand sensor and health frame chooses health sensor;
  matched target accuracy >=0.90 and exceeds mismatched-sensor accuracy by >=0.20.
M causal_model:
  policy disagreement >=0.05; misspecified-model prediction L1 error >=0.01;
  correct-model prediction error <=1e-10.
Hpred prediction_horizon:
  policy disagreement >=0.05; Hpred=10 true base objective is >=0.005 lower than Hpred=1.
Heval evaluation_horizon:
  policy disagreement >=0.05; short evaluation lowers early shortage;
  long evaluation lowers full-horizon risk by >=0.005.
Risk risk_sensitivity:
  policy disagreement >=0.05; high sensitivity lowers dangerous-state occupancy by >=0.02;
  shortage is no better than under expectation-only evaluation.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results_a1")
    return parser.parse_args()


def dangerous_next_cost(p):
    bad = np.array([1.0 if stock <= 1 and worn == 1 else 0.0 for stock, _, worn in STATES])
    return np.einsum("ask,k->as", p, bad)


def first_action_for_horizon(p_model, stage_cost, horizon, discount):
    n_states = len(STATES)
    value = np.zeros(n_states, dtype=float)
    first = None
    for depth in range(horizon):
        q = stage_cost + discount * np.einsum("ask,k->as", p_model, value)
        action = np.argmin(q, axis=0)
        value = q[action, np.arange(n_states)]
        first = action
    return first


def receding_policy(cfg, p_model, c_model, prediction_horizon, discount, risk_lambda):
    stage = 0.5 * c_model[:, :, 0] + 0.5 * c_model[:, :, 1]
    stage = stage + risk_lambda * dangerous_next_cost(p_model)
    stage = stage + cfg.effort_tiebreak * c_model[:, :, 2]
    policy = np.zeros((cfg.horizon, len(STATES)), dtype=np.int16)
    for t in range(cfg.horizon):
        lookahead = min(prediction_horizon, cfg.horizon - t)
        policy[t] = first_action_for_horizon(p_model, stage, lookahead, discount)
    return policy


def evaluate_detailed(cfg, p, c, policy):
    occupancy = initial_distribution(cfg)
    time_rows = []
    dangerous = np.array([1.0 if stock <= 1 and worn == 1 else 0.0 for stock, _, worn in STATES])
    totals = np.zeros(3, dtype=float)
    danger_total = 0.0
    for t in range(cfg.horizon):
        acts = policy[t]
        step = np.sum(occupancy[:, None] * c[acts, np.arange(len(STATES))], axis=0)
        danger_total += float(occupancy @ dangerous)
        totals += step
        time_rows.append({"step": t, "shortage": float(step[0]), "risk": float(step[1]),
                          "effort": float(step[2]), "dangerous_occupancy": float(occupancy @ dangerous)})
        nxt = np.zeros_like(occupancy)
        for si in range(len(STATES)):
            nxt += occupancy[si] * p[acts[si], si]
        occupancy = nxt
    return {
        "shortage": float(totals[0] / cfg.horizon),
        "risk": float(totals[1] / cfg.horizon),
        "effort": float(totals[2] / cfg.horizon),
        "base_objective": float((totals[0] + totals[1]) / (2 * cfg.horizon)),
        "early_shortage": float(sum(r["shortage"] for r in time_rows[:3]) / 3),
        "dangerous_occupancy": float(danger_total / cfg.horizon),
        "time": time_rows,
    }


def disagreement(a, b):
    return float(np.mean(a != b))


def policy_hash(policy):
    return hashlib.sha256(policy.tobytes()).hexdigest()[:16]


def intervention_row(name, policy, metrics):
    return {"condition": name, "policy_hash": policy_hash(policy),
            **{k: v for k, v in metrics.items() if k != "time"}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    actions, p_true, c_true = transition_and_cost(cfg, False)
    if not (Path(__file__).parent / "results_a0" / "result.json").exists():
        raise RuntimeError("III-A0の結果が見つかりません。先にIII-A0を実行してください。")
    a0 = json.loads((Path(__file__).parent / "results_a0" / "result.json").read_text(encoding="utf-8"))
    if not all(v["pass"] for v in a0["gates"].values()):
        raise RuntimeError("III-A0が全仮説群を通過していません。")

    # κ: both zero-cost sensor options are equally available; only target formation changes.
    frames = []
    for target in ("demand", "health"):
        candidates = []
        for sensor in ("demand", "health"):
            acc = bayes_accuracy(cfg, target, (sensor,))
            candidates.append((acc, sensor))
        acc, selected = max(candidates)
        mismatch = bayes_accuracy(cfg, target, (("health" if target == "demand" else "demand"),))
        frames.append({"frame": target, "selected_sensor": selected,
                       "target_accuracy": acc, "mismatched_sensor_accuracy": mismatch,
                       "accuracy_gain": acc - mismatch})

    # Common baseline: correct model, long prediction/evaluation, expectation-only.
    p_base, c_base = p_true, c_true
    policies = {}
    policies["baseline"] = receding_policy(cfg, p_base, c_base, 10, 1.0, 0.0)

    cfg_wrong = replace(cfg, wear_per_delivery=0.0)
    _, p_wrong, c_wrong = transition_and_cost(cfg_wrong, False)
    policies["causal_model_misspecified"] = receding_policy(cfg, p_wrong, c_wrong, 10, 1.0, 0.0)
    policies["prediction_horizon_1"] = receding_policy(cfg, p_base, c_base, 1, 1.0, 0.0)
    policies["evaluation_discount_035"] = receding_policy(cfg, p_base, c_base, 10, 0.35, 0.0)
    policies["risk_sensitive"] = receding_policy(cfg, p_base, c_base, 10, 1.0, 0.35)

    metrics_true = {name: evaluate_detailed(cfg, p_true, c_true, pol) for name, pol in policies.items()}
    rows = [intervention_row(name, policies[name], metrics_true[name]) for name in policies]

    # Model-calibration diagnostic: predicted versus true outcomes under the same policy.
    predicted_correct = evaluate_detailed(cfg, p_true, c_true, policies["baseline"])
    predicted_wrong = evaluate_detailed(cfg, p_wrong, c_wrong, policies["causal_model_misspecified"])
    true_wrong = metrics_true["causal_model_misspecified"]
    correct_l1 = abs(predicted_correct["shortage"] - metrics_true["baseline"]["shortage"]) + \
                 abs(predicted_correct["risk"] - metrics_true["baseline"]["risk"])
    wrong_l1 = abs(predicted_wrong["shortage"] - true_wrong["shortage"]) + \
               abs(predicted_wrong["risk"] - true_wrong["risk"])

    base = metrics_true["baseline"]
    short_pred = metrics_true["prediction_horizon_1"]
    short_eval = metrics_true["evaluation_discount_035"]
    risk_high = metrics_true["risk_sensitive"]

    gates = {
        "K_task_frame": {
            "pass": all(r["selected_sensor"] == r["frame"] and r["target_accuracy"] >= 0.90 and
                        r["accuracy_gain"] >= 0.20 for r in frames), "rows": frames,
        },
        "M_causal_model": {
            "pass": disagreement(policies["baseline"], policies["causal_model_misspecified"]) >= 0.05 and
                    wrong_l1 >= 0.01 and correct_l1 <= 1e-10,
            "policy_disagreement": disagreement(policies["baseline"], policies["causal_model_misspecified"]),
            "misspecified_prediction_l1": wrong_l1, "correct_prediction_l1": correct_l1,
        },
        "Hpred_prediction_horizon": {
            "pass": disagreement(policies["baseline"], policies["prediction_horizon_1"]) >= 0.05 and
                    short_pred["base_objective"] - base["base_objective"] >= 0.005,
            "policy_disagreement": disagreement(policies["baseline"], policies["prediction_horizon_1"]),
            "long_horizon_objective_gain": short_pred["base_objective"] - base["base_objective"],
        },
        "Heval_evaluation_horizon": {
            "pass": disagreement(policies["baseline"], policies["evaluation_discount_035"]) >= 0.05 and
                    short_eval["early_shortage"] < base["early_shortage"] and
                    short_eval["risk"] - base["risk"] >= 0.005,
            "policy_disagreement": disagreement(policies["baseline"], policies["evaluation_discount_035"]),
            "short_eval_early_shortage_gain": base["early_shortage"] - short_eval["early_shortage"],
            "long_eval_risk_gain": short_eval["risk"] - base["risk"],
        },
        "Risk_sensitivity": {
            "pass": disagreement(policies["baseline"], policies["risk_sensitive"]) >= 0.05 and
                    base["dangerous_occupancy"] - risk_high["dangerous_occupancy"] >= 0.02 and
                    risk_high["shortage"] >= base["shortage"] - 1e-12,
            "policy_disagreement": disagreement(policies["baseline"], policies["risk_sensitive"]),
            "dangerous_occupancy_reduction": base["dangerous_occupancy"] - risk_high["dangerous_occupancy"],
            "shortage_change": risk_high["shortage"] - base["shortage"],
        },
    }

    write_csv(out / "task_frame.csv", frames)
    write_csv(out / "intervention_outcomes.csv", rows)
    time_rows = []
    for condition, metrics in metrics_true.items():
        for row in metrics["time"]:
            time_rows.append({"condition": condition, **row})
    write_csv(out / "time_series.csv", time_rows)
    completeness = {"frame_rows": len(frames), "expected_frame_rows": 2,
                    "intervention_rows": len(rows), "expected_intervention_rows": 5,
                    "time_rows": len(time_rows), "expected_time_rows": 5 * cfg.horizon}
    completeness["pass"] = (completeness["frame_rows"] == 2 and len(rows) == 5 and
                             len(time_rows) == 5 * cfg.horizon)
    if not completeness["pass"]:
        raise RuntimeError(f"完全性検査に失敗: {completeness}")

    result = {"experiment": "III-A1", "config": asdict(cfg),
              "preregistered_gates": PREREGISTERED_GATES.strip(),
              "gates": gates, "task_frames": frames, "outcomes": rows,
              "completeness": completeness}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 実験III-A1 結果：五つの一要因介入", "", "## 仮説群", ""]
    for name, gate in gates.items():
        lines.append(f"- **{name}**: {'PASS' if gate['pass'] else 'FAIL'}")
    lines += ["", "## 課題枠組み", "", "|枠組み|選択センサ|対象正答率|非対応正答率|増分|",
              "|---|---|---:|---:|---:|"]
    for r in frames:
        lines.append(f"|{r['frame']}|{r['selected_sensor']}|{r['target_accuracy']:.4f}|"
                     f"{r['mismatched_sensor_accuracy']:.4f}|{r['accuracy_gain']:.4f}|")
    lines += ["", "## 方策介入の真の帰結", "",
              "|条件|現在不足|障害リスク|危険状態率|初期不足|基本目的|", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['condition']}|{r['shortage']:.6f}|{r['risk']:.6f}|"
                     f"{r['dangerous_occupancy']:.6f}|{r['early_shortage']:.6f}|{r['base_objective']:.6f}|")
    lines += ["", "## 進行判断", ""]
    failed = [k for k, v in gates.items() if not v["pass"]]
    lines.append("全仮説群が成立した。III-A2へ進む。" if not failed else
                 "次段階の前に不成立要因を監査する: " + ", ".join(failed))
    lines += ["", "## 主張の範囲", "",
              "各介入が有限状態環境で識別可能な異なる読み出しを生むかを校正した結果である。"
              "人間の対立や志向性の原因を特定したものではない。"]
    (out / "result_a1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                  "script_sha256": sha256(Path(__file__).resolve()),
                  "a0_script_sha256": sha256(Path(__file__).parent / "experiment_iii_a0.py"),
                  "preregistration_sha256": sha256(Path(__file__).parent / "preregistration_a1.md")}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gates": {k: v["pass"] for k, v in gates.items()},
                      "completeness": completeness, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
