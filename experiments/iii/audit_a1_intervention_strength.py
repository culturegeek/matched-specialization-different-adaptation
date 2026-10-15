#!/usr/bin/env python3
"""III-A1 v0.1 の不成立要因を調べる事後的な介入強度監査。

この走査は結果を見た後の探索であり、仮説検定ではない。確認実験の介入水準を選ぶためだけに使う。
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from experiment_iii_a0 import Config, transition_and_cost
from experiment_iii_a1 import receding_policy, evaluate_detailed, disagreement


def main():
    cfg = Config()
    _, p, c = transition_and_cost(cfg, False)
    baseline = receding_policy(cfg, p, c, 10, 1.0, 0.0)
    base_met = evaluate_detailed(cfg, p, c, baseline)
    rows = []

    for wear in (0.0, 0.03, 0.06):
        for recharge in (0.50, 0.70, 0.90, 0.98):
            wrong_cfg = replace(cfg, wear_per_delivery=wear,
                                recharge_low_demand=recharge, recharge_high_demand=recharge)
            _, pw, cw = transition_and_cost(wrong_cfg, False)
            pol = receding_policy(cfg, pw, cw, 10, 1.0, 0.0)
            pred = evaluate_detailed(cfg, pw, cw, pol)
            true = evaluate_detailed(cfg, p, c, pol)
            l1 = abs(pred["shortage"] - true["shortage"]) + abs(pred["risk"] - true["risk"])
            rows.append({"factor": "causal_model", "level": f"wear={wear};recharge={recharge}",
                         "policy_disagreement": disagreement(baseline, pol), "signal": l1,
                         "shortage": true["shortage"], "risk": true["risk"]})

    for gamma in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        pol = receding_policy(cfg, p, c, 10, gamma, 0.0)
        met = evaluate_detailed(cfg, p, c, pol)
        rows.append({"factor": "evaluation_horizon", "level": f"gamma={gamma}",
                     "policy_disagreement": disagreement(baseline, pol),
                     "signal": met["risk"] - base_met["risk"],
                     "shortage": met["shortage"], "risk": met["risk"]})

    for lam in (0.35, 0.50, 0.75, 1.00, 1.50, 2.00):
        pol = receding_policy(cfg, p, c, 10, 1.0, lam)
        met = evaluate_detailed(cfg, p, c, pol)
        rows.append({"factor": "risk_sensitivity", "level": f"lambda={lam}",
                     "policy_disagreement": disagreement(baseline, pol),
                     "signal": base_met["dangerous_occupancy"] - met["dangerous_occupancy"],
                     "shortage": met["shortage"], "risk": met["risk"]})

    path = Path("audit_a1_intervention_strength.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
