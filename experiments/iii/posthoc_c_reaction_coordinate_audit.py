#!/usr/bin/env python3
"""実験III-Cの候補反応座標 q=κ を摂動量Kで条件づけて監査する。

この解析はIII-Cの結果取得後に設計した事後解析であり、事前登録判定ではない。
既存の endpoints と summary を読み、K水準内AUCと、seedを分離した
5分割交差検証による K+s 対 K+s+κ の予測比較を保存する。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


K_LEVELS = (4000, 4800, 5600)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = score[y == 1]
    negative = score[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def logistic_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    test_design = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    weights = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0

    for _ in range(100):
        probability = sigmoid(design @ weights)
        variance = probability * (1.0 - probability)
        gradient = design.T @ (probability - train_y) + penalty @ weights
        hessian = (
            design.T @ (design * variance[:, None])
            + penalty
            + np.eye(design.shape[1]) * 1e-8
        )
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) < 1e-9:
            break
    return sigmoid(test_design @ weights)


def cross_validated_metrics(
    features: np.ndarray,
    outcome: np.ndarray,
    seeds: np.ndarray,
) -> dict[str, float]:
    prediction = np.empty(len(outcome))
    for fold in range(5):
        test = seeds % 5 == fold
        prediction[test] = logistic_predict(
            features[~test], outcome[~test], features[test]
        )
    epsilon = 1e-12
    log_loss = -np.mean(
        outcome * np.log(prediction + epsilon)
        + (1.0 - outcome) * np.log(1.0 - prediction + epsilon)
    )
    brier = np.mean((prediction - outcome) ** 2)
    return {
        "auc": auc(outcome, prediction),
        "log_loss": float(log_loss),
        "brier_score": float(brier),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results_c"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_c_posthoc_coordinate_audit")
    )
    args = parser.parse_args()

    endpoints = read_rows(args.input_dir / "perturbation_endpoints.csv")
    summaries = read_rows(args.input_dir / "summary.csv")
    switched = {
        (int(row["seed"]), row["history"], int(float(row["K"]))): int(row["switched"])
        for row in summaries
        if row["measure"] == "perturbation"
    }
    rows = [row for row in endpoints if int(float(row["K"])) in K_LEVELS]

    seeds = np.array([int(row["seed"]) for row in rows])
    k_level = np.array([int(float(row["K"])) for row in rows])
    outcome = np.array(
        [switched[(int(row["seed"]), row["history"], int(float(row["K"])))] for row in rows],
        dtype=float,
    )
    aligned_s = np.array([float(row["aligned_s"]) for row in rows])
    aligned_kappa = np.array([float(row["aligned_kappa"]) for row in rows])

    strata = []
    for level in K_LEVELS:
        selected = k_level == level
        strata.append(
            {
                "K": level,
                "n": int(selected.sum()),
                "switch_rate": float(outcome[selected].mean()),
                "auc_kappa": auc(outcome[selected], -aligned_kappa[selected]),
                "auc_s": auc(outcome[selected], -aligned_s[selected]),
            }
        )
    strata.append(
        {
            "K": "pooled",
            "n": len(rows),
            "switch_rate": float(outcome.mean()),
            "auc_kappa": auc(outcome, -aligned_kappa),
            "auc_s": auc(outcome, -aligned_s),
        }
    )

    k_dummy = np.column_stack(
        [(k_level == K_LEVELS[1]).astype(float), (k_level == K_LEVELS[2]).astype(float)]
    )
    models = {
        "K+s": np.column_stack([k_dummy, aligned_s]),
        "K+s+kappa": np.column_stack([k_dummy, aligned_s, aligned_kappa]),
    }
    comparisons = [
        {"model": name, **cross_validated_metrics(features, outcome, seeds)}
        for name, features in models.items()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "stratified_auc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=strata[0].keys())
        writer.writeheader()
        writer.writerows(strata)
    with (args.output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)

    result = {
        "analysis_type": "posthoc",
        "k_levels": list(K_LEVELS),
        "stratified_auc": strata,
        "seed_split_cross_validation": comparisons,
        "interpretation": (
            "The pooled association of kappa was not preserved after conditioning on K; "
            "q=kappa is not supported as an individual-level reaction coordinate."
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# 実験III-C 候補反応座標の事後監査",
        "",
        "この解析は結果取得後に設計した事後解析であり、事前登録判定ではない。",
        "",
        "## K水準内AUC",
        "",
        "| K | n | 転換率 | AUC(kappa) | AUC(s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in strata:
        report.append(
            f"| {row['K']} | {row['n']} | {row['switch_rate']:.3f} | "
            f"{row['auc_kappa']:.3f} | {row['auc_s']:.3f} |"
        )
    report.extend(
        [
            "",
            "## seed分離5分割交差検証",
            "",
            "| 予測変数 | AUC | log loss | Brier score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        report.append(
            f"| {row['model']} | {row['auc']:.4f} | {row['log_loss']:.4f} | "
            f"{row['brier_score']:.4f} |"
        )
    report.extend(
        [
            "",
            "プール解析で得られたkappaの予測連関は、Kを条件づけると保たれなかった。",
            "q=kappaは個体差レベルの転換反応座標として支持されない。",
            "",
        ]
    )
    (args.output_dir / "result_c_posthoc_coordinate_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
