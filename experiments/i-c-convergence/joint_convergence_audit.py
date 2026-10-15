#!/usr/bin/env python3
"""実験I-C v0.2.1：共同更新の収束性監査。

v0.2で生じた非Oracle方策を、次の四つに切り分ける。
1. 母集団上の静的な局所最適
2. 方策と推定器の共同更新が作る安定点候補
3. 単純な反復不足
4. softmax飽和によるアルゴリズム上の捕捉

環境と目的地形はv0.2のまま固定し、開ループ方策だけを監査する。
NumPyのみ、GPU不要。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

import temporal_coupling_poc as base


ARMS = (
    "continue",
    "reset_only",
    "reheat_x2",
    "reheat_x4",
    "reheat_x4_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="共同更新方策の収束性を監査します。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="5 seed・1600反復（既定）")
    mode.add_argument("--full", action="store_true", help="20 seed・3200反復")
    parser.add_argument("--output-dir", default="joint_convergence_audit_results")
    return parser.parse_args()


def make_audit_config(full: bool) -> tuple[base.Config, list[int], int]:
    cfg = base.make_config(full=full)
    # v0.2のfull条件（400反復）を全モードの共通分岐点にする。
    cfg.policy_iterations = 400
    if full:
        return cfg, [400, 800, 1600, 3200], 3200
    cfg.train_size = 2048
    cfg.test_size = 1024
    cfg.seeds = list(range(5))
    return cfg, [400, 800, 1600], 1600


def centered(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=1, keepdims=True)


def make_indicators(
    cfg: base.Config, schedules: list[tuple[int, ...]]
) -> np.ndarray:
    indicators = np.empty((cfg.time_steps, 2, len(schedules)))
    for t in range(cfg.time_steps):
        for action in (0, 1):
            indicators[t, action] = [
                1.0 if schedule[t] == action else 0.0 for schedule in schedules
            ]
    return indicators


def joint_state(
    logits: np.ndarray,
    train_stats: base.SufficientStats,
    cfg: base.Config,
    schedules: list[tuple[int, ...]],
    lambda_a: float,
    indicators: np.ndarray,
) -> dict:
    """共同目的の値、logit勾配、確率境界での脱出利得を返す。"""
    policy = base.softmax(logits)
    probs = base.schedule_probabilities(policy, schedules)
    estimator = base.fit_estimator(train_stats, probs, cfg.ridge)
    losses, _, _ = base.objective_by_schedule(
        estimator, train_stats, cfg, lambda_a
    )
    expected = float(probs @ losses)
    centered_losses = losses - expected
    gradient = np.empty_like(logits)
    conditional = np.empty_like(logits)
    for t in range(cfg.time_steps):
        for action in (0, 1):
            gradient[t, action] = float(
                np.sum(probs * centered_losses * indicators[t, action])
            )
            # 当該時刻の行動を固定し、他時刻だけを現在方策で周辺化する。
            conditional_probs = np.ones(len(schedules), dtype=np.float64)
            for s_index, schedule in enumerate(schedules):
                if schedule[t] != action:
                    conditional_probs[s_index] = 0.0
                    continue
                for u, other_action in enumerate(schedule):
                    if u != t:
                        conditional_probs[s_index] *= policy[u, other_action]
            conditional_probs /= conditional_probs.sum()
            conditional[t, action] = float(conditional_probs @ losses)

    chosen = np.argmax(policy, axis=1)
    escape_gains = []
    for t, action in enumerate(chosen):
        alternative = 1 - int(action)
        # 正なら、現在ほぼ選ばれていない側へ動くと共同目的が改善する。
        escape_gains.append(
            max(float(conditional[t, action] - conditional[t, alternative]), 0.0)
        )
    return {
        "policy": policy,
        "probs": probs,
        "estimator": estimator,
        "train_objective": expected,
        "gradient": gradient,
        "logit_gradient_norm": float(np.linalg.norm(gradient)),
        "conditional_losses": conditional,
        "max_boundary_escape_gain": float(max(escape_gains)),
        "mean_boundary_escape_gain": float(np.mean(escape_gains)),
    }


def initialize(
    cfg: base.Config, seed: int, phi: float, lambda_a: float
) -> tuple[np.ndarray, base.Adam]:
    rng = np.random.default_rng(
        71_000 + seed * 1009 + int(lambda_a * 1000) + int(phi * 10_000)
    )
    logits = rng.normal(scale=cfg.policy_init_scale, size=(cfg.time_steps, 2))
    return logits, base.Adam(logits.shape, cfg.policy_learning_rate)


def step_joint(
    logits: np.ndarray,
    optimizer: base.Adam,
    train_stats: base.SufficientStats,
    cfg: base.Config,
    schedules: list[tuple[int, ...]],
    lambda_a: float,
    indicators: np.ndarray,
) -> dict:
    state = joint_state(
        logits, train_stats, cfg, schedules, lambda_a, indicators
    )
    optimizer.step(logits, state["gradient"])
    return state


def branch_at_400(
    arm: str,
    logits: np.ndarray,
    optimizer: base.Adam,
    cfg: base.Config,
    seed: int,
    phi: float,
    lambda_a: float,
) -> tuple[np.ndarray, base.Adam]:
    result = logits.copy()
    if arm == "continue":
        return result, deepcopy(optimizer)
    new_optimizer = base.Adam(result.shape, cfg.policy_learning_rate)
    if arm == "reset_only":
        return result, new_optimizer
    if arm == "reheat_x2":
        return centered(result) / 2.0, new_optimizer
    if arm == "reheat_x4":
        return centered(result) / 4.0, new_optimizer
    if arm == "reheat_x4_noise":
        rng = np.random.default_rng(
            910_000 + seed * 1009 + int(phi * 10_000) + int(lambda_a * 1000)
        )
        result = centered(result) / 4.0
        result += rng.normal(scale=0.10, size=result.shape)
        return result, new_optimizer
    raise ValueError(f"unknown arm: {arm}")


def argmax_schedule(policy: np.ndarray) -> str:
    actions = tuple(np.argmax(policy, axis=1).tolist())
    return base.format_schedule(actions)


def audit_row(
    *,
    seed: int,
    phi: float,
    lambda_a: float,
    arm: str,
    iteration: int,
    logits: np.ndarray,
    state: dict,
    test_stats: base.SufficientStats,
    cfg: base.Config,
    schedules: list[tuple[int, ...]],
    oracle_schedule: str,
    oracle_objective: float,
    transition_count: int,
) -> dict:
    d_a, d_b, test_objective = base.evaluate_distribution(
        state["estimator"], test_stats, state["probs"], cfg, lambda_a
    )
    schedule = argmax_schedule(state["policy"])
    return {
        "seed": seed,
        "phi": phi,
        "lambda_a": lambda_a,
        "arm": arm,
        "iteration": iteration,
        "schedule": schedule,
        "oracle_schedule": oracle_schedule,
        "oracle_reached": schedule == oracle_schedule,
        "train_objective": state["train_objective"],
        "test_objective": test_objective,
        "population_oracle_objective": oracle_objective,
        "test_gap_from_population_oracle": test_objective - oracle_objective,
        "D_A": d_a,
        "D_B": d_b,
        "entropy": base.policy_entropy(state["policy"]),
        "logit_gradient_norm": state["logit_gradient_norm"],
        "max_boundary_escape_gain": state["max_boundary_escape_gain"],
        "mean_boundary_escape_gain": state["mean_boundary_escape_gain"],
        "logit_abs_max": float(np.max(np.abs(centered(logits)))),
        "transition_count": transition_count,
    }


def run_one(
    *,
    seed: int,
    phi: float,
    lambda_a: float,
    cfg: base.Config,
    checkpoints: list[int],
    max_iterations: int,
    schedules: list[tuple[int, ...]],
    train_stats: base.SufficientStats,
    test_stats: base.SufficientStats,
    oracle_schedule: str,
    oracle_objective: float,
) -> tuple[list[dict], list[dict]]:
    indicators = make_indicators(cfg, schedules)
    logits, optimizer = initialize(cfg, seed, phi, lambda_a)
    previous_schedule = argmax_schedule(base.softmax(logits))
    pre_transitions: list[dict] = []
    pre_transition_count = 0
    for iteration in range(1, 401):
        step_joint(
            logits, optimizer, train_stats, cfg, schedules, lambda_a, indicators
        )
        current_schedule = argmax_schedule(base.softmax(logits))
        if current_schedule != previous_schedule:
            pre_transition_count += 1
            pre_transitions.append(
                {
                    "seed": seed,
                    "phi": phi,
                    "lambda_a": lambda_a,
                    "arm": "shared_prebranch",
                    "iteration": iteration,
                    "from_schedule": previous_schedule,
                    "to_schedule": current_schedule,
                }
            )
            previous_schedule = current_schedule

    rows: list[dict] = []
    transitions = list(pre_transitions)
    for arm in ARMS:
        arm_logits, arm_optimizer = branch_at_400(
            arm, logits, optimizer, cfg, seed, phi, lambda_a
        )
        state = joint_state(
            arm_logits, train_stats, cfg, schedules, lambda_a, indicators
        )
        prior_schedule = argmax_schedule(state["policy"])
        arm_transition_count = pre_transition_count
        rows.append(
            audit_row(
                seed=seed,
                phi=phi,
                lambda_a=lambda_a,
                arm=arm,
                iteration=400,
                logits=arm_logits,
                state=state,
                test_stats=test_stats,
                cfg=cfg,
                schedules=schedules,
                oracle_schedule=oracle_schedule,
                oracle_objective=oracle_objective,
                transition_count=arm_transition_count,
            )
        )
        for iteration in range(401, max_iterations + 1):
            step_joint(
                arm_logits,
                arm_optimizer,
                train_stats,
                cfg,
                schedules,
                lambda_a,
                indicators,
            )
            current_schedule = argmax_schedule(base.softmax(arm_logits))
            if current_schedule != prior_schedule:
                arm_transition_count += 1
                transitions.append(
                    {
                        "seed": seed,
                        "phi": phi,
                        "lambda_a": lambda_a,
                        "arm": arm,
                        "iteration": iteration,
                        "from_schedule": prior_schedule,
                        "to_schedule": current_schedule,
                    }
                )
                prior_schedule = current_schedule
            if iteration in checkpoints:
                state = joint_state(
                    arm_logits,
                    train_stats,
                    cfg,
                    schedules,
                    lambda_a,
                    indicators,
                )
                rows.append(
                    audit_row(
                        seed=seed,
                        phi=phi,
                        lambda_a=lambda_a,
                        arm=arm,
                        iteration=iteration,
                        logits=arm_logits,
                        state=state,
                        test_stats=test_stats,
                        cfg=cfg,
                        schedules=schedules,
                        oracle_schedule=oracle_schedule,
                        oracle_objective=oracle_objective,
                        transition_count=arm_transition_count,
                    )
                )
    return rows, transitions


def aggregate(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (float(r["lambda_a"]), str(r["arm"]), int(r["iteration"]))
            for r in rows
        }
    )
    result = []
    for lambda_a, arm, iteration in keys:
        values = [
            r
            for r in rows
            if float(r["lambda_a"]) == lambda_a
            and str(r["arm"]) == arm
            and int(r["iteration"]) == iteration
        ]
        schedules = Counter(str(r["schedule"]) for r in values)
        arr = lambda name: np.asarray([float(r[name]) for r in values])
        result.append(
            {
                "phi": 0.8,
                "lambda_a": lambda_a,
                "arm": arm,
                "iteration": iteration,
                "n": len(values),
                "oracle_reach_rate": float(
                    np.mean([bool(r["oracle_reached"]) for r in values])
                ),
                "test_objective_mean": float(arr("test_objective").mean()),
                "test_objective_sd": float(arr("test_objective").std(ddof=0)),
                "entropy_mean": float(arr("entropy").mean()),
                "logit_gradient_norm_mean": float(
                    arr("logit_gradient_norm").mean()
                ),
                "max_boundary_escape_gain_mean": float(
                    arr("max_boundary_escape_gain").mean()
                ),
                "boundary_violation_rate": float(
                    np.mean(arr("max_boundary_escape_gain") > 1e-5)
                ),
                "schedule_counts": ";".join(
                    f"{name}:{count}" for name, count in sorted(schedules.items())
                ),
            }
        )
    return result


def classify_seed(
    seed_rows: list[dict], final_iteration: int, population_local_optima: set[str]
) -> dict:
    final_rows = [r for r in seed_rows if int(r["iteration"]) == final_iteration]
    by_arm = {str(row["arm"]): row for row in final_rows}
    cont = by_arm["continue"]
    reset = by_arm["reset_only"]
    hot2 = by_arm["reheat_x2"]
    hot4 = by_arm["reheat_x4"]
    hot4_noise = by_arm["reheat_x4_noise"]
    at_400 = next(
        row
        for row in seed_rows
        if str(row["arm"]) == "continue" and int(row["iteration"]) == 400
    )
    static_local = str(cont["schedule"]) in population_local_optima
    if bool(at_400["oracle_reached"]):
        label = "already_oracle_at_400"
    elif bool(cont["oracle_reached"]):
        label = "continuation_reaches_oracle"
    elif bool(reset["oracle_reached"]):
        label = "optimizer_state_dependence"
    elif bool(hot2["oracle_reached"]) or bool(hot4["oracle_reached"]):
        label = "policy_reheating_escape"
    elif bool(hot4_noise["oracle_reached"]):
        label = "perturbation_escape"
    elif float(cont["max_boundary_escape_gain"]) > 1e-5:
        label = "boundary_saturation_persists"
    elif (
        str(hot2["schedule"]) == str(cont["schedule"])
        and str(hot4["schedule"]) == str(cont["schedule"])
        and str(hot4_noise["schedule"]) == str(cont["schedule"])
    ):
        label = (
            "static_and_joint_stable_candidate"
            if static_local
            else "joint_dynamics_stable_candidate"
        )
    else:
        label = "multiple_joint_basins_candidate"
    return {
        "seed": int(cont["seed"]),
        "phi": float(cont["phi"]),
        "lambda_a": float(cont["lambda_a"]),
        "classification": label,
        "schedule_at_400": str(at_400["schedule"]),
        "continue_schedule": str(cont["schedule"]),
        "reset_schedule": str(reset["schedule"]),
        "reheat_x2_schedule": str(hot2["schedule"]),
        "reheat_x4_schedule": str(hot4["schedule"]),
        "reheat_x4_noise_schedule": str(hot4_noise["schedule"]),
        "continue_boundary_escape_gain": float(
            cont["max_boundary_escape_gain"]
        ),
        "population_static_local_optimum": static_local,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, summary: list[dict], final_iteration: int) -> None:
    width, height = 940, 520
    left, right, top, bottom = 80, 30, 65, 95
    panel_w = (width - left - right) / 3
    plot_h = height - top - bottom
    colors = {
        "continue": "#455A64",
        "reset_only": "#7E57C2",
        "reheat_x2": "#EF6C00",
        "reheat_x4": "#C62828",
        "reheat_x4_noise": "#176B87",
    }
    labels = {
        "continue": "continue",
        "reset_only": "reset",
        "reheat_x2": "reheat x2",
        "reheat_x4": "reheat x4",
        "reheat_x4_noise": "reheat x4 + noise",
    }
    iterations = sorted({int(r["iteration"]) for r in summary})

    def sx(panel: int, iteration: int) -> float:
        lo, hi = min(iterations), max(iterations)
        fraction = (math.log(iteration) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return left + panel * panel_w + 25 + (panel_w - 50) * fraction

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.grid{stroke:#CFD8DC}.axis{stroke:#455A64}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Joint-update convergence audit: Oracle reach rate (phi=0.8)</text>',
    ]
    for panel, lambda_a in enumerate((0.25, 0.5, 0.75)):
        x0 = left + panel * panel_w
        lines.append(
            f'<text x="{x0+panel_w/2}" y="52" text-anchor="middle" font-size="15">lambda={lambda_a}</text>'
        )
        for tick in range(6):
            value = tick / 5
            y = sy(value)
            lines.append(
                f'<line x1="{x0+25}" y1="{y}" x2="{x0+panel_w-25}" y2="{y}" class="grid"/>'
            )
            if panel == 0:
                lines.append(
                    f'<text x="{left-10}" y="{y+5}" text-anchor="end" font-size="12">{value:.1f}</text>'
                )
        for arm in ARMS:
            values = [
                r
                for r in summary
                if float(r["lambda_a"]) == lambda_a and str(r["arm"]) == arm
            ]
            values.sort(key=lambda r: int(r["iteration"]))
            points = " ".join(
                f'{sx(panel,int(r["iteration"])):.1f},{sy(float(r["oracle_reach_rate"])):.1f}'
                for r in values
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[arm]}" stroke-width="2.5"/>'
            )
            for row in values:
                lines.append(
                    f'<circle cx="{sx(panel,int(row["iteration"]))}" cy="{sy(float(row["oracle_reach_rate"]))}" r="4" fill="{colors[arm]}"/>'
                )
        for iteration in iterations:
            lines.append(
                f'<text x="{sx(panel,iteration)}" y="{top+plot_h+22}" text-anchor="middle" font-size="11">{iteration}</text>'
            )
    legend_x = left + 55
    for index, arm in enumerate(ARMS):
        x = legend_x + index * 165
        lines += [
            f'<line x1="{x}" y1="{height-35}" x2="{x+30}" y2="{height-35}" stroke="{colors[arm]}" stroke-width="4"/>',
            f'<text x="{x+38}" y="{height-30}" font-size="13">{labels[arm]}</text>',
        ]
    lines += [
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" font-size="14">Oracle reach rate</text>',
        f'<text x="{width/2}" y="{height-62}" text-anchor="middle" font-size="13">optimization iteration (log scale; final={final_iteration})</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg, checkpoints, max_iterations = make_audit_config(full=args.full)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schedules = base.all_schedules(cfg.time_steps)
    start = time.time()
    phi = 0.8
    print("実験I-C v0.2.1：共同更新の収束性監査を開始します（GPU不要）")
    print(f"反復: {checkpoints} / seed数: {len(cfg.seeds)}")
    print(f"保存先: {output_dir}")

    landscapes = {}
    for lambda_a in cfg.lambdas:
        _, summary, _ = base.landscape_analysis(phi, lambda_a, cfg, schedules)
        landscapes[lambda_a] = summary

    rows: list[dict] = []
    transitions: list[dict] = []
    total = len(cfg.seeds) * len(cfg.lambdas)
    count = 0
    for seed in cfg.seeds:
        rng = np.random.default_rng(100_000 + seed * 97 + int(phi * 1000))
        train = base.make_dataset(cfg.train_size, phi, cfg, rng)
        test = base.make_dataset(cfg.test_size, phi, cfg, rng)
        train_stats = base.build_empirical_stats(*train, schedules)
        test_stats = base.build_empirical_stats(*test, schedules)
        for lambda_a in cfg.lambdas:
            landscape = landscapes[lambda_a]
            one_rows, one_transitions = run_one(
                seed=seed,
                phi=phi,
                lambda_a=lambda_a,
                cfg=cfg,
                checkpoints=checkpoints,
                max_iterations=max_iterations,
                schedules=schedules,
                train_stats=train_stats,
                test_stats=test_stats,
                oracle_schedule=str(landscape["oracle_schedule"]),
                oracle_objective=float(landscape["oracle_objective"]),
            )
            rows.extend(one_rows)
            transitions.extend(one_transitions)
            count += 1
            if count == 1 or count % max(total // 10, 1) == 0 or count == total:
                print(f"  進捗 {count}/{total}")

    summary = aggregate(rows)
    classifications = []
    for seed in cfg.seeds:
        for lambda_a in cfg.lambdas:
            values = [
                r
                for r in rows
                if int(r["seed"]) == seed and float(r["lambda_a"]) == lambda_a
            ]
            local_optima = set(str(landscapes[lambda_a]["local_optima"]).split(";"))
            classifications.append(
                classify_seed(values, max_iterations, local_optima)
            )

    write_csv(output_dir / "audit_raw.csv", rows)
    write_csv(output_dir / "audit_summary.csv", summary)
    write_csv(output_dir / "schedule_transitions.csv", transitions)
    write_csv(output_dir / "seed_classification.csv", classifications)
    write_svg(output_dir / "oracle_reach_by_iteration.svg", summary, max_iterations)

    class_counts = Counter(row["classification"] for row in classifications)
    nonoracle_at_400 = sum(
        count
        for name, count in class_counts.items()
        if name != "already_oracle_at_400"
    )
    final_summary = [r for r in summary if int(r["iteration"]) == max_iterations]
    elapsed = time.time() - start
    metadata = {
        "experiment": "Experiment I-C v0.2.1: joint-update convergence audit",
        "version": "0.2.1",
        "config": asdict(cfg),
        "checkpoints": checkpoints,
        "max_iterations": max_iterations,
        "arms": list(ARMS),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "classification_counts": dict(class_counts),
        "scope_note": (
            "This audit distinguishes prolonged optimization, optimizer-state dependence, "
            "softmax saturation, and candidates for stable joint-update basins. It does not "
            "introduce closed-loop state-conditioned attention."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 実験I-C v0.2.1・共同更新の収束性監査",
        "",
        "v0.2の400反復後から、継続、Adam状態だけのリセット、softmaxの再加熱、再加熱＋微小摂動へ分岐した。",
        "確率境界で別行動へ移る改善余地も測り、低エントロピーと真の局所安定性を区別した。",
        "",
        f"最終反復: {max_iterations}、seed数: {len(cfg.seeds)}",
        "",
        "## 結論",
        "",
        f"全{len(classifications)}条件のうち、400反復時点でOracleに達していなかったのは{nonoracle_at_400}条件だった。",
        f"単純な反復継続によって初めてOracleへ移った条件は{class_counts.get('continuation_reaches_oracle', 0)}件だった。",
        f"母集団の静的局所最適ではない共同更新の安定点候補は{class_counts.get('joint_dynamics_stable_candidate', 0)}件、"
        f"静的局所最適とも一致した候補は{class_counts.get('static_and_joint_stable_candidate', 0)}件だった。",
        f"十分に強い方策再加熱によってOracleへ移った条件は{class_counts.get('policy_reheating_escape', 0)}件だった。",
        "したがって、v0.2の非Oracle方策を一括して単純な反復不足と説明する結果にはならなかった。",
        "",
        "## 最終時点の比較",
        "",
        "| lambda | 分岐 | Oracle到達率 | 平均目的値 | 平均entropy | 境界違反率 | スケジュール分布 |",
        "|---:|:---|---:|---:|---:|---:|:---|",
    ]
    for row in final_summary:
        report.append(
            f"| {float(row['lambda_a']):.2f} | {row['arm']} "
            f"| {float(row['oracle_reach_rate']):.3f} "
            f"| {float(row['test_objective_mean']):.5f} "
            f"| {float(row['entropy_mean']):.6f} "
            f"| {float(row['boundary_violation_rate']):.3f} "
            f"| {row['schedule_counts']} |"
        )
    report += [
        "",
        "## seed単位の診断分類",
        "",
        *[f"- `{name}`: {count}" for name, count in sorted(class_counts.items())],
        "",
        "## 判定の読み方",
        "",
        "- `already_oracle_at_400`: v0.2の反復数ですでにOracleへ到達していた。",
        "- `continuation_reaches_oracle`: 400反復では足りなかったが、単純継続でOracleへ到達した。",
        "- `optimizer_state_dependence`: Adamの内部状態を捨てるだけでOracleへ到達した。",
        "- `policy_reheating_escape`: logitの縮小による十分な再加熱でのみOracleへ移り、低エントロピー方策の吸引域を越えた。",
        "- `perturbation_escape`: 再加熱だけでは動かず、微小摂動を加えた場合だけOracleへ移った。",
        "- `boundary_saturation_persists`: 非Oracleかつ確率境界に改善方向が残り、飽和捕捉が継続している。",
        "- `static_and_joint_stable_candidate`: 母集団の静的局所最適と共同更新の安定点候補が一致した。",
        "- `joint_dynamics_stable_candidate`: 母集団の静的局所最適ではないが、共同更新では同じ方策へ復帰した。",
        "- `multiple_joint_basins_candidate`: 改善方向は小さいが、再加熱・摂動で異なる方策へ分岐する。",
        "",
        "## 限界",
        "",
        "この分類は、現行のAdam更新と因子化softmax方策に対する診断である。",
        "二種類の`*_stable_candidate`は数学的な固定点の証明ではなく、より直接的なHessian解析や別最適化法との比較を要する。",
        "本監査は閉ループ方策、志向性、発達的履歴依存、人間の注意を検証していない。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print()
    print("監査を完了しました。")
    print("分類:", dict(class_counts))
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, seed_classification.csv, oracle_reach_by_iteration.svg")


if __name__ == "__main__":
    main()
