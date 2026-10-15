#!/usr/bin/env python3
"""実験I-C v0.2.2：共有推定器が作る方策遷移障壁の監査。

Hamming距離1の全方策対について、悪い決定的方策から良い方策へ
一つの観測確率を連続的に動かす。混合方策ごとに単一推定器を
当て直す条件と、方策ごとに専用推定器を持つ条件を比較する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

import temporal_coupling_poc as base


EXEMPLARS = (
    (0.8, 0.5, "ABBB", "AABB"),
    (0.8, 0.5, "ABAB", "AAAB"),
    (0.8, 0.5, "AABA", "AABB"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="共有推定器の混合障壁を測ります。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="alphaを101点で走査（既定）")
    mode.add_argument("--full", action="store_true", help="alphaを1001点で走査")
    parser.add_argument("--output-dir", default="shared_estimator_barrier_results")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_solution(
    index: int,
    stats: base.SufficientStats,
    cfg: base.Config,
    lambda_a: float,
    n_schedules: int,
) -> tuple[np.ndarray, float, float, float]:
    probs = base.deterministic_probs(index, n_schedules)
    estimator = base.fit_estimator(stats, probs, cfg.ridge)
    d_a, d_b, objective = base.evaluate_distribution(
        estimator, stats, probs, cfg, lambda_a
    )
    return estimator, d_a, d_b, objective


def path_profile(
    source: int,
    target: int,
    stats: base.SufficientStats,
    cfg: base.Config,
    schedules: list[tuple[int, ...]],
    lambda_a: float,
    alphas: np.ndarray,
) -> tuple[dict, list[dict]]:
    names = [base.format_schedule(s) for s in schedules]
    n_schedules = len(schedules)
    source_est, _, _, source_obj = deterministic_solution(
        source, stats, cfg, lambda_a, n_schedules
    )
    target_est, _, _, target_obj = deterministic_solution(
        target, stats, cfg, lambda_a, n_schedules
    )
    shared_values = np.empty(len(alphas))
    dedicated_values = (1.0 - alphas) * source_obj + alphas * target_obj
    d_as = np.empty(len(alphas))
    d_bs = np.empty(len(alphas))
    for k, alpha in enumerate(alphas):
        probs = np.zeros(n_schedules)
        probs[source] = 1.0 - alpha
        probs[target] = alpha
        estimator = base.fit_estimator(stats, probs, cfg.ridge)
        d_a, d_b, objective = base.evaluate_distribution(
            estimator, stats, probs, cfg, lambda_a
        )
        d_as[k], d_bs[k], shared_values[k] = d_a, d_b, objective

    mixing_penalty = shared_values - dedicated_values
    shared_peak = int(np.argmax(shared_values))
    penalty_peak = int(np.argmax(mixing_penalty))
    barrier = max(float(shared_values[shared_peak] - source_obj), 0.0)

    # source用推定器を固定したままtarget観測へ切り替える費用。
    # 混合ごとに推定器を再最適化したJ_sharedの方向微分ではない。
    target_probs = base.deterministic_probs(target, n_schedules)
    target_loss_under_source = base.evaluate_distribution(
        source_est, stats, target_probs, cfg, lambda_a
    )[2]
    frozen_estimator_switch_cost = target_loss_under_source - source_obj

    summary = {
        "source": names[source],
        "target": names[target],
        "source_objective": source_obj,
        "target_objective": target_obj,
        "target_improvement": source_obj - target_obj,
        "shared_barrier_height": barrier,
        "shared_peak_alpha": float(alphas[shared_peak]),
        "shared_peak_objective": float(shared_values[shared_peak]),
        "max_mixing_penalty": float(np.max(mixing_penalty)),
        "mixing_penalty_peak_alpha": float(alphas[penalty_peak]),
        "min_mixing_penalty": float(np.min(mixing_penalty)),
        "frozen_estimator_switch_cost": float(frozen_estimator_switch_cost),
        "shared_barrier_exists": barrier > 1e-8,
        "dedicated_barrier_height": 0.0,
    }
    profile = [
        {
            "source": names[source],
            "target": names[target],
            "alpha": float(alpha),
            "D_A": float(d_a),
            "D_B": float(d_b),
            "shared_objective": float(shared),
            "dedicated_objective": float(dedicated),
            "mixing_penalty": float(penalty),
        }
        for alpha, d_a, d_b, shared, dedicated, penalty in zip(
            alphas, d_as, d_bs, shared_values, dedicated_values, mixing_penalty
        )
    ]
    return summary, profile


def analyze_condition(
    phi: float,
    lambda_a: float,
    cfg: base.Config,
    schedules: list[tuple[int, ...]],
    alphas: np.ndarray,
) -> tuple[list[dict], dict, dict[tuple[str, str], list[dict]]]:
    stats = base.build_population_stats(phi, cfg, schedules)
    names = [base.format_schedule(s) for s in schedules]
    objectives = []
    for index in range(len(schedules)):
        objectives.append(
            deterministic_solution(
                index, stats, cfg, lambda_a, len(schedules)
            )[3]
        )

    rows = []
    profiles: dict[tuple[str, str], list[dict]] = {}
    for source in range(len(schedules)):
        for target in base.hamming_neighbors(source, schedules):
            # 各無向辺を、悪い端点から良い端点への一方向だけで数える。
            if objectives[target] >= objectives[source] - 1e-12:
                continue
            row, profile = path_profile(
                source,
                target,
                stats,
                cfg,
                schedules,
                lambda_a,
                alphas,
            )
            row = {"phi": phi, "lambda_a": lambda_a, **row}
            rows.append(row)
            profiles[(names[source], names[target])] = profile

    barriers = np.asarray([float(row["shared_barrier_height"]) for row in rows])
    penalties = np.asarray([float(row["max_mixing_penalty"]) for row in rows])
    frozen_switch_costs = np.asarray(
        [float(row["frozen_estimator_switch_cost"]) for row in rows]
    )
    summary = {
        "phi": phi,
        "lambda_a": lambda_a,
        "improving_hamming1_edges": len(rows),
        "shared_barrier_edges": int(np.sum(barriers > 1e-8)),
        "shared_barrier_fraction": float(np.mean(barriers > 1e-8)),
        "mean_shared_barrier": float(np.mean(barriers)),
        "max_shared_barrier": float(np.max(barriers)),
        "mean_max_mixing_penalty": float(np.mean(penalties)),
        "max_mixing_penalty": float(np.max(penalties)),
        "min_mixing_penalty_across_edges": float(
            min(float(row["min_mixing_penalty"]) for row in rows)
        ),
        "positive_frozen_switch_cost_edges": int(
            np.sum(frozen_switch_costs > 1e-8)
        ),
        "dedicated_barrier_edges": 0,
    }
    return rows, summary, profiles


def write_profile_svg(path: Path, exemplar_rows: list[dict]) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 75, 30, 65, 80
    panel_w = (width - left - right) / 3
    plot_h = height - top - bottom
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in exemplar_rows:
        groups[(str(row["source"]), str(row["target"]))].append(row)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.grid{stroke:#CFD8DC}.axis{stroke:#455A64}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Shared estimator creates a barrier above schedule-specific experts</text>',
    ]
    for panel, key in enumerate(groups):
        rows = sorted(groups[key], key=lambda r: float(r["alpha"]))
        all_values = [
            float(r[field])
            for r in rows
            for field in ("shared_objective", "dedicated_objective")
        ]
        ymin, ymax = min(all_values), max(all_values)
        pad = max((ymax - ymin) * 0.15, 0.002)
        ymin -= pad
        ymax += pad
        x0 = left + panel * panel_w

        def sx(alpha: float) -> float:
            return x0 + 25 + (panel_w - 50) * alpha

        def sy(value: float) -> float:
            return top + plot_h * (1.0 - (value - ymin) / (ymax - ymin))

        lines.append(
            f'<text x="{x0+panel_w/2}" y="52" text-anchor="middle" font-size="15">{key[0]} → {key[1]}</text>'
        )
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = sy(value)
            lines.append(
                f'<line x1="{x0+25}" y1="{y}" x2="{x0+panel_w-25}" y2="{y}" class="grid"/>'
            )
            lines.append(
                f'<text x="{x0+20}" y="{y-4}" text-anchor="end" font-size="10">{value:.3f}</text>'
            )
        for field, color in (
            ("shared_objective", "#C62828"),
            ("dedicated_objective", "#176B87"),
        ):
            points = " ".join(
                f'{sx(float(r["alpha"])):.1f},{sy(float(r[field])):.1f}'
                for r in rows
            )
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
            )
        for alpha in (0.0, 0.5, 1.0):
            lines.append(
                f'<text x="{sx(alpha)}" y="{top+plot_h+23}" text-anchor="middle" font-size="12">{alpha:.1f}</text>'
            )
    lines += [
        f'<line x1="{left+220}" y1="{height-30}" x2="{left+255}" y2="{height-30}" stroke="#C62828" stroke-width="4"/>',
        f'<text x="{left+265}" y="{height-25}" font-size="13">one shared estimator</text>',
        f'<line x1="{left+510}" y1="{height-30}" x2="{left+545}" y2="{height-30}" stroke="#176B87" stroke-width="4"/>',
        f'<text x="{left+555}" y="{height-25}" font-size="13">schedule-specific estimators</text>',
        f'<text x="{width/2}" y="{height-53}" text-anchor="middle" font-size="13">target-policy mixture α</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prevalence_svg(path: Path, summaries: list[dict]) -> None:
    width, height = 850, 500
    left, right, top, bottom = 80, 35, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    phis = [0.0, 0.4, 0.8]
    lambdas = [0.25, 0.5, 0.75]
    colors = {0.25: "#7E57C2", 0.5: "#176B87", 0.75: "#EF6C00"}

    def sx(i: int) -> float:
        return left + plot_w * i / 2

    def sy(v: float) -> float:
        return top + plot_h * (1.0 - v)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.grid{stroke:#CFD8DC}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Fraction of improving Hamming-1 moves blocked by a shared-estimator barrier</text>',
    ]
    for tick in range(6):
        v = tick / 5
        y = sy(v)
        lines += [
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="12">{v:.1f}</text>',
        ]
    for lam in lambdas:
        values = [
            next(
                float(r["shared_barrier_fraction"])
                for r in summaries
                if float(r["phi"]) == phi and float(r["lambda_a"]) == lam
            )
            for phi in phis
        ]
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[lam]}" stroke-width="3"/>'
        )
        for i, value in enumerate(values):
            lines.append(
                f'<circle cx="{sx(i)}" cy="{sy(value)}" r="6" fill="{colors[lam]}"/>'
            )
    for i, phi in enumerate(phis):
        lines.append(
            f'<text x="{sx(i)}" y="{top+plot_h+25}" text-anchor="middle" font-size="14">phi={phi}</text>'
        )
    for i, lam in enumerate(lambdas):
        x = left + 190 + i * 150
        lines += [
            f'<line x1="{x}" y1="{height-30}" x2="{x+30}" y2="{height-30}" stroke="{colors[lam]}" stroke-width="4"/>',
            f'<text x="{x+38}" y="{height-25}" font-size="13">lambda={lam}</text>',
        ]
    lines += [
        f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+plot_h/2})" font-size="14">barrier fraction</text>',
        '</svg>',
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    alpha_points = 1001 if args.full else 101
    alphas = np.linspace(0.0, 1.0, alpha_points)
    cfg = base.make_config(full=True)
    schedules = base.all_schedules(cfg.time_steps)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("実験I-C v0.2.2：共有推定器の方策遷移障壁を測定します（GPU不要）")
    print(f"alpha点数: {alpha_points}")
    print(f"保存先: {output_dir}")

    edge_rows: list[dict] = []
    summaries: list[dict] = []
    exemplar_rows: list[dict] = []
    exemplar_summaries: list[dict] = []
    total = len(cfg.phis) * len(cfg.lambdas)
    count = 0
    for phi in cfg.phis:
        for lambda_a in cfg.lambdas:
            rows, summary, profiles = analyze_condition(
                phi, lambda_a, cfg, schedules, alphas
            )
            edge_rows.extend(rows)
            summaries.append(summary)
            for e_phi, e_lambda, source, target in EXEMPLARS:
                if phi == e_phi and lambda_a == e_lambda:
                    names = [base.format_schedule(s) for s in schedules]
                    stats = base.build_population_stats(phi, cfg, schedules)
                    exemplar_summary, exemplar_profile = path_profile(
                        names.index(source),
                        names.index(target),
                        stats,
                        cfg,
                        schedules,
                        lambda_a,
                        alphas,
                    )
                    exemplar_summaries.append(
                        {"phi": phi, "lambda_a": lambda_a, **exemplar_summary}
                    )
                    for row in exemplar_profile:
                        exemplar_rows.append(
                            {"phi": phi, "lambda_a": lambda_a, **row}
                        )
            count += 1
            print(f"  進捗 {count}/{total}")

    write_csv(output_dir / "edge_barriers.csv", edge_rows)
    write_csv(output_dir / "barrier_summary.csv", summaries)
    write_csv(output_dir / "exemplar_paths.csv", exemplar_rows)
    write_profile_svg(output_dir / "exemplar_barrier_profiles.svg", exemplar_rows)
    write_prevalence_svg(output_dir / "barrier_prevalence.svg", summaries)

    s00 = next(r for r in summaries if r["phi"] == 0.0 and r["lambda_a"] == 0.5)
    s08 = next(r for r in summaries if r["phi"] == 0.8 and r["lambda_a"] == 0.5)
    criteria = {
        "phi0_shared_penalty_near_zero": bool(
            max(float(r["max_mixing_penalty"]) for r in summaries if r["phi"] == 0.0)
            < 1e-8
        ),
        "phi08_barriers_present": bool(int(s08["shared_barrier_edges"]) >= 12),
        "phi08_shared_penalty_substantial": bool(
            float(s08["max_mixing_penalty"]) > 0.02
        ),
        "dedicated_experts_have_no_barrier": bool(
            all(int(r["dedicated_barrier_edges"]) == 0 for r in summaries)
        ),
        "mixing_penalty_nonnegative_with_tolerance": bool(
            min(float(r["min_mixing_penalty_across_edges"]) for r in summaries)
            > -1e-8
        ),
    }
    criteria["passed"] = all(criteria.values())
    elapsed = time.time() - start
    metadata = {
        "experiment": "Experiment I-C v0.2.2: shared-estimator transition barriers",
        "version": "0.2.2",
        "config": asdict(cfg),
        "alpha_points": alpha_points,
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "criteria": criteria,
        "scope_note": (
            "The barrier is a property of fitting one estimator to a mixture of observation "
            "schedules. Schedule-specific estimators remove this mixing barrier."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 実験I-C v0.2.2・共有推定器が作る方策遷移障壁",
        "",
        (
            "校正成功。" if criteria["passed"] else "校正条件の一部が未達。"
        )
        + " 時間相関がある条件では、より良い隣接方策へ移る経路に共有推定器由来の障壁が生じた。",
        "",
        "## 機構",
        "",
        "二つの決定的方策をs,t、target方策の混合率をalphaとする。共有条件では、混合分布全体に一つの推定器Wを当てる。",
        "",
        "`J_shared(alpha) = min_W [(1-alpha)L_s(W) + alpha L_t(W)]`",
        "",
        "専用条件では各方策が別の最適推定器を持つ。",
        "",
        "`J_dedicated(alpha) = (1-alpha)min_W L_s(W) + alpha min_W L_t(W)`",
        "",
        "一般に `J_shared(alpha) >= J_dedicated(alpha)` であり、差が混合ペナルティになる。target端点がより良くても、この差が十分に大きければ途中に山ができる。",
        "",
        "## 全Hamming-1遷移の集計",
        "",
        "| phi | lambda | 改善辺数 | 障壁辺数 | 障壁率 | 最大障壁 | 最大混合ペナルティ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {float(row['phi']):.1f} | {float(row['lambda_a']):.2f} "
            f"| {int(row['improving_hamming1_edges'])} "
            f"| {int(row['shared_barrier_edges'])} "
            f"| {float(row['shared_barrier_fraction']):.3f} "
            f"| {float(row['max_shared_barrier']):.6f} "
            f"| {float(row['max_mixing_penalty']):.6f} |"
        )
    report += [
        "",
        "## v0.2.1で注目された三経路",
        "",
        "| 遷移 | 出発目的値 | 到着目的値 | 障壁 | 最大混合ペナルティ | 山のalpha |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in exemplar_summaries:
        report.append(
            f"| {row['source']} → {row['target']} "
            f"| {float(row['source_objective']):.6f} "
            f"| {float(row['target_objective']):.6f} "
            f"| {float(row['shared_barrier_height']):.6f} "
            f"| {float(row['max_mixing_penalty']):.6f} "
            f"| {float(row['shared_peak_alpha']):.3f} |"
        )
    report += [
        "",
        "ABAB → AAABは到着点の方が悪い。この経路は、より良い隣接方策への移動阻害ではなく、静的局所最適ABABから一ビット離れる際の離脱費用を示す。ほかの二経路は到着点の方が良く、共有推定器によって改善経路の途中に山が生じる例である。",
        "",
        "## 結論",
        "",
        f"phi=0, lambda=0.5では障壁辺は{s00['shared_barrier_edges']}/32だった。",
        f"phi=0.8, lambda=0.5では障壁辺は{s08['shared_barrier_edges']}/32だった。",
        "したがって障壁は、有限容量だけで自動的に現れたのではなく、時間的に相関した観測のもとで、一つの推定器が複数の観測分布へ同時適応することから生じた。",
        "方策ごとに専用推定器を置く対照では経路は端点間の直線となり、障壁は0だった。",
        "phi=0で混合ペナルティが数値精度内の0になることは、観測パターン間の結合が実際に働いているかを調べる対照指標として利用できる。",
        "",
        "## 理論との対応範囲",
        "",
        "本実験が分離したのは、現在の観測方策に推定器が適応することで、その方策からの離脱に局所的な費用が生じる成分である。",
        "推定器は各混合率で最適に当て直されており、過去の観測履歴を重みに蓄積していない。したがって、観測頻度の反復が表現解像度を履歴依存的に固定する不可逆性は、まだ検証していない。",
        "",
        "## 限界",
        "",
        "この結果は単一の線形共有推定器を仮定した構成的事例である。共有された内部表現一般や、人間の認知に同じ障壁が存在することを直接示さない。",
        "専用推定器対照はMixture-of-Expertsに近い極端な上限であり、部分共有モデルでは中間的な障壁が予想されるが、本実験では未検証である。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print()
    print("校正成功" if criteria["passed"] else "校正条件の一部が未達")
    print(
        f"phi=0.8, lambda=0.5: barrier {s08['shared_barrier_edges']}/"
        f"{s08['improving_hamming1_edges']} edges"
    )
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, edge_barriers.csv, exemplar_barrier_profiles.svg")


if __name__ == "__main__":
    main()
