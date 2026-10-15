#!/usr/bin/env python3
"""
実験I-B v0.2：観測予算 B_A と表現容量 C_R の分離PoC。

観測予算は「利用できるセンサーチャネル数」、表現容量は
「取得後の線形状態表現の次元」として独立に操作する。
外部依存は NumPy だけで、GPU は不要。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    train_size: int
    test_size: int
    source_dim: int
    correlation: float
    observation_noise: float
    observation_budgets: list[int]
    representation_capacities: list[int]
    lambdas: list[float]
    seeds: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="観測予算 B_A と表現容量 C_R を独立に操作します。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="短時間設定（既定）")
    mode.add_argument("--full", action="store_true", help="seedと標本数を増やす")
    parser.add_argument("--output-dir", default="observation_budget_results")
    return parser.parse_args()


def make_config(full: bool) -> Config:
    if full:
        return Config(
            train_size=8192,
            test_size=4096,
            source_dim=3,
            correlation=0.35,
            observation_noise=0.02,
            observation_budgets=[2, 3, 4, 6],
            representation_capacities=[1, 2, 3, 4, 5, 6],
            lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
            seeds=list(range(20)),
        )
    return Config(
        train_size=2048,
        test_size=1024,
        source_dim=3,
        correlation=0.35,
        observation_noise=0.02,
        observation_budgets=[2, 3, 4, 6],
        representation_capacities=[1, 2, 3, 4, 5, 6],
        lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
        seeds=list(range(5)),
    )


def make_dataset(
    n: int, cfg: Config, rng: np.random.Generator, mixing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = rng.normal(size=(n, cfg.source_dim))
    private_b = rng.normal(size=(n, cfg.source_dim))
    rho = cfg.correlation
    b = rho * a + math.sqrt(1.0 - rho**2) * private_b
    sources = np.concatenate([a, b], axis=1)
    x = sources @ mixing
    x += cfg.observation_noise * rng.normal(size=x.shape)
    return x.astype(np.float64), a.astype(np.float64), b.astype(np.float64)


def standardize_triplet(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + 1e-10
    return (train - mean) / scale, (test - mean) / scale


def optimal_readout(
    z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray
) -> np.ndarray:
    train_aug = np.concatenate([z_train, np.ones((len(z_train), 1))], axis=1)
    test_aug = np.concatenate([z_test, np.ones((len(z_test), 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(train_aug, y_train, rcond=None)
    return test_aug @ coefficients


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2) / (np.var(target) + 1e-12))


def population_covariances(
    cfg: Config, mixing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成過程から Cov(X), Cov(A,X), Cov(B,X) を正確に計算する。"""
    d = cfg.source_dim
    eye = np.eye(d)
    sigma_sources = np.block(
        [[eye, cfg.correlation * eye], [cfg.correlation * eye, eye]]
    )
    sigma_x = (
        mixing.T @ sigma_sources @ mixing
        + cfg.observation_noise**2 * np.eye(2 * d)
    )
    cov_a_sources = np.concatenate([eye, cfg.correlation * eye], axis=1)
    cov_b_sources = np.concatenate([cfg.correlation * eye, eye], axis=1)
    return sigma_x, cov_a_sources @ mixing, cov_b_sources @ mixing


def population_observation_distortion(
    cfg: Config, mixing: np.ndarray, mask: tuple[int, ...]
) -> tuple[float, float]:
    """選択した観測だけからの最適線形推定誤差（母集団値）。"""
    sigma_x, cov_ax, cov_bx = population_covariances(cfg, mixing)
    index = np.array(mask, dtype=int)
    sigma_obs = sigma_x[np.ix_(index, index)]
    inverse = np.linalg.pinv(sigma_obs)
    cov_a_obs = cov_ax[:, index]
    cov_b_obs = cov_bx[:, index]
    residual_a = np.eye(cfg.source_dim) - cov_a_obs @ inverse @ cov_a_obs.T
    residual_b = np.eye(cfg.source_dim) - cov_b_obs @ inverse @ cov_b_obs.T
    return (
        float(np.trace(residual_a) / cfg.source_dim),
        float(np.trace(residual_b) / cfg.source_dim),
    )


def select_oracle_mask(
    cfg: Config, mixing: np.ndarray, budget: int, lambda_a: float
) -> tuple[tuple[int, ...], float, float]:
    """下流圧縮がない場合に最良な観測チャネル集合を全探索する。"""
    input_dim = cfg.source_dim * 2
    candidates = []
    for mask in itertools.combinations(range(input_dim), budget):
        d_a, d_b = population_observation_distortion(cfg, mixing, mask)
        objective = lambda_a * d_a + (1.0 - lambda_a) * d_b
        candidates.append((objective, mask, d_a, d_b))
    _, mask, d_a, d_b = min(candidates, key=lambda row: (row[0], row[1]))
    return mask, d_a, d_b


def fit_reduced_rank_representation(
    x_train: np.ndarray,
    a_train: np.ndarray,
    b_train: np.ndarray,
    capacity: int,
    lambda_a: float,
) -> np.ndarray | None:
    """
    重み付き二課題に対する最適な経験的線形部分空間を求める。

    capacity >= 観測次元なら None を返し、観測値をそのまま状態表現とする。
    これにより「観測後に圧縮しない」条件を明示的に作る。
    """
    observed_dim = x_train.shape[1]
    if capacity >= observed_dim:
        return None
    weighted_targets = np.concatenate(
        [math.sqrt(lambda_a) * a_train, math.sqrt(1.0 - lambda_a) * b_train],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(x_train, weighted_targets, rcond=None)
    fitted = x_train @ coefficients
    _, singular_values, vt = np.linalg.svd(fitted, full_matrices=False)
    rank = min(capacity, int(np.sum(singular_values > 1e-10)))
    if rank == 0:
        return np.zeros((observed_dim, 1))
    target_directions = vt[:rank].T
    return coefficients @ target_directions


def evaluate_condition(
    x_train: np.ndarray,
    a_train: np.ndarray,
    b_train: np.ndarray,
    x_test: np.ndarray,
    a_test: np.ndarray,
    b_test: np.ndarray,
    mask: tuple[int, ...],
    capacity: int,
    lambda_a: float,
) -> tuple[float, float, int]:
    x_train_obs = x_train[:, mask]
    x_test_obs = x_test[:, mask]
    encoder = fit_reduced_rank_representation(
        x_train_obs, a_train, b_train, capacity, lambda_a
    )
    if encoder is None:
        z_train, z_test = x_train_obs, x_test_obs
        effective_capacity = x_train_obs.shape[1]
    else:
        z_train, z_test = x_train_obs @ encoder, x_test_obs @ encoder
        effective_capacity = encoder.shape[1]
    pred_a = optimal_readout(z_train, a_train, z_test)
    pred_b = optimal_readout(z_train, b_train, z_test)
    return (
        normalized_mse(pred_a, a_test),
        normalized_mse(pred_b, b_test),
        effective_capacity,
    )


def aggregate(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (int(r["observation_budget"]), int(r["representation_capacity"]), float(r["lambda_a"]))
            for r in rows
        }
    )
    summary = []
    for budget, capacity, lambda_a in keys:
        values = [
            r
            for r in rows
            if int(r["observation_budget"]) == budget
            and int(r["representation_capacity"]) == capacity
            and float(r["lambda_a"]) == lambda_a
        ]
        da = np.array([float(v["D_A"]) for v in values])
        db = np.array([float(v["D_B"]) for v in values])
        summary.append(
            {
                "observation_budget": budget,
                "representation_capacity": capacity,
                "lambda_a": lambda_a,
                "D_A_mean": float(da.mean()),
                "D_A_sd": float(da.std(ddof=0)),
                "D_B_mean": float(db.mean()),
                "D_B_sd": float(db.std(ddof=0)),
                "mean_D": float(0.5 * (da.mean() + db.mean())),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_summary(
    summary: list[dict], budget: int, capacity: int, lambda_a: float = 0.5
) -> dict:
    return next(
        row
        for row in summary
        if int(row["observation_budget"]) == budget
        and int(row["representation_capacity"]) == capacity
        and float(row["lambda_a"]) == lambda_a
    )


def make_decomposition(summary: list[dict], cfg: Config) -> list[dict]:
    full_budget = max(cfg.observation_budgets)
    full_capacity = max(cfg.representation_capacities)
    budget = 4
    capacity = 2
    baseline = float(find_summary(summary, full_budget, full_capacity)["mean_D"])
    observation_only = float(find_summary(summary, budget, full_capacity)["mean_D"])
    compression_only = float(find_summary(summary, full_budget, capacity)["mean_D"])
    joint = float(find_summary(summary, budget, capacity)["mean_D"])
    return [
        {
            "lambda_a": 0.5,
            "limited_observation_budget": budget,
            "limited_representation_capacity": capacity,
            "baseline_D": baseline,
            "observation_only_D": observation_only,
            "compression_only_D": compression_only,
            "joint_D": joint,
            "observation_penalty": observation_only - baseline,
            "compression_penalty": compression_only - baseline,
            "interaction": joint - observation_only - compression_only + baseline,
        }
    ]


def jaccard(mask_a: str, mask_b: str) -> float:
    a = set(int(v) for v in mask_a.split(";") if v != "")
    b = set(int(v) for v in mask_b.split(";") if v != "")
    return len(a & b) / len(a | b)


def expected_random_jaccard(n_channels: int, budget: int) -> float:
    """同じ大きさの二集合を独立一様に選ぶ帰無モデルの期待Jaccard。"""
    subsets = [set(values) for values in itertools.combinations(range(n_channels), budget)]
    values = [len(a & b) / len(a | b) for a in subsets for b in subsets]
    return float(np.mean(values))


def summarize_mask_overlap(rows: list[dict], cfg: Config) -> list[dict]:
    """lambda=0/1 のOracle観測マスクを予算別に偶然水準と比較する。"""
    result = []
    n_channels = cfg.source_dim * 2
    for budget in cfg.observation_budgets:
        values = []
        for seed in cfg.seeds:
            mask_b = next(
                row
                for row in rows
                if int(row["seed"]) == seed
                and int(row["observation_budget"]) == budget
                and float(row["lambda_a"]) == 0.0
            )["mask"]
            mask_a = next(
                row
                for row in rows
                if int(row["seed"]) == seed
                and int(row["observation_budget"]) == budget
                and float(row["lambda_a"]) == 1.0
            )["mask"]
            values.append(jaccard(str(mask_b), str(mask_a)))
        chance = expected_random_jaccard(n_channels, budget)
        observed = float(np.mean(values))
        result.append(
            {
                "observation_budget": budget,
                "n_seeds": len(values),
                "observed_jaccard": observed,
                "observed_sd": float(np.std(values, ddof=0)),
                "random_expectation": chance,
                "difference_from_random": observed - chance,
                "below_random_expectation": bool(observed < chance),
            }
        )
    return result


def judge(
    rows: list[dict], summary: list[dict], mask_overlap: list[dict], cfg: Config
) -> tuple[str, dict]:
    full_budget = max(cfg.observation_budgets)
    full_capacity = max(cfg.representation_capacities)
    baseline = float(find_summary(summary, full_budget, full_capacity)["mean_D"])
    obs_limited = float(find_summary(summary, 3, full_capacity)["mean_D"])
    repr_limited = float(find_summary(summary, full_budget, 3)["mean_D"])

    floor_errors = []
    plateau_errors = []
    for row in rows:
        if int(row["representation_capacity"]) >= int(row["observation_budget"]):
            floor_errors.extend(
                [
                    float(row["D_A"]) - float(row["oracle_D_A"]),
                    float(row["D_B"]) - float(row["oracle_D_B"]),
                ]
            )
    for seed in cfg.seeds:
        for budget in cfg.observation_budgets:
            for lambda_a in cfg.lambdas:
                values = [
                    r
                    for r in rows
                    if int(r["seed"]) == seed
                    and int(r["observation_budget"]) == budget
                    and float(r["lambda_a"]) == lambda_a
                    and int(r["representation_capacity"]) >= budget
                ]
                if len(values) > 1:
                    plateau_errors.append(
                        max(float(v["mean_D"]) for v in values)
                        - min(float(v["mean_D"]) for v in values)
                    )

    floor_rmse = float(np.sqrt(np.mean(np.square(floor_errors))))
    max_plateau_error = float(max(plateau_errors, default=0.0))
    limited_overlap = [
        row for row in mask_overlap if int(row["observation_budget"]) < full_budget
    ]
    criteria = {
        "baseline_D": baseline,
        "observation_limited_D": obs_limited,
        "representation_limited_D": repr_limited,
        "observation_penalty": obs_limited - baseline,
        "representation_penalty": repr_limited - baseline,
        "oracle_floor_rmse": floor_rmse,
        "max_plateau_error": max_plateau_error,
        "endpoint_mask_jaccard_mean": float(
            np.mean([float(row["observed_jaccard"]) for row in limited_overlap])
        ),
        "all_limited_mask_overlaps_below_random": bool(
            all(bool(row["below_random_expectation"]) for row in limited_overlap)
        ),
        "full_condition_reconstructed": bool(baseline < 0.02),
        "observation_effect_detected": bool(obs_limited - baseline > 0.10),
        "representation_effect_detected": bool(repr_limited - baseline > 0.10),
        "oracle_floor_reproduced": bool(floor_rmse < 0.04),
        "irrecoverability_plateau_reproduced": bool(max_plateau_error < 1e-10),
    }
    passed = all(
        criteria[key]
        for key in [
            "full_condition_reconstructed",
            "observation_effect_detected",
            "representation_effect_detected",
            "oracle_floor_reproduced",
            "irrecoverability_plateau_reproduced",
        ]
    )
    if passed:
        message = (
            "観測・表現制約の分離に成功：未観測による歪みと、取得後の圧縮による歪みを"
            "独立に検出し、観測後に容量を増やしても前者が回復しないことを再現しました。"
        )
    else:
        message = (
            "分離校正は未達：生成過程、Oracle観測選択、低ランク表現または読み出しを点検してください。"
        )
    criteria["passed"] = passed
    return message, criteria


def write_heatmap_svg(path: Path, summary: list[dict], cfg: Config) -> None:
    budgets = cfg.observation_budgets
    capacities = cfg.representation_capacities
    values = np.array(
        [
            [float(find_summary(summary, b, c)["mean_D"]) for c in capacities]
            for b in budgets
        ]
    )
    width, height = 800, 520
    left, top, cell_w, cell_h = 145, 70, 105, 78
    vmax = max(0.7, float(values.max()))

    def color(value: float) -> str:
        t = min(max(value / vmax, 0.0), 1.0)
        r = int(235 - 105 * t)
        g = int(247 - 155 * t)
        b = int(250 - 65 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:700}.label{font-size:16px}.value{font-size:15px;font-weight:700}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Balanced distortion: observation budget × representation capacity</text>',
    ]
    for j, capacity in enumerate(capacities):
        x = left + j * cell_w + cell_w / 2
        lines.append(f'<text x="{x}" y="58" text-anchor="middle" class="label">C_R={capacity}</text>')
    for i, budget in enumerate(budgets):
        y = top + i * cell_h
        lines.append(
            f'<text x="{left-18}" y="{y+cell_h/2+5}" text-anchor="end" class="label">B_A={budget}</text>'
        )
        for j, capacity in enumerate(capacities):
            x = left + j * cell_w
            value = values[i, j]
            lines += [
                f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" rx="6" fill="{color(value)}"/>',
                f'<text x="{x+(cell_w-4)/2}" y="{y+(cell_h-4)/2+5}" text-anchor="middle" class="value">{value:.3f}</text>',
            ]
    lines += [
        f'<text x="{width/2}" y="{height-25}" text-anchor="middle" class="label">lower is better; lambda = 0.5</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_curves_svg(path: Path, summary: list[dict], cfg: Config) -> None:
    width, height = 800, 530
    left, right, top, bottom = 90, 40, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    capacities = cfg.representation_capacities
    ymax = 0.75
    colors = ["#7E57C2", "#EF8354", "#3D9970", "#176B87"]

    def sx(index: int) -> float:
        return left + plot_w * index / (len(capacities) - 1)

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value / ymax)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC}.title{font-size:21px;font-weight:700}.label{font-size:16px}</style>',
        f'<text x="{width/2}" y="29" text-anchor="middle" class="title">Recoverability after observation: lambda = 0.5</text>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = sy(value)
        lines += [
            f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid"/>',
            f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="13">{value:.2f}</text>',
        ]
    for j, capacity in enumerate(capacities):
        x = sx(j)
        lines.append(f'<text x="{x}" y="{top+plot_h+25}" text-anchor="middle" font-size="13">{capacity}</text>')
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
    ]
    for idx, budget in enumerate(cfg.observation_budgets):
        points = []
        for j, capacity in enumerate(capacities):
            value = float(find_summary(summary, budget, capacity)["mean_D"])
            points.append(f"{sx(j):.1f},{sy(value):.1f}")
        color = colors[idx]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for point in points:
            x, y = point.split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
        ly = top + 20 + idx * 24
        lines += [
            f'<line x1="{left+plot_w-140}" y1="{ly}" x2="{left+plot_w-105}" y2="{ly}" stroke="{color}" stroke-width="4"/>',
            f'<text x="{left+plot_w-95}" y="{ly+5}" font-size="14">B_A={budget}</text>',
        ]
    lines += [
        f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">representation capacity C_R</text>',
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" class="label">mean distortion</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = make_config(full=args.full)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("実験I-B：観測予算と表現容量の分離PoCを開始します（GPU不要）")
    print(f"保存先: {output_dir}")

    rows: list[dict] = []
    total = (
        len(cfg.seeds)
        * len(cfg.lambdas)
        * len(cfg.observation_budgets)
        * len(cfg.representation_capacities)
    )
    count = 0
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
        mixing, _ = np.linalg.qr(raw)
        x_train, a_train, b_train = make_dataset(cfg.train_size, cfg, rng, mixing)
        x_test, a_test, b_test = make_dataset(cfg.test_size, cfg, rng, mixing)
        x_train, x_test = standardize_triplet(x_train, x_test)
        a_train, a_test = standardize_triplet(a_train, a_test)
        b_train, b_test = standardize_triplet(b_train, b_test)

        for lambda_a in cfg.lambdas:
            for budget in cfg.observation_budgets:
                mask, oracle_da, oracle_db = select_oracle_mask(
                    cfg, mixing, budget, lambda_a
                )
                mask_text = ";".join(str(value) for value in mask)
                for capacity in cfg.representation_capacities:
                    d_a, d_b, effective = evaluate_condition(
                        x_train,
                        a_train,
                        b_train,
                        x_test,
                        a_test,
                        b_test,
                        mask,
                        capacity,
                        lambda_a,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "lambda_a": lambda_a,
                            "observation_budget": budget,
                            "representation_capacity": capacity,
                            "effective_capacity": effective,
                            "mask": mask_text,
                            "oracle_D_A": oracle_da,
                            "oracle_D_B": oracle_db,
                            "D_A": d_a,
                            "D_B": d_b,
                            "mean_D": 0.5 * (d_a + d_b),
                        }
                    )
                    count += 1
                    if count == 1 or count % max(total // 10, 1) == 0 or count == total:
                        print(f"  進捗 {count}/{total}")

    summary = aggregate(rows)
    decomposition = make_decomposition(summary, cfg)
    mask_overlap = summarize_mask_overlap(rows, cfg)
    message, criteria = judge(rows, summary, mask_overlap, cfg)
    elapsed = time.time() - start
    metadata = {
        "experiment": "Experiment I-B: observation-representation separation PoC",
        "version": "0.2",
        "config": asdict(cfg),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "criteria": criteria,
        "scope_note": (
            "Observation masks are population-Oracle selections. This PoC separates constraints; "
            "it does not demonstrate learned attention, history dependence, or human intentionality."
        ),
    }
    write_csv(output_dir / "raw_results.csv", rows)
    write_csv(output_dir / "summary_results.csv", summary)
    write_csv(output_dir / "distortion_decomposition.csv", decomposition)
    write_csv(output_dir / "mask_overlap.csv", mask_overlap)
    write_heatmap_svg(output_dir / "factorial_heatmap.svg", summary, cfg)
    write_curves_svg(output_dir / "recoverability_curves.svg", summary, cfg)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 実験I-B・観測予算／表現容量分離PoC 実行結果",
        "",
        message,
        "",
        "## 自動判定",
        "",
        f"- 無制約条件の平均歪み: {criteria['baseline_D']:.4f}",
        f"- 観測制約のみ（B_A=3）の平均歪み: {criteria['observation_limited_D']:.4f}",
        f"- 表現制約のみ（C_R=3）の平均歪み: {criteria['representation_limited_D']:.4f}",
        f"- 観測制約による増分: {criteria['observation_penalty']:.4f}",
        f"- 表現制約による増分: {criteria['representation_penalty']:.4f}",
        f"- 母集団Oracle観測床とのRMSE: {criteria['oracle_floor_rmse']:.4f}",
        f"- C_RをB_A以上に増やしたときの最大変化: {criteria['max_plateau_error']:.3e}",
        f"- lambda=0/1の観測マスク平均Jaccard: {criteria['endpoint_mask_jaccard_mean']:.3f}",
        f"- 制約予算の全てで偶然期待値を下回った: {criteria['all_limited_mask_overlaps_below_random']}",
        "",
        "## 観測マスクの探索的比較",
        "",
        "| B_A | 実測Jaccard | 偶然期待値 | 差 |",
        "|---:|---:|---:|---:|",
        *[
            (
                f"| {int(row['observation_budget'])} "
                f"| {float(row['observed_jaccard']):.4f} "
                f"| {float(row['random_expectation']):.4f} "
                f"| {float(row['difference_from_random']):+.4f} |"
            )
            for row in mask_overlap
        ],
        "",
        "この比較は校正ゲートではなく探索的結果です。Oracle最適化された観測先が、保存対象に応じて取得段階から分岐することを示します。",
        "",
        "## 解釈の限界",
        "",
        "観測チャネルは生成過程を知るOracleが選択しており、観測方策を学習した結果ではありません。",
        "本結果は、未観測による情報欠落と観測後の圧縮を操作的に分離できることを示します。",
        "負の交互作用には情報欠落の重複と正規化歪みの飽和が混在するため、その大きさを冗長性の量とは解釈しません。",
        "人間の注意、志向性、発達的固定、心理志向／客観志向を支持する結果ではありません。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text("\n".join(report), encoding="utf-8")
    print()
    print(message)
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, factorial_heatmap.svg, recoverability_curves.svg")


if __name__ == "__main__":
    main()
