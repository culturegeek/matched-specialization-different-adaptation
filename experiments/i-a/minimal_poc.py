#!/usr/bin/env python3
"""
実験I・最小PoC v0.3：解析解を持つ線形ガウス系による測定基盤の校正。

外部依存は NumPy だけです。GPU は使いません。
実行例:
    python minimal_poc.py --quick
    python minimal_poc.py --full
"""

from __future__ import annotations

import argparse
import csv
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
    epochs: int
    learning_rate: float
    weight_decay: float
    bottlenecks: list[int]
    lambdas: list[float]
    calibration_correlations: list[float]
    scaling_train_sizes: list[int]
    scaling_seeds: list[int]
    scaling_epochs: int
    seeds: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPUとNumPyだけで、共有表現の情報保存トレードオフを確かめます。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="初回用の短時間設定（既定）")
    mode.add_argument("--full", action="store_true", help="seedと条件を増やした確認設定")
    parser.add_argument("--output-dir", default="poc_results", help="結果の保存先")
    return parser.parse_args()


def make_config(full: bool) -> Config:
    if full:
        return Config(
            train_size=4096,
            test_size=2048,
            source_dim=3,
            correlation=0.35,
            observation_noise=0.02,
            epochs=900,
            learning_rate=0.025,
            weight_decay=1e-5,
            bottlenecks=[2, 3, 4, 6],
            lambdas=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
            calibration_correlations=[0.0, 0.2, 0.35, 0.5, 0.7, 0.9],
            scaling_train_sizes=[1024, 2048, 4096, 8192, 16384],
            scaling_seeds=list(range(12)),
            scaling_epochs=500,
            seeds=[0, 1, 2, 3, 4],
        )
    return Config(
        train_size=2048,
        test_size=1024,
        source_dim=3,
        correlation=0.35,
        observation_noise=0.02,
        epochs=500,
        learning_rate=0.03,
        weight_decay=1e-5,
        bottlenecks=[3, 6],
        lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
        calibration_correlations=[0.0, 0.2, 0.35, 0.5, 0.7, 0.9],
        scaling_train_sizes=[1024, 2048, 4096, 8192],
        scaling_seeds=list(range(8)),
        scaling_epochs=500,
        seeds=[0, 1, 2],
    )


def make_dataset(
    n: int,
    cfg: Config,
    rng: np.random.Generator,
    mixing: np.ndarray,
    correlation: float | None = None,
):
    """AとBは一部を共有しつつ、互いに固有の情報も持つ。"""
    a = rng.normal(size=(n, cfg.source_dim))
    private_b = rng.normal(size=(n, cfg.source_dim))
    rho = cfg.correlation if correlation is None else correlation
    b = rho * a + math.sqrt(1.0 - rho**2) * private_b
    sources = np.concatenate([a, b], axis=1)
    x = sources @ mixing
    x += cfg.observation_noise * rng.normal(size=x.shape)
    return x.astype(np.float64), a.astype(np.float64), b.astype(np.float64)


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / scale, (test - mean) / scale


def optimal_linear_readout(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray):
    """エンコーダを固定し、切片付き最小二乗で読み出しの下限を近似する。"""
    train_aug = np.concatenate([z_train, np.ones((z_train.shape[0], 1))], axis=1)
    test_aug = np.concatenate([z_test, np.ones((z_test.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(train_aug, y_train, rcond=None)
    return test_aug @ coefficients


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray], lr: float):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.v = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.t = 0

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]):
        self.t += 1
        for key in parameters:
            grad = gradients[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad * grad
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            parameters[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_one(
    x_train: np.ndarray,
    a_train: np.ndarray,
    b_train: np.ndarray,
    x_test: np.ndarray,
    a_test: np.ndarray,
    b_test: np.ndarray,
    bottleneck: int,
    lambda_a: float,
    seed: int,
    cfg: Config,
) -> dict[str, float | int]:
    """1層の共有エンコーダと2つの線形読み出しヘッドを学習する。"""
    rng = np.random.default_rng(10_000 + seed * 101 + bottleneck * 17 + int(lambda_a * 100))
    input_dim = x_train.shape[1]
    output_dim = a_train.shape[1]
    scale = 0.15
    p = {
        "we": rng.normal(scale=scale, size=(input_dim, bottleneck)),
        "wa": rng.normal(scale=scale, size=(bottleneck, output_dim)),
        "wb": rng.normal(scale=scale, size=(bottleneck, output_dim)),
        "ba": np.zeros((1, output_dim)),
        "bb": np.zeros((1, output_dim)),
    }
    opt = Adam(p, cfg.learning_rate)
    n = x_train.shape[0]

    for _ in range(cfg.epochs):
        z = x_train @ p["we"]
        pred_a = z @ p["wa"] + p["ba"]
        pred_b = z @ p["wb"] + p["bb"]

        grad_pred_a = (2.0 * lambda_a / (n * output_dim)) * (pred_a - a_train)
        grad_pred_b = (2.0 * (1.0 - lambda_a) / (n * output_dim)) * (pred_b - b_train)

        grad_z = grad_pred_a @ p["wa"].T + grad_pred_b @ p["wb"].T
        gradients = {
            "we": x_train.T @ grad_z + 2 * cfg.weight_decay * p["we"],
            "wa": z.T @ grad_pred_a + 2 * cfg.weight_decay * p["wa"],
            "wb": z.T @ grad_pred_b + 2 * cfg.weight_decay * p["wb"],
            "ba": grad_pred_a.sum(axis=0, keepdims=True),
            "bb": grad_pred_b.sum(axis=0, keepdims=True),
        }
        opt.step(p, gradients)

    # 理論稿の D_k = inf_{readout} E[d_k] に合わせ、学習後の Z に
    # 両方の最適線形読み出しを当て直す。lambda=0/1 の未学習ヘッドを
    # 「情報が失われた」と誤認しないためである。
    z_train = x_train @ p["we"]
    z_test = x_test @ p["we"]
    pred_a = optimal_linear_readout(z_train, a_train, z_test)
    pred_b = optimal_linear_readout(z_train, b_train, z_test)
    # 1.0 は「平均値を予測するだけ」の基準値。小さいほどよく保存できた。
    d_a = float(np.mean((pred_a - a_test) ** 2) / (np.var(a_test) + 1e-12))
    d_b = float(np.mean((pred_b - b_test) ** 2) / (np.var(b_test) + 1e-12))
    return {
        "seed": seed,
        "bottleneck": bottleneck,
        "lambda_a": lambda_a,
        "D_A": d_a,
        "D_B": d_b,
        "mean_D": 0.5 * (d_a + d_b),
    }


def aggregate(rows: list[dict[str, float | int]]):
    groups: dict[tuple[int, float], list[dict[str, float | int]]] = {}
    for row in rows:
        key = (int(row["bottleneck"]), float(row["lambda_a"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (bottleneck, lambda_a), values in sorted(groups.items()):
        da = np.array([float(v["D_A"]) for v in values])
        db = np.array([float(v["D_B"]) for v in values])
        result.append(
            {
                "bottleneck": bottleneck,
                "lambda_a": lambda_a,
                "D_A_mean": float(da.mean()),
                "D_A_sd": float(da.std(ddof=0)),
                "D_B_mean": float(db.mean()),
                "D_B_sd": float(db.std(ddof=0)),
                "mean_D": float(0.5 * (da.mean() + db.mean())),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


COLORS = ["#176B87", "#EF8354", "#3D9970", "#7E57C2", "#B23A48"]


def write_tradeoff_svg(path: Path, agg: list[dict]):
    width, height = 760, 560
    left, right, top, bottom = 90, 35, 45, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(1.05, max(max(r["D_A_mean"], r["D_B_mean"]) for r in agg) * 1.05)

    def sx(x):
        return left + plot_w * x / max_value

    def sy(y):
        return top + plot_h * (1.0 - y / max_value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC;stroke-width:1}.legend{font-size:15px}.label{font-size:16px}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="{width/2}" y="26" text-anchor="middle" class="title">Shared bottleneck trade-off (lower is better)</text>',
    ]
    for i in range(6):
        val = max_value * i / 5
        x, y = sx(val), sy(val)
        lines += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="13">{val:.2f}</text>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{val:.2f}</text>',
        ]
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">D_A: information A distortion</text>',
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" class="label">D_B: information B distortion</text>',
    ]
    bottlenecks = sorted({int(r["bottleneck"]) for r in agg})
    for idx, bottleneck in enumerate(bottlenecks):
        color = COLORS[idx % len(COLORS)]
        subset = [r for r in agg if int(r["bottleneck"]) == bottleneck]
        points = " ".join(f'{sx(r["D_A_mean"]):.1f},{sy(r["D_B_mean"]):.1f}' for r in subset)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for r in subset:
            x, y = sx(r["D_A_mean"]), sy(r["D_B_mean"])
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            lines.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-size="11">λ={r["lambda_a"]:.2g}</text>')
        ly = top + 20 + idx * 23
        lines += [
            f'<line x1="{left+plot_w-150}" y1="{ly}" x2="{left+plot_w-120}" y2="{ly}" stroke="{color}" stroke-width="4"/>',
            f'<text x="{left+plot_w-112}" y="{ly+5}" class="legend">C_R = {bottleneck}</text>',
        ]
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def analytic_balanced_distortion(rho: float, observation_noise: float) -> float:
    """C_R=source_dim, lambda=.5 の母集団歪み。Xは標準化しない前提。"""
    noise_var = observation_noise**2
    return 1.0 - ((1.0 + rho) ** 2) / (2.0 * (1.0 + rho + noise_var))


def run_correlation_calibration(cfg: Config) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    for rho in cfg.calibration_correlations:
        for seed in cfg.seeds:
            rng = np.random.default_rng(50_000 + seed)
            raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
            q, _ = np.linalg.qr(raw)
            x_train, a_train, b_train = make_dataset(
                cfg.train_size, cfg, rng, q, correlation=rho
            )
            x_test, a_test, b_test = make_dataset(
                cfg.test_size, cfg, rng, q, correlation=rho
            )
            a_train, a_test = standardize(a_train, a_test)
            b_train, b_test = standardize(b_train, b_test)
            row = train_one(
                x_train,
                a_train,
                b_train,
                x_test,
                a_test,
                b_test,
                cfg.source_dim,
                0.5,
                seed,
                cfg,
            )
            row["rho"] = rho
            raw_rows.append(row)

    summary = []
    for rho in cfg.calibration_correlations:
        values = [row for row in raw_rows if float(row["rho"]) == rho]
        empirical = np.array([float(row["mean_D"]) for row in values])
        theory = analytic_balanced_distortion(rho, cfg.observation_noise)
        summary.append(
            {
                "rho": rho,
                "empirical_mean_D": float(empirical.mean()),
                "empirical_sd": float(empirical.std(ddof=0)),
                "analytic_D": theory,
                "absolute_error": abs(float(empirical.mean()) - theory),
            }
        )
    return raw_rows, summary


def run_sample_size_calibration(cfg: Config) -> tuple[list[dict], list[dict], dict]:
    """有限標本誤差と実装バイアスを分けるため、nと誤差の縮小を測る。"""
    raw_rows: list[dict] = []
    theory = analytic_balanced_distortion(cfg.correlation, cfg.observation_noise)
    original_epochs = cfg.epochs
    cfg.epochs = cfg.scaling_epochs
    try:
        for n_train in cfg.scaling_train_sizes:
            n_test = max(512, n_train // 2)
            for seed in cfg.scaling_seeds:
                rng = np.random.default_rng(90_000 + seed * 10_003 + n_train)
                raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
                q, _ = np.linalg.qr(raw)
                x_train, a_train, b_train = make_dataset(n_train, cfg, rng, q)
                x_test, a_test, b_test = make_dataset(n_test, cfg, rng, q)
                a_train, a_test = standardize(a_train, a_test)
                b_train, b_test = standardize(b_train, b_test)
                row = train_one(
                    x_train,
                    a_train,
                    b_train,
                    x_test,
                    a_test,
                    b_test,
                    cfg.source_dim,
                    0.5,
                    seed,
                    cfg,
                )
                row.update(
                    {
                        "train_size": n_train,
                        "test_size": n_test,
                        "analytic_D": theory,
                        "signed_error": float(row["mean_D"]) - theory,
                    }
                )
                raw_rows.append(row)
    finally:
        cfg.epochs = original_epochs

    summary: list[dict] = []
    for n_train in cfg.scaling_train_sizes:
        values = [row for row in raw_rows if int(row["train_size"]) == n_train]
        empirical = np.array([float(row["mean_D"]) for row in values])
        errors = empirical - theory
        summary.append(
            {
                "train_size": n_train,
                "test_size": max(512, n_train // 2),
                "n_seeds": len(values),
                "empirical_mean_D": float(empirical.mean()),
                "signed_bias": float(errors.mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mse": float(np.mean(errors**2)),
            }
        )

    log_n = np.log(np.array([float(row["train_size"]) for row in summary]))
    log_rmse = np.log(np.array([max(float(row["rmse"]), 1e-12) for row in summary]))
    rmse_slope = float(np.polyfit(log_n, log_rmse, 1)[0])
    diagnostics = {
        "rmse_loglog_slope": rmse_slope,
        "mse_loglog_slope": 2.0 * rmse_slope,
        "smallest_n_rmse": float(summary[0]["rmse"]),
        "largest_n_rmse": float(summary[-1]["rmse"]),
        "error_decreases": bool(float(summary[-1]["rmse"]) < float(summary[0]["rmse"])),
        # RMSEの理論的オーダーは n^{-1/2}。seed数が限られるため幅を持たせる。
        "sampling_scaling_consistent": bool(
            float(summary[-1]["rmse"]) < float(summary[0]["rmse"])
            and -0.9 < rmse_slope < -0.1
        ),
    }
    return raw_rows, summary, diagnostics


def write_calibration_svg(path: Path, rows: list[dict]):
    width, height = 760, 500
    left, right, top, bottom = 85, 35, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(x):
        return left + plot_w * x

    def sy(y):
        return top + plot_h * (1.0 - y / 0.55)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC;stroke-width:1}.label{font-size:16px}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="{width/2}" y="26" text-anchor="middle" class="title">Calibration against analytic solution</text>',
    ]
    for i in range(6):
        x = i / 5
        y = 0.55 * i / 5
        lines += [
            f'<line x1="{sx(x):.1f}" y1="{top}" x2="{sx(x):.1f}" y2="{top+plot_h}" class="grid"/>',
            f'<line x1="{left}" y1="{sy(y):.1f}" x2="{left+plot_w}" y2="{sy(y):.1f}" class="grid"/>',
            f'<text x="{sx(x):.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="13">{x:.1f}</text>',
            f'<text x="{left-12}" y="{sy(y)+5:.1f}" text-anchor="end" font-size="13">{y:.2f}</text>',
        ]
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" class="label">correlation rho</text>',
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" class="label">balanced distortion</text>',
    ]
    theory_points = " ".join(
        f'{sx(float(r["rho"])):.1f},{sy(float(r["analytic_D"])):.1f}' for r in rows
    )
    empirical_points = " ".join(
        f'{sx(float(r["rho"])):.1f},{sy(float(r["empirical_mean_D"])):.1f}' for r in rows
    )
    lines += [
        f'<polyline points="{theory_points}" fill="none" stroke="#455A64" stroke-width="3" stroke-dasharray="8 6"/>',
        f'<polyline points="{empirical_points}" fill="none" stroke="#176B87" stroke-width="3"/>',
    ]
    for row in rows:
        lines.append(
            f'<circle cx="{sx(float(row["rho"])):.1f}" cy="{sy(float(row["empirical_mean_D"])):.1f}" r="5" fill="#176B87"/>'
        )
    lines += [
        f'<line x1="{left+plot_w-180}" y1="{top+18}" x2="{left+plot_w-145}" y2="{top+18}" stroke="#455A64" stroke-width="3" stroke-dasharray="8 6"/>',
        f'<text x="{left+plot_w-135}" y="{top+23}" font-size="14">analytic</text>',
        f'<line x1="{left+plot_w-180}" y1="{top+43}" x2="{left+plot_w-145}" y2="{top+43}" stroke="#176B87" stroke-width="3"/>',
        f'<text x="{left+plot_w-135}" y="{top+48}" font-size="14">empirical</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sample_size_svg(path: Path, rows: list[dict], diagnostics: dict):
    width, height = 760, 500
    left, right, top, bottom = 90, 35, 50, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = np.log10(np.array([float(row["train_size"]) for row in rows]))
    ys = np.log10(np.array([float(row["rmse"]) for row in rows]))
    x_min, x_max = float(xs.min()), float(xs.max())
    y_pad = max(0.08, float(ys.max() - ys.min()) * 0.15)
    y_min, y_max = float(ys.min() - y_pad), float(ys.max() + y_pad)

    def sx(x):
        return left + plot_w * (x - x_min) / max(x_max - x_min, 1e-12)

    def sy(y):
        return top + plot_h * (1.0 - (y - y_min) / max(y_max - y_min, 1e-12))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.axis{stroke:#455A64;stroke-width:1.5}.grid{stroke:#CFD8DC;stroke-width:1}.label{font-size:16px}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="{width/2}" y="27" text-anchor="middle" class="title">Finite-sample calibration: RMSE versus n</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
    ]
    for row, x, y in zip(rows, xs, ys):
        px, py = sx(float(x)), sy(float(y))
        lines += [
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+plot_h}" class="grid"/>',
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#176B87"/>',
            f'<text x="{px:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-size="13">{int(row["train_size"])}</text>',
            f'<text x="{px+8:.1f}" y="{py-8:.1f}" font-size="12">{float(row["rmse"]):.4f}</text>',
        ]
    points = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys))
    lines += [
        f'<polyline points="{points}" fill="none" stroke="#176B87" stroke-width="3"/>',
        f'<text x="{left+12}" y="{top+22}" font-size="14">log-log slope = {float(diagnostics["rmse_loglog_slope"]):.3f} (RMSE target: about -0.5)</text>',
        f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">training sample size n (log scale)</text>',
        f'<text x="24" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})" class="label">RMSE (log scale)</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def verdict(
    agg: list[dict], calibration: list[dict], scaling: dict, cfg: Config
) -> tuple[str, dict]:
    limited = min(cfg.bottlenecks, key=lambda value: abs(value - cfg.source_dim))
    expanded = max(cfg.bottlenecks)
    limited_rows = [r for r in agg if int(r["bottleneck"]) == limited]
    expanded_rows = [r for r in agg if int(r["bottleneck"]) == expanded]
    best_a = min(r["D_A_mean"] for r in limited_rows)
    best_b = min(r["D_B_mean"] for r in limited_rows)
    balanced_limited = min(limited_rows, key=lambda r: abs(r["lambda_a"] - 0.5))
    balanced_expanded = min(expanded_rows, key=lambda r: abs(r["lambda_a"] - 0.5))
    gain = balanced_limited["mean_D"] - balanced_expanded["mean_D"]
    errors = np.array([float(row["absolute_error"]) for row in calibration])
    calibration_rmse = float(np.sqrt(np.mean(errors**2)))
    calibration_max_error = float(errors.max())
    criteria = {
        "limited_bottleneck": limited,
        "expanded_bottleneck": expanded,
        "best_D_A_limited": best_a,
        "best_D_B_limited": best_b,
        "balanced_mean_D_limited": balanced_limited["mean_D"],
        "balanced_mean_D_expanded": balanced_expanded["mean_D"],
        "capacity_gain": gain,
        "individual_learnability": bool(best_a < 0.20 and best_b < 0.20),
        "capacity_sensitive_conflict": bool(gain > 0.15),
        "calibration_rmse": calibration_rmse,
        "calibration_max_error": calibration_max_error,
        "analytic_calibration_passed": bool(
            calibration_rmse < 0.03 and calibration_max_error < 0.05
        ),
        **scaling,
    }
    if criteria["analytic_calibration_passed"] and criteria["sampling_scaling_consistent"]:
        message = "測定基盤の校正に成功：解析曲線を再現し、残る誤差も標本数とともに縮小しました。"
    else:
        message = "測定基盤の校正は未達：環境設計ではなく、実装、前処理、収束または最適読み出しの計算を先に点検してください。"
    return message, criteria


def main():
    args = parse_args()
    cfg = make_config(full=args.full)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("実験I・最小PoCを開始します（GPU不要）")
    print(f"保存先: {output_dir}")
    print(f"実行条件: bottleneck={cfg.bottlenecks}, lambda={cfg.lambdas}, seeds={cfg.seeds}")

    rows = []
    total = len(cfg.bottlenecks) * len(cfg.lambdas) * len(cfg.seeds)
    count = 0
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
        q, _ = np.linalg.qr(raw)
        x_train, a_train, b_train = make_dataset(cfg.train_size, cfg, rng, q)
        x_test, a_test, b_test = make_dataset(cfg.test_size, cfg, rng, q)
        a_train, a_test = standardize(a_train, a_test)
        b_train, b_test = standardize(b_train, b_test)
        for bottleneck in cfg.bottlenecks:
            for lambda_a in cfg.lambdas:
                rows.append(
                    train_one(
                        x_train,
                        a_train,
                        b_train,
                        x_test,
                        a_test,
                        b_test,
                        bottleneck,
                        lambda_a,
                        seed,
                        cfg,
                    )
                )
                count += 1
                if count == 1 or count % max(1, total // 10) == 0 or count == total:
                    print(f"  進捗 {count}/{total}")

    agg = aggregate(rows)
    print("  相関 rho の校正走査を実行します")
    calibration_raw, calibration = run_correlation_calibration(cfg)
    print("  標本サイズ n の校正走査を実行します")
    scaling_raw, scaling_summary, scaling_diagnostics = run_sample_size_calibration(cfg)
    message, criteria = verdict(agg, calibration, scaling_diagnostics, cfg)
    elapsed = time.time() - start
    metadata = {
        "config": asdict(cfg),
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "criteria": criteria,
    }
    write_csv(output_dir / "raw_results.csv", rows)
    write_csv(output_dir / "summary_results.csv", agg)
    write_csv(output_dir / "rho_calibration_raw.csv", calibration_raw)
    write_csv(output_dir / "rho_calibration.csv", calibration)
    write_csv(output_dir / "sample_size_calibration_raw.csv", scaling_raw)
    write_csv(output_dir / "sample_size_calibration.csv", scaling_summary)
    write_tradeoff_svg(output_dir / "tradeoff.svg", agg)
    write_calibration_svg(output_dir / "rho_calibration.svg", calibration)
    write_sample_size_svg(
        output_dir / "sample_size_calibration.svg", scaling_summary, scaling_diagnostics
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 実験I・測定基盤校正PoC 実行結果",
        "",
        message,
        "",
        "## 自動判定の内訳",
        "",
        f"- 制約した表現次元: {criteria['limited_bottleneck']}",
        f"- 拡張した表現次元: {criteria['expanded_bottleneck']}",
        f"- 要求Aの最良歪み（制約下）: {criteria['best_D_A_limited']:.4f}",
        f"- 要求Bの最良歪み（制約下）: {criteria['best_D_B_limited']:.4f}",
        f"- 釣り合い条件の平均歪み（制約下）: {criteria['balanced_mean_D_limited']:.4f}",
        f"- 釣り合い条件の平均歪み（拡張時）: {criteria['balanced_mean_D_expanded']:.4f}",
        f"- 容量拡張による改善量: {criteria['capacity_gain']:.4f}",
        f"- 解析解とのRMSE: {criteria['calibration_rmse']:.4f}",
        f"- 解析解との最大絶対誤差: {criteria['calibration_max_error']:.4f}",
        f"- RMSEの log-log 傾き: {criteria['rmse_loglog_slope']:.3f}（理論的オーダーは約 -0.5）",
        f"- MSEの log-log 傾き: {criteria['mse_loglog_slope']:.3f}（理論的オーダーは約 -1）",
        f"- 最小標本数のRMSE: {criteria['smallest_n_rmse']:.4f}",
        f"- 最大標本数のRMSE: {criteria['largest_n_rmse']:.4f}",
        "",
        "## 注意",
        "",
        "この結果は、解析解を持つ線形ガウス系で歪み測定パイプラインを校正したものです。",
        "情報保存トレードオフの一般的な実在や、本理論に固有の予測を経験的に支持するものではありません。",
        "人間の志向性、発達、準安定レジーム、心理志向／客観志向の実在を証明するものではありません。",
        "",
        f"実行時間: {elapsed:.1f}秒",
    ]
    (output_dir / "result_report.md").write_text("\n".join(report), encoding="utf-8")

    print()
    print(message)
    print(f"実行時間: {elapsed:.1f}秒")
    print("主な結果: result_report.md, rho_calibration.svg, sample_size_calibration.svg")


if __name__ == "__main__":
    main()
