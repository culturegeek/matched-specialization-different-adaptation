#!/usr/bin/env python3
"""実験II v0.2: 表現精度と誤差選択を結ぶ最小閉ループPoC。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


CONDITIONS = ("all_targets", "balanced_sampled", "fixed_propensity", "stabilization", "exploration")


@dataclass
class Config:
    train_size: int
    selector_size: int
    test_size: int
    source_dim: int
    bottleneck: int
    correlation: float
    observation_noise: float
    hidden_dim: int
    batch_size: int
    history_steps: int
    common_steps: int
    history_lambda_a: float
    history_lambda_b: float
    checkpoints: list[int]
    learning_rate: float
    weight_decay: float
    seeds: list[int]
    probe_ridge: float
    selector_interval: int
    selector_gain: float
    selector_max_bias: float
    conditions: list[str]


def parse_args():
    parser = argparse.ArgumentParser(description="現在の表現精度が次の誤差選択を変える閉ループを検証します。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="操作確認用")
    mode.add_argument("--full", action="store_true", help="主要確認用")
    parser.add_argument("--output-dir", default="closed_loop_poc_results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def make_config(full: bool) -> Config:
    if full:
        return Config(
            train_size=4096, selector_size=1024, test_size=2048,
            source_dim=3, bottleneck=3, correlation=0.35, observation_noise=0.02,
            hidden_dim=32, batch_size=256, history_steps=1200, common_steps=3000,
            history_lambda_a=0.9, history_lambda_b=0.1,
            checkpoints=[0, 20, 100, 300, 750, 1500, 3000],
            learning_rate=2e-3, weight_decay=1e-5, seeds=list(range(10)),
            probe_ridge=1e-3, selector_interval=10, selector_gain=1.5,
            selector_max_bias=0.40, conditions=list(CONDITIONS),
        )
    return Config(
        train_size=2048, selector_size=512, test_size=1024,
        source_dim=3, bottleneck=3, correlation=0.35, observation_noise=0.02,
        hidden_dim=24, batch_size=256, history_steps=400, common_steps=800,
        history_lambda_a=0.9, history_lambda_b=0.1,
        checkpoints=[0, 20, 100, 300, 800],
        learning_rate=2e-3, weight_decay=1e-5, seeds=list(range(4)),
        probe_ridge=1e-3, selector_interval=10, selector_gain=1.5,
        selector_max_bias=0.40, conditions=list(CONDITIONS),
    )


def choose_device(request: str) -> torch.device:
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが見つかりません。--device auto または cpu を使ってください。")
        return torch.device("cuda")
    if request == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(cfg.source_dim * 2, cfg.source_dim * 2))
    q, _ = np.linalg.qr(raw)

    def sample(n):
        a = rng.normal(size=(n, cfg.source_dim))
        private_b = rng.normal(size=(n, cfg.source_dim))
        b = cfg.correlation * a + math.sqrt(1 - cfg.correlation**2) * private_b
        x = np.concatenate([a, b], axis=1) @ q
        x += cfg.observation_noise * rng.normal(size=x.shape)
        return x.astype("float32"), a.astype("float32"), b.astype("float32")

    train = sample(cfg.train_size)
    selector = sample(cfg.selector_size)
    test = sample(cfg.test_size)

    standardized = []
    for train_value, selector_value, test_value in zip(train, selector, test):
        mean = train_value.mean(0, keepdims=True)
        std = train_value.std(0, keepdims=True) + 1e-7
        standardized.append(((train_value - mean) / std, (selector_value - mean) / std, (test_value - mean) / std))
    train_out = tuple(torch.from_numpy(v[0]) for v in standardized)
    selector_out = tuple(torch.from_numpy(v[1]) for v in standardized)
    test_out = tuple(torch.from_numpy(v[2]) for v in standardized)
    return train_out, selector_out, test_out


class SharedRepresentation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck), nn.Tanh(),
        )
        self.head_a = nn.Linear(bottleneck, output_dim)
        self.head_b = nn.Linear(bottleneck, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.head_a(z), self.head_b(z)


def train_weighted(model, optimizer, x, a, b, lambda_a):
    optimizer.zero_grad(set_to_none=True)
    _, pred_a, pred_b = model(x)
    d_a = torch.mean((pred_a - a) ** 2)
    d_b = torch.mean((pred_b - b) ** 2)
    loss = lambda_a * d_a + (1 - lambda_a) * d_b
    loss.backward()
    optimizer.step()


def train_selected(model, optimizer, x, a, b, select_a: bool):
    optimizer.zero_grad(set_to_none=True)
    _, pred_a, pred_b = model(x)
    loss = torch.mean((pred_a - a) ** 2) if select_a else torch.mean((pred_b - b) ** 2)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def decoder_metrics(model, data, device):
    x, a, b = (v.to(device) for v in data)
    z, pred_a, pred_b = model(x)
    d_a = (torch.mean((pred_a - a) ** 2) / (torch.var(a, unbiased=False) + 1e-12)).item()
    d_b = (torch.mean((pred_b - b) ** 2) / (torch.var(b, unbiased=False) + 1e-12)).item()
    index = (d_b - d_a) / (d_a + d_b + 1e-12)
    return z, d_a, d_b, index


@torch.no_grad()
def ridge_probe(z_train, y_train, z_test, y_test, ridge):
    ones_train = torch.ones((z_train.shape[0], 1), device=z_train.device)
    ones_test = torch.ones((z_test.shape[0], 1), device=z_test.device)
    design = torch.cat([z_train, ones_train], dim=1)
    design_test = torch.cat([z_test, ones_test], dim=1)
    eye = torch.eye(design.shape[1], device=z_train.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y_train)
    pred = design_test @ weights
    return (torch.mean((pred - y_test) ** 2) / (torch.var(y_test, unbiased=False) + 1e-12)).item()


@torch.no_grad()
def full_metrics(model, train, test, cfg, device):
    xtr, atr, btr = (v.to(device) for v in train)
    xte, ate, bte = (v.to(device) for v in test)
    ztr, _, _, _ = decoder_metrics(model, train, device)
    zte, d_a, d_b, index = decoder_metrics(model, test, device)
    probe_d_a = ridge_probe(ztr, atr, zte, ate, cfg.probe_ridge)
    probe_d_b = ridge_probe(ztr, btr, zte, bte, cfg.probe_ridge)
    probe_index = (probe_d_b - probe_d_a) / (probe_d_a + probe_d_b + 1e-12)
    return {
        "D_A": d_a, "D_B": d_b, "allocation_index": index,
        "probe_D_A": probe_d_a, "probe_D_B": probe_d_b, "probe_index": probe_index,
    }, zte.detach()


@torch.no_grad()
def representation_selector_index(model, selector_data, ridge, device):
    """共有表現を固定した線形プローブで、A/Bの復号精度差を測る。"""
    x, a, b = (value.to(device) for value in selector_data)
    z = model.encoder(x)
    split = z.shape[0] // 2
    d_a = ridge_probe(z[:split], a[:split], z[split:], a[split:], ridge)
    d_b = ridge_probe(z[:split], b[:split], z[split:], b[split:], ridge)
    return (d_b - d_a) / (d_a + d_b + 1e-12)


@torch.no_grad()
def linear_cka(z1, z2):
    z1 = z1 - z1.mean(0, keepdim=True)
    z2 = z2 - z2.mean(0, keepdim=True)
    cross = torch.sum((z1.T @ z2) ** 2)
    self1 = torch.sum((z1.T @ z1) ** 2)
    self2 = torch.sum((z2.T @ z2) ** 2)
    return (cross / torch.sqrt(self1 * self2 + 1e-20)).item()


def selector_probability(index: float, condition: str, cfg: Config, frozen_probability: float | None = None):
    if condition == "balanced_sampled":
        return 0.5
    if condition == "fixed_propensity":
        if frozen_probability is None:
            raise ValueError("fixed_propensityには初期確率が必要です。")
        return frozen_probability
    direction = 1.0 if condition == "stabilization" else -1.0
    probability = 0.5 + direction * cfg.selector_max_bias * math.tanh(cfg.selector_gain * index)
    return float(np.clip(probability, 0.5 - cfg.selector_max_bias, 0.5 + cfg.selector_max_bias))


def ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def run_seed(cfg: Config, seed: int, condition: str, device: torch.device):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(cfg, seed)
    train = tuple(v.to(device) for v in train_cpu)
    n = train[0].shape[0]

    base = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    initial_state = copy.deepcopy(base.state_dict())
    model_a = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model_b = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model_a.load_state_dict(initial_state)
    model_b.load_state_dict(initial_state)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    batch_rng = np.random.default_rng(100_000 + seed)
    for _ in range(cfg.history_steps):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(v[idx] for v in train)
        train_weighted(model_a, opt_a, *batch, cfg.history_lambda_a)
        train_weighted(model_b, opt_b, *batch, cfg.history_lambda_b)

    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    initial_index_a = representation_selector_index(model_a, selector_cpu, cfg.probe_ridge, device)
    initial_index_b = representation_selector_index(model_b, selector_cpu, cfg.probe_ridge, device)
    frozen_a = selector_probability(initial_index_a, "stabilization", cfg)
    frozen_b = selector_probability(initial_index_b, "stabilization", cfg)
    probability_a = 0.5 if condition == "all_targets" else selector_probability(initial_index_a, condition, cfg, frozen_a)
    probability_b = 0.5 if condition == "all_targets" else selector_probability(initial_index_b, condition, cfg, frozen_b)
    selected_a_count_a = selected_a_count_b = 0
    selector_updates = []
    rows = []
    checkpoints = set(cfg.checkpoints)
    selection_rng = np.random.default_rng(200_000 + seed)

    def record(step):
        metrics_a, z_a = full_metrics(model_a, train_cpu, test_cpu, cfg, device)
        metrics_b, z_b = full_metrics(model_b, train_cpu, test_cpu, cfg, device)
        denominator = max(step, 1)
        for history, metrics, probability, selected_count in (
            ("A_history", metrics_a, probability_a, selected_a_count_a),
            ("B_history", metrics_b, probability_b, selected_a_count_b),
        ):
            rows.append({
                "seed": seed, "condition": condition, "common_step": step, "history": history,
                **metrics, "selector_p_A": probability,
                "realized_A_fraction": selected_count / denominator if condition != "all_targets" else 0.5,
            })
        rows.append({
            "seed": seed, "condition": condition, "common_step": step, "history": "pair_CKA",
            "D_A": float("nan"), "D_B": float("nan"), "allocation_index": linear_cka(z_a, z_b),
            "probe_D_A": float("nan"), "probe_D_B": float("nan"), "probe_index": float("nan"),
            "selector_p_A": float("nan"), "realized_A_fraction": float("nan"),
        })

    record(0)
    for step in range(1, cfg.common_steps + 1):
        if condition in ("stabilization", "exploration") and (step == 1 or (step - 1) % cfg.selector_interval == 0):
            index_a = representation_selector_index(model_a, selector_cpu, cfg.probe_ridge, device)
            index_b = representation_selector_index(model_b, selector_cpu, cfg.probe_ridge, device)
            probability_a = selector_probability(index_a, condition, cfg)
            probability_b = selector_probability(index_b, condition, cfg)
            selector_updates.append((step, probability_a, probability_b))

        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        batch = tuple(v[idx] for v in train)
        if condition == "all_targets":
            train_weighted(model_a, opt_a, *batch, 0.5)
            train_weighted(model_b, opt_b, *batch, 0.5)
        else:
            # 対応する二つの履歴群には同じ乱数を使い、履歴以外の差を減らす。
            uniform = selection_rng.random()
            choose_a_for_a = uniform < probability_a
            choose_a_for_b = uniform < probability_b
            selected_a_count_a += int(choose_a_for_a)
            selected_a_count_b += int(choose_a_for_b)
            train_selected(model_a, opt_a, *batch, choose_a_for_a)
            train_selected(model_b, opt_b, *batch, choose_a_for_b)
        if step in checkpoints:
            record(step)
    return rows, selector_updates


def summarize(rows, cfg):
    summaries = []
    for condition in cfg.conditions:
        steps = sorted({int(r["common_step"]) for r in rows if r["condition"] == condition})
        for step in steps:
            arows = sorted([r for r in rows if r["condition"] == condition and r["common_step"] == step and r["history"] == "A_history"], key=lambda r: r["seed"])
            brows = sorted([r for r in rows if r["condition"] == condition and r["common_step"] == step and r["history"] == "B_history"], key=lambda r: r["seed"])
            crows = [r for r in rows if r["condition"] == condition and r["common_step"] == step and r["history"] == "pair_CKA"]
            gap = [a["allocation_index"] - b["allocation_index"] for a, b in zip(arows, brows)]
            probe_gap = [a["probe_index"] - b["probe_index"] for a, b in zip(arows, brows)]
            gmean, glo, ghi = ci95(gap)
            pmean, plo, phi = ci95(probe_gap)
            summaries.append({
                "condition": condition, "common_step": step,
                "A_index_mean": float(np.mean([r["allocation_index"] for r in arows])),
                "B_index_mean": float(np.mean([r["allocation_index"] for r in brows])),
                "paired_gap_mean": gmean, "paired_gap_ci_low": glo, "paired_gap_ci_high": ghi,
                "probe_gap_mean": pmean, "probe_gap_ci_low": plo, "probe_gap_ci_high": phi,
                "pair_CKA_mean": float(np.mean([r["allocation_index"] for r in crows])),
                "A_history_p_A_mean": float(np.mean([r["selector_p_A"] for r in arows])),
                "B_history_p_A_mean": float(np.mean([r["selector_p_A"] for r in brows])),
                "A_history_realized_A_mean": float(np.mean([r["realized_A_fraction"] for r in arows])),
                "B_history_realized_A_mean": float(np.mean([r["realized_A_fraction"] for r in brows])),
            })
    return summaries


def trapezoid_auc(items, key):
    items = sorted(items, key=lambda row: row["common_step"])
    xs = np.asarray([row["common_step"] for row in items], dtype=float)
    ys = np.asarray([abs(row[key]) for row in items], dtype=float)
    if xs[-1] == xs[0]:
        return float(ys[0])
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def evaluate(summaries, cfg):
    by_condition = {}
    for condition in cfg.conditions:
        items = [row for row in summaries if row["condition"] == condition]
        start = min(items, key=lambda row: row["common_step"])
        final = max(items, key=lambda row: row["common_step"])
        initial_gap = start["paired_gap_mean"]
        final_gap = final["paired_gap_mean"]
        initial_probe = start["probe_gap_mean"]
        final_probe = final["probe_gap_mean"]
        by_condition[condition] = {
            "initial_gap": initial_gap,
            "final_gap": final_gap,
            "behavioral_retention": abs(final_gap) / (abs(initial_gap) + 1e-12),
            "behavioral_ci_excludes_zero": bool(final["paired_gap_ci_low"] > 0 or final["paired_gap_ci_high"] < 0),
            "probe_retention": abs(final_probe) / (abs(initial_probe) + 1e-12),
            "probe_ci_excludes_zero": bool(final["probe_gap_ci_low"] > 0 or final["probe_gap_ci_high"] < 0),
            "gap_auc": trapezoid_auc(items, "paired_gap_mean"),
            "probe_gap_auc": trapezoid_auc(items, "probe_gap_mean"),
            "final_pair_CKA": final["pair_CKA_mean"],
            "final_selector_gap": final["A_history_p_A_mean"] - final["B_history_p_A_mean"],
        }
    all_targets = by_condition["all_targets"]
    stabilization = by_condition["stabilization"]
    exploration = by_condition["exploration"]
    fixed = by_condition["fixed_propensity"]
    gates = {
        "history_manipulation": abs(all_targets["initial_gap"]) > 0.5,
        # v0.1で持続候補とした残存率0.20を下回れば、負の基準結果を再現したと判定する。
        "open_loop_replicates_erasure": all_targets["behavioral_retention"] < 0.20,
        "stabilization_preserves_history": stabilization["behavioral_retention"] > 0.20 and stabilization["behavioral_ci_excludes_zero"],
        "exploration_reduces_history_relative_to_stabilization": exploration["gap_auc"] < 0.70 * stabilization["gap_auc"],
    }
    diagnostics = {
        "dynamic_to_fixed_auc_ratio": stabilization["gap_auc"] / (fixed["gap_auc"] + 1e-12),
        "stabilization_minus_fixed_final_retention": stabilization["behavioral_retention"] - fixed["behavioral_retention"],
    }
    return {"gates": gates, "diagnostics": diagnostics, "by_condition": by_condition}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path, summaries, cfg):
    width, height = 900, 560
    left, right, top, bottom = 85, 35, 45, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = {"all_targets":"#455A64", "balanced_sampled":"#78909C", "fixed_propensity":"#F9A825", "stabilization":"#C62828", "exploration":"#1565C0"}
    def sx(x): return left + plot_width * x / cfg.common_steps
    def sy(y): return top + plot_height * (1 - (y + 2.0) / 4.0)
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#263238}.grid{stroke:#CFD8DC}.axis{stroke:#455A64;stroke-width:1.5}</style>', '<text x="450" y="27" text-anchor="middle" font-size="20" font-weight="bold">History gap under error-selection policies</text>']
    for value in [-2, -1, 0, 1, 2]:
        y = sy(value)
        output += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" class="grid"/>', f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-size="13">{value}</text>']
    output += [f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" class="axis"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" class="axis"/>', f'<text x="{left+plot_width/2}" y="{height-22}" text-anchor="middle" font-size="16">common learning step</text>', f'<text x="18" y="{top+plot_height/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_height/2})" font-size="16">paired allocation-index gap</text>']
    for index, condition in enumerate(cfg.conditions):
        values = [row for row in summaries if row["condition"] == condition]
        color = colors[condition]
        points = " ".join(f'{sx(row["common_step"]):.1f},{sy(row["paired_gap_mean"]):.1f}' for row in values)
        output.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in values:
            output.append(f'<circle cx="{sx(row["common_step"]):.1f}" cy="{sy(row["paired_gap_mean"]):.1f}" r="3.5" fill="{color}"/>')
        legend_y = top + 18 + index * 23
        output += [f'<line x1="{left+plot_width-205}" y1="{legend_y}" x2="{left+plot_width-170}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>', f'<text x="{left+plot_width-162}" y="{legend_y+5}" font-size="13">{condition}</text>']
    output.append('</svg>')
    path.write_text("\n".join(output), encoding="utf-8")


def make_report(evaluation, cfg, elapsed, device_name):
    lines = [
        "# 実験II v0.2・閉ループ誤差選択PoC 実行結果", "",
        "## 事前判定", "",
    ]
    for gate, passed in evaluation["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {gate}")
    lines += ["", "## 条件別結果", "", "| 条件 | 最終残存率 | 履歴差AUC | 最終CKA | 最終選択確率差 |", "|---|---:|---:|---:|---:|"]
    for condition in cfg.conditions:
        value = evaluation["by_condition"][condition]
        lines.append(f"| {condition} | {value['behavioral_retention']:.3f} | {value['gap_auc']:.3f} | {value['final_pair_CKA']:.4f} | {value['final_selector_gap']:.3f} |")
    lines += [
        "", "## 読み方", "",
        "all_targets は実験II v0.1の負の結果を再現する基準です。balanced_sampled は一度に一方の誤差だけを学ぶこと自体の影響を測ります。",
        "fixed_propensity は後半開始時の選択確率を固定し、継続する外部配分の効果を測ります。stabilization と exploration だけが現在の表現精度から選択確率を更新します。",
        "", "## 固定配分との差", "",
        f"- 安定化型／固定配分の履歴差AUC比: {evaluation['diagnostics']['dynamic_to_fixed_auc_ratio']:.3f}",
        f"- 安定化型－固定配分の最終残存率差: {evaluation['diagnostics']['stabilization_minus_fixed_final_retention']:.3f}",
        "この差が小さい場合、履歴の持続は確認できても、選択確率を継続更新することの上乗せ効果は小さいと解釈します。",
        "", "## 限界", "",
        "本実験が選択するのは観測対象そのものではなく、取得済みデータのうちどちらの差異を誤差として学習に入れるかです。",
        "選択指標は正解ラベルを使う線形プローブで測り、系自身が精度推定を学習する機構は含みません。",
        "人間の発達、性格、志向性、準安定レジームを直接検証するものではありません。",
        "", f"- 実行デバイス: {device_name}", f"- 実行時間: {elapsed:.1f}秒",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = make_config(full=args.full)
    device = choose_device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"実験II v0.2・閉ループ誤差選択PoCを開始します: {device_name}")
    start = time.time()
    rows = []
    selector_rows = []
    total = len(cfg.seeds) * len(cfg.conditions)
    completed = 0
    for condition in cfg.conditions:
        for seed in cfg.seeds:
            seed_rows, selector_updates = run_seed(cfg, seed, condition, device)
            rows.extend(seed_rows)
            selector_rows.extend({"seed": seed, "condition": condition, "step": step, "A_history_p_A": probability_a, "B_history_p_A": probability_b} for step, probability_a, probability_b in selector_updates)
            completed += 1
            print(f"  進捗 {completed}/{total}: {condition}, seed={seed}")
    summaries = summarize(rows, cfg)
    evaluation = evaluate(summaries, cfg)
    elapsed = time.time() - start
    write_csv(output_dir / "raw_timecourse.csv", rows)
    write_csv(output_dir / "summary_timecourse.csv", summaries)
    if selector_rows:
        write_csv(output_dir / "selector_updates.csv", selector_rows)
    write_svg(output_dir / "history_gap.svg", summaries, cfg)
    metadata = {
        "config": asdict(cfg), "device": device_name, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "python": sys.version,
        "platform": platform.platform(), "runtime_seconds": elapsed, "evaluation": evaluation,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = make_report(evaluation, cfg, elapsed, device_name)
    (output_dir / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
