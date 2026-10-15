#!/usr/bin/env python3
"""実験II v0.7 本実験: 内点二枝の吸引域 — 摂動後の復帰・転換の確認実験。

位置づけ
--------
v0.7 パイロット（seed 1100–1105、水準探索）と v0.7.1 機構監査（Adam の役割の切り分け）を
経た確認実験である。判定条件は本ファイルの PREREGISTERED_GATES に実行前に固定した。
seed 1200–1239 は既存のどの実験とも重複しない。

設計（2026-08-16 に三者で確定）
------------------------------
- 容量2・安定化型・m=0.40。基底走行 8000 ステップ（対称事前学習 25 を含む前段）。
- 分岐点でモデルと Adam の状態を複製し、K ∈ {0, 800, 1600, 3200} の反対側更新を与える。
- 全条件で摂動終了後 8000 ステップの回復期間（回復時間の交絡を排除）。
- 乱数は三系列に分離する（v0.7.1 の絶対時刻参照を修正）:
  (1) 摂動期用の共通ミニバッチ列（摂動ステップ 1..K で添字づけ、全 K で共有）
  (2) 回復期用の共通ミニバッチ列（回復相対時刻 1..8000 で添字づけ、全 K・全条件で共有）
  (3) 回復期用の共通選択乱数列（同上）
  これにより、全 K が同じ回復相対時刻で同じ乱数を使う。
- Adam keep を主条件とし、K=1600 のみ Adam reset を追加（機構診断用）。
- 記録: プローブ s / D_A / D_B、デコーダ D_A / D_B、エンコーダ・両ヘッドの重み距離、
  潜在表現の線形CKA（分岐点比）、Adam モーメントノルム、選択確率。

用語
----
- 転換: 回復終了時（rel=8000）の s の符号が分岐時の符号と異なる。
- 復帰: 同符号。
- 回復時間: 摂動終了後、s の符号が分岐時の符号へ最初に戻った記録時点（復帰個体のみ、
  記録格子上の上界）。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from history_closed_loop_poc import (
    SharedRepresentation, choose_device, ci95, linear_cka, make_config, make_dataset,
    representation_selector_index, selector_probability, set_seed,
    train_selected, train_weighted, full_metrics,
)

PRETRAIN_STEPS = 25
BASE_STEPS = 8000
RECOVERY_STEPS = 8000
BOTTLENECK = 2
SELECTOR_MAX_BIAS = 0.40
K_LEVELS = (0, 800, 1600, 3200)
RESET_K = 1600
REC_MARKS = (0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 4800, 6400, 8000)

PREREGISTERED_GATES = """
統計単位: seed（各 seed 1 モデル、40 独立 seed）。区間は ci95。
転換 = rel8000 の s の符号が分岐時と異なる。主条件は Adam keep。
得意側／不得意側の誤差ラベルは分岐時の符号 pre_sign で固定する。

G1. 後期定常性（K=0, keep）
 1 control_spontaneous_switch_below_0_10   : 自発転換率 < 0.10
 2 control_index_stationary                : |s| の rel0→rel8000 変化の95%区間が (-0.03, +0.03)
 3 control_errors_stationary               : 得意側・不得意側誤差の同変化の95%区間がともに (-0.05, +0.05)

G2. 有限の吸引域（keep）
 4 switch_gap_strong_vs_control            : 転換率(K=3200) − 転換率(K=0) ≥ 0.50
 5 switch_rate_nondecreasing_in_K          : 転換率が K に対して非減少（許容 0.05）
 6 weak_returns_strong_switches            : 転換率(K=800) ≤ 0.40 かつ 転換率(K=3200) ≥ 0.60

G3. 枝の対称性（keep）
 7 returned_recover_pre_level              : 復帰個体の |s_T|/分岐時|s| の平均が (0.75, 1.15)
 8 switched_reach_opposite_interior        : 転換個体の90%以上が |s_T| ∈ (0.20, 0.70)、
                                             かつ |s_T|/分岐時|s| の平均が (0.75, 1.25)

G4. 表面指標の不十分性（keep）
 9 pert_end_surface_similar_across_K       : rel0 の（分岐方向へ揃えた）s の K∈{800,1600,3200} 間の
                                             平均の最大差 < 0.03、かつ得意側・不得意側誤差の
                                             平均の最大差 < 0.05
10 outcomes_differ_nonetheless             : 転換率(K=3200) − 転換率(K=800) ≥ 0.30

G5. 機構診断
11 head_distance_predicts_within_K1600     : K=1600 keep の内部で、rel0 の不得意ヘッド距離が
                                             転換を予測する（AUC ≥ 0.70）。転換・復帰の一方が
                                             5 seed 未満なら N/A とし判定しない
12 adam_agreement_at_K1600                 : K=1600 の keep/reset 帰結一致率 ≥ 0.80。
                                             回復時間差は診断量として報告のみ

A〜G5 は別個の仮説群であり、合計PASS数では評価しない。主判定は G1・G2。
G5-11 が不成立でも G1〜G4 の結論には影響しない（機構の候補指標の検証である）。
"""


FIELDS = ["row_key", "seed", "K", "adam_mode", "rel_step",
          "probe_index", "probe_D_A", "probe_D_B", "decoder_D_A", "decoder_D_B",
          "selector_p_A",
          "wdist_encoder_vs_branch", "wdist_head_fav_vs_branch", "wdist_head_neg_vs_branch",
          "cka_vs_branch", "adam_m1_norm", "adam_m2_norm",
          "pre_sign", "pre_abs_index"]


def build_model(cfg, train, device):
    return SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)


def flat_params(model, prefix):
    return torch.cat([p.detach().flatten() for n, p in model.named_parameters() if n.startswith(prefix)])


def adam_norms(opt):
    m1 = m2 = 0.0
    for state in opt.state.values():
        if "exp_avg" in state:
            m1 += float(state["exp_avg"].norm() ** 2)
            m2 += float(state["exp_avg_sq"].norm() ** 2)
    return m1 ** 0.5, m2 ** 0.5


@torch.no_grad()
def latent(model, x):
    return model.encoder(x)


def run_seed(cfg, seed, device, writer, handle):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(cfg, seed)
    train = tuple(v.to(device) for v in train_cpu)
    x_test = test_cpu[0].to(device)
    n = train[0].shape[0]

    # ---- 基底走行
    model = build_model(cfg, train, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(700_000 + seed)
    for _ in range(PRETRAIN_STEPS):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(model, opt, *(v[idx] for v in train), 0.5)
    batch_rng = np.random.default_rng(730_000 + seed)
    sel_rng = np.random.default_rng(740_000 + seed)
    p = selector_probability(
        representation_selector_index(model, selector_cpu, cfg.probe_ridge, device), "stabilization", cfg)
    for step in range(1, BASE_STEPS + 1):
        if step == 1 or (step - 1) % cfg.selector_interval == 0:
            p = selector_probability(
                representation_selector_index(model, selector_cpu, cfg.probe_ridge, device), "stabilization", cfg)
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_selected(model, opt, *(v[idx] for v in train), sel_rng.random() < p)

    # ---- 分岐点の基準量
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    pre_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
    pre_sign = 1.0 if pre_index > 0 else -1.0
    pre_abs = abs(pre_index)
    ref_enc = flat_params(model, "encoder")
    fav_prefix = "head_a" if pre_sign > 0 else "head_b"
    neg_prefix = "head_b" if pre_sign > 0 else "head_a"
    ref_fav = flat_params(model, fav_prefix)
    ref_neg = flat_params(model, neg_prefix)
    z_ref = latent(model, x_test)

    # ---- 三系列の共通乱数（摂動期は摂動ステップで、回復期は回復相対時刻で添字づけ）
    pert_batch = np.random.default_rng(810_000 + seed).integers(0, n, size=(max(K_LEVELS), cfg.batch_size))
    rec_batch = np.random.default_rng(820_000 + seed).integers(0, n, size=(RECOVERY_STEPS, cfg.batch_size))
    rec_uniform = np.random.default_rng(830_000 + seed).random(RECOVERY_STEPS)

    def record(m, o, K, adam_mode, rel, p):
        metrics, _ = full_metrics(m, train_cpu, test_cpu, cfg, device)
        enc = flat_params(m, "encoder")
        m1, m2 = adam_norms(o)
        writer.writerow({
            "row_key": f"{seed}|{K}|{adam_mode}|{rel}",
            "seed": seed, "K": K, "adam_mode": adam_mode, "rel_step": rel,
            "probe_index": metrics["probe_index"],
            "probe_D_A": metrics["probe_D_A"], "probe_D_B": metrics["probe_D_B"],
            "decoder_D_A": metrics["D_A"], "decoder_D_B": metrics["D_B"],
            "selector_p_A": p,
            "wdist_encoder_vs_branch": float((enc - ref_enc).norm()),
            "wdist_head_fav_vs_branch": float((flat_params(m, fav_prefix) - ref_fav).norm()),
            "wdist_head_neg_vs_branch": float((flat_params(m, neg_prefix) - ref_neg).norm()),
            "cka_vs_branch": linear_cka(latent(m, x_test), z_ref),
            "adam_m1_norm": m1, "adam_m2_norm": m2,
            "pre_sign": pre_sign, "pre_abs_index": pre_abs,
        })

    forced_choose_a = pre_sign < 0
    for K in K_LEVELS:
        # ---- 摂動（共通摂動系列 1..K）
        pert = build_model(cfg, train, device)
        pert.load_state_dict(model_state)
        popt = torch.optim.Adam(pert.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        popt.load_state_dict(copy.deepcopy(opt_state))
        for t in range(1, K + 1):
            idx = torch.from_numpy(pert_batch[t - 1]).to(device)
            train_selected(pert, popt, *(v[idx] for v in train), forced_choose_a)
        pert_model_state = copy.deepcopy(pert.state_dict())
        pert_opt_state = copy.deepcopy(popt.state_dict())

        modes = ("keep", "reset") if K == RESET_K else ("keep",)
        for adam_mode in modes:
            m = build_model(cfg, train, device)
            m.load_state_dict(copy.deepcopy(pert_model_state))
            o = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            if adam_mode == "keep":
                o.load_state_dict(copy.deepcopy(pert_opt_state))
            p = selector_probability(
                representation_selector_index(m, selector_cpu, cfg.probe_ridge, device), "stabilization", cfg)
            record(m, o, K, adam_mode, 0, p)
            marks = set(REC_MARKS)
            for rel in range(1, RECOVERY_STEPS + 1):
                if rel == 1 or (rel - 1) % cfg.selector_interval == 0:
                    p = selector_probability(
                        representation_selector_index(m, selector_cpu, cfg.probe_ridge, device),
                        "stabilization", cfg)
                idx = torch.from_numpy(rec_batch[rel - 1]).to(device)
                train_selected(m, o, *(v[idx] for v in train), rec_uniform[rel - 1] < p)
                if rel in marks:
                    record(m, o, K, adam_mode, rel, p)
            handle.flush()


# ---------------------------------------------------------------- 集計

def load(paths):
    seen, rows, duplicates, conflicts = {}, [], 0, []
    for p in paths:
        with open(p, encoding="utf-8") as h:
            for r in csv.DictReader(h):
                key = r["row_key"]
                parsed = {"row_key": key, "seed": int(r["seed"]), "K": int(r["K"]),
                          "adam_mode": r["adam_mode"], "rel_step": int(r["rel_step"]),
                          "probe_index": float(r["probe_index"]),
                          "probe_D_A": float(r["probe_D_A"]), "probe_D_B": float(r["probe_D_B"]),
                          "wdist_head_neg_vs_branch": float(r["wdist_head_neg_vs_branch"]),
                          "wdist_encoder_vs_branch": float(r["wdist_encoder_vs_branch"]),
                          "cka_vs_branch": float(r["cka_vs_branch"]),
                          "pre_sign": float(r["pre_sign"]), "pre_abs_index": float(r["pre_abs_index"])}
                if key in seen:
                    duplicates += 1
                    if seen[key]["probe_index"] != parsed["probe_index"]:
                        conflicts.append(key)
                    continue
                seen[key] = parsed
                rows.append(parsed)
    return rows, {"duplicate_rows": duplicates, "n_conflicts": len(conflicts), "n_rows": len(rows)}


def validate(rows, expected_seeds):
    problems = []
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) != expected_seeds:
        problems.append(f"seed 数が {len(seeds)}、期待は {expected_seeds}")
    want = set()
    for K in K_LEVELS:
        for mode in (("keep", "reset") if K == RESET_K else ("keep",)):
            for rel in REC_MARKS:
                want.add((K, mode, rel))
    for seed in seeds:
        got = {(r["K"], r["adam_mode"], r["rel_step"]) for r in rows if r["seed"] == seed}
        missing = want - got
        if missing:
            problems.append(f"seed {seed}: {len(missing)} 件欠損")
    return problems


def pick(rows, K, mode, rel):
    return sorted([r for r in rows if r["K"] == K and r["adam_mode"] == mode and r["rel_step"] == rel],
                  key=lambda r: r["seed"])


def aligned(r):
    return r["probe_index"] * r["pre_sign"]


def fav_err(r):
    return r["probe_D_A"] if r["pre_sign"] > 0 else r["probe_D_B"]


def neg_err(r):
    return r["probe_D_B"] if r["pre_sign"] > 0 else r["probe_D_A"]


def switched(r):
    return aligned(r) < 0


def interval(values):
    m, lo, hi = ci95(np.asarray(values, dtype=float))
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(values)}


def auc(scores_pos, scores_neg):
    if not scores_pos or not scores_neg:
        return float("nan")
    wins = sum((sp > sn) + 0.5 * (sp == sn) for sp in scores_pos for sn in scores_neg)
    return float(wins / (len(scores_pos) * len(scores_neg)))


def summarize(rows):
    out = {"switch_rates": {}, "timecourse": {}, "gates": {}, "diagnostics": {}}
    term = {K: pick(rows, K, "keep", RECOVERY_STEPS) for K in K_LEVELS}
    rates = {K: float(np.mean([switched(r) for r in term[K]])) for K in K_LEVELS}
    out["switch_rates"] = {str(K): rates[K] for K in K_LEVELS}

    for K in K_LEVELS:
        out["timecourse"][str(K)] = {
            str(rel): {"aligned_s_mean": float(np.mean([aligned(r) for r in pick(rows, K, "keep", rel)])) }
            for rel in REC_MARKS if pick(rows, K, "keep", rel)
        }

    g = out["gates"]
    # G1
    c0 = pick(rows, 0, "keep", 0)
    cT = pick(rows, 0, "keep", RECOVERY_STEPS)
    g["control_spontaneous_switch_below_0_10"] = bool(rates[0] < 0.10)
    di = interval([abs(b["probe_index"]) - abs(a["probe_index"]) for a, b in zip(c0, cT)])
    df = interval([fav_err(b) - fav_err(a) for a, b in zip(c0, cT)])
    dn = interval([neg_err(b) - neg_err(a) for a, b in zip(c0, cT)])
    g["control_index_stationary"] = bool(di["ci_low"] > -0.03 and di["ci_high"] < 0.03)
    g["control_errors_stationary"] = bool(df["ci_low"] > -0.05 and df["ci_high"] < 0.05
                                          and dn["ci_low"] > -0.05 and dn["ci_high"] < 0.05)
    # G2
    g["switch_gap_strong_vs_control"] = bool(rates[3200] - rates[0] >= 0.50)
    ordered = [rates[K] for K in K_LEVELS]
    g["switch_rate_nondecreasing_in_K"] = bool(all(b + 0.05 >= a for a, b in zip(ordered, ordered[1:])))
    g["weak_returns_strong_switches"] = bool(rates[800] <= 0.40 and rates[3200] >= 0.60)
    # G3
    returned = [r for K in (800, 1600, 3200) for r in term[K] if not switched(r)]
    swit = [r for K in (800, 1600, 3200) for r in term[K] if switched(r)]
    ret_ratio = [abs(r["probe_index"]) / r["pre_abs_index"] for r in returned]
    sw_ratio = [abs(r["probe_index"]) / r["pre_abs_index"] for r in swit]
    sw_interior = [0.20 < abs(r["probe_index"]) < 0.70 for r in swit]
    g["returned_recover_pre_level"] = bool(returned and 0.75 < float(np.mean(ret_ratio)) < 1.15)
    g["switched_reach_opposite_interior"] = bool(swit and float(np.mean(sw_interior)) >= 0.90
                                                 and 0.75 < float(np.mean(sw_ratio)) < 1.25)
    # G4
    s0 = {K: float(np.mean([aligned(r) for r in pick(rows, K, "keep", 0)])) for K in (800, 1600, 3200)}
    f0 = {K: float(np.mean([fav_err(r) for r in pick(rows, K, "keep", 0)])) for K in (800, 1600, 3200)}
    n0 = {K: float(np.mean([neg_err(r) for r in pick(rows, K, "keep", 0)])) for K in (800, 1600, 3200)}
    spread = lambda d: max(d.values()) - min(d.values())
    g["pert_end_surface_similar_across_K"] = bool(spread(s0) < 0.03 and spread(f0) < 0.05 and spread(n0) < 0.05)
    g["outcomes_differ_nonetheless"] = bool(rates[3200] - rates[800] >= 0.30)
    # G5
    k16 = pick(rows, 1600, "keep", 0)
    k16T = {r["seed"]: switched(r) for r in term[1600]}
    pos = [r["wdist_head_neg_vs_branch"] for r in k16 if k16T[r["seed"]]]
    neg = [r["wdist_head_neg_vs_branch"] for r in k16 if not k16T[r["seed"]]]
    if len(pos) >= 5 and len(neg) >= 5:
        a = auc(pos, neg)
        g["head_distance_predicts_within_K1600"] = bool(a >= 0.70)
    else:
        a = auc(pos, neg)
        g["head_distance_predicts_within_K1600"] = None  # N/A
    reset_T = {r["seed"]: switched(r) for r in pick(rows, 1600, "reset", RECOVERY_STEPS)}
    agree = [k16T[s] == reset_T[s] for s in k16T if s in reset_T]
    g["adam_agreement_at_K1600"] = bool(float(np.mean(agree)) >= 0.80) if agree else None

    def recovery_times(K, mode):
        times = []
        by = {}
        for r in rows:
            if r["K"] == K and r["adam_mode"] == mode:
                by.setdefault(r["seed"], {})[r["rel_step"]] = aligned(r)
        for s, d in by.items():
            if d.get(RECOVERY_STEPS, -1) < 0:
                continue
            t = next((m for m in sorted(d) if m > 0 and d[m] > 0), None)
            times.append(t)
        return times

    out["diagnostics"] = {
        "pert_end_surface": {"aligned_s": s0, "favored_err": f0, "neglected_err": n0},
        "control_changes": {"abs_index": di, "favored": df, "neglected": dn},
        "returned_ratio": interval(ret_ratio) if ret_ratio else None,
        "switched_ratio": interval(sw_ratio) if sw_ratio else None,
        "switched_interior_rate": float(np.mean(sw_interior)) if swit else None,
        "auc_head_distance_K1600": a,
        "n_switch_K1600": len(pos), "n_return_K1600": len(neg),
        "adam_agreement_rate_K1600": float(np.mean(agree)) if agree else None,
        "recovery_times_keep": {str(K): recovery_times(K, "keep") for K in (800, 1600, 3200)},
        "recovery_times_reset_K1600": recovery_times(1600, "reset"),
        "sign_positive_rate_all_arms": float(np.mean([r["probe_index"] > 0 for K in K_LEVELS for r in term[K]])),
    }
    return out


def provenance():
    source = Path(__file__).read_text(encoding="utf-8")
    segment = source[source.index("PRETRAIN_STEPS = 25"):
                     source.index("# ---------------------------------------------------------------- 集計")]
    segment = re.sub(r'PREREGISTERED_GATES = """.*?"""', "PREREGISTERED_GATES = <doc>", segment, flags=re.S)
    return {"script_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "execution_core_sha256": hashlib.sha256(segment.encode()).hexdigest(),
            "constants": {"K_LEVELS": list(K_LEVELS), "RESET_K": RESET_K,
                          "BASE_STEPS": BASE_STEPS, "RECOVERY_STEPS": RECOVERY_STEPS,
                          "BOTTLENECK": BOTTLENECK, "SELECTOR_MAX_BIAS": SELECTOR_MAX_BIAS,
                          "REC_MARKS": list(REC_MARKS)},
            "preregistered_gates": PREREGISTERED_GATES}


def completed_seeds(path: Path):
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if int(r["K"]) == K_LEVELS[-1] and r["adam_mode"] == "keep" and int(r["rel_step"]) == RECOVERY_STEPS:
                done.add(int(r["seed"]))
    return done


def main():
    ap = argparse.ArgumentParser(description="v0.7 本実験: 内点二枝の吸引域（事前登録済み）")
    ap.add_argument("--output-dir", default="perturbation_main_results")
    ap.add_argument("--seed-start", type=int, default=1200)
    ap.add_argument("--seed-count", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--summarize", nargs="+", default=None)
    ap.add_argument("--expect-seeds", type=int, default=40)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.summarize:
        rows, load_report = load(a.summarize)
        problems = validate(rows, a.expect_seeds)
        if load_report["n_conflicts"]:
            problems.append(f"row_key の矛盾 {load_report['n_conflicts']} 件")
        if problems and not a.allow_incomplete:
            print("完全性検査に失敗したため集計を中止します:")
            for x in problems[:20]:
                print("  -", x)
            raise SystemExit(2)
        result = summarize(rows)
        result["completeness"] = {"problems": problems, "load_report": load_report}
        result["provenance"] = provenance()
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 実験II v0.7 本実験 結果", "",
                 "転換率: " + "  ".join(f"K={k}: {v:.3f}" for k, v in result["switch_rates"].items()),
                 "", "## 事前判定", ""]
        for k, v in result["gates"].items():
            lines.append(f"- {'N/A' if v is None else ('PASS' if v else 'FAIL')}: {k}")
        (out / "result_report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    torch.set_num_threads(a.threads)
    device = choose_device("cpu")
    cfg = make_config(full=False)
    cfg.bottleneck = BOTTLENECK
    cfg.selector_max_bias = SELECTOR_MAX_BIAS

    (out / f"provenance_run_{a.seed_start}.json").write_text(
        json.dumps({**provenance(), "seeds": list(range(a.seed_start, a.seed_start + a.seed_count))},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    raw = out / f"main_{a.seed_start}_{a.seed_start + a.seed_count - 1}.csv"
    done = completed_seeds(raw)
    if done:
        print(f"再開: 完了済み {len(done)} seed をスキップします")
    new_file = not raw.exists() or raw.stat().st_size == 0
    with raw.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        seeds = list(range(a.seed_start, a.seed_start + a.seed_count))
        for i, seed in enumerate(seeds, 1):
            if seed in done:
                continue
            run_seed(cfg, seed, device, writer, handle)
            print(f"seed {i}/{len(seeds)} (seed={seed}) を完了", flush=True)
    print(f"生データ: {raw}")


if __name__ == "__main__":
    main()
