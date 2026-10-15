#!/usr/bin/env python3
"""実験II v0.8 本実験: 外部バイアス往復に対するヒステリシスと二枝の共存（事前登録済み）。

設計（2026-08-16 三者で確定・ChatGPT 承認済み）
----------------------------------------------
- seed 1400–1439（独立40 seed、既存集合と重複なし）
- 容量2・安定化型・m=0.40。基底走行8000ステップ。
- 主走査: h を 0 → -0.20 →（+0.20 まで上昇）→ 0 と往復。格子 0.025、滞在8000。
- 枝準備（v0.8.2 の対称方式）: 同じ基底から複製し、正枝に h=+0.20、負枝に h=-0.20 を
  同じ8000ステップ。両方 Adam リセット。両方厳密な h=0 で4000ステップ緩和。
  同じバッチ系列、選択乱数は鏡像化（正枝 u<p、負枝 (1-u)<p）。
- 固定条件: h ∈ {-0.05, -0.025, 0, +0.025, +0.05}、両枝から8000ステップ保持。
  準備に失敗した seed も保持試験を実行し、維持失敗として分母40に数える。
- 滞在500・2000の診断走査: 事前固定した seed 1400–1409 のみ、基底状態から独立に実行。
  乱数は水準ごとに固定し滞在ステップで添字づけるため、短い滞在は滞在8000の共通接頭辞。
- 枝分類: 各段階の終盤三時点（0.75T, 0.875T, T）の平均 s~ が +0.20 超で正枝、
  -0.20 未満で負枝、それ以外は未確定。
- 統計単位: seed。区間は固定乱数（seed 20260816）による10000反復の対応付きブートストラップ。

中断再開
--------
seed ごとの一時ファイル（tmp_seed_<seed>.csv）へ書き、全条件完了後に
seed_<seed>.csv へ改名する。再開時は正式ファイルの存在で完了を判定し、
残っている一時ファイルは削除して最初からやり直す。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from history_closed_loop_poc import (
    SharedRepresentation, choose_device, make_config, make_dataset,
    representation_selector_index, selector_probability, set_seed,
    train_selected, train_weighted, full_metrics,
)

PRETRAIN_STEPS = 25
BASE_STEPS = 8000
BOTTLENECK = 2
SELECTOR_MAX_BIAS = 0.40
H_STEP = 0.025
H_MAX = 0.20
MAIN_DWELL = 8000
DIAG_DWELLS = (500, 2000)
DIAG_SEEDS = tuple(range(1400, 1410))
PREP_H = 0.20
PREP_STEPS = 8000
RELAX_STEPS = 4000
HOLD_H = (-0.05, -0.025, 0.0, 0.025, 0.05)
HOLD_STEPS = 8000
BRANCH_THRESHOLD = 0.20
BOOT_SEED = 20260816
BOOT_N = 10000
IN_LEVEL_MARKS = lambda T: (T // 2, (3 * T) // 4, (7 * T) // 8, T)

PREREGISTERED_GATES = """
統計単位: seed（40）。区間は固定乱数 seed=20260816 による 10000 反復の
seed 単位対応付きブートストラップ（パーセンタイル法）。

枝分類: 各水準・各段階の終盤三時点（0.75T, 0.875T, T）の s~ の平均が
  +0.20 超 → 正枝、-0.20 未満 → 負枝、それ以外 → 未確定。

転換点・復帰点の規則:
  転換点 = 下降走査で分類が初めて「負枝」になった h。未確定の水準は飛ばし、
  次に確定した反対枝の水準を採る。走査端まで未確定または正枝のままなら打ち切り。
  復帰点 = 転換後、上昇走査（back 区間を含む）で分類が初めて「正枝」に戻った h。
  同じ規則で打ち切りを定義する。転換点差 = 復帰点 − 転換点。

主判定（三つ。合計数では評価しない）:
 1 prep_success_pos / prep_success_neg   : 緩和後の分類が正枝／負枝である seed が
                                           正負それぞれ 36/40 以上。
 2 coexistence_at_h0                     : h=0 の保持後、正枝開始で正枝のまま、
                                           負枝開始で負枝のままの seed が正負それぞれ
                                           36/40 以上。枝準備に失敗した seed は
                                           その枝の維持失敗として分母 40 に数える。
 3 hysteresis_at_slowest                 : 滞在8000で全 seed の転換・復帰が走査範囲内に
                                           観測され（打ち切り 0 件。1件でもあれば本判定は
                                           不成立とし主解析を成立させない）、かつ
                                           seed 別転換点差の平均の95%ブートストラップ
                                           区間下限が 0.025（格子幅）を上回る。

副判定:
 4 loop_area_positive                    : 滞在8000のループ面積（h ∈ [-0.175, 0] の各水準で
                                           下降時 s~ − 上昇時 s~ をとり 0.025 を乗じて総和）の
                                           95%区間下限が 0 を上回る。
 5 coexistence_range                     : h ∈ {-0.025, 0, +0.025} のうち正負両枝がともに
                                           36/40 以上維持される水準の範囲（報告値）。

診断量（合否にしない）:
 - 滞在 500 / 2000 / 8000 に伴う転換点差の変化（seed 1400–1409、事前固定）。
   速度依存の単調性は合否条件にしない。
 - 枝準備の緩和後水準、保持後水準の分布。
"""

FIELDS = ["row_key", "seed", "protocol", "branch", "leg", "dwell", "h_aligned",
          "level_index", "step_in_phase", "probe_index", "aligned_s", "selector_p_A",
          "pre_sign", "pre_abs_index"]


def sweep_levels():
    n = int(round(H_MAX / H_STEP))
    down = [(-i * H_STEP) for i in range(0, n + 1)]                    # 0 .. -0.20 (9)
    up = [-H_MAX + i * H_STEP for i in range(1, 2 * n + 1)]            # -0.175 .. +0.20 (16)
    back = [H_MAX - i * H_STEP for i in range(1, n + 1)]               # +0.175 .. 0 (8)
    return [("down", round(h, 4)) for h in down] + [("up", round(h, 4)) for h in up] \
         + [("back", round(h, 4)) for h in back]


def build_model(cfg, train, device):
    return SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)


def stream_rngs(seed, name, leg, h):
    """乱数キー。滞在時間を含めないため、短い滞在は長い滞在の共通接頭辞になる。"""
    code = int(round((h + 1.0) * 1000))
    leg_id = {"down": 1, "up": 2, "back": 3, "prep": 4, "relax": 5, "hold": 6}[leg]
    name_id = {"sweep": 1, "brancher": 2, "holder": 3}[name]
    base = 970_000_000 + seed * 100_000 + name_id * 10_000 + leg_id * 1_000 + code % 1_000
    return np.random.default_rng(base), np.random.default_rng(base + 11_000_000_000)


def run_seed(cfg, seed, device, path_tmp, path_final):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(cfg, seed)
    train = tuple(v.to(device) for v in train_cpu)
    n = train[0].shape[0]

    if path_tmp.exists():
        path_tmp.unlink()          # 中断の残骸は破棄して最初からやり直す
    handle = path_tmp.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()

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

    base_model = copy.deepcopy(model.state_dict())
    base_opt = copy.deepcopy(opt.state_dict())
    pre_index = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
    pre_sign = 1.0 if pre_index > 0 else -1.0
    pre_abs = abs(pre_index)

    def biased_probability(m, h_aligned):
        s_now = representation_selector_index(m, selector_cpu, cfg.probe_ridge, device)
        base = selector_probability(s_now, "stabilization", cfg)
        return float(np.clip(base + pre_sign * h_aligned, 0.02, 0.98))

    def record(m, protocol, branch, leg, dwell, h, li, sip, p):
        met, _ = full_metrics(m, train_cpu, test_cpu, cfg, device)
        writer.writerow({
            "row_key": f"{seed}|{protocol}|{branch}|{leg}|{dwell}|{h:+.3f}|{li}|{sip}",
            "seed": seed, "protocol": protocol, "branch": branch, "leg": leg,
            "dwell": dwell, "h_aligned": h, "level_index": li, "step_in_phase": sip,
            "probe_index": met["probe_index"], "aligned_s": met["probe_index"] * pre_sign,
            "selector_p_A": p, "pre_sign": pre_sign, "pre_abs_index": pre_abs,
        })

    def run_phase(m, o, protocol, branch, name, leg, dwell_label, h, li, steps):
        b_rng, u_rng = stream_rngs(seed, name, leg, h)
        marks = set(IN_LEVEL_MARKS(steps))
        p = 0.5
        for t in range(1, steps + 1):
            if t == 1 or (t - 1) % cfg.selector_interval == 0:
                p = biased_probability(m, h)
            idx = torch.from_numpy(b_rng.integers(0, n, size=cfg.batch_size)).to(device)
            u = u_rng.random()
            draw = u if branch != "neg" else (1.0 - u)     # 鏡像化（負枝のみ）
            train_selected(m, o, *(v[idx] for v in train), draw < p)
            if t in marks:
                record(m, protocol, branch, leg, dwell_label, h, li, t, p)

    # ---- 主走査（滞在8000）と診断走査（滞在500・2000、DIAG_SEEDS のみ）
    dwells = (MAIN_DWELL,) + (DIAG_DWELLS if seed in DIAG_SEEDS else ())
    for dwell in dwells:
        m = build_model(cfg, train, device)
        m.load_state_dict(copy.deepcopy(base_model))
        o = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        o.load_state_dict(copy.deepcopy(base_opt))
        for li, (leg, h) in enumerate(sweep_levels()):
            run_phase(m, o, "sweep", "main", "sweep", leg, dwell, h, li, dwell)
        handle.flush()

    # ---- 枝準備（対称）と固定 h 保持
    states = {}
    for branch, h_prep in (("pos", +PREP_H), ("neg", -PREP_H)):
        m = build_model(cfg, train, device)
        m.load_state_dict(copy.deepcopy(base_model))
        o = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        o.load_state_dict(copy.deepcopy(base_opt))
        run_phase(m, o, "prep", branch, "brancher", "prep", PREP_STEPS, h_prep, 0, PREP_STEPS)
        o = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)  # Adamリセット
        run_phase(m, o, "prep", branch, "brancher", "relax", RELAX_STEPS, 0.0, 1, RELAX_STEPS)
        states[branch] = (copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict()))
        handle.flush()

    for branch in ("pos", "neg"):
        ms, os_ = states[branch]
        for h in HOLD_H:
            m = build_model(cfg, train, device)
            m.load_state_dict(copy.deepcopy(ms))
            o = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            o.load_state_dict(copy.deepcopy(os_))
            run_phase(m, o, "hold", branch, "holder", "hold", HOLD_STEPS, h, 0, HOLD_STEPS)
        handle.flush()

    # ---- 完了記録 → 改名（原子的な完了マーク）
    writer.writerow({"row_key": f"{seed}|seed_complete|-|-|0|+0.000|0|0",
                     "seed": seed, "protocol": "seed_complete", "branch": "-", "leg": "-",
                     "dwell": 0, "h_aligned": 0.0, "level_index": 0, "step_in_phase": 0,
                     "probe_index": 0.0, "aligned_s": 0.0, "selector_p_A": 0.0,
                     "pre_sign": pre_sign, "pre_abs_index": pre_abs})
    handle.close()
    os.replace(path_tmp, path_final)


# ---------------------------------------------------------------- 集計

def expected_keys(seed):
    keys = set()
    for dwell in (MAIN_DWELL,) + (DIAG_DWELLS if seed in DIAG_SEEDS else ()):
        for li, (leg, h) in enumerate(sweep_levels()):
            for sip in IN_LEVEL_MARKS(dwell):
                keys.add(f"{seed}|sweep|main|{leg}|{dwell}|{h:+.3f}|{li}|{sip}")
    for branch in ("pos", "neg"):
        for sip in IN_LEVEL_MARKS(PREP_STEPS):
            keys.add(f"{seed}|prep|{branch}|prep|{PREP_STEPS}|{+PREP_H if branch=='pos' else -PREP_H:+.3f}|0|{sip}")
        for sip in IN_LEVEL_MARKS(RELAX_STEPS):
            keys.add(f"{seed}|prep|{branch}|relax|{RELAX_STEPS}|+0.000|1|{sip}")
        for h in HOLD_H:
            for sip in IN_LEVEL_MARKS(HOLD_STEPS):
                keys.add(f"{seed}|hold|{branch}|hold|{HOLD_STEPS}|{h:+.3f}|0|{sip}")
    keys.add(f"{seed}|seed_complete|-|-|0|+0.000|0|0")
    return keys


def load_dir(out):
    rows, problems = [], []
    seeds = []
    for f in sorted(out.glob("seed_*.csv")):
        seed = int(f.stem.split("_")[1])
        seeds.append(seed)
        with f.open(encoding="utf-8") as h:
            got = list(csv.DictReader(h))
        keys = [r["row_key"] for r in got]
        if len(keys) != len(set(keys)):
            problems.append(f"seed {seed}: row_key 重複")
        missing = expected_keys(seed) - set(keys)
        extra = set(keys) - expected_keys(seed)
        if missing:
            problems.append(f"seed {seed}: 欠損 {len(missing)} 件")
        if extra:
            problems.append(f"seed {seed}: 想定外 {len(extra)} 件")
        rows.extend(got)
    return rows, sorted(seeds), problems


def classify(x):
    if x > BRANCH_THRESHOLD:
        return "pos"
    if x < -BRANCH_THRESHOLD:
        return "neg"
    return "mid"


def tail_means(rows):
    """(seed, protocol, branch, leg, dwell, h, li) -> 終盤三時点平均の s~"""
    acc = {}
    for r in rows:
        if r["protocol"] == "seed_complete":
            continue
        T = int(r["dwell"])
        sip = int(r["step_in_phase"])
        if sip not in ((3 * T) // 4, (7 * T) // 8, T):
            continue
        k = (int(r["seed"]), r["protocol"], r["branch"], r["leg"], T,
             round(float(r["h_aligned"]), 4), int(r["level_index"]))
        acc.setdefault(k, []).append(float(r["aligned_s"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def sweep_points(tm, seed, dwell):
    """転換点・復帰点。事前登録した規則（未確定は飛ばす、端まで確定しなければ打ち切り）。"""
    levels = sweep_levels()
    h_switch = h_return = None
    for li, (leg, h) in enumerate(levels):
        v = tm.get((seed, "sweep", "main", leg, dwell, round(h, 4), li))
        if v is None:
            return None, None, True
        c = classify(v)
        if leg == "down" and c == "neg" and h_switch is None:
            h_switch = h
        if leg in ("up", "back") and h_switch is not None and c == "pos" and h_return is None:
            h_return = h
    censored = h_switch is None or h_return is None
    return h_switch, h_return, censored


def boot_ci(values, rng):
    values = np.asarray(values, dtype=float)
    stats = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOT_N)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(rows, seeds):
    tm = tail_means(rows)
    rng = np.random.default_rng(BOOT_SEED)
    out = {"gates": {}, "primary": {}, "secondary": {}, "diagnostics": {}}

    # 主判定1: 枝準備
    prep = {"pos": 0, "neg": 0}
    prep_ok = {}
    for s in seeds:
        for br in ("pos", "neg"):
            v = tm.get((s, "prep", br, "relax", RELAX_STEPS, 0.0, 1))
            ok = classify(v) == br if v is not None else False
            prep_ok[(s, br)] = ok
            prep[br] += ok
    out["gates"]["prep_success_pos"] = bool(prep["pos"] >= 36)
    out["gates"]["prep_success_neg"] = bool(prep["neg"] >= 36)
    out["primary"]["prep_success"] = {k: f"{v}/{len(seeds)}" for k, v in prep.items()}

    # 主判定2: h=0 の共存（準備失敗も失敗に数える）
    keep = {"pos": 0, "neg": 0}
    hold_table = {}
    for h in HOLD_H:
        hold_table[h] = {"pos": 0, "neg": 0}
        for s in seeds:
            for br in ("pos", "neg"):
                v = tm.get((s, "hold", br, "hold", HOLD_STEPS, round(h, 4), 0))
                ok = prep_ok[(s, br)] and v is not None and classify(v) == br
                hold_table[h][br] += ok
                if h == 0.0:
                    keep[br] += ok
    out["gates"]["coexistence_at_h0"] = bool(keep["pos"] >= 36 and keep["neg"] >= 36)
    out["primary"]["hold_maintenance"] = {f"{h:+.3f}": {k: f"{v}/{len(seeds)}" for k, v in d.items()}
                                          for h, d in hold_table.items()}

    # 主判定3: 滞在8000のヒステリシス
    gaps, censored_seeds, points = [], [], {}
    for s in seeds:
        hs, hr, cen = sweep_points(tm, s, MAIN_DWELL)
        points[s] = {"switch": hs, "return": hr, "censored": cen}
        if cen:
            censored_seeds.append(s)
        else:
            gaps.append(hr - hs)
    if gaps and not censored_seeds:
        lo, hi = boot_ci(gaps, rng)
        out["gates"]["hysteresis_at_slowest"] = bool(lo > H_STEP)
    else:
        lo = hi = float("nan")
        out["gates"]["hysteresis_at_slowest"] = False
    out["primary"]["hysteresis"] = {
        "n_censored": len(censored_seeds), "censored_seeds": censored_seeds,
        "gap_mean": float(np.mean(gaps)) if gaps else None,
        "gap_ci95": [lo, hi], "gap_min": float(min(gaps)) if gaps else None,
        "gap_max": float(max(gaps)) if gaps else None,
        "switch_points": {str(s): points[s]["switch"] for s in seeds},
        "return_points": {str(s): points[s]["return"] for s in seeds},
    }

    # 副判定4: ループ面積（h ∈ [-0.175, 0]、下降 − 上昇）
    areas = []
    levels = sweep_levels()
    down_idx = {round(h, 4): li for li, (leg, h) in enumerate(levels) if leg == "down"}
    up_idx = {round(h, 4): li for li, (leg, h) in enumerate(levels) if leg == "up"}
    for s in seeds:
        a = 0.0
        valid = True
        for h in [round(-i * H_STEP, 4) for i in range(0, 8)]:      # 0 .. -0.175
            vd = tm.get((s, "sweep", "main", "down", MAIN_DWELL, h, down_idx.get(h)))
            vu = tm.get((s, "sweep", "main", "up", MAIN_DWELL, h, up_idx.get(h)))
            if vd is None or vu is None:
                valid = False
                break
            a += (vd - vu) * H_STEP
        if valid:
            areas.append(a)
    if areas:
        alo, ahi = boot_ci(areas, rng)
        out["gates"]["loop_area_positive"] = bool(alo > 0)
        out["secondary"]["loop_area"] = {"mean": float(np.mean(areas)), "ci95": [alo, ahi], "n": len(areas)}
    else:
        out["gates"]["loop_area_positive"] = False

    # 副判定5: 共存区間
    coexist = [f"{h:+.3f}" for h in (-0.025, 0.0, 0.025)
               if hold_table[h]["pos"] >= 36 and hold_table[h]["neg"] >= 36]
    out["secondary"]["coexistence_range_levels"] = coexist

    # 診断: 速度依存（seed 1400–1409）
    diag = {}
    for dwell in DIAG_DWELLS + (MAIN_DWELL,):
        g = []
        nc = 0
        for s in DIAG_SEEDS:
            if s not in seeds:
                continue
            hs, hr, cen = sweep_points(tm, s, dwell)
            if cen:
                nc += 1
            else:
                g.append(hr - hs)
        diag[str(dwell)] = {"gap_mean": float(np.mean(g)) if g else None,
                            "n": len(g), "n_censored": nc}
    out["diagnostics"]["gap_by_dwell_seeds_1400_1409"] = diag
    return out


def provenance():
    source = Path(__file__).read_text(encoding="utf-8")
    segment = source[source.index("PRETRAIN_STEPS = 25"):
                     source.index("# ---------------------------------------------------------------- 集計")]
    segment = re.sub(r'PREREGISTERED_GATES = """.*?"""', "PREREGISTERED_GATES = <doc>", segment, flags=re.S)
    return {"script_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "execution_core_sha256": hashlib.sha256(segment.encode()).hexdigest(),
            "preregistered_gates": PREREGISTERED_GATES}


def main():
    ap = argparse.ArgumentParser(description="v0.8 本実験（事前登録済み）")
    ap.add_argument("--output-dir", default="hysteresis_main_results")
    ap.add_argument("--seed-start", type=int, default=1400)
    ap.add_argument("--seed-count", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--expect-seeds", type=int, default=40)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.summarize:
        rows, seeds, problems = load_dir(out)
        if len(seeds) != a.expect_seeds:
            problems.append(f"完了 seed が {len(seeds)}、期待は {a.expect_seeds}")
        if problems and not a.allow_incomplete:
            print("完全性検査に失敗したため集計を中止します:")
            for x in problems[:20]:
                print("  -", x)
            raise SystemExit(2)
        for r in rows:
            pass
        result = summarize(rows, seeds)
        result["completeness"] = {"problems": problems, "n_rows": len(rows), "n_seeds": len(seeds)}
        result["provenance"] = provenance()
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 実験II v0.8 本実験 結果", "", "## 事前判定", ""]
        for k, v in result["gates"].items():
            lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
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
    seeds = list(range(a.seed_start, a.seed_start + a.seed_count))
    for i, seed in enumerate(seeds, 1):
        final = out / f"seed_{seed}.csv"
        if final.exists():
            print(f"seed {seed} は完了済み。スキップ", flush=True)
            continue
        run_seed(cfg, seed, device, out / f"tmp_seed_{seed}.csv", final)
        print(f"seed {i}/{len(seeds)} (seed={seed}) を完了", flush=True)
    print("全 seed 完了")


if __name__ == "__main__":
    main()
