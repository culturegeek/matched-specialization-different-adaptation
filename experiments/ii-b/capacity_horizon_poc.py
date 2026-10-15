#!/usr/bin/env python3
"""実験II v0.6: 共有容量と時間地平線による、境界特化と内部定常点の分離。

背景
----
v0.4.1 は、安定化型の最終偏差が約0.87へ集中することを「共通した大きさへの飽和」と
記述した。しかし 800 ステップは収束点ではない。探索的な 6 seed の走査では、地平線を
4000 まで延ばすと bottleneck=3 で |s| が 0.979 まで上昇し続け、不得意側の誤差が
1-rho^2 付近で頭打ちになる一方、得意側の誤差だけが 0 へ向かった。角解への暴走である。

source_dim=3 に対して bottleneck=3 では、片方の情報源だけなら完全に符号化できる。
片側特化に容量上の費用がないため、内点で止まる理由がない。同じ探索走査で
bottleneck=2 にすると |s| が 0.396 で平坦化した。本実験はこれを独立 seed と
長い地平線で確認する。

測定の要点
----------
比 s = (D_bad - D_good)/(D_bad + D_good) だけを見ると、
(a) 不得意側が底を打った と (b) 得意側が上限まで良くなった を区別できない。
さらに、分子と分母が同時に動いても比だけが一定になりうる。
したがって D_A, D_B を分けて記録し、比と両誤差の双方について定常性を判定する。

統計単位
--------
鏡像二アームは同一 seed のデータ・初期重み・ミニバッチ順を共有するため独立ではない。
すべての信頼区間は、アームを seed 内で平均した 40 単位で計算する。

事前登録
--------
判定条件は本ファイル内の PREREGISTERED_GATES に、実行前に固定した。
生データは逐次書き出し、完了 seed を記録して中断後に重複なく再開できるようにする。

改訂履歴
--------
v1 (2026-08-16): 初版。
v2 (2026-08-16): 結果を見る前に改訂。ChatGPT のレビューを反映。
  - 旧判定1（4000→8000 増分の区間下限 > 0）を撤回。境界 |s|=1 へ接近すれば
    増分は必然的にゼロへ向かうため、正しい仮説に対して失敗しうる判定だった。
    状態そのもの（|s|、両誤差、実質的な逆行のなさ）を検査する四条件へ置換。
  - 容量2について、比だけでなく D_good, D_bad の定常性も判定に追加。
  - 旧判定9（低域+高域 > 0.80）を撤回。全 seed が低域に入っても合格する、
    一峰性でも通る条件だった。各帯の下限を個別に置く形へ置換。
  - 旧判定10（平均の大小）を対応差の区間判定へ置換。
  - すべての信頼区間の単位を、モデル 80 個から seed 40 個へ修正。
v2.1 (2026-08-16): 実行中・結果を見る前に、診断量のみ追加。判定の閾値は変更していない。
  - e の帯だけでは個体状態の二極化を判定できない。二アームが連続的に共変していても
    差 e はゼロ付近へ集まるためである。各アームの符号付き s_T の分布、L 付近への
    集中割合、正負両枝の占有率、鏡像ペアの異符号率、操作方向との一致率を追加した。
  - 判定12 FAIL 時の読み分けを明文化した（READING_RULE_GATE12）。
  - 用語: 「離散化器」は、複数の内部平衡とその吸引域が確認されるまで用いない。
    現段階の記述は「二枝への収束」または「二極化」とする。
  - 報告は A〜E を別個の仮説群として扱い、「13件中N件PASS」という総合評価はしない。
    主判定は A（境界特化）と B（内点定常点候補）、D は副次的な機構診断である。
v2.2 (2026-08-16): 実行中・結果を見る前に集計側のみ改訂。実行経路のコードは v2.1 と
  バイト同一であり、実行中プロセスの挙動は変わらない（provenance.json に両者のハッシュを記録）。
  - 得意側の定義を修正。旧 FAVORED=min(D_A,D_B) は各時点で得意側を選び直すため、
    4000 と 8000 のあいだで符号が反転した個体では別の情報源の誤差を比較していた。
    終端の符号で A/B のどちらを得意側とするかを個体ごとに固定し、過去時点も同じ
    ラベルで読む方式へ変更した。判定6・7はこの修正後の定義を主とする。
    旧定義の値も diagnostics に併記する。
  - seed 平均の変化だけでは、半数が正・半数が負へ動いても平均変化はゼロになる。
    分位点（4000/6000/8000）、アーム単位の変化量の絶対値、4000→8000 の符号反転率、
    両アームとも 0.20<|s|<0.70 に入る seed の割合を診断量として追加した。
  - 判定5〜9が通った段階の記述は「内点の定常点」ではなく「内点の定常分布候補」とする。
  - --summarize に完全性検査を追加。40 seed、各 seed に 2容量×4条件×2アーム×9時点、
    row_key の欠損・矛盾がないことを確認し、不完全なら停止する（--allow-incomplete で解除）。
  - 事前登録文・設定・スクリプトのハッシュを provenance.json へ保存する。
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
    SharedRepresentation, choose_device, ci95, full_metrics, make_config, make_dataset,
    representation_selector_index, selector_probability, set_seed, train_selected, train_weighted,
)

PRETRAIN_STEPS = 25
FORCED_STEPS = 100
FORCED_A_COUNT = 55
BOTTLENECKS = (2, 3)
SELECTOR_MAX_BIAS = 0.40
CONDITIONS = ("balanced_sampled", "fixed_propensity", "stabilization", "exploration")
CHECKPOINTS = (0, 100, 300, 800, 1500, 2500, 4000, 6000, 8000)
TOTAL_STEPS = 8000
PLATEAU_FROM = 4000

PREREGISTERED_GATES = """
統計単位: 鏡像二アームを seed 内で平均した 40 seed。区間はすべてこの単位で計算する。
rho = 0.35 なので、相関のみで説明できる誤差の下限は 1 - rho^2 = 0.8775。

A. 容量3は境界特化へ向かう（内点で止まらない）
 1 capacity3_terminal_index_above_0_95      : bottleneck=3 stabilization の step8000 平均 |s| > 0.95
 2 capacity3_favored_error_near_zero        : 同 得意側平均誤差 < 0.02
 3 capacity3_neglected_error_at_corr_floor  : 同 不得意側平均誤差が 1-rho^2 の ±0.05 以内
 4 capacity3_no_material_reversal           : 同 |s| の 4000→8000 増分の95%区間の下限 > -0.02

B. 容量2は内点で定常化する
 5 capacity2_index_plateaus                 : bottleneck=2 stabilization の |s| の 4000→8000 増分の
                                              95%区間が (-0.03, +0.03) に収まる
 6 capacity2_favored_error_stationary       : 同 得意側誤差の 4000→8000 変化の95%区間が (-0.05, +0.05)
 7 capacity2_neglected_error_stationary     : 同 不得意側誤差の 4000→8000 変化の95%区間が (-0.05, +0.05)
 8 capacity2_favored_error_has_floor        : 同 得意側平均誤差が step8000 で > 0.20
 9 capacity2_terminal_is_interior           : 同 step8000 |s| の95%区間が (0.20, 0.70) に収まる

C. 二条件の質的差は容量2でも残るか
10 capacity2_stabilization_exceeds_fixed    : bottleneck=2 で stabilization の |s_T| が
                                              fixed propensity の1.5倍超、かつ対応差の95%区間の下限 > 0
11 capacity2_stabilization_concentrated     : 同 変動係数が fixed propensity の半分未満

D. 二峰性は指標の上限圧縮か、力学の性質か
12 capacity2_bimodality_persists            : bottleneck=2 stabilization の方向効果 e について、
                                              L = |s_T| の中央値 として
                                              低域 |e| < 0.25L が 20% 以上、
                                              高域 |e| > 0.75L が 15% 以上、
                                              中間域が 20% 未満、
                                              低域+高域が 80% 以上。
                                              不成立なら、v0.4.1 の二峰性は s の上限圧縮による見かけであり、
                                              Claude の「離散化」という記述は撤回する。

E. 探索型の復元力
13 exploration_restorative_both_capacities  : 両容量で、対応差 (exploration - balanced sampled) の
                                              |s_T| の95%区間の上限 < 0

判定6・7の得意側／不得意側は、終端の符号で個体ごとに固定したラベルで測る（v2.2）。
判定5〜9が通っても言えるのは「内点に定常分布候補がある」までである。
seed 平均の停止は個体の停止を含意しないため、分位点・アーム単位変化量・符号反転率を併読する。
安定平衡には摂動後の復帰が、準安定レジームには復帰・離脱・転換費用が別途要る。
"""

DIAGNOSTICS_NOTE = """
以下は事前判定ではなく、記録のみの診断量である。事後の解釈に使う場合は探索的と明記する。
すべて結果を見る前に宣言した。
- 収縮比: |s_T| の seed 間標準偏差 / |s_0| の seed 間標準偏差
- 選択確率 p_A の終端値
- 方向効果 e の帯別割合（容量3でも算出し、容量2と比較する）
- 各アームの符号付き s_T の枝別占有率（正枝 s_T > 0.75L、負枝 s_T < -0.75L、中間）
- |s_T| が L の ±25% に入る割合（大きさの集中）
- 鏡像ペアの異符号率
- 操作方向との一致率（A寄り操作アームが正、B寄り操作アームが負になった割合）
- |s| の分位点（10/25/50/75/90）を 4000・6000・8000 で算出
- アーム単位の 4000→8000 変化量の絶対値（seed 平均の相殺を検出する）
- 4000→8000 で s の符号が反転した割合
- 両アームとも 0.20<|s|<0.70 に入る seed の割合
- 旧定義（各時点で min/max を取り直す）による得意側・不得意側の値
注: 枝別占有率と集中割合はアーム単位（80モデル）である。同一 seed の二アームは
独立ではないため、これらの割合に信頼区間は付さない。
"""

READING_RULE_GATE12 = """
判定12 の結果は、e の帯だけでなく各アームの s_T 分布と併せて読む。
e の中間域が空であることは、個体状態の二極化を含意しない。二つのアームが
連続的に共変していても、差である e はゼロ付近へ集まりうるためである。

  各アームの s_T        e の帯              解釈
  ------------------   -----------------   --------------------------------------------
  ±L 付近へ集中、両枝   低域と高域に分離     状態が二枝へ収束し、操作が分岐方向を決めている
  ±L 付近へ集中、両枝   ほぼ低域のみ         状態は二枝へ収束するが、方向は背景偶発差が支配する
  中間値へ広く分布      中間域も多い         二極化の解釈は支持されない。「二枝への収束」を撤回する
  片方の符号だけに集中  ほぼ低域のみ         二極化ではなく共通方向への単一収束の可能性

したがって、判定12 の FAIL だけで記述を撤回せず、また低域が大半を占めるだけで
記述を維持もしない。各アームの s_T が正負二枝へ集中していることが必要条件である。
"""


# ---------------------------------------------------------------- 実行

def pretrain(cfg, seed, train, device):
    set_seed(seed)
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(700_000 + seed)
    n = train[0].shape[0]
    for _ in range(PRETRAIN_STEPS):
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(model, opt, *(v[idx] for v in train), 0.5)
    return copy.deepcopy(model.state_dict())


def manipulate(cfg, seed, state, train, schedule, device):
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(720_000 + seed)
    n = train[0].shape[0]
    for choose_a in schedule:
        idx = torch.from_numpy(rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_selected(model, opt, *(v[idx] for v in train), bool(choose_a))
    return copy.deepcopy(model.state_dict())


def continue_learning(cfg, seed, bottleneck, arm, condition, state,
                      train, train_cpu, selector_cpu, test_cpu, device):
    model = SharedRepresentation(train[0].shape[1], cfg.hidden_dim, cfg.bottleneck, train[1].shape[1]).to(device)
    model.load_state_dict(state)
    # 操作期の Adam 状態は持ち越さない。
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch_rng = np.random.default_rng(730_000 + seed)
    sel_rng = np.random.default_rng(740_000 + seed)
    n = train[0].shape[0]
    release = representation_selector_index(model, selector_cpu, cfg.probe_ridge, device)
    frozen = selector_probability(release, "stabilization", cfg)
    p = selector_probability(release, condition, cfg, frozen)
    choices_a = 0
    rows = []

    def record(step):
        m, _ = full_metrics(model, train_cpu, test_cpu, cfg, device)
        rows.append({
            "row_key": f"{seed}|{bottleneck}|{condition}|{arm}|{step}",
            "seed": seed, "bottleneck": bottleneck, "arm": arm, "condition": condition,
            "selector_max_bias": cfg.selector_max_bias, "step": step,
            "probe_index": m["probe_index"],
            "probe_D_A": m["probe_D_A"], "probe_D_B": m["probe_D_B"],
            "decoder_D_A": m["D_A"], "decoder_D_B": m["D_B"],
            "decoder_index": m["allocation_index"],
            "selector_p_A": p, "realized_A_fraction": choices_a / max(step, 1),
        })

    checkpoints = set(CHECKPOINTS)
    record(0)
    for step in range(1, TOTAL_STEPS + 1):
        if condition in ("stabilization", "exploration") and (step == 1 or (step - 1) % cfg.selector_interval == 0):
            p = selector_probability(representation_selector_index(model, selector_cpu, cfg.probe_ridge, device), condition, cfg)
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        choose_a = sel_rng.random() < p
        choices_a += int(choose_a)
        train_selected(model, opt, *(v[idx] for v in train), choose_a)
        if step in checkpoints:
            record(step)
    return rows


def run_seed(base, seed, device):
    set_seed(seed)
    train_cpu, selector_cpu, test_cpu = make_dataset(base, seed)
    train = tuple(v.to(device) for v in train_cpu)
    rng = np.random.default_rng(710_000 + seed)
    order = rng.permutation(FORCED_STEPS)
    sched = np.zeros(FORCED_STEPS, dtype=bool)
    sched[order[:FORCED_A_COUNT]] = True
    rows = []
    for bottleneck in BOTTLENECKS:
        bcfg = copy.deepcopy(base)
        bcfg.bottleneck = bottleneck
        bcfg.selector_max_bias = SELECTOR_MAX_BIAS
        state = pretrain(bcfg, seed, train, device)
        states = {"plus": manipulate(bcfg, seed, state, train, sched, device),
                  "minus": manipulate(bcfg, seed, state, train, ~sched, device)}
        for condition in CONDITIONS:
            for arm in ("plus", "minus"):
                rows.extend(continue_learning(bcfg, seed, bottleneck, arm, condition, states[arm],
                                              train, train_cpu, selector_cpu, test_cpu, device))
    return rows


# ---------------------------------------------------------------- 集計
# すべての統計は seed 単位（鏡像二アームを seed 内で平均）で計算する。

EXPECTED_ROWS_PER_SEED = len(BOTTLENECKS) * len(CONDITIONS) * 2 * len(CHECKPOINTS)


def load(paths):
    """row_key で重複を除く。同一 row_key で値が食い違う場合は矛盾として記録する。"""
    seen, rows, duplicates, conflicts = {}, [], 0, []
    for p in paths:
        with open(p, encoding="utf-8") as h:
            for r in csv.DictReader(h):
                key = r["row_key"]
                parsed = {
                    "row_key": key, "seed": int(r["seed"]), "bottleneck": int(r["bottleneck"]),
                    "arm": r["arm"], "condition": r["condition"], "step": int(r["step"]),
                    "probe_index": float(r["probe_index"]),
                    "probe_D_A": float(r["probe_D_A"]), "probe_D_B": float(r["probe_D_B"]),
                    "selector_p_A": float(r["selector_p_A"]),
                }
                if key in seen:
                    duplicates += 1
                    if seen[key]["probe_index"] != parsed["probe_index"]:
                        conflicts.append(key)
                    continue
                seen[key] = parsed
                rows.append(parsed)
    return rows, {"duplicate_rows": duplicates, "conflicting_row_keys": conflicts[:20],
                  "n_conflicts": len(conflicts), "n_rows": len(rows)}


def validate(rows, expected_seeds):
    """不完全なデータでの集計を拒否するための検査。"""
    problems = []
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) != expected_seeds:
        problems.append(f"seed 数が {len(seeds)}、期待は {expected_seeds}")
    want = {(b, c, a, s) for b in BOTTLENECKS for c in CONDITIONS
            for a in ("plus", "minus") for s in CHECKPOINTS}
    for seed in seeds:
        got = {(r["bottleneck"], r["condition"], r["arm"], r["step"])
               for r in rows if r["seed"] == seed}
        missing = want - got
        if missing:
            problems.append(f"seed {seed}: {len(missing)} 件欠損（例 {sorted(missing)[:3]}）")
        extra = got - want
        if extra:
            problems.append(f"seed {seed}: 想定外の組 {len(extra)} 件")
    return problems


def pick(rows, bottleneck, condition, step):
    return [r for r in rows if r["bottleneck"] == bottleneck and r["condition"] == condition and r["step"] == step]


def pick_all(rows, bottleneck, condition):
    return [r for r in rows if r["bottleneck"] == bottleneck and r["condition"] == condition]


def per_seed(rows, bottleneck, condition, step, field):
    """seed ごとに二アームを平均した配列を、seed 昇順で返す。"""
    by = {}
    for r in pick(rows, bottleneck, condition, step):
        by.setdefault(r["seed"], {})[r["arm"]] = field(r)
    seeds = sorted(s for s, v in by.items() if len(v) == 2)
    return np.array(seeds), np.array([0.5 * (by[s]["plus"] + by[s]["minus"]) for s in seeds])


ABS_INDEX = lambda r: abs(r["probe_index"])
P_DEV = lambda r: abs(r["selector_p_A"] - 0.5)
LEGACY_FAVORED = lambda r: min(r["probe_D_A"], r["probe_D_B"])
LEGACY_NEGLECTED = lambda r: max(r["probe_D_A"], r["probe_D_B"])


def terminal_favored_map(rows):
    """個体（seed, 容量, 条件, アーム）ごとに、終端の符号で得意側を A/B に固定する。

    s = (D_B - D_A)/(D_A + D_B) なので、s > 0 は A の方が精密＝A が得意側である。
    """
    fmap = {}
    for r in rows:
        if r["step"] == TOTAL_STEPS:
            fmap[(r["seed"], r["bottleneck"], r["condition"], r["arm"])] = "A" if r["probe_index"] > 0 else "B"
    return fmap


def make_error_field(fmap, which):
    def field(r):
        favored = fmap.get((r["seed"], r["bottleneck"], r["condition"], r["arm"]))
        if favored is None:                                  # 終端が無ければ旧定義へ退避
            return LEGACY_FAVORED(r) if which == "favored" else LEGACY_NEGLECTED(r)
        good = r["probe_D_A"] if favored == "A" else r["probe_D_B"]
        bad = r["probe_D_B"] if favored == "A" else r["probe_D_A"]
        return good if which == "favored" else bad
    return field


def paired_change(rows, bottleneck, condition, field, step_a=PLATEAU_FROM, step_b=TOTAL_STEPS):
    sa, va = per_seed(rows, bottleneck, condition, step_a, field)
    sb, vb = per_seed(rows, bottleneck, condition, step_b, field)
    shared = np.intersect1d(sa, sb)
    return vb[np.isin(sb, shared)] - va[np.isin(sa, shared)]


def paired_contrast(rows, bottleneck, cond_a, cond_b, field=ABS_INDEX, step=TOTAL_STEPS):
    sa, va = per_seed(rows, bottleneck, cond_a, step, field)
    sb, vb = per_seed(rows, bottleneck, cond_b, step, field)
    shared = np.intersect1d(sa, sb)
    return va[np.isin(sa, shared)] - vb[np.isin(sb, shared)]


def directional(rows, bottleneck, condition, step):
    by = {}
    for r in pick(rows, bottleneck, condition, step):
        by.setdefault(r["seed"], {})[r["arm"]] = r["probe_index"]
    seeds = sorted(s for s, v in by.items() if len(v) == 2)
    return np.array([0.5 * (by[s]["plus"] - by[s]["minus"]) for s in seeds])


def arm_pairs_over_time(rows, bottleneck, condition, field, step_a=PLATEAU_FROM, step_b=TOTAL_STEPS):
    """アーム単位で二時点の値を対応づける。seed 平均では相殺される動きを見るため。"""
    by = {}
    for r in pick_all(rows, bottleneck, condition):
        if r["step"] in (step_a, step_b):
            by.setdefault((r["seed"], r["arm"]), {})[r["step"]] = field(r)
    return [v for v in by.values() if len(v) == 2]


def arm_dynamics(rows, bottleneck, condition, favored_field):
    """seed 平均の停止が個体の停止を意味するかを見るための診断量。"""
    idx_pairs = arm_pairs_over_time(rows, bottleneck, condition, lambda r: r["probe_index"])
    abs_change = np.array([abs(abs(v[TOTAL_STEPS]) - abs(v[PLATEAU_FROM])) for v in idx_pairs])
    signed = np.array([v[TOTAL_STEPS] for v in idx_pairs])
    flips = np.array([np.sign(v[PLATEAU_FROM]) != np.sign(v[TOTAL_STEPS]) for v in idx_pairs])
    fav_pairs = arm_pairs_over_time(rows, bottleneck, condition, favored_field)
    fav_change = np.array([abs(v[TOTAL_STEPS] - v[PLATEAU_FROM]) for v in fav_pairs])
    by = {}
    for r in pick(rows, bottleneck, condition, TOTAL_STEPS):
        by.setdefault(r["seed"], {})[r["arm"]] = abs(r["probe_index"])
    both_interior = [0.20 < v["plus"] < 0.70 and 0.20 < v["minus"] < 0.70
                     for v in by.values() if len(v) == 2]
    return {
        "n_arms": int(len(idx_pairs)),
        "arm_abs_index_change_mean": float(abs_change.mean()) if len(abs_change) else float("nan"),
        "arm_abs_index_change_max": float(abs_change.max()) if len(abs_change) else float("nan"),
        "arm_favored_error_change_mean": float(fav_change.mean()) if len(fav_change) else float("nan"),
        "sign_flip_rate_4000_to_8000": float(flips.mean()) if len(flips) else float("nan"),
        "both_arms_interior_rate": float(np.mean(both_interior)) if both_interior else float("nan"),
        "terminal_signed_positive_rate": float(np.mean(signed > 0)) if len(signed) else float("nan"),
    }


def quantiles(values):
    return {f"q{q}": float(np.quantile(values, q / 100.0)) for q in (10, 25, 50, 75, 90)}


def bands(e, level):
    low = float(np.mean(np.abs(e) < 0.25 * level))
    high = float(np.mean(np.abs(e) > 0.75 * level))
    return {"low": low, "high": high, "middle": 1.0 - low - high, "level": level}


def arm_structure(rows, bottleneck, condition, level, step=TOTAL_STEPS):
    """各アームの符号付き s_T の構造。判定ではなく診断量（アーム単位、区間は付さない）。"""
    by = {}
    for r in pick(rows, bottleneck, condition, step):
        by.setdefault(r["seed"], {})[r["arm"]] = r["probe_index"]
    pairs = [(v["plus"], v["minus"]) for v in by.values() if len(v) == 2]
    values = np.array([x for pair in pairs for x in pair])
    plus_arm = np.array([p for p, _ in pairs])
    minus_arm = np.array([m for _, m in pairs])
    return {
        "n_arms": int(len(values)), "n_pairs": int(len(pairs)),
        "level_L": level,
        "positive_branch_rate": float(np.mean(values > 0.75 * level)),
        "negative_branch_rate": float(np.mean(values < -0.75 * level)),
        "middle_rate": float(np.mean(np.abs(values) <= 0.75 * level)),
        "magnitude_concentrated_rate": float(np.mean(np.abs(np.abs(values) - level) <= 0.25 * level)),
        "sign_positive_rate": float(np.mean(values > 0)),
        "pair_opposite_sign_rate": float(np.mean(plus_arm * minus_arm < 0)),
        "pair_tracks_manipulation_rate": float(np.mean((plus_arm > 0) & (minus_arm < 0))),
        "pair_effect_positive_rate": float(np.mean(plus_arm - minus_arm > 0)),
        "signed_mean": float(values.mean()), "signed_sd": float(values.std(ddof=1)),
    }


def cv(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / max(abs(values.mean()), 1e-12))


def interval(values):
    m, lo, hi = ci95(values)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": int(len(values))}


def summarize(rows, correlation=0.35):
    corr_floor = 1.0 - correlation ** 2
    fmap = terminal_favored_map(rows)
    FAVORED = make_error_field(fmap, "favored")
    NEGLECTED = make_error_field(fmap, "neglected")
    out = {"correlation_error_floor": corr_floor, "timecourse": {}, "terminal": {},
           "gates": {}, "diagnostics": {}}

    for b in BOTTLENECKS:
        for c in CONDITIONS:
            key = f"b{b}_{c}"
            out["timecourse"][key] = {}
            for s in CHECKPOINTS:
                _, idx = per_seed(rows, b, c, s, ABS_INDEX)
                if len(idx) == 0:
                    continue
                _, good = per_seed(rows, b, c, s, FAVORED)
                _, bad = per_seed(rows, b, c, s, NEGLECTED)
                _, lgood = per_seed(rows, b, c, s, LEGACY_FAVORED)
                _, lbad = per_seed(rows, b, c, s, LEGACY_NEGLECTED)
                out["timecourse"][key][str(s)] = {
                    "n_seeds": int(len(idx)), "abs_index_mean": float(idx.mean()),
                    "favored_error_mean": float(good.mean()), "neglected_error_mean": float(bad.mean()),
                    "legacy_favored_error_mean": float(lgood.mean()),
                    "legacy_neglected_error_mean": float(lbad.mean()),
                }
            _, idx = per_seed(rows, b, c, TOTAL_STEPS, ABS_INDEX)
            _, pdev = per_seed(rows, b, c, TOTAL_STEPS, P_DEV)
            e = directional(rows, b, c, TOTAL_STEPS)
            level = float(np.median(idx))
            quant = {}
            for s in (PLATEAU_FROM, 6000, TOTAL_STEPS):
                _, v = per_seed(rows, b, c, s, ABS_INDEX)
                if len(v):
                    quant[str(s)] = quantiles(v)
            out["terminal"][key] = {
                "abs_index": interval(idx), "abs_index_cv": cv(idx),
                "abs_index_quantiles": quant,
                "directional_effect": interval(e),
                "index_change_4000_8000": interval(paired_change(rows, b, c, ABS_INDEX)),
                "favored_change_4000_8000": interval(paired_change(rows, b, c, FAVORED)),
                "neglected_change_4000_8000": interval(paired_change(rows, b, c, NEGLECTED)),
                "legacy_favored_change_4000_8000": interval(paired_change(rows, b, c, LEGACY_FAVORED)),
                "legacy_neglected_change_4000_8000": interval(paired_change(rows, b, c, LEGACY_NEGLECTED)),
                "mean_p_A_deviation": float(pdev.mean()),
                "bands": bands(e, level),
                "arm_structure": arm_structure(rows, b, c, level),
                "arm_dynamics": arm_dynamics(rows, b, c, FAVORED),
            }

    g, t, tc = out["gates"], out["terminal"], out["timecourse"]

    # A. 容量3は境界特化へ向かう
    s3, tc3 = t["b3_stabilization"], tc["b3_stabilization"][str(TOTAL_STEPS)]
    g["capacity3_terminal_index_above_0_95"] = bool(s3["abs_index"]["mean"] > 0.95)
    g["capacity3_favored_error_near_zero"] = bool(tc3["favored_error_mean"] < 0.02)
    g["capacity3_neglected_error_at_corr_floor"] = bool(abs(tc3["neglected_error_mean"] - corr_floor) < 0.05)
    g["capacity3_no_material_reversal"] = bool(s3["index_change_4000_8000"]["ci_low"] > -0.02)

    # B. 容量2は内点で定常化する（得意側は終端符号で固定したラベル）
    s2, tc2 = t["b2_stabilization"], tc["b2_stabilization"][str(TOTAL_STEPS)]
    ic, fc, nc = (s2["index_change_4000_8000"], s2["favored_change_4000_8000"],
                  s2["neglected_change_4000_8000"])
    g["capacity2_index_plateaus"] = bool(ic["ci_low"] > -0.03 and ic["ci_high"] < 0.03)
    g["capacity2_favored_error_stationary"] = bool(fc["ci_low"] > -0.05 and fc["ci_high"] < 0.05)
    g["capacity2_neglected_error_stationary"] = bool(nc["ci_low"] > -0.05 and nc["ci_high"] < 0.05)
    g["capacity2_favored_error_has_floor"] = bool(tc2["favored_error_mean"] > 0.20)
    g["capacity2_terminal_is_interior"] = bool(s2["abs_index"]["ci_low"] > 0.20 and s2["abs_index"]["ci_high"] < 0.70)

    # C. 二条件の質的差
    f2 = t["b2_fixed_propensity"]
    ci_contrast = interval(paired_contrast(rows, 2, "stabilization", "fixed_propensity"))
    g["capacity2_stabilization_exceeds_fixed"] = bool(
        s2["abs_index"]["mean"] > 1.50 * f2["abs_index"]["mean"] and ci_contrast["ci_low"] > 0)
    g["capacity2_stabilization_concentrated"] = bool(s2["abs_index_cv"] < 0.50 * f2["abs_index_cv"])

    # D. 二峰性
    band = s2["bands"]
    g["capacity2_bimodality_persists"] = bool(
        band["low"] >= 0.20 and band["high"] >= 0.15 and band["middle"] < 0.20
        and band["low"] + band["high"] >= 0.80)

    # E. 探索型の復元力
    restorative = {b: interval(paired_contrast(rows, b, "exploration", "balanced_sampled"))
                   for b in BOTTLENECKS}
    g["exploration_restorative_both_capacities"] = bool(all(restorative[b]["ci_high"] < 0 for b in BOTTLENECKS))

    contraction = {}
    for b in BOTTLENECKS:
        for c in CONDITIONS:
            _, i0 = per_seed(rows, b, c, 0, ABS_INDEX)
            _, iT = per_seed(rows, b, c, TOTAL_STEPS, ABS_INDEX)
            contraction[f"b{b}_{c}"] = float(iT.std(ddof=1) / max(i0.std(ddof=1), 1e-12))

    out["hypothesis_groups"] = {
        "A_boundary_specialization_capacity3": {k: g[k] for k in g if k.startswith("capacity3_")},
        "B_interior_stationary_distribution_capacity2": {k: g[k] for k in g if k in (
            "capacity2_index_plateaus", "capacity2_favored_error_stationary",
            "capacity2_neglected_error_stationary", "capacity2_favored_error_has_floor",
            "capacity2_terminal_is_interior")},
        "C_dynamic_vs_fixed_capacity2": {k: g[k] for k in g if k in (
            "capacity2_stabilization_exceeds_fixed", "capacity2_stabilization_concentrated")},
        "D_bipolarity_secondary_diagnosis": {k: g[k] for k in g if k == "capacity2_bimodality_persists"},
        "E_exploration_restorative": {k: g[k] for k in g if k == "exploration_restorative_both_capacities"},
        "note": "A〜Eは別個の仮説群である。合計PASS数で評価しない。"
                "主判定はAとB、Dは副次的な機構診断である。"
                "Bが通っても言えるのは内点の定常分布候補までであり、"
                "個体の停止は arm_dynamics を併読して判断する。",
    }
    out["diagnostics"] = {
        "note": DIAGNOSTICS_NOTE,
        "reading_rule_gate12": READING_RULE_GATE12,
        "capacity2_stabilization_minus_fixed": ci_contrast,
        "exploration_minus_balanced": {f"b{b}": restorative[b] for b in BOTTLENECKS},
        "terminal_over_initial_sd_ratio": contraction,
    }
    return out


# ---------------------------------------------------------------- CLI

def provenance():
    """事前登録文・設定・スクリプトのハッシュ。実行経路のみのハッシュも別に取る。

    実行経路 = 定数ブロックから run_seed の終端まで（事前登録文などの文字列は除外）。
    集計側の改訂で全体ハッシュが変わっても、この値が同じなら実行結果は同一である。
    """
    source = Path(__file__).read_text(encoding="utf-8")
    segment = source[source.index("PRETRAIN_STEPS = 25"):
                     source.index("# ---------------------------------------------------------------- 集計")]
    for name in ("PREREGISTERED_GATES", "DIAGNOSTICS_NOTE", "READING_RULE_GATE12"):
        segment = re.sub(rf'{name} = """.*?"""', f"{name} = <doc>", segment, flags=re.S)
    return {
        "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "execution_core_sha256": hashlib.sha256(segment.encode()).hexdigest(),
        "constants": {
            "PRETRAIN_STEPS": PRETRAIN_STEPS, "FORCED_STEPS": FORCED_STEPS,
            "FORCED_A_COUNT": FORCED_A_COUNT, "BOTTLENECKS": list(BOTTLENECKS),
            "SELECTOR_MAX_BIAS": SELECTOR_MAX_BIAS, "CONDITIONS": list(CONDITIONS),
            "CHECKPOINTS": list(CHECKPOINTS), "TOTAL_STEPS": TOTAL_STEPS,
            "PLATEAU_FROM": PLATEAU_FROM,
        },
        "preregistered_gates": PREREGISTERED_GATES,
        "diagnostics_note": DIAGNOSTICS_NOTE,
        "reading_rule_gate12": READING_RULE_GATE12,
    }


FIELDS = ["row_key", "seed", "bottleneck", "arm", "condition", "selector_max_bias", "step",
          "probe_index", "probe_D_A", "probe_D_B", "decoder_D_A", "decoder_D_B",
          "decoder_index", "selector_p_A", "realized_A_fraction"]


def completed_seeds(path: Path):
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as h:
        for r in csv.DictReader(h):
            if int(r["step"]) == TOTAL_STEPS and r["condition"] == CONDITIONS[-1] \
               and r["arm"] == "minus" and int(r["bottleneck"]) == BOTTLENECKS[-1]:
                done.add(int(r["seed"]))
    return done


def main():
    ap = argparse.ArgumentParser(description="共有容量と時間地平線による境界特化と内部定常点の分離")
    ap.add_argument("--output-dir", default="capacity_horizon_results")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--seed-count", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--summarize", nargs="+", default=None, help="raw CSV を読んで集計だけ行う")
    ap.add_argument("--expect-seeds", type=int, default=40, help="集計時に要求する seed 数")
    ap.add_argument("--allow-incomplete", action="store_true", help="不完全データでも集計する（既定は拒否）")
    a = ap.parse_args()

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.summarize:
        rows, load_report = load(a.summarize)
        problems = validate(rows, a.expect_seeds)
        if load_report["n_conflicts"]:
            problems.append(f"同一 row_key で値が食い違う行が {load_report['n_conflicts']} 件")
        if problems and not a.allow_incomplete:
            print("完全性検査に失敗したため集計を中止します:")
            for x in problems[:20]:
                print("  -", x)
            print("（意図的に途中集計する場合は --allow-incomplete を付けてください）")
            raise SystemExit(2)
        result = summarize(rows)
        result["completeness"] = {"problems": problems, "load_report": load_report,
                                  "expected_rows_per_seed": EXPECTED_ROWS_PER_SEED,
                                  "allow_incomplete": bool(a.allow_incomplete)}
        result["provenance"] = provenance()
        result["preregistered_gates"] = PREREGISTERED_GATES
        (out / "provenance.json").write_text(
            json.dumps(result["provenance"], ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 実験II v0.6・容量×時間 結果", "", "## 事前判定（仮説群ごと）", ""]
        for group, items in result["hypothesis_groups"].items():
            if group == "note":
                continue
            lines += ["", f"### {group}", ""]
            lines += [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in items.items()]
        lines += ["", "合計PASS数では評価しない。A〜Eは別個の仮説群である。"]
        (out / "result_report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    torch.set_num_threads(a.threads)
    device = choose_device("cpu")
    base = make_config(full=False)
    base.history_steps = PRETRAIN_STEPS
    base.common_steps = TOTAL_STEPS
    base.checkpoints = list(CHECKPOINTS)
    base.conditions = list(CONDITIONS)
    base.seeds = list(range(a.seed_start, a.seed_start + a.seed_count))

    (out / f"provenance_run_{a.seed_start}.json").write_text(
        json.dumps({**provenance(), "seeds": base.seeds}, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = out / f"raw_{a.seed_start}_{a.seed_start + a.seed_count - 1}.csv"
    done = completed_seeds(raw)
    if done:
        print(f"再開: 完了済み {len(done)} seed をスキップします")
    with raw.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not done and raw.stat().st_size == 0:
            writer.writeheader()
        for index, seed in enumerate(base.seeds, 1):
            if seed in done:
                continue
            writer.writerows(run_seed(base, seed, device))
            handle.flush()
            print(f"seed {index}/{len(base.seeds)} (seed={seed}) を完了", flush=True)
    print(f"生データ: {raw}")


if __name__ == "__main__":
    main()
