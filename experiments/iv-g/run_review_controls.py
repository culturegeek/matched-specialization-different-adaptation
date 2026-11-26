#!/usr/bin/env python3
"""IV-G Review Controls and Saved-Checkpoint Decompositions (2026-11-20).

Reproduces:
1. Within-batch stationary balanced update-route control:
   - Pilot calibration on disjoint pilot seeds 8000..8015 using endpoints only.
   - Confirmatory comparison on 40 seeds 20000..20039 paired with saved acquisition checkpoints,
     including formation cancellation accounting, m/v exchange, reset, and common-scale / layerwise
     pattern exchanges.
2. Multi-granularity scale vs. coordinate-pattern decompositions on saved checkpoints (seeds 20000..20039):
   - 4-layer, 8-tensor, Layer-1 per-neuron (row), and Layer-1 per-input (column) scale normalizations.
   - 4-layer per-step equal-movement control.
   - Geometric-mean and median inverse-scale normalizations.
   - Scalar-v + common-scale m-swap and zero-m checks.
   - Pure discounted ancestor (beta2^500 * v0) transplant.
   - Layer-1 orthogonal-subspace (Q) and bottleneck-coupling diagnostics.
3. PCGrad donor-v common-scale control on saved checkpoints (seeds 38000..38039).
4. Target-specific readout-error (target_D, other_D, error_sum) summaries from archived IV-D and follow-up CSVs,
   plus ancestor CKA reference on IV-D seeds 8100..8139 and follow-up seeds 20000..20039.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
FOLLOW_DIR = EXPERIMENTS / "iv-followup"
PCGRAD_DIR = EXPERIMENTS / "iv-cancellation"

sys.path.insert(0, str(FOLLOW_DIR))
import run_followup as F  # noqa: E402

B, D, E = F.B, F.D, F.E
DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_ci(vals, seed: int = 20260903, n_boot: int = 5000) -> dict:
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "low": float(np.quantile(b, 0.025)),
        "high": float(np.quantile(b, 0.975)),
    }


def subset_q_stat(state, indices, scale: float = 1.0, stat: str = "mean") -> float:
    g = state["param_groups"][0]
    beta2, eps, lr = g["betas"][1], g["eps"], g["lr"]
    qs = []
    for i in indices:
        s = state["state"][g["params"][i]]
        bc = 1.0 - beta2 ** E.scalar_step(s["step"])
        qs.append(lr / (torch.sqrt(s["exp_avg_sq"].double().reshape(-1) * scale / bc) + eps))
    qcat = torch.cat(qs)
    if stat == "mean":
        return float(qcat.mean())
    if stat == "gmean":
        return float(torch.exp(torch.log(qcat).mean()))
    if stat == "median":
        return float(qcat.median())
    raise ValueError(stat)


def scale_subset_to_target(state, indices, target_q: float, stat: str = "mean") -> None:
    g = state["param_groups"][0]
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if subset_q_stat(state, indices, float(np.exp(mid)), stat=stat) > target_q:
            lo = mid
        else:
            hi = mid
    factor = float(np.exp(0.5 * (lo + hi)))
    for i in indices:
        state["state"][g["params"][i]]["exp_avg_sq"].mul_(factor)


def run_10step_pulse(cfg, seed: int, direction: str, ms, opt_state, splits):
    train, probe, test = splits
    model = B.make_model(cfg, DEVICE)
    model.load_state_dict(ms)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    if opt_state is not None:
        opt.load_state_dict(copy.deepcopy(opt_state))
    start_m = D.aligned_metrics(model, probe, test, cfg, DEVICE, direction)
    rng = np.random.default_rng(700_000 + seed)
    order_rng = np.random.default_rng(710_000 + seed)
    schedule = []
    for _ in range(6):
        pair = [0, 1]
        order_rng.shuffle(pair)
        schedule.extend(pair)
    acq, w = E.pulse_condition("combined_switch", direction)
    for step in range(1, 11):
        idx = torch.from_numpy(rng.integers(0, len(train.a), size=cfg.batch_size))
        x = D.acquisition_tensor(train, acq, schedule[step - 1])[idx]
        B.train_weighted(model, opt, x, train.a[idx], train.b[idx], w)
    end_m = D.aligned_metrics(model, probe, test, cfg, DEVICE, direction)
    return start_m["aligned_s"] - end_m["aligned_s"], start_m, end_m


def form_within_batch_update(cfg, seed: int, direction: str, weight_favored: float, nm, no, train, record_cancellation: bool = False):
    model = B.make_model(cfg, DEVICE)
    model.load_state_dict(nm)
    opt = B.clone_optimizer(model, cfg, no)
    weight_a = weight_favored if direction == "A" else (1.0 - weight_favored)
    batch_rng = np.random.default_rng(600_000 + seed)
    n = train.a.shape[0]
    half = cfg.batch_size // 2
    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    enc1_idx = [i for i, name in enumerate(names) if name.startswith("encoder.0")]
    enc2_idx = [i for i, name in enumerate(names) if name.startswith("encoder.2")]
    S1 = C1 = S2 = C2 = 0.0
    beta2 = 0.999

    for _ in range(cfg.formation_steps):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size))
        x = torch.cat([train.x_a_precise[idx[:half]], train.x_b_precise[idx[half:]]], dim=0)
        a_b, b_b = train.a[idx], train.b[idx]
        if record_cancellation:
            opt.zero_grad(set_to_none=True)
            _, pa, pb = model(x)
            la = torch.mean((pa - a_b) ** 2)
            lb = torch.mean((pb - b_b) ** 2)
            ga = torch.autograd.grad(la, params, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(lb, params, retain_graph=True, allow_unused=True)
            wf = weight_a if direction == "A" else (1.0 - weight_a)
            wo = 1.0 - wf
            gf_list = ga if direction == "A" else gb
            go_list = gb if direction == "A" else ga
            s1_step = sum(float(((wf * gf_list[i].double()) ** 2 + (wo * go_list[i].double()) ** 2).sum()) for i in enc1_idx)
            c1_step = sum(float((2.0 * wf * wo * gf_list[i].double() * go_list[i].double()).sum()) for i in enc1_idx)
            s2_step = sum(float(((wf * gf_list[i].double()) ** 2 + (wo * go_list[i].double()) ** 2).sum()) for i in enc2_idx)
            c2_step = sum(float((2.0 * wf * wo * gf_list[i].double() * go_list[i].double()).sum()) for i in enc2_idx)
            S1 = beta2 * S1 + s1_step
            C1 = beta2 * C1 + c1_step
            S2 = beta2 * S2 + s2_step
            C2 = beta2 * C2 + c2_step
        B.train_weighted(model, opt, x, a_b, b_b, weight_a)

    canc = {"canc_enc1": -C1 / S1, "canc_enc2": -C2 / S2, "S1": S1, "S2": S2} if record_cancellation else None
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), canc


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for reproduced outputs")
    out_dir = parser.parse_args().out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = D.config_for_stage("confirm")
    param_names = [n for n, _ in B.make_model(cfg, DEVICE).named_parameters()]
    all_indices = list(range(len(param_names)))
    LAYER4 = {
        "enc1": ["encoder.0.weight", "encoder.0.bias"],
        "enc2": ["encoder.2.weight", "encoder.2.bias"],
        "head_a": ["head_a.weight", "head_a.bias"],
        "head_b": ["head_b.weight", "head_b.bias"],
    }
    layer4_indices = {k: [i for i, n in enumerate(param_names) if n in v] for k, v in LAYER4.items()}
    tensor8_indices = {n: [i] for i, n in enumerate(param_names)}

    # 1. Pilot calibration for stationary within-batch update route (seeds 8000..8015)
    pilot_seeds = list(range(8000, 8016))
    tolerances = {"aligned_s": 0.02, "target_D": 0.02, "other_D": 0.02, "error_sum": 0.02, "aligned_low_s": 0.01}
    candidate_weights = [0.74, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.84]
    pilot_acq, pilot_splits, pilot_anc = {}, {}, {}
    for seed in pilot_seeds:
        splits = B.make_splits(cfg, seed)
        pilot_splits[seed] = splits
        nm, no = D.prepare_neutral(cfg, seed, splits[0], DEVICE)
        pilot_anc[seed] = (nm, no)
        for direction in ["A", "B"]:
            ms_a, _ = D.form_state(cfg, seed, "acquisition", direction, 0.80, nm, no, splits[0], DEVICE)
            m_a = B.make_model(cfg, DEVICE)
            m_a.load_state_dict(ms_a)
            pilot_acq[(seed, direction)] = D.aligned_metrics(m_a, splits[1], splits[2], cfg, DEVICE, direction)

    pilot_grid = {}
    best_w, best_score = None, 1e9
    for w in candidate_weights:
        diffs = {k: [] for k in tolerances}
        for seed in pilot_seeds:
            splits = pilot_splits[seed]
            nm, no = pilot_anc[seed]
            d_dir = {k: [] for k in tolerances}
            for direction in ["A", "B"]:
                ms_u, _, _ = form_within_batch_update(cfg, seed, direction, w, nm, no, splits[0], False)
                m_u = B.make_model(cfg, DEVICE)
                m_u.load_state_dict(ms_u)
                mu = D.aligned_metrics(m_u, splits[1], splits[2], cfg, DEVICE, direction)
                ma = pilot_acq[(seed, direction)]
                for k in tolerances:
                    d_dir[k].append(ma[k] - mu[k])
            for k in tolerances:
                diffs[k].append(0.5 * sum(d_dir[k]))
        max_std = float(max(abs(np.mean(diffs[k])) / tolerances[k] for k in tolerances))
        pilot_grid[f"{w:.2f}"] = {"max_standardized_discrepancy": max_std, "mean_aligned_s_diff": float(np.mean(diffs["aligned_s"]))}
        if max_std < best_score:
            best_score = max_std
            best_w = w

    # 2. Confirmatory stationary within-batch update + multi-granularity saved-checkpoint analyses (seeds 20000..20039)
    confirm_seeds = list(range(20000, 20040))
    stat_match = {k: [] for k in tolerances}
    stat_contrasts = {k: [] for k in ["own", "m_other", "v_other", "reset", "v_equal_scale", "v_other_equal_scale", "common_scale_pattern", "layer4_pattern"]}
    stat_v_ratios, stat_q_ratios, stat_canc_e1, stat_canc_e2 = [], [], [], []
    stat_seed_rows = []
    sw_acq_stat, sw_upd_stat = 0, 0

    decomp_modes = [
        "global_common_scale_own", "global_common_scale_swap", "global_pattern_effect",
        "layer4_equal_scale_own", "layer4_equal_scale_swap", "layer4_pattern_effect",
        "tensor8_equal_scale_own", "tensor8_equal_scale_swap", "tensor8_pattern_effect",
        "row_equal_scale_own", "row_equal_scale_swap", "row_pattern_effect",
        "col_equal_scale_own", "col_equal_scale_swap", "col_pattern_effect",
        "layer4_norm_own", "layer4_norm_swap", "layer4_norm_effect",
        "gmean_pattern_effect", "median_pattern_effect",
        "scalar_v_equal", "scalar_v_equal_m_swap", "scalar_v_equal_m_zero",
        "ancestor_discounted_v_swap",
    ]
    decomp_vals = {k: [] for k in decomp_modes}
    decomp_seed_rows = []
    cka_pair_f, cka_anc_acq_f, cka_anc_upd_f = [], [], []
    corr_w2_acq, corr_w2_upd = [], []

    for seed in confirm_seeds:
        splits = B.make_splits(cfg, seed)
        train, probe, test = splits
        nm, no = D.prepare_neutral(cfg, seed, train, DEVICE)
        ep = torch.load(FOLLOW_DIR / f"results_confirm/seed_{seed}/formation_checkpoints.pt", map_location="cpu", weights_only=True)

        # --- Stationary within-batch update branch ---
        md_dir = {k: [] for k in tolerances}
        sc_dir = {k: [] for k in stat_contrasts}
        vr_d, qr_d, ce1_d, ce2_d = [], [], [], []
        # --- Saved alternating branch decompositions ---
        dc_dir = {k: [] for k in decomp_modes}
        ckp_d, ckaa_d, ckau_d = [], [], []
        cwa_d, cwu_d = [], []

        for direction in ["A", "B"]:
            ms_a, os_a = ep[("acquisition", direction)]
            ms_u_alt, os_u_alt = ep[("update", direction)]

            # CKA & neuron coupling on saved alternating checkpoints
            ckp_d.append(D.endpoint_cka(ms_a, ms_u_alt, cfg, probe, DEVICE))
            ckaa_d.append(D.endpoint_cka(ms_a, nm, cfg, probe, DEVICE))
            ckau_d.append(D.endpoint_cka(ms_u_alt, nm, cfg, probe, DEVICE))
            bc2 = 1.0 - 0.999 ** 600
            for ms_r, os_r, bucket in [(ms_a, os_a, cwa_d), (ms_u_alt, os_u_alt, cwu_d)]:
                w2_norm = torch.linalg.vector_norm(ms_r["encoder.2.weight"].double(), dim=0)
                v1 = os_r["state"][os_r["param_groups"][0]["params"][0]]["exp_avg_sq"].double()
                inv_row = (0.002 / (torch.sqrt(v1 / bc2) + 1e-8)).mean(dim=1)
                bucket.append(float(np.corrcoef(w2_norm.numpy(), inv_row.numpy())[0, 1]))

            # Stationary within-batch formation & pulses
            ms_u_st, os_u_st, canc = form_within_batch_update(cfg, seed, direction, best_w, nm, no, train, True)
            ce1_d.append(canc["canc_enc1"])
            ce2_d.append(canc["canc_enc2"])
            ma_m = B.make_model(cfg, DEVICE); ma_m.load_state_dict(ms_a)
            mu_m = B.make_model(cfg, DEVICE); mu_m.load_state_dict(ms_u_st)
            ma = D.aligned_metrics(ma_m, probe, test, cfg, DEVICE, direction)
            mu = D.aligned_metrics(mu_m, probe, test, cfg, DEVICE, direction)
            for k in tolerances:
                md_dir[k].append(ma[k] - mu[k])
            va_sum = sum(float(s["exp_avg_sq"].double().sum()) for s in os_a["state"].values())
            vu_sum = sum(float(s["exp_avg_sq"].double().sum()) for s in os_u_st["state"].values())
            vr_d.append(va_sum / vu_sum)
            qa_st, qu_st = F.q_mean(os_a), F.q_mean(os_u_st)
            qr_d.append(qa_st / qu_st)
            qc_st = float(np.sqrt(qa_st * qu_st))
            for mode in ["own", "m_other", "v_other", "reset", "v_equal_scale", "v_other_equal_scale"]:
                sa = F.adjusted_state(os_a, os_u_st, mode, qc_st)
                su = F.adjusted_state(os_u_st, os_a, mode, qc_st)
                da, _, end_a = run_10step_pulse(cfg, seed, direction, ms_a, sa, splits)
                du, _, end_u = run_10step_pulse(cfg, seed, direction, ms_u_st, su, splits)
                if mode == "own":
                    if end_a["aligned_s"] <= 0: sw_acq_stat += 1
                    if end_u["aligned_s"] <= 0: sw_upd_stat += 1
                sc_dir[mode].append(da - du)
            sc_dir["common_scale_pattern"].append(sc_dir["v_equal_scale"][-1] - sc_dir["v_other_equal_scale"][-1])

            sa_l4_st = copy.deepcopy(os_a); su_l4_st = copy.deepcopy(os_u_st)
            sa_l4_st_sw = E.compose_optimizer_state(os_a, os_u_st, "v_other")
            su_l4_st_sw = E.compose_optimizer_state(os_u_st, os_a, "v_other")
            for idxs in layer4_indices.values():
                qlc = float(np.sqrt(subset_q_stat(os_a, idxs) * subset_q_stat(os_u_st, idxs)))
                scale_subset_to_target(sa_l4_st, idxs, qlc); scale_subset_to_target(su_l4_st, idxs, qlc)
                scale_subset_to_target(sa_l4_st_sw, idxs, qlc); scale_subset_to_target(su_l4_st_sw, idxs, qlc)
            cl4_st_o = run_10step_pulse(cfg, seed, direction, ms_a, sa_l4_st, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_st, su_l4_st, splits)[0]
            cl4_st_s = run_10step_pulse(cfg, seed, direction, ms_a, sa_l4_st_sw, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_st, su_l4_st_sw, splits)[0]
            sc_dir["layer4_pattern"].append(cl4_st_o - cl4_st_s)

            # --- Multi-granularity pattern decompositions on saved alternating checkpoints ---
            qc_alt = float(np.sqrt(F.q_mean(os_a) * F.q_mean(os_u_alt)))
            cg_o = run_10step_pulse(cfg, seed, direction, ms_a, F.adjusted_state(os_a, os_u_alt, "v_equal_scale", qc_alt), splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, F.adjusted_state(os_u_alt, os_a, "v_equal_scale", qc_alt), splits)[0]
            cg_s = run_10step_pulse(cfg, seed, direction, ms_a, F.adjusted_state(os_a, os_u_alt, "v_other_equal_scale", qc_alt), splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, F.adjusted_state(os_u_alt, os_a, "v_other_equal_scale", qc_alt), splits)[0]
            dc_dir["global_common_scale_own"].append(cg_o)
            dc_dir["global_common_scale_swap"].append(cg_s)
            dc_dir["global_pattern_effect"].append(cg_o - cg_s)

            for prefix_name, grp_dict in [("layer4", layer4_indices), ("tensor8", tensor8_indices)]:
                sa_g = copy.deepcopy(os_a); su_g = copy.deepcopy(os_u_alt)
                sa_g_sw = E.compose_optimizer_state(os_a, os_u_alt, "v_other")
                su_g_sw = E.compose_optimizer_state(os_u_alt, os_a, "v_other")
                for idxs in grp_dict.values():
                    qt = float(np.sqrt(subset_q_stat(os_a, idxs) * subset_q_stat(os_u_alt, idxs)))
                    scale_subset_to_target(sa_g, idxs, qt); scale_subset_to_target(su_g, idxs, qt)
                    scale_subset_to_target(sa_g_sw, idxs, qt); scale_subset_to_target(su_g_sw, idxs, qt)
                co = run_10step_pulse(cfg, seed, direction, ms_a, sa_g, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_g, splits)[0]
                cs = run_10step_pulse(cfg, seed, direction, ms_a, sa_g_sw, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_g_sw, splits)[0]
                dc_dir[f"{prefix_name}_equal_scale_own"].append(co)
                dc_dir[f"{prefix_name}_equal_scale_swap"].append(cs)
                dc_dir[f"{prefix_name}_pattern_effect"].append(co - cs)

            # Row/Col within encoder.0.weight
            def make_rc(own_s, donor_s, mode_rc):
                bs = copy.deepcopy(own_s) if donor_s is None else E.compose_optimizer_state(own_s, donor_s, "v_other")
                g = bs["param_groups"][0]
                lr, eps = g["lr"], g["eps"]
                for idx_p, k in enumerate(g["params"]):
                    va = os_a["state"][os_a["param_groups"][0]["params"][idx_p]]["exp_avg_sq"].double()
                    vu = os_u_alt["state"][os_u_alt["param_groups"][0]["params"][idx_p]]["exp_avg_sq"].double()
                    vb = bs["state"][k]["exp_avg_sq"].double()
                    if idx_p == 0:
                        n_sl = vb.shape[0] if mode_rc == "row" else vb.shape[1]
                        for sl in range(n_sl):
                            sl_i = (sl, slice(None)) if mode_rc == "row" else (slice(None), sl)
                            qa_s = float((lr / (torch.sqrt(va[sl_i] / bc2) + eps)).mean())
                            qu_s = float((lr / (torch.sqrt(vu[sl_i] / bc2) + eps)).mean())
                            qt_s = np.sqrt(qa_s * qu_s)
                            lo, hi = -40.0, 40.0
                            for _ in range(60):
                                mid = 0.5 * (lo + hi)
                                if float((lr / (torch.sqrt(vb[sl_i] * np.exp(mid) / bc2) + eps)).mean()) > qt_s: lo = mid
                                else: hi = mid
                            vb[sl_i] *= np.exp(0.5 * (lo + hi))
                        bs["state"][k]["exp_avg_sq"].copy_(vb.to(bs["state"][k]["exp_avg_sq"].dtype))
                    else:
                        qt_s = float(np.sqrt(subset_q_stat(os_a, [idx_p]) * subset_q_stat(os_u_alt, [idx_p])))
                        scale_subset_to_target(bs, [idx_p], qt_s)
                return bs

            for mrc in ["row", "col"]:
                co = run_10step_pulse(cfg, seed, direction, ms_a, make_rc(os_a, None, mrc), splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, make_rc(os_u_alt, None, mrc), splits)[0]
                cs = run_10step_pulse(cfg, seed, direction, ms_a, make_rc(os_a, os_u_alt, mrc), splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, make_rc(os_u_alt, os_a, mrc), splits)[0]
                dc_dir[f"{mrc}_equal_scale_own"].append(co)
                dc_dir[f"{mrc}_equal_scale_swap"].append(cs)
                dc_dir[f"{mrc}_pattern_effect"].append(co - cs)

            # Layer-4 equal movement per step
            branches = []
            for rname, ms_r, own_r, oth_r in [("acquisition", ms_a, os_a, os_u_alt), ("update", ms_u_alt, os_u_alt, os_a)]:
                for md in ["own", "v_other"]:
                    m_b = B.make_model(cfg, DEVICE); m_b.load_state_dict(ms_r)
                    o_b = torch.optim.Adam(m_b.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay)
                    o_b.load_state_dict(E.compose_optimizer_state(own_r, oth_r, md))
                    s0_b = D.aligned_metrics(m_b, probe, test, cfg, DEVICE, direction)["aligned_s"]
                    branches.append((rname, md, m_b, o_b, s0_b))
            rng_p = np.random.default_rng(700_000 + seed)
            order_p = np.random.default_rng(710_000 + seed)
            sched_p = []
            for _ in range(6):
                pr = [0, 1]; order_p.shuffle(pr); sched_p.extend(pr)
            acq_p, w_p = E.pulse_condition("combined_switch", direction)
            for step_p in range(1, 11):
                idx_p = torch.from_numpy(rng_p.integers(0, len(train.a), size=cfg.batch_size))
                x_p = D.acquisition_tensor(train, acq_p, sched_p[step_p - 1])[idx_p]
                befs, dlts = [], []
                for _, _, m_b, o_b, _ in branches:
                    ps_b = list(m_b.parameters())
                    bf = [p.detach().clone() for p in ps_b]
                    B.train_weighted(m_b, o_b, x_p, train.a[idx_p], train.b[idx_p], w_p)
                    dl = [p.detach() - b for p, b in zip(ps_b, bf)]
                    befs.append(bf); dlts.append(dl)
                for grp in layer4_indices.values():
                    nrms = [float(torch.sqrt(sum((dl[i].double() ** 2).sum() for i in grp))) for dl in dlts]
                    tgt_n = float(np.exp(np.mean(np.log(np.maximum(nrms, 1e-30)))))
                    for (_, _, m_b, _, _), bf, dl, nr in zip(branches, befs, dlts, nrms):
                        ps_b = list(m_b.parameters())
                        sc_f = tgt_n / nr if nr > 0 else 1.0
                        with torch.no_grad():
                            for i in grp: ps_b[i].copy_(bf[i] + dl[i] * sc_f)
            lm_res = {(rn, md): s0_b - D.aligned_metrics(m_b, probe, test, cfg, DEVICE, direction)["aligned_s"] for rn, md, m_b, _, s0_b in branches}
            ln_o = lm_res[("acquisition", "own")] - lm_res[("update", "own")]
            ln_s = lm_res[("acquisition", "v_other")] - lm_res[("update", "v_other")]
            dc_dir["layer4_norm_own"].append(ln_o)
            dc_dir["layer4_norm_swap"].append(ln_s)
            dc_dir["layer4_norm_effect"].append(ln_o - ln_s)

            # Geometric mean & median global inverse scale
            for st_n in ["gmean", "median"]:
                qt_st = float(np.sqrt(subset_q_stat(os_a, all_indices, stat=st_n) * subset_q_stat(os_u_alt, all_indices, stat=st_n)))
                sa_s = copy.deepcopy(os_a); su_s = copy.deepcopy(os_u_alt)
                sa_ssw = E.compose_optimizer_state(os_a, os_u_alt, "v_other")
                su_ssw = E.compose_optimizer_state(os_u_alt, os_a, "v_other")
                scale_subset_to_target(sa_s, all_indices, qt_st, stat=st_n)
                scale_subset_to_target(su_s, all_indices, qt_st, stat=st_n)
                scale_subset_to_target(sa_ssw, all_indices, qt_st, stat=st_n)
                scale_subset_to_target(su_ssw, all_indices, qt_st, stat=st_n)
                co_s = run_10step_pulse(cfg, seed, direction, ms_a, sa_s, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_s, splits)[0]
                cs_s = run_10step_pulse(cfg, seed, direction, ms_a, sa_ssw, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_ssw, splits)[0]
                dc_dir[f"{st_n}_pattern_effect"].append(co_s - cs_s)

            # Scalar v + equal a_v checks (own m, swapped m, zero m)
            sa_eq = F.adjusted_state(os_a, os_u_alt, "scalar_v_equal", qc_alt)
            su_eq = F.adjusted_state(os_u_alt, os_a, "scalar_v_equal", qc_alt)
            dc_dir["scalar_v_equal"].append(run_10step_pulse(cfg, seed, direction, ms_a, sa_eq, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_eq, splits)[0])
            sa_eq_msw = E.compose_optimizer_state(sa_eq, su_eq, "m_other")
            su_eq_msw = E.compose_optimizer_state(su_eq, sa_eq, "m_other")
            dc_dir["scalar_v_equal_m_swap"].append(run_10step_pulse(cfg, seed, direction, ms_a, sa_eq_msw, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_eq_msw, splits)[0])
            sa_eq_m0 = copy.deepcopy(sa_eq); su_eq_m0 = copy.deepcopy(su_eq)
            for s in sa_eq_m0["state"].values(): s["exp_avg"].zero_()
            for s in su_eq_m0["state"].values(): s["exp_avg"].zero_()
            dc_dir["scalar_v_equal_m_zero"].append(run_10step_pulse(cfg, seed, direction, ms_a, sa_eq_m0, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_eq_m0, splits)[0])

            # Pure discounted ancestor v0 transplant (give Acq Update's v, give Update beta2^500 * v0)
            su_anc = copy.deepcopy(os_u_alt)
            disc = 0.999 ** 500
            for ku, k0 in zip(su_anc["param_groups"][0]["params"], no["param_groups"][0]["params"]):
                su_anc["state"][ku]["exp_avg_sq"] = no["state"][k0]["exp_avg_sq"].detach().clone() * disc
            sa_from_u = E.compose_optimizer_state(os_a, os_u_alt, "v_other")
            dc_dir["ancestor_discounted_v_swap"].append(run_10step_pulse(cfg, seed, direction, ms_a, sa_from_u, splits)[0] - run_10step_pulse(cfg, seed, direction, ms_u_alt, su_anc, splits)[0])

        for k in tolerances: stat_match[k].append(0.5 * sum(md_dir[k]))
        for k in stat_contrasts: stat_contrasts[k].append(0.5 * sum(sc_dir[k]))
        stat_v_ratios.append(0.5 * sum(vr_d)); stat_q_ratios.append(0.5 * sum(qr_d))
        stat_canc_e1.append(0.5 * sum(ce1_d)); stat_canc_e2.append(0.5 * sum(ce2_d))
        stat_seed_rows.append({"seed": seed, **{f"match_{k}": stat_match[k][-1] for k in tolerances}, **{f"contrast_{k}": stat_contrasts[k][-1] for k in stat_contrasts}})

        for k in decomp_modes: decomp_vals[k].append(0.5 * sum(dc_dir[k]))
        decomp_seed_rows.append({"seed": seed, **{k: decomp_vals[k][-1] for k in decomp_modes}})
        cka_pair_f.append(0.5 * sum(ckp_d)); cka_anc_acq_f.append(0.5 * sum(ckaa_d)); cka_anc_upd_f.append(0.5 * sum(ckau_d))
        corr_w2_acq.append(0.5 * sum(cwa_d)); corr_w2_upd.append(0.5 * sum(cwu_d))

    # 3. PCGrad donor-v common-scale check (seeds 38000..38039)
    pc_dir = PCGRAD_DIR / "runs"
    enc_indices = [i for i, n in enumerate(param_names) if n.startswith("encoder.")]
    pc_raw, pc_global_eq, pc_enc_eq = [], [], []
    for seed in range(38000, 38040):
        splits = B.make_splits(cfg, seed)
        ep_pc = torch.load(pc_dir / f"seed_{seed}/formation_checkpoints.pt", map_location="cpu", weights_only=True)
        dr, dg, de = [], [], []
        for direction in ["A", "B"]:
            ms_n, os_n = ep_pc[("acquisition", direction)]
            _, os_p = ep_pc[("projected", direction)]
            r_own = run_10step_pulse(cfg, seed, direction, ms_n, os_n, splits)[0]
            st_raw = E.compose_optimizer_state(os_n, os_p, "v_other")
            r_raw = run_10step_pulse(cfg, seed, direction, ms_n, st_raw, splits)[0]
            st_g = copy.deepcopy(st_raw); scale_subset_to_target(st_g, all_indices, subset_q_stat(os_n, all_indices))
            r_g = run_10step_pulse(cfg, seed, direction, ms_n, st_g, splits)[0]
            st_e = copy.deepcopy(st_raw); scale_subset_to_target(st_e, enc_indices, subset_q_stat(os_n, enc_indices))
            r_e = run_10step_pulse(cfg, seed, direction, ms_n, st_e, splits)[0]
            dr.append(r_raw - r_own); dg.append(r_g - r_own); de.append(r_e - r_own)
        pc_raw.append(float(np.mean(dr))); pc_global_eq.append(float(np.mean(dg))); pc_enc_eq.append(float(np.mean(de)))

    # 4. Target-error trajectories from archived follow-up CSVs (seeds 20000..20039)
    all_tc = []
    for seed in confirm_seeds:
        with (FOLLOW_DIR / f"results_confirm/seed_{seed}/timecourse.csv").open() as f:
            all_tc.extend(csv.DictReader(f))
    target_error_summary = {}
    for step in [0, 10, 50, 500]:
        target_error_summary[str(step)] = {}
        for mode in ["own", "v_other", "reset", "sgd_calibrated", "v_equal_scale", "v_other_equal_scale", "scalar_v_equal"]:
            sub = [r for r in all_tc if r["trajectory"] == "continued_opposition" and int(r["step"]) == step and r["mode"] == mode]
            if not sub: continue
            by_s = {}
            for r in sub:
                by_s.setdefault((int(r["seed"]), r["route"]), []).append((float(r["target_D"]), float(r["other_D"]), float(r["error_sum"])))
            at = [np.mean([x[0] for x in by_s[(s, "acquisition")]]) for s in confirm_seeds]
            ut = [np.mean([x[0] for x in by_s[(s, "update")]]) for s in confirm_seeds]
            ao = [np.mean([x[1] for x in by_s[(s, "acquisition")]]) for s in confirm_seeds]
            uo = [np.mean([x[1] for x in by_s[(s, "update")]]) for s in confirm_seeds]
            target_error_summary[str(step)][mode] = {
                "acquisition_orig_D": boot_ci(at),
                "update_orig_D": boot_ci(ut),
                "diff_orig_D": boot_ci([a - u for a, u in zip(at, ut)]),
                "acquisition_dest_D": boot_ci(ao),
                "update_dest_D": boot_ci(uo),
                "diff_dest_D": boot_ci([a - u for a, u in zip(ao, uo)]),
            }

    summary = {
        "stationary_within_batch_update": {
            "selected_weight": best_w,
            "pilot_grid": pilot_grid,
            "n_confirm_seeds": 40,
            "formation_match": {k: {**boot_ci(stat_match[k]), "margin": tolerances[k], "pass": bool(boot_ci(stat_match[k])["low"] >= -tolerances[k] and boot_ci(stat_match[k])["high"] <= tolerances[k])} for k in tolerances},
            "contrasts": {k: boot_ci(stat_contrasts[k]) for k in stat_contrasts},
            "switch_counts_n80": {"acquisition": sw_acq_stat, "stationary_update": sw_upd_stat},
            "v_sum_ratio_acq_over_stat_upd": boot_ci(stat_v_ratios),
            "mean_inv_scale_ratio_acq_over_stat_upd": boot_ci(stat_q_ratios),
            "cancellation_enc1": boot_ci(stat_canc_e1),
            "cancellation_enc2": boot_ci(stat_canc_e2),
        },
        "saved_checkpoint_decompositions": {
            "n_seeds": 40,
            "effects": {k: boot_ci(decomp_vals[k]) for k in decomp_modes},
            "cka_followup": {
                "acquisition_vs_update": boot_ci(cka_pair_f),
                "acquisition_vs_ancestor": boot_ci(cka_anc_acq_f),
                "update_vs_ancestor": boot_ci(cka_anc_upd_f),
            },
            "layer1_neuron_bottleneck_coupling_corr": {
                "acquisition": boot_ci(corr_w2_acq),
                "update": boot_ci(corr_w2_upd),
            },
        },
        "pcgrad_scale_control": {
            "n_seeds": 40,
            "native_recipient_projected_v_raw": boot_ci(pc_raw, seed=2026101704, n_boot=10000),
            "native_recipient_projected_v_equal_global_scale": boot_ci(pc_global_eq, seed=2026101704, n_boot=10000),
            "native_recipient_projected_v_equal_encoder_scale": boot_ci(pc_enc_eq, seed=2026101704, n_boot=10000),
        },
        "target_error_trajectories": target_error_summary,
    }

    E.write_csv(out_dir / "stationary_update_seed_summary.csv", stat_seed_rows)
    E.write_csv(out_dir / "saved_decomposition_seed_summary.csv", decomp_seed_rows)
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    prov = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 1,
        "deterministic": True,
        "elapsed_seconds": round(time.time() - t0, 2),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "result_sha256": sha256_file(out_dir / "result.json"),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print("Completed IV-G review controls in", prov["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
