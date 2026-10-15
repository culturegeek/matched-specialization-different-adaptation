#!/usr/bin/env python3
"""実験III-Dの測定解像度と中立継続を監査する。

元の確認結果は変更せず、同じseed・生成条件を再構成して、
1) x_noneとx_bothによる同一学習軌道の再測定、
2) x_bothを用いたA/B等重み更新での中立継続、
を実行する。
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiment_iii_d import (
    Config,
    FORMED_BRANCHES,
    PRIMARY_ACQUISITION,
    acquisition_tensor,
    clone_optimizer,
    make_model,
    make_splits,
    prepare_branches,
    resolved_side,
    ridge_error,
    train_selected,
    train_weighted,
    write_csv,
)


MEASUREMENTS = ("x_none", "x_both")
UPDATE_SIDES = ("reference", "alternative")
SEEDS = tuple(range(6200, 6240))
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_SAMPLES = 10_000
OUTPUT_DIR = Path(__file__).resolve().parent / "results_measurement_neutral_audit"


def confirm_config() -> Config:
    cfg = Config()
    cfg.train_size = 3072
    cfg.probe_size = 768
    cfg.test_size = 1536
    cfg.source_dim = 3
    cfg.hidden_dim = 32
    cfg.bottleneck = 2
    cfg.correlation = 0.35
    cfg.low_noise = 1.20
    cfg.high_noise = 0.05
    cfg.formation_noise = 0.15
    cfg.batch_size = 256
    cfg.balanced_steps = 100
    cfg.formation_steps = 500
    cfg.trajectory_steps = 400
    cfg.checkpoint_interval = 25
    cfg.early_checkpoint_until = 100
    cfg.early_checkpoint_interval = 5
    cfg.formation_weight = 0.85
    cfg.learning_rate = 2e-3
    cfg.weight_decay = 1e-5
    cfg.probe_ridge = 1e-3
    cfg.branch_threshold = 0.08
    cfg.old_requirement_weight = 0.80
    cfg.bootstrap_samples = BOOTSTRAP_SAMPLES
    cfg.bootstrap_seed = BOOTSTRAP_SEED
    return cfg


def checkpoints(cfg: Config):
    values = set(range(0, min(cfg.early_checkpoint_until, cfg.trajectory_steps) + 1, cfg.early_checkpoint_interval))
    values.update(range(cfg.checkpoint_interval, cfg.trajectory_steps + 1, cfg.checkpoint_interval))
    values.add(cfg.trajectory_steps)
    return sorted(values)


@torch.no_grad()
def aligned_metrics(model, probe, test, cfg, device, measurement: str, reference: str):
    probe_x = getattr(probe, measurement).to(device)
    test_x = getattr(test, measurement).to(device)
    z_probe = model.encoder(probe_x)
    z_test = model.encoder(test_x)
    d_a = ridge_error(z_probe, probe.a.to(device), z_test, test.a.to(device), cfg.probe_ridge)
    d_b = ridge_error(z_probe, probe.b.to(device), z_test, test.b.to(device), cfg.probe_ridge)
    s_ab = (d_b - d_a) / (d_a + d_b + 1e-12)
    if reference == "A":
        return d_a, d_b, d_a, d_b, s_ab
    return d_a, d_b, d_b, d_a, -s_ab


def record_rows(model, probe, test, cfg, device, seed, reference, condition, step):
    rows = []
    for measurement in MEASUREMENTS:
        d_a, d_b, d_ref, d_alt, s_ref = aligned_metrics(
            model, probe, test, cfg, device, measurement, reference
        )
        rows.append({
            "seed": seed,
            "reference": reference,
            "condition": condition,
            "measurement": measurement,
            "step": step,
            "probe_D_A": d_a,
            "probe_D_B": d_b,
            "D_ref": d_ref,
            "D_alt": d_alt,
            "s_ref": s_ref,
        })
    return rows


def run_primary_dual(
    cfg,
    seed,
    reference,
    acquisition,
    update,
    model_state,
    optimizer_state,
    train,
    probe,
    test,
    device,
):
    model = make_model(cfg, device)
    model.load_state_dict(model_state)
    optimizer = clone_optimizer(model, cfg, optimizer_state)
    batch_rng = np.random.default_rng(300_000 + seed)
    sensor_rng = np.random.default_rng(400_000 + seed)
    n = train.a.shape[0]
    update_absolute = resolved_side(update, reference)
    record_at = set(checkpoints(cfg))
    condition = f"acq_{acquisition}__upd_{update}"
    rows = record_rows(model, probe, test, cfg, device, seed, reference, condition, 0)
    for step in range(1, cfg.trajectory_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        random_choose_a = bool(sensor_rng.integers(0, 2))
        x_source = acquisition_tensor(train, acquisition, reference, random_choose_a).to(device)
        train_selected(
            model,
            optimizer,
            x_source[idx],
            train.a.to(device)[idx],
            train.b.to(device)[idx],
            update_absolute == "A",
        )
        if step in record_at:
            rows.extend(record_rows(model, probe, test, cfg, device, seed, reference, condition, step))
    return rows


def run_neutral(
    cfg,
    seed,
    reference,
    model_state,
    optimizer_state,
    train,
    probe,
    test,
    device,
):
    model = make_model(cfg, device)
    model.load_state_dict(model_state)
    optimizer = clone_optimizer(model, cfg, optimizer_state)
    batch_rng = np.random.default_rng(300_000 + seed)
    n = train.a.shape[0]
    record_at = set(checkpoints(cfg))
    rows = record_rows(model, probe, test, cfg, device, seed, reference, "neutral_equal_update", 0)
    for step in range(1, cfg.trajectory_steps + 1):
        idx = torch.from_numpy(batch_rng.integers(0, n, size=cfg.batch_size)).to(device)
        train_weighted(
            model,
            optimizer,
            train.x_both.to(device)[idx],
            train.a.to(device)[idx],
            train.b.to(device)[idx],
            0.5,
        )
        if step in record_at:
            rows.extend(record_rows(model, probe, test, cfg, device, seed, reference, "neutral_equal_update", step))
    return rows


def percentile_ci(values, rng):
    values = np.asarray(values, dtype=float)
    sampled = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(sampled, 0.025)),
        "high": float(np.quantile(sampled, 0.975)),
    }


def first_cross(rows, boundary, comparison):
    for row in sorted(rows, key=lambda item: item["step"]):
        if row["step"] == 0:
            continue
        if comparison(row["s_ref"], boundary):
            return float(row["step"]), True
    return 400.0, False


def time_average(rows, end_step=100):
    selected = sorted((r for r in rows if r["step"] <= end_step), key=lambda item: item["step"])
    steps = np.asarray([r["step"] for r in selected], dtype=float)
    values = np.asarray([r["s_ref"] for r in selected], dtype=float)
    return float(np.trapezoid(values, steps) / end_step)


def group_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["seed"], row["reference"], row["condition"], row["measurement"])
        grouped.setdefault(key, []).append(row)
    return grouped


def seed_average(branch_values):
    result = {}
    for (seed, reference), value in branch_values.items():
        result.setdefault(seed, []).append(value)
    return {seed: float(np.mean(values)) for seed, values in result.items()}


def summarize(primary_rows, neutral_rows, cfg):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = group_rows(primary_rows)
    neutral = group_rows(neutral_rows)
    summary = {"measurement_audit": {}, "neutral_continuation": {}}

    for measurement in MEASUREMENTS:
        subset = [r for r in primary_rows if r["measurement"] == measurement]
        d_values = [r["probe_D_A"] for r in subset] + [r["probe_D_B"] for r in subset]
        s_values = [r["s_ref"] for r in subset]
        measurement_result = {
            "D_range": [float(min(d_values)), float(max(d_values))],
            "s_ref_range": [float(min(s_values)), float(max(s_values))],
        }

        zero_effect = {}
        auc_effect = {}
        formation = {}
        finals = {}
        for seed in SEEDS:
            for reference in FORMED_BRANCHES:
                ref_key = (seed, reference, "acq_reference__upd_alternative", measurement)
                alt_key = (seed, reference, "acq_alternative__upd_alternative", measurement)
                ref_time, _ = first_cross(primary[ref_key], 0.0, lambda value, boundary: value <= boundary)
                alt_time, _ = first_cross(primary[alt_key], 0.0, lambda value, boundary: value <= boundary)
                zero_effect[(seed, reference)] = ref_time - alt_time
                auc_effect[(seed, reference)] = time_average(primary[ref_key]) - time_average(primary[alt_key])
                formation[(seed, reference)] = primary[ref_key][0]["s_ref"]
                finals[(seed, reference)] = {
                    "reference_acquisition": primary[ref_key][-1]["s_ref"],
                    "alternative_acquisition": primary[alt_key][-1]["s_ref"],
                }

        zero_seed = seed_average(zero_effect)
        auc_seed = seed_average(auc_effect)
        form_seed = seed_average(formation)
        measurement_result["formation_s_ref"] = percentile_ci(list(form_seed.values()), rng)
        measurement_result["zero_crossing_time_effect"] = percentile_ci(list(zero_seed.values()), rng)
        measurement_result["first_100_s_auc_effect"] = percentile_ci(list(auc_seed.values()), rng)
        measurement_result["audit_direction_pass"] = (
            measurement_result["zero_crossing_time_effect"]["low"] > 0
            and measurement_result["first_100_s_auc_effect"]["low"] > 0
        )

        threshold_sensitivity = {}
        for threshold in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            effects = {}
            events = []
            for seed in SEEDS:
                for reference in FORMED_BRANCHES:
                    ref_key = (seed, reference, "acq_reference__upd_alternative", measurement)
                    alt_key = (seed, reference, "acq_alternative__upd_alternative", measurement)
                    ref_time, ref_event = first_cross(
                        primary[ref_key], -threshold, lambda value, boundary: value <= boundary
                    )
                    alt_time, alt_event = first_cross(
                        primary[alt_key], -threshold, lambda value, boundary: value <= boundary
                    )
                    effects[(seed, reference)] = ref_time - alt_time
                    events.extend([ref_event, alt_event])
            effect_seed = seed_average(effects)
            threshold_sensitivity[f"{threshold:.2f}"] = {
                "time_effect": percentile_ci(list(effect_seed.values()), rng),
                "event_rate": float(np.mean(events)),
            }
        measurement_result["threshold_sensitivity"] = threshold_sensitivity
        summary["measurement_audit"][measurement] = measurement_result

    for measurement in MEASUREMENTS:
        initial = {}
        final = {}
        ratio = {}
        retained = {}
        half_time = {}
        for seed in SEEDS:
            for reference in FORMED_BRANCHES:
                key = (seed, reference, "neutral_equal_update", measurement)
                ordered = sorted(neutral[key], key=lambda item: item["step"])
                initial_value = ordered[0]["s_ref"]
                final_value = ordered[-1]["s_ref"]
                initial[(seed, reference)] = initial_value
                final[(seed, reference)] = final_value
                ratio[(seed, reference)] = final_value / initial_value
                retained[(seed, reference)] = float(final_value >= cfg.branch_threshold)
                half, _ = first_cross(
                    ordered,
                    0.5 * initial_value,
                    lambda value, boundary: value <= boundary,
                )
                half_time[(seed, reference)] = half
        initial_seed = seed_average(initial)
        final_seed = seed_average(final)
        ratio_seed = seed_average(ratio)
        retained_seed = seed_average(retained)
        half_seed = seed_average(half_time)
        retained_mean = float(np.mean(list(retained_seed.values())))
        if retained_mean >= 0.90:
            classification = "original_threshold_retained"
        elif retained_mean < 0.50:
            classification = "formed_bias_mostly_decayed"
        else:
            classification = "intermediate"
        summary["neutral_continuation"][measurement] = {
            "initial_s_ref": percentile_ci(list(initial_seed.values()), rng),
            "final_s_ref": percentile_ci(list(final_seed.values()), rng),
            "final_over_initial": percentile_ci(list(ratio_seed.values()), rng),
            "retention_rate_at_0_08": percentile_ci(list(retained_seed.values()), rng),
            "half_decay_time_restricted": percentile_ci(list(half_seed.values()), rng),
            "classification": classification,
        }
    return summary


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_text(summary, runtime):
    low = summary["measurement_audit"]["x_none"]
    common = summary["measurement_audit"]["x_both"]
    neutral_low = summary["neutral_continuation"]["x_none"]
    neutral_common = summary["neutral_continuation"]["x_both"]
    lines = [
        "# 実験III-D：測定解像度と中立継続の追加監査",
        "",
        "## 確認1：共通低雑音入力での再測定",
        "",
        f"- 元の低精度入力のD範囲: {low['D_range'][0]:.4f}--{low['D_range'][1]:.4f}",
        f"- 共通低雑音入力のD範囲: {common['D_range'][0]:.4f}--{common['D_range'][1]:.4f}",
        f"- 元の低精度入力のs範囲: {low['s_ref_range'][0]:.4f}--{low['s_ref_range'][1]:.4f}",
        f"- 共通低雑音入力のs範囲: {common['s_ref_range'][0]:.4f}--{common['s_ref_range'][1]:.4f}",
        "",
        f"- 共通低雑音入力のゼロ交差時間差: {common['zero_crossing_time_effect']['mean']:.3f} "
        f"(95%区間 [{common['zero_crossing_time_effect']['low']:.3f}, {common['zero_crossing_time_effect']['high']:.3f}])",
        f"- 共通低雑音入力の最初の100更新s時間平均差: {common['first_100_s_auc_effect']['mean']:.4f} "
        f"(95%区間 [{common['first_100_s_auc_effect']['low']:.4f}, {common['first_100_s_auc_effect']['high']:.4f}])",
        f"- 取得効果方向の二指標確認: {'PASS' if common['audit_direction_pass'] else 'FAIL'}",
        "",
        "## 確認2：両側等重みの中立継続",
        "",
        f"- 元の低精度測定: 初期s={neutral_low['initial_s_ref']['mean']:.4f}, "
        f"最終s={neutral_low['final_s_ref']['mean']:.4f}, 残存比={neutral_low['final_over_initial']['mean']:.4f}, "
        f"閾値保持率={neutral_low['retention_rate_at_0_08']['mean']:.3f}",
        f"- 共通低雑音測定: 初期s={neutral_common['initial_s_ref']['mean']:.4f}, "
        f"最終s={neutral_common['final_s_ref']['mean']:.4f}, 残存比={neutral_common['final_over_initial']['mean']:.4f}, "
        f"閾値保持率={neutral_common['retention_rate_at_0_08']['mean']:.3f}",
        f"- 元の操作的基準による分類: {neutral_low['classification']}",
        "",
        "## 読み方",
        "",
        "この追加監査は元の事前登録結果を置き換えない。共通低雑音入力による測定は取得効果の測定依存性を、"
        "中立継続は形成済み偏差が片側更新なしでも残るかを調べる。",
        "",
        f"- 実行時間: {runtime:.1f}秒",
    ]
    return "\n".join(lines) + "\n"


def main():
    cfg = confirm_config()
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    primary_rows = []
    neutral_rows = []
    for index, seed in enumerate(SEEDS, 1):
        train, probe, test = make_splits(cfg, seed)
        branches = prepare_branches(cfg, seed, train, device)
        for reference in FORMED_BRANCHES:
            model_state, optimizer_state = branches[reference]
            for acquisition in PRIMARY_ACQUISITION:
                for update in UPDATE_SIDES:
                    primary_rows.extend(
                        run_primary_dual(
                            cfg,
                            seed,
                            reference,
                            acquisition,
                            update,
                            model_state,
                            optimizer_state,
                            train,
                            probe,
                            test,
                            device,
                        )
                    )
            neutral_rows.extend(
                run_neutral(
                    cfg,
                    seed,
                    reference,
                    model_state,
                    optimizer_state,
                    train,
                    probe,
                    test,
                    device,
                )
            )
        print(f"seed {index}/{len(SEEDS)} 完了", flush=True)

    expected_primary = len(SEEDS) * len(FORMED_BRANCHES) * 4 * len(checkpoints(cfg)) * len(MEASUREMENTS)
    expected_neutral = len(SEEDS) * len(FORMED_BRANCHES) * len(checkpoints(cfg)) * len(MEASUREMENTS)
    if len(primary_rows) != expected_primary or len(neutral_rows) != expected_neutral:
        raise RuntimeError(
            f"不完全な結果: primary={len(primary_rows)}/{expected_primary}, "
            f"neutral={len(neutral_rows)}/{expected_neutral}"
        )
    summary = summarize(primary_rows, neutral_rows, cfg)
    runtime = time.time() - started
    result = {
        "experiment": "III-D measurement-resolution and neutral-continuation audit",
        "design": {
            "seeds": list(SEEDS),
            "measurements": list(MEASUREMENTS),
            "primary_conditions": 4,
            "neutral_condition": "x_both acquisition; 0.5/0.5 weighted update",
            "statistical_unit": "seed; A/B formed states averaged within seed",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "completeness": {
                "primary_rows": len(primary_rows),
                "expected_primary_rows": expected_primary,
                "neutral_rows": len(neutral_rows),
                "expected_neutral_rows": expected_neutral,
                "pass": True,
            },
        },
        "config": asdict(cfg),
        "summary": summary,
        "runtime_seconds": runtime,
    }
    write_csv(OUTPUT_DIR / "primary_dual_probe_timecourse.csv", primary_rows)
    write_csv(OUTPUT_DIR / "neutral_timecourse.csv", neutral_rows)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    script_path = Path(__file__).resolve()
    protocol_path = script_path.parent / "audit_protocol_measurement_neutral_20260827.md"
    source_path = script_path.parent / "experiment_iii_d.py"
    provenance = {
        "audit_script_sha256": sha256(script_path),
        "audit_protocol_sha256": sha256(protocol_path),
        "original_experiment_script_sha256": sha256(source_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
    }
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report = report_text(summary, runtime)
    (OUTPUT_DIR / "result_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
