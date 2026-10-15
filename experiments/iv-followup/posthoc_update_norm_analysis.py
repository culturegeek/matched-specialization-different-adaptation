"""Exploratory link from Adam scaling to realized parameter motion.

This analysis reads only the frozen IV-E follow-up CSV outputs.  It asks how
the pre-intervention mean inverse-second-moment scaling, the realized first
parameter update, and the cumulative ten-update path length relate to the
observed change in the allocation index.  Because realized update norms are
measured during the intervention, they are mechanistic descriptors rather
than pre-treatment causal predictors.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results_confirm"
OUT_JSON = ROOT / "posthoc_update_norm_results.json"
OUT_CSV = ROOT / "posthoc_update_norm_seed_table.csv"
BOOTSTRAP_SEED = 20260904
N_BOOT = 20_000


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "n_seed": int(len(values)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = []
    while len(values) < N_BOOT:
        idx = rng.integers(0, len(x), size=len(x))
        value = corr(x[idx], y[idx], method)
        if np.isfinite(value):
            values.append(value)
    return {
        "estimate": corr(x, y, method),
        "ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
        "n_seed": int(len(x)),
    }


def loocv_linear(feature: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    feature = np.asarray(feature, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    pred = np.empty_like(outcome)
    baseline = np.empty_like(outcome)
    for i in range(len(outcome)):
        keep = np.arange(len(outcome)) != i
        design = np.column_stack([np.ones(keep.sum()), feature[keep]])
        coef, *_ = np.linalg.lstsq(design, outcome[keep], rcond=None)
        pred[i] = coef[0] + coef[1] * feature[i]
        baseline[i] = outcome[keep].mean()
    mse = float(np.mean((pred - outcome) ** 2))
    baseline_mse = float(np.mean((baseline - outcome) ** 2))
    return {
        "mse": mse,
        "baseline_mse": baseline_mse,
        "mse_reduction_fraction": float(1.0 - mse / baseline_mse),
        "rmse": float(np.sqrt(mse)),
        "baseline_rmse": float(np.sqrt(baseline_mse)),
    }


def load_diagnostics() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA.glob("seed_*/update_diagnostics.csv")):
        frames.append(pd.read_csv(path))
    diagnostics = pd.concat(frames, ignore_index=True)
    return diagnostics[
        (diagnostics["trajectory"] == "continued_opposition")
        & diagnostics["step"].between(1, 10)
    ].copy()


def load_timecourse() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA.glob("seed_*/timecourse.csv")):
        frames.append(pd.read_csv(path))
    timecourse = pd.concat(frames, ignore_index=True)
    return timecourse[
        (timecourse["trajectory"] == "continued_opposition")
        & timecourse["step"].isin([0, 10])
    ].copy()


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diagnostics = load_diagnostics()
    timecourse = load_timecourse()
    predictors = pd.read_csv(DATA / "seed_predictors.csv")
    moments = pd.read_csv(DATA / "optimizer_moments.csv")

    expected_modes = {
        "own", "m_other", "v_other", "reset", "v_equal_scale",
        "v_other_equal_scale", "scalar_v_equal", "scalar_v_own",
        "sgd_base", "sgd_calibrated",
    }
    assert set(diagnostics["mode"].unique()) == expected_modes
    assert diagnostics.groupby(["seed", "route", "direction", "mode"]).size().eq(10).all()
    assert timecourse.groupby(["seed", "route", "direction", "mode"]).size().eq(2).all()
    assert not diagnostics.isna().any().any()
    assert not timecourse.isna().any().any()

    grouped = diagnostics.groupby(["seed", "route", "direction", "mode"], as_index=False).agg(
        first_update_norm=("update_norm", "first"),
        update_path_length=("update_norm", "sum"),
        gradient_path_length=("gradient_norm", "sum"),
    )
    grouped["path_scale"] = grouped["update_path_length"] / grouped["gradient_path_length"]
    grouped = grouped.groupby(["seed", "route", "mode"], as_index=False).agg(
        first_update_norm=("first_update_norm", "mean"),
        update_path_length=("update_path_length", "mean"),
        gradient_path_length=("gradient_path_length", "mean"),
        path_scale=("path_scale", "mean"),
    )

    aligned = timecourse.groupby(["seed", "route", "direction", "mode", "step"], as_index=False)["aligned_s"].mean()
    wide_s = aligned.pivot(index=["seed", "route", "direction", "mode"], columns="step", values="aligned_s").reset_index()
    wide_s["response_drop"] = wide_s[0] - wide_s[10]
    response = wide_s.groupby(["seed", "route", "mode"], as_index=False)["response_drop"].mean()
    grouped = grouped.merge(response, on=["seed", "route", "mode"], validate="one_to_one")

    registered_route_effects = pd.read_csv(DATA / "seed_route_effects.csv")
    registered_route_effects = registered_route_effects[
        (registered_route_effects["trajectory"] == "continued_opposition")
        & (registered_route_effects["step"] == 10)
    ][["seed", "mode", "route_drop_difference"]]
    raw_route_effects = (
        grouped.pivot(index=["seed", "mode"], columns="route", values="response_drop")
        .reset_index()
    )
    raw_route_effects["raw_route_drop_difference"] = (
        raw_route_effects["acquisition"] - raw_route_effects["update"]
    )
    audit = registered_route_effects.merge(raw_route_effects, on=["seed", "mode"], validate="one_to_one")
    audit_max_difference = float(
        np.max(np.abs(audit["route_drop_difference"] - audit["raw_route_drop_difference"]))
    )
    assert audit_max_difference < 1e-12

    own_grouped = grouped[grouped["mode"] == "own"]
    acq = own_grouped[own_grouped["route"] == "acquisition"].set_index("seed")
    upd = own_grouped[own_grouped["route"] == "update"].set_index("seed")
    seeds = sorted(set(acq.index) & set(upd.index))
    table = pd.DataFrame({"seed": seeds})
    for field in [
        "first_update_norm",
        "update_path_length",
        "gradient_path_length",
        "path_scale",
        "response_drop",
    ]:
        a = acq.loc[seeds, field].to_numpy()
        u = upd.loc[seeds, field].to_numpy()
        table[f"{field}_acquisition"] = a
        table[f"{field}_update"] = u
        table[f"{field}_difference"] = a - u
        table[f"{field}_ratio"] = a / u
    table = table.merge(predictors[["seed", "v_ratio", "q_ratio", "m_ratio"]], on="seed", validate="one_to_one")
    pooled_q = (
        moments.groupby(["seed", "route"], as_index=False)["effective_lr_mean"].mean()
        .pivot(index="seed", columns="route", values="effective_lr_mean")
    )
    pooled_q["q_pooled_ratio"] = pooled_q["acquisition"] / pooled_q["update"]
    table = table.merge(
        pooled_q[["q_pooled_ratio"]].reset_index(), on="seed", validate="one_to_one"
    )
    table.to_csv(OUT_CSV, index=False)

    result: dict[str, object] = {
        "status": "post_hoc_exploratory",
        "data_scope": "frozen 20000--20039 confirmation CSVs; own Adam; first 10 opposing updates",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOT,
        "n_seed": int(len(table)),
        "audit": {
            "raw_timecourse_vs_registered_seed_route_effects_max_abs_difference": audit_max_difference,
            "diagnostic_group_rows": 10,
            "timecourse_group_rows": 2,
            "source_tree_sha256": None,
        },
        "descriptive": {},
        "correlations": {},
        "loocv_linear_prediction": {},
        "intervention_modes": {},
    }

    summary_fields = [
        "response_drop_difference",
        "response_drop_ratio",
        "first_update_norm_difference",
        "first_update_norm_ratio",
        "update_path_length_difference",
        "update_path_length_ratio",
        "gradient_path_length_difference",
        "gradient_path_length_ratio",
        "path_scale_difference",
        "path_scale_ratio",
        "q_ratio",
        "q_pooled_ratio",
    ]
    for field in summary_fields:
        result["descriptive"][field] = bootstrap_mean(table[field].to_numpy(), rng)

    pairs = {
        "q_ratio_to_first_update_norm_ratio": ("q_ratio", "first_update_norm_ratio"),
        "q_ratio_to_ten_step_update_path_ratio": ("q_ratio", "update_path_length_ratio"),
        "q_ratio_to_response_difference": ("q_ratio", "response_drop_difference"),
        "q_ratio_to_response_ratio": ("q_ratio", "response_drop_ratio"),
        "first_update_norm_difference_to_response_difference": (
            "first_update_norm_difference",
            "response_drop_difference",
        ),
        "first_update_norm_ratio_to_response_ratio": ("first_update_norm_ratio", "response_drop_ratio"),
        "ten_step_update_path_difference_to_response_difference": (
            "update_path_length_difference",
            "response_drop_difference",
        ),
        "ten_step_update_path_ratio_to_response_ratio": ("update_path_length_ratio", "response_drop_ratio"),
        "path_scale_difference_to_response_difference": ("path_scale_difference", "response_drop_difference"),
    }
    for name, (x_name, y_name) in pairs.items():
        x = table[x_name].to_numpy()
        y = table[y_name].to_numpy()
        result["correlations"][name] = {
            "pearson": bootstrap_corr(x, y, "pearson", rng),
            "spearman": bootstrap_corr(x, y, "spearman", rng),
        }

    for feature in [
        "q_ratio",
        "first_update_norm_difference",
        "update_path_length_difference",
        "path_scale_difference",
    ]:
        result["loocv_linear_prediction"][f"{feature}_to_response_difference"] = loocv_linear(
            table[feature].to_numpy(), table["response_drop_difference"].to_numpy()
        )
    for feature in ["q_ratio", "first_update_norm_ratio", "update_path_length_ratio"]:
        result["loocv_linear_prediction"][f"{feature}_to_response_ratio"] = loocv_linear(
            table[feature].to_numpy(), table["response_drop_ratio"].to_numpy()
        )

    mode_means = []
    for mode in sorted(grouped["mode"].unique()):
        mode_data = grouped[grouped["mode"] == mode]
        mode_acq = mode_data[mode_data["route"] == "acquisition"].set_index("seed")
        mode_upd = mode_data[mode_data["route"] == "update"].set_index("seed")
        mode_seeds = sorted(set(mode_acq.index) & set(mode_upd.index))
        mode_result = {}
        for field in ["response_drop", "first_update_norm", "update_path_length", "gradient_path_length", "path_scale"]:
            difference = mode_acq.loc[mode_seeds, field].to_numpy() - mode_upd.loc[mode_seeds, field].to_numpy()
            ratio = mode_acq.loc[mode_seeds, field].to_numpy() / mode_upd.loc[mode_seeds, field].to_numpy()
            mode_result[f"{field}_difference"] = bootstrap_mean(difference, rng)
            mode_result[f"{field}_ratio"] = bootstrap_mean(ratio, rng)
        path_diff = (
            mode_acq.loc[mode_seeds, "update_path_length"].to_numpy()
            - mode_upd.loc[mode_seeds, "update_path_length"].to_numpy()
        )
        response_diff = (
            mode_acq.loc[mode_seeds, "response_drop"].to_numpy()
            - mode_upd.loc[mode_seeds, "response_drop"].to_numpy()
        )
        mode_result["update_path_difference_to_response_difference"] = {
            "pearson": bootstrap_corr(path_diff, response_diff, "pearson", rng),
            "spearman": bootstrap_corr(path_diff, response_diff, "spearman", rng),
        }
        result["intervention_modes"][mode] = mode_result
        mode_means.append([
            mode_result["update_path_length_difference"]["mean"],
            mode_result["response_drop_difference"]["mean"],
        ])
    mode_means = np.asarray(mode_means)
    result["across_mode_mean_association"] = {
        "n_modes": int(len(mode_means)),
        "update_path_difference_to_response_difference_pearson": corr(
            mode_means[:, 0], mode_means[:, 1], "pearson"
        ),
        "description": "descriptive association across intervention-arm means, not a seed-level causal estimate",
    }

    digest = hashlib.sha256()
    source_paths = sorted(DATA.glob("seed_*/update_diagnostics.csv")) + sorted(DATA.glob("seed_*/timecourse.csv"))
    source_paths += [
        DATA / "seed_predictors.csv",
        DATA / "seed_route_effects.csv",
        DATA / "optimizer_moments.csv",
    ]
    for path in source_paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    result["audit"]["source_tree_sha256"] = digest.hexdigest()

    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
