#!/usr/bin/env python3
"""Equal-weight stationarity follow-up. Run with --archive-root and a fresh --out directory."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    parent = root / "experiments/iv-g"
    sys.path.insert(0, str(parent))
    import run_review_controls as base

    here = Path(__file__).resolve().parent
    plan_path = here / "plan.json"
    plan = json.loads(plan_path.read_text())
    tolerances = plan["endpoint_margins"]
    modes = plan["pulse_modes"]
    seeds = list(range(plan["seed_first"], plan["seed_last"] + 1))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshots = out / "checkpoints"
    snapshots.mkdir()
    direction_rows, seed_rows = [], []
    followup = root / "experiments/iv-followup"
    alt_csv = parent / "results/saved_decomposition_seed_summary.csv"
    with alt_csv.open() as handle:
        alt_record = {int(row["seed"]): row for row in csv.DictReader(handle)}
    cfg = base.D.config_for_stage("confirm")
    assert (cfg.formation_steps, cfg.initial_steps, cfg.batch_size) == (500, 100, 256)
    started = datetime.now(timezone.utc).isoformat()
    inputs = {alt_csv.relative_to(root).as_posix(): digest(alt_csv)}
    for path in [parent / "run_review_controls.py", followup / "run_followup.py",
                 *sorted((followup / "vendor").rglob("experiment_iv_*.py"))]:
        inputs[path.relative_to(root).as_posix()] = digest(path)
    max_alt_error = 0.0

    for seed in seeds:
        splits = base.B.make_splits(cfg, seed)
        train, probe, test = splits
        neutral = base.D.prepare_neutral(cfg, seed, train, base.DEVICE)
        source = followup / f"results_confirm/seed_{seed}/formation_checkpoints.pt"
        inputs[source.relative_to(root).as_posix()] = digest(source)
        stored = torch.load(source, map_location="cpu", weights_only=True)
        new_states = {}
        pairs = []
        for direction in ("A", "B"):
            acq_state = stored[("acquisition", direction)]
            alt_state = stored[("update", direction)]
            model, optimizer, cancellation = base.form_within_batch_update(
                cfg, seed, direction, plan["favored_weight"], *neutral, train, True)
            mixed_state = model, optimizer
            new_states[direction] = mixed_state
            row = {"seed": seed, "direction": direction}
            for name, state in (("acq", acq_state), ("alt", alt_state), ("mix", mixed_state)):
                learner = base.B.make_model(cfg, base.DEVICE)
                learner.load_state_dict(state[0])
                measures = base.D.aligned_metrics(learner, probe, test, cfg, base.DEVICE, direction)
                for key in tolerances:
                    row[f"{name}_start_{key}"] = measures[key]
            for key in tolerances:
                row[f"acq_minus_mix_{key}"] = row[f"acq_start_{key}"] - row[f"mix_start_{key}"]
                row[f"alt_minus_mix_{key}"] = row[f"alt_start_{key}"] - row[f"mix_start_{key}"]
            row["mixed_cancellation_encoder1"] = cancellation["canc_enc1"]
            row["mixed_cancellation_encoder2"] = cancellation["canc_enc2"]
            for name, update in (("alt", alt_state), ("mix", mixed_state)):
                target = float(np.sqrt(base.F.q_mean(acq_state[1]) * base.F.q_mean(update[1])))
                row[f"{name}_common_multiplier"] = target
                for mode in modes:
                    for recipient, own, other in (("acq", acq_state, update), ("upd", update, acq_state)):
                        state = base.F.adjusted_state(own[1], other[1], mode, target)
                        if mode in ("v_equal_scale", "v_other_equal_scale"):
                            assert np.isclose(base.F.q_mean(state), target, rtol=1e-6, atol=0)
                        response, start, end = base.run_10step_pulse(cfg, seed, direction, own[0], state, splits)
                        row[f"{name}_{mode}_{recipient}_response"] = response
                        if mode == "own":
                            row[f"{name}_{recipient}_end_s"] = end["aligned_s"]
                            assert abs(start["aligned_s"] - row[f"{'acq' if recipient == 'acq' else name}_start_aligned_s"]) < 1e-12
                    row[f"{name}_{mode}"] = row[f"{name}_{mode}_acq_response"] - row[f"{name}_{mode}_upd_response"]
                row[f"{name}_pattern"] = row[f"{name}_v_equal_scale"] - row[f"{name}_v_other_equal_scale"]
            row["alt_minus_mix_pattern"] = row["alt_pattern"] - row["mix_pattern"]
            row["alt_minus_mix_own_contrast"] = row["alt_own"] - row["mix_own"]
            direction_rows.append(row)
            pairs.append(row)
        torch.save(new_states, snapshots / f"seed_{seed}.pt")
        avg = {"seed": seed}
        for key in pairs[0]:
            if key not in ("seed", "direction"):
                avg[key] = 0.5 * (pairs[0][key] + pairs[1][key])
        for new, old in (("alt_v_equal_scale", "global_common_scale_own"),
                         ("alt_v_other_equal_scale", "global_common_scale_swap"),
                         ("alt_pattern", "global_pattern_effect")):
            error = abs(avg[new] - float(alt_record[seed][old]))
            max_alt_error = max(max_alt_error, error)
            assert error <= 1e-6, (seed, new, error)
        seed_rows.append(avg)
        if (len(seed_rows) % 10) == 0:
            print(f"{len(seed_rows)} / {len(seeds)} seeds", flush=True)
    write_rows(out / "directional.csv", direction_rows)
    write_rows(out / "per_seed.csv", seed_rows)
    estimates = {key: base.boot_ci([row[key] for row in seed_rows], plan["bootstrap_seed"], plan["bootstrap_draws"])
                 for key in seed_rows[0] if key != "seed"}
    matches = {}
    for pair in ("acq_minus_mix", "alt_minus_mix"):
        matches[pair] = {}
        for key, margin in tolerances.items():
            est = estimates[f"{pair}_{key}"]
            matches[pair][key] = (-margin <= est["low"] and est["high"] <= margin)
    switches = {name: sum(row[f"{name}_end_s"] <= 0 for row in direction_rows)
                for name in ("alt_acq", "alt_upd", "mix_acq", "mix_upd")}
    result = {"classification": plan["classification"], "n_seeds": len(seeds), "n_directions": len(direction_rows),
              "favored_update_weight": plan["favored_weight"], "estimates": estimates, "endpoint_pass": matches,
              "switches_at_update_10": switches, "max_alternating_seed_reference_error": max_alt_error}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    provenance = {"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                  "device": "cpu", "threads": torch.get_num_threads(), "deterministic": True,
                  "plan_sha256": digest(plan_path), "runner_sha256": digest(Path(__file__)),
                  "source_hashes": inputs, "output_hashes": {
                      path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

