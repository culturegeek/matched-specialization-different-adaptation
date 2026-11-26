"""Verify previously reported update-ten counts at w=.79, optionally regenerating the directional records."""
import argparse
import csv
import json
from pathlib import Path
import sys

def replay(root, output):
    import torch
    parent = root / "experiments/iv-g"
    sys.path.insert(0, str(parent))
    import run_review_controls as study

    cfg = study.D.config_for_stage("confirm")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "direction", "acq_start_s", "acq_end_s", "mixed_start_s", "mixed_end_s"))
        for seed in range(20000, 20040):
            splits = study.B.make_splits(cfg, seed)
            ancestor = study.D.prepare_neutral(cfg, seed, splits[0], study.DEVICE)
            saved = root / f"experiments/iv-followup/results_confirm/seed_{seed}/formation_checkpoints.pt"
            original = torch.load(saved, map_location="cpu", weights_only=True)
            for direction in ("A", "B"):
                mixed = study.form_within_batch_update(cfg, seed, direction, 0.79, *ancestor, splits[0])
                row = [seed, direction]
                for model, optimizer in (original["acquisition", direction], mixed[:2]):
                    _, start, end = study.run_10step_pulse(cfg, seed, direction, model, optimizer, splits)
                    row += [start["aligned_s"], end["aligned_s"]]
                writer.writerow(row)

def verify(root, path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r["seed"]),r["direction"]) for r in rows] == [
        (seed, direction) for seed in range(20000,20040) for direction in ("A","B")]
    original = root / "experiments/iv-g/results"
    records = json.loads((original/"result.json").read_text())["stationary_within_batch_update"]
    with (original/"stationary_update_seed_summary.csv").open(newline="") as handle:
        by_seed = {int(r["seed"]): float(r["contrast_own"]) for r in csv.DictReader(handle)}
    for seed in by_seed:
        same = [r for r in rows if int(r["seed"]) == seed]
        contrast = sum(float(r["acq_start_s"])-float(r["acq_end_s"])-float(r["mixed_start_s"])+float(r["mixed_end_s"]) for r in same) / len(same)
        assert abs(contrast-by_seed[seed]) < 1e-12, seed
    found = {k: sum(float(r[k+"_end_s"])<=0 for r in rows) for k in ("acq","mixed")}
    archived = records["switch_counts_n80"]
    assert found == {"acq":archived["acquisition"],"mixed":archived["stationary_update"]}, (found, archived)
    print(f"PASS: calibrated w=0.79, update 10: {found['acq']}/80 acquisition and {found['mixed']}/80 within-batch; all 40 contrasts match the archived seed table.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--replay-to", type=Path, help="Recreate the CSV at a new path before verification")
    args = parser.parse_args()
    csv_path = args.replay_to or Path(__file__).with_name("directional.csv")
    if args.replay_to:
        replay(args.archive_root.resolve(),csv_path)
    verify(args.archive_root.resolve(),csv_path)
