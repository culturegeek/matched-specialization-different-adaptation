"""Reconstruct three exploratory intervals from archived seed-level observations.

This adds a dated, explicit analysis; it does not recover an undocumented old
random stream or alter any registered experiment. No training is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_root
    paths = [
        'experiments/ii-a/results/forced_selection_causal_results/raw_timecourse.csv',
        'experiments/ii-a/results/forced_selection_causal_confirmatory_results/raw_timecourse.csv',
        'experiments/iv/IV-A/results_measurement_neutral_audit/primary_dual_probe_timecourse.csv',
    ]
    frames = [pd.read_csv(root / path) for path in paths]
    data = pd.concat(frames[:2], ignore_index=True)
    data = data[data.common_step == 800].copy()
    assert data.seed.nunique() == 80
    assert data.groupby(['seed', 'condition']).size().eq(2).all()
    data['abs_s'] = data.test_probe_index.abs()
    wide = data.groupby(['seed', 'condition']).abs_s.mean().unstack().sort_index()
    depth = wide.stabilization - wide.fixed_propensity

    data = frames[2]
    data = data[(data.measurement == 'x_both') & (data.step == 400)].copy()
    assert data.seed.nunique() == 40
    assert data.groupby(['seed', 'condition']).size().eq(2).all()
    data['abs_s'] = data.s_ref.abs()
    wide = data.groupby(['seed', 'condition']).abs_s.mean().unstack().sort_index()
    rr = wide.acq_reference__upd_reference
    ra = wide.acq_reference__upd_alternative
    ar = wide.acq_alternative__upd_reference
    aa = wide.acq_alternative__upd_alternative
    contrasts = {
        'ii_mirrored_absolute_depth': (depth, 2026101601,
            'At update 800, average absolute test-probe allocation over the two mirrored arms within seed; subtract fixed propensity from state-following. Pool the two disjoint 40-seed cohorts.'),
        'iv_a_terminal_alignment': ((rr + aa - ra - ar) / 2, 2026101602,
            'At update 400 under x_both, average absolute aligned allocation over the two formed target labels within seed; aligned acquisition/update mean minus crossed mean.'),
        'iv_a_acquisition_main_effect': ((rr + ra - ar - aa) / 2, 2026101603,
            'At update 400 under x_both, reference-acquisition minus alternative-acquisition absolute allocation, averaging the two update conditions and formed labels within seed.'),
    }
    result = {'analysis_date': '2026-10-16', 'status': 'exploratory_reconstruction',
        'interval_method': 'Paired seed bootstrap, percentile 95%, 20000 resamples; each contrast has an independent fixed random stream. Mirrored arms are never independent resampling units.',
        'scope': 'Reconstructed intervals supersede the three undocumented manuscript intervals. Historical raw data and registered result files remain unchanged.',
        'input_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'contrasts': {}}
    rows = []
    for name, (series, seed, definition) in contrasts.items():
        values = series.to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        boot = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
        result['contrasts'][name] = {'mean': float(values.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist(), 'n_seed': len(values),
            'bootstrap_seed': seed, 'definition': definition}
        rows.extend({'contrast': name, 'seed': int(s), 'value': float(v)} for s, v in series.items())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'derived_intervals.json').write_text(json.dumps(result, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(args.out_dir / 'derived_interval_seed_values.csv', index=False)
    print(json.dumps(result['contrasts'], indent=2))


if __name__ == '__main__':
    main()
