"""Verify the supplied IV-G records without training or overwriting archived results."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
DATA = EXP / 'iv-g/results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def verify_bootstrap(seed_rows, column, reference):
    values = np.array([float(row[column]) for row in seed_rows])
    rng = np.random.default_rng(20260903)
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    np.testing.assert_allclose([values.mean(),lo,hi],[reference['mean'],reference['low'],reference['high']],
                               rtol=0,atol=1e-12,err_msg=column)


def main():
    recorded = json.loads((DATA/'provenance.json').read_text())
    adapted = json.loads((DATA.parent/'archive_runner.json').read_text())
    assert adapted['original_script_sha256'] == recorded['script_sha256']
    assert adapted['archive_runner_sha256'] == sha(DATA.parent/'run_review_controls.py')
    assert recorded['result_sha256'] == sha(DATA/'result.json')
    data = json.loads((DATA/'result.json').read_text())
    stationary = data['stationary_within_batch_update']
    station_rows = rows(DATA/'stationary_update_seed_summary.csv')
    decomp = rows(DATA/'saved_decomposition_seed_summary.csv')
    for items in (station_rows, decomp):
        assert [int(item['seed']) for item in items] == list(range(20000, 20040))
    tests = 0
    for prefix, group in [('match_',stationary['formation_match']),('contrast_',stationary['contrasts'])]:
        for key, ref in group.items():
            verify_bootstrap(station_rows,prefix+key,ref)
            if prefix=='match_':
                assert ref['pass'] == (-ref['margin'] <= ref['low'] and ref['high'] <= ref['margin'])
            tests += 1
    for key, ref in data['saved_checkpoint_decompositions']['effects'].items():
        verify_bootstrap(decomp,key,ref)
        tests += 1
    for i in range(20000,20040):
        folder=EXP/f'iv-followup/results_confirm/seed_{i}'
        assert (folder/'formation_checkpoints.pt').is_file()
        assert (folder/'timecourse.csv').is_file()
    for i in range(38000,38040):
        folder=EXP/f'iv-cancellation/runs/seed_{i}'
        assert (folder/'formation_checkpoints.pt').is_file()
    assert (EXP/'iv-followup/vendor/experiment_iv_complete_v05/IV-E/experiment_iv_e.py').is_file()
    print(f'PASS: {tests} estimates, source hashes, and inputs for all 80 checkpoint sets; no training')


if __name__ == '__main__':
    main()
