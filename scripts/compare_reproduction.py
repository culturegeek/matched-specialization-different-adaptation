"""Compare a newly reproduced seed-9100 trajectory with the archived trajectory."""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
FOLLOW = ROOT / 'experiments/iv-followup'
KEYS = ('seed', 'route', 'direction', 'mode', 'trajectory', 'step')

def read(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[k] for k in KEYS): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate trajectory keys')
    return indexed

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=FOLLOW/'results_reproduce_new')
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    old = read(FOLLOW/'results_reproduce/seed_9100/timecourse.csv')
    new = read(args.folder/'seed_9100/timecourse.csv')
    if old.keys() != new.keys():
        raise ValueError('Trajectory keys do not match')
    maximum = 0.0
    for key, row in old.items():
        if row.keys() != new[key].keys():
            raise ValueError('Columns do not match')
        for column, value in row.items():
            if column in KEYS:
                continue
            a, b = float(value), float(new[key][column])
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError('Non-finite measurement')
            maximum = max(maximum, abs(a-b))
    report = {'pass': maximum <= args.atol, 'seed': 9100, 'rows': len(old),
              'maximum_absolute_difference': maximum, 'absolute_tolerance': args.atol}
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
