"""Verify printed intervals against explicit, reviewed source bindings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re


PATTERN = re.compile(r'\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]')


def flatten(value, prefix='$', output=None):
    output = {} if output is None else output
    output[prefix] = value
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(v, prefix + '.' + k, output)
    elif isinstance(value, list):
        for k, v in enumerate(value):
            flatten(v, prefix + f'[{k}]', output)
    return output


def occurrences(path):
    pid, count = '', 0
    found = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith('% P:'):
            pid, count = line[4:], 0
        for match in PATTERN.finditer(line):
            count += 1
            found[(pid, count)] = (list(match.groups()), n)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-root', type=Path, required=True)
    ap.add_argument('--tex-root', type=Path, required=True)
    ap.add_argument('--bindings', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    bindings = json.loads(a.bindings.read_text())
    printed = {name: occurrences(a.tex_root / name) for name in
        ['sections_en.tex', 'appendices_en.tex', 'sections_ja.tex', 'appendices_ja.tex']}
    records = []
    for b in bindings:
        interval, line = printed[b['source_tex']].pop((b['paragraph_id'], b['interval_in_paragraph']))
        ja = b['source_tex'].replace('_en.', '_ja.')
        ja_interval, ja_line = printed[ja].pop((b['paragraph_id'], b['interval_in_paragraph']))
        assert interval == ja_interval, ('translation', b, interval, ja_interval)
        source = a.archive_root / b['source_file']
        obj = flatten(json.loads(source.read_text()))[b['source_key']]
        if isinstance(obj, dict):
            for low, high in [('low', 'high'), ('ci_low', 'ci_high'), ('ci95_low', 'ci95_high')]:
                if low in obj and high in obj:
                    obj = [obj[low], obj[high]]
                    break
        assert isinstance(obj, list) and len(obj) == 2, b
        digits = [len(v.split('.')[1]) for v in interval]
        rounded = [f'{v:.{d}f}' for v, d in zip(obj, digits)]
        assert rounded == interval, ('source_mismatch', b, obj, interval)
        records.append({**b, 'english_line': line, 'japanese_line': ja_line,
            'raw_lower': obj[0], 'raw_upper': obj[1], 'printed_lower': interval[0],
            'printed_upper': interval[1], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'status': 'matched'})
    assert all(not v for v in printed.values()), ('unbound_intervals', printed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'interval_ledger.json').write_text(json.dumps(records, indent=2) + '\n')
    with (a.out_dir / 'INTERVAL_LEDGER.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps({'confidence_interval_occurrences_per_language': sum(x['kind']=='confidence_interval' for x in records),
        'observed_range_occurrences_per_language': sum(x['kind']=='observed_range' for x in records),
        'bilingual_match': True, 'source_and_rounding_match': True}))


if __name__ == '__main__':
    main()
