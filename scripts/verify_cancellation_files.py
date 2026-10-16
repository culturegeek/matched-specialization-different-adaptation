from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[1]
m=json.loads((r/'provenance/cancellation_files.json').read_text())
for p,h in m.items():
 assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
print(f'PASS: {len(m)} cancellation-intervention files')
