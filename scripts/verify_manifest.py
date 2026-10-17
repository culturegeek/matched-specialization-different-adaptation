"""Verify every supplied file listed in the release SHA256SUMS."""
from pathlib import Path
import argparse
import hashlib
import re

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
a = p.parse_args()
root = a.root.resolve()
seen = set()
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    expected, rel = line.split("  ", 1)
    assert re.fullmatch(r"[0-9a-f]{64}", expected), rel
    f = root / rel
    assert not Path(rel).is_absolute() and ".." not in Path(rel).parts, rel
    assert f.resolve().is_relative_to(root) and rel not in seen, rel
    seen.add(rel)
    assert f.is_file(), f"Missing: {rel}"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == expected, f"Hash mismatch: {rel}"
print(f"PASS: {len(seen)} supplied files; manifest itself excluded")
