"""Read-only integrity checks for released experiment files; standard library only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / "provenance/source_files.json").read_text())
    failures = []
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            failures.append(record["path"])
    follow = ROOT / "experiments/iv-followup"
    registration = json.loads((follow / "registration.json").read_text())
    frozen = json.loads((follow / "results_confirm/manifest.json").read_text())
    for records in [registration["file_hashes"], frozen]:
        for name, expected in records.items():
            path = follow / name
            if not path.is_file() or digest(path) != expected:
                failures.append("frozen:" + name)
    report = {"pass": not failures, "preserved_source_files": len(manifest["files"]),
              "registration_files": len(registration["file_hashes"]),
              "frozen_result_files": len(frozen), "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
