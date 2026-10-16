# Printed-interval audit — v1.5.0

All 92 printed confidence intervals in each language are bound to exact source JSON keys in interval_bindings.json. verify_manuscript_intervals.py checks source values, decimal rounding, bilingual agreement and completeness. No interval is inferred by numerical proximity during verification. The source hash is stored per occurrence in INTERVAL_LEDGER.csv and interval_ledger.json. Removed narrative intervals remain in earlier release tags and original result files.

```sh
python scripts/verify_manuscript_intervals.py --archive-root . --tex-root paper/interval_sources --bindings analyses/manuscript_intervals/interval_bindings.json --out-dir /tmp/recomputed-interval-ledger
```
