# Manuscript interval audit

The ledger covers 164 printed confidence-interval occurrences and two observed ranges in each language. Each row records its source JSON key, raw endpoints, printed precision, and source hash. Numerical manuscript fragments in paper/interval_sources are anonymous audit inputs.

Three exploratory intervals are newly reconstructed from archived paired seed values: the pooled mirrored-history depth contrast (80 seeds from two disjoint cohorts), IV-A terminal alignment (40 seeds), and IV-A acquisition main effect (40 seeds). They use 20,000 percentile bootstrap resamples, with mirrored conditions averaged within seed. Their original printed intervals lacked a recoverable saved calculation. The new calculations preserve point estimates and change three rounded endpoints; they do not recover the original random streams. Existing registered criteria and frozen results are unchanged.

From the archive root:

```sh
python scripts/audit_manuscript_intervals.py --archive-root . --out-dir /tmp/recomputed_intervals
python scripts/verify_manuscript_intervals.py --archive-root . --tex-root paper/interval_sources --bindings analyses/manuscript_intervals/interval_bindings.json --out-dir /tmp/verified_intervals
```

Compare the regenerated derived_intervals.json and seed values with analyses/manuscript_intervals. Figure 3 also omits the crowded tick at 200; plotted data are unchanged. The new IV-F intervals retain their frozen analysis streams. The original core intervals remain separately identified.

## v1.2.0 intervals

The ledger additionally binds every interval in Appendices K and L to the V-B/V-C result JSONs. The independent experiment audits reproduce their bootstrap intervals from per-seed CSVs. English/Japanese parity and all printed interval occurrences are checked by the same complete-coverage verifier.

## v1.3.0 intervals

The reset column adds six occurrences, the exploratory paired reset-minus-own comparison adds two, and the head-gradient table adds ten. Coverage is 155 confidence intervals plus two observed ranges per language. Moved tables retain their source bindings under the revised manuscript paragraph locations. The head estimates and intervals use the documented 40-seed analysis, separate from the earlier confirmatory experiments.

## v1.4.0 intervals

Nine exploratory occurrences are added: three endpoint associations and six formation-accounting estimates. The current ledger binds 164 confidence intervals plus two observed ranges per language, preserving all earlier registered results.
