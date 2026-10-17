# Formation projection and crossed second-moment transplantation

This study asks whether changing a formation rule produces a second moment that changes a fixed recipient's subsequent response. It uses 40 new paired seeds (38000–38039), two mirrored directions and 500 formation updates. The new protocol was locally hash-fixed before runs, after exploratory diagnostics; it is not a public preregistration.

PCGrad operates on the two weighted shared-encoder gradients. It changes direction, magnitude, subsequent weights and optimizer state. The crossed transplant holds recipient weights, first moment and step count fixed while changing the second-moment donor. It is not an isolated intervention on a scalar cancellation measure, nor a natural mediation analysis.

Encoder second moment rises by a geometric factor 1.4226; projected-history v changes the native recipient's ten-update response by −0.0472. Projected endpoints fail matching. All endpoints, unsuccessful criteria, both donors, both recipients and the secondary 50-update responses are retained.

`protocol_clarification.json` documents a wording mismatch: the local projection plan fixes uniform ±0.02 endpoint tolerances, whereas the original low-precision tolerance was ±0.01. Matching fails under both rules; locked plans and outputs are unchanged.

## Reproduce from the repository root

```sh
python experiments/iv-cancellation/run_intervention.py --help
python experiments/iv-cancellation/analyze_intervention.py --help
python experiments/iv-cancellation/audit_intervention.py --archive-root . --input experiments/iv-cancellation/runs --out /tmp/projection-audit.json
python experiments/iv-cancellation/analyze_saved.py --archive-root . --out /tmp/new-saved-analysis
python experiments/iv-cancellation/make_figures.py --out /tmp/projection-figures
```

Use a new output directory for training; preserve the archived inputs. Python 3.12.14, NumPy 2.4.6 and PyTorch 2.13.0, single-thread CPU, generated these data. The run registration preserves source hashes. Independent checks cover native endpoint reproduction, analytic encoder gradients and projection, v updates, finite records and identical pulse starting states.

`saved_results/` is a separate exploratory analysis of existing formation records and nine distinct endpoint settings (six in the decay/duration grid). It reports paired individual-square differences, observation strata and duration changes in geometric route ratios. It does not create new training seeds. The ratio widens at beta2=0.999 but contracts at 0.9 and 0.99; this limits a universal mean-scale explanation.

## Audit portability (v1.5.1)

`audit_intervention.py` now supports `--tolerance 1e-6` and an optional declared `--seeds` subset. Default comparisons remain strict. Only replayed floating weights and moments use the documented tensor-specific tolerance; source hashes, schedules, step counts and state-transplant identity stay exact. Inspect `numerical_comparison` and `full_cohort` in the output. The cross-version checks and rejection tests are in `analyses/final_checks_20261017/README.md`.
