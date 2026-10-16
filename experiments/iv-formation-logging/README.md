# Formation-time gradient accounting (exploratory)

The actual formation histories show strong cancellation between the two task gradients in the acquisition route. At 500 updates, the accumulated cross term offsets 94.2% and 93.0% of the weighted individual-square sums in encoder layers 1 and 2; update-route values are 28.4% and 25.2%. Final acquisition/update second-moment ratios are 0.242 and 0.340. This is accounting along observed trajectories, not the causal effect of changing cancellation.

## Design and data

All 40 original follow-up seeds (20000--20039), both favored-target directions and both routes are replayed. The original 100-update ancestor and 500 formation updates use unchanged batches, observations, weights, loss, Adam settings and CPU float32 arithmetic. All 160 final model/m/v states match their archived checkpoints bitwise. The 80000 formation updates produce 320000 layer-step records. These are repeated histories, not additional independent seeds.

`runs/seed_*/records.npz` contains a `values` matrix and its `fields` array. Direction 0/1 is A/B; route 0/1 is acquisition/update; layer 0/1/2/3 is encoder1/encoder2/head_a/head_b. Each seed has 8000 rows. Gradients and squared norms are recorded before each original update; v is recorded before and after it. Cosine is zero by convention for disjoint head-parameter gradients. `weighted_self` sums the two weighted individual squares; `weighted_cross` is twice their weighted inner product. `v_self`, `v_cross`, and `v_regularization` accumulate these terms with the original beta2 recursion. `v_initial` is the discounted shared ancestor. `v_rounding` retains finite-precision residuals in gradient combination and state updates. These five components sum to `v_after`.

`snapshots_*.npz` stores weights, m, v at 0,1,10,50,100,250,500, all batch indices and the precision schedule. Stored parameter names/shapes and layer sizes define the flattened coordinate order. `results/seed_final.csv`, `seed_windows.csv`, `trajectory.csv`, and `result.json` retain all layers, prespecified windows and observation strata. Positive route ratios use log averaging within direction/seed and geometric means over seeds; cancellation and ancestor fractions are direction-specific ratios averaged within seed. Individual intervals use 5000 paired-seed resamples (RNG 2026101702); they are not simultaneous confidence regions.

## Reproduction from the archive root

Use the extension dependencies. Every output directory must be new.

```sh
python experiments/iv-formation-logging/run_logging.py --archive-root . --out generated/formation-replay
python experiments/iv-formation-logging/analyze_logging.py --input generated/formation-replay --out generated/formation-analysis
python experiments/iv-formation-logging/audit_logging.py --archive-root . --input experiments/iv-formation-logging/runs --out generated/formation-audit.json
python experiments/iv-formation-logging/audit_summary.py --input experiments/iv-formation-logging/runs --result experiments/iv-formation-logging/results/result.json --out generated/formation-summary-audit.json
python experiments/iv-formation-logging/analyze_endpoint_associations.py --archive-root . --out generated/endpoint-associations
python experiments/iv-formation-logging/make_figures.py --input experiments/iv-formation-logging/runs --out generated/formation-figures
```

For a smaller replay, add `--seeds 20000`; full inference requires the original 40 seeds. Python 3.12.14, NumPy 2.4.6, PyTorch 2.13.0, CPU and one thread were used. The Japanese figure uses Hiragino when available; the English figure uses DejaVu Sans. No download or new data is needed.

## Validation and limits

`independent_audit.json` records analytic NumPy chain-rule checks on 3840 snapshot/layer records, scalar second-moment recursions on all 320000 records, and direct checks against the 160 archived endpoints. `summary_audit.json` independently checks all 108 final estimates and intervals from reshaped raw records. The largest final rounding term is below 7e-6 of layer v.

The locally fixed plan is exploratory because the original endpoint results and response hypotheses were known. `implementation_correction.json` documents a logging-only coefficient error detected by the independent audit in the first replay. Original training and endpoints were unaffected. The coefficient was corrected and every seed replayed; only corrected records are distributed as the current result. The initial specification and source are retained in `audit_history`; the full initial records remain in the private development archive. This is a technical correction, not outcome-driven seed or metric selection.

The endpoint association analysis is separately exploratory after previous estimates were known. `endpoint_associations` retains all 24 associations (four probes, two layers, three predictors), source hashes and seed values. Correlations are across 40 seeds after within-seed averaging. Small cosine variation does not prove absence of an association. RMS pooling and averaging checkpoint-specific ratios differ: under both-high encoder1, pooled other/favored ratios are 1.83/2.05; mean checkpoint-specific ratios are 1.83/2.17. The manuscript specifies the pooled convention. None of these correlations isolates the Adam numerator or excludes optimizer-state contributions.

Code: MIT. Generated data, figures and documentation: CC BY 4.0, under the repository licenses.

## Reproduction on another environment

The archived 160 endpoints were bitwise identical in the recorded CPU environment (Python 3.12.14, PyTorch 2.13.0). This is an observed validation, not a guarantee for every machine or library release. PyTorch does not guarantee identical results across releases/platforms. The original locked `run_logging.py` remains unchanged. The supported reproduction entry point `reproduce_logging.py` checks the same original lock and defaults to strict endpoint equality:

```sh
python experiments/iv-formation-logging/reproduce_logging.py --archive-root . --seeds 20000 --out /tmp/formation-strict
python experiments/iv-formation-logging/reproduce_logging.py --archive-root . --seeds 20000 --out /tmp/formation-tolerant --tolerance 1e-6
```

Tolerance is opt-in and is recorded along with actual bitwise status and maximum differences for weights, m, and v. Weights/m use max absolute difference <= tolerance * (1 + reference tensor max); v uses <= 1e-12 + tolerance * reference tensor max; step counters and non-tensor configuration stay exact. Finite values, shapes and dtypes must agree. Passing the tolerance mode is numerical agreement under these declared bounds, not bitwise equality or automatic equivalence of every subsequent scientific estimate. Always retain the validation file and compare the reported estimates. The archived analyses and their strict audits remain unchanged. See https://docs.pytorch.org/docs/stable/notes/randomness.html .
