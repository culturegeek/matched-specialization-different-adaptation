# IV-F: endpoint prediction, transfer, and mechanism

This extension tests whether measured endpoint differences predict response, whether component effects transfer across model and target changes, and whether they survive equal parameter movement.

## Evidence

- `results_saved`: exploratory analysis of three disjoint 40-seed cohorts and the 40 saved follow-up checkpoints. Fixed endpoint features provide little improvement over the held-out training-mean predictor. Correlation alone does not exclude confounding.
- `results_general`: the same 32 new seeds at baseline, width 64, ReLU, and independent-target settings. All pass endpoint tolerances and reverse the route contrast under v exchange. Width 64 fails reset equivalence.
- `results_beta`: five additional settings, completing beta2 {0.9,0.99,0.999} crossed with 500/1500 formation updates (the baseline is in results_general). All pass endpoint tolerances and show a positive exchange effect at common mean inverse scaling. Reversal and reset equivalence fail in some settings.
- `results_mechanism`: interventions on 40 existing checkpoint pairs. Encoder-v exchanges change the response strongly; head-v exchanges have little effect. An exchange effect remains under equal global update norms. This is a controlled learning rule, not a natural-trajectory mediation fraction.

New training comprises 288 setting-seed pairs, but only 32 unique new seeds; conditions share those seeds. Mirrored directions are averaged within seed. No seeds were dropped or added based on outcomes. The locally frozen protocol and registration hashes precede all new runs; they are not public preregistration. `runs_*` preserve every seed and mode; individual 95% intervals do not provide simultaneous coverage.

## Reproduce

From the repository or anonymous archive root, with the extension dependencies installed:

```sh
python -m pip install -r requirements-extension.txt
python experiments/iv-extension/audit_extension.py
python experiments/iv-extension/analyze_saved.py --archive-root . --out generated/ivf_saved_new
python experiments/iv-extension/run_extension.py --phase general --out generated/ivf_general_new --workers 2
python experiments/iv-extension/run_extension.py --phase beta --out generated/ivf_beta_new --workers 2
python experiments/iv-extension/run_extension.py --phase mechanism --archive-root . --out generated/ivf_mechanism_new --workers 1
python experiments/iv-extension/analyze_extension.py --input generated/ivf_general_new --out generated/ivf_general_analysis_new
python experiments/iv-extension/analyze_extension.py --input generated/ivf_beta_new --out generated/ivf_beta_analysis_new
python experiments/iv-extension/analyze_extension.py --input generated/ivf_mechanism_new --out generated/ivf_mechanism_analysis_new
```

Each output directory must be new. Keep registration files and their bound source files unchanged; do not run `--phase lock` again. Each worker uses CPU and one PyTorch thread. Exact environment versions are in `registration_experiments.json`. The saved checkpoint and six pulse modes for seed 20000 were reproduced exactly before new runs; `pre_run_validation.json` records this check. The independent audit checks means, completeness, finite values, starts, norm constraints, and frozen hashes; it does not rerun training.

This extension retains a two-target synthetic regression task. Three-target adaptation, natural datasets, SGD formation, and weight-layer swaps remain future experiments.
