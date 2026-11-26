# Equal-weight observation schedules

This exploratory control holds the favored loss weight in both update histories at 0.80. The acquisition checkpoint is shared by both comparisons. One update history assigns a single precision condition to each batch and balances it across the 500 updates. The other presents 128 examples of each precision condition in every 256-example batch. The same ancestor, latent examples, batch indices and ten-update opposing pulse are used for each seed and target direction.

The analysis plan was recorded before this run, after the original 0.79 within-batch results and the alternating results had been inspected. Evaluation uses all 40 previous follow-up seeds (20000–20039). Each seed contributes the average of the two target directions. Each acquisition/update pair is normalized to its own geometric mean initial multiplier before comparing own and exchanged patterns.

The 0.80 within-batch route passes all five endpoint tolerances against both existing routes. Under pairwise mean-scale matching the alternating route contrast changes from +0.0875 to −0.0110, and the within-batch contrast from −0.0035 to +0.0025. The paired difference in the own-minus-exchanged pattern effect is +0.1045 (95% percentile interval [0.0987, 0.1102]). For the mixed route the two contrasts have intervals spanning zero, while their paired change has a resolved direction. The original calibrated 0.79 control remains under the parent directory.

## Files and reproduction

`plan.json` fixes the estimands, seed range, tolerances and bootstrap. `results/directional.csv` and `per_seed.csv` (in the same directory) give the 80 directional records and 40 seed pairs. `results/checkpoints/` contains the two new formation states per seed. `results/result.json` gives all estimates, endpoint decisions and update-ten switch counts. `results/provenance.json` binds the plan, runner, inputs and outputs to SHA-256 hashes. The ordinary and scale-matched effects use the same sign convention as the paper.

From the root of the submission supplement, verify the archived numbers without training, or repeat the experiment in a fresh directory:

```sh
python experiments/iv-g/equal-weight-080/audit.py --archive-root .
python experiments/iv-g/equal-weight-080/run.py --archive-root . --out /tmp/ivg-equal-weight-rerun
```

The new run used CPU PyTorch 2.13.0, NumPy 2.4.6, one thread and deterministic algorithms. Each alternating scale-matched seed contrast was also checked against the original saved table.

As a reproduction of the earlier calibrated control, `calibrated-replay/directional.csv` gives the inherited-state readout for all 80 directions before and after ten updates at weight 0.79. To check the reported counts (62/80 and 8/80) and all 40 prior seed means using only Python, run `python experiments/iv-g/equal-weight-080/calibrated-replay/check.py --archive-root .`. To recreate that directional CSV, add `--replay-to /tmp/calibrated-directions.csv` with the training environment.
