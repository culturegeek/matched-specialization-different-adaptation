# V-B: three-target adaptation and optimizer-component exchange

The question is whether similar measured endpoints conceal different responses with three targets, and whether second-moment exchange changes that response. The experiment extends V-A's symmetric three-target Gaussian system to all six ordered adaptation directions, at bottleneck capacities two and three.

Eight calibration seeds (33000–33007) select a target loss weight using endpoints only. Both capacities select 0.65 from nine candidates. Forty previously unused confirmation seeds (34000–34039) are shared across the two capacities; this is 40 independent seeds, not 80. All endpoint tolerances pass at both capacities. The ten-update contrast changes from 0.1945 to -0.1005 at capacity two, and from 0.4054 to -0.2193 at capacity three, under v exchange. Exchange effects at common mean inverse scaling remain positive. First-moment exchange changes the contrast within the fixed margin. Reset equivalence passes at capacity two; the capacity-three interval crosses the upper margin, leaving equivalence unresolved. All six directions and the secondary 50-update horizon are saved.

## Reproduce

Use the repository's requirements-extension.txt environment (Python 3.12.14, NumPy 2.4.6, pandas 2.2.3, PyTorch 2.13.0, CPU). From this experiment directory:

```sh
python audit_adaptation.py
python run_adaptation.py --phase confirm --out /tmp/vb_rerun --workers 2
python analyze_adaptation.py --phase analyze --input /tmp/vb_rerun --out /tmp/vb_reanalysis
```

Each output directory must be new. Preserve registration.json and its bound files; do not rerun the lock step. The calibration command is `python analyze_adaptation.py --phase calibrate --input runs_pilot --out /tmp/vb_calibration.json`. The resulting JSON must match calibration.json. Rerunning training needs no downloaded dataset.

## Records and limits

- pilot_protocol.md fixes the calibration rule, seeds, comparisons, margins, and stopping rule. runs_pilot preserves every candidate; calibration.json records the selected endpoints.
- registration.json locally freezes the plan, code, analysis, and selection before confirmation. This is not public preregistration.
- runs_confirm stores all formation states, six-direction trajectories, actual update norms, mean inverse-scaling diagnostics, and per-run metadata. No seed or failed criterion is excluded.
- results_confirm/result.json stores the estimates and criteria; its seed-level and direction-level CSVs expose the statistical unit and the third-target response.
- audit_adaptation.py independently checks raw-CSV means and bootstrap intervals, row completeness, identical mode starts, metric identities, scaling constraints, and hashes. independent_audit.json records the 1,603 passing checks.
- pre_run_validation.json records exact agreement with the archived V-A seed 8200 for both formation routes, and recipient-start checks on excluded smoke seed 32999.

The response uses the change in pairwise allocation s_ij=(D_j-D_i)/(D_j+D_i), with directions averaged within seed; all three centered log-error scores are also stored. Individual 95% bootstrap intervals are not simultaneous-coverage guarantees. These artificial observations remain three-dimensional Gaussian targets, and formation and adaptation use Adam. Reset equivalence and reversal have distinct criteria. Capacity effects are descriptive; the calibration rule allows a weight per capacity, although both selected the same value.
