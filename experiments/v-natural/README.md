# V-C: natural three-room sensor reconstruction

Does the optimizer-component effect observed with three Gaussian targets also occur with real cross-target and temporal structure? This experiment uses recorded temperature and humidity in the kitchen, living room, and laundry room of one house. It reconstructs recorded measurements under controlled input noise; it is not future forecasting or the original appliance-energy benchmark.

Eight pilot seeds select room-specific weights without pulse responses or confirmation blocks. Even the best endpoint discrepancies exceed the matching tolerances. The fixed protocol requires confirmation without extending the search. Forty new seeds (36000–36039), reused across capacities two and three, confirm that endpoints do not match: 1/24 and 0/24 criteria pass. The v-exchange effects on the ten-update route contrast are 0.0854 and 0.1261; at common mean inverse scale they are 0.0103 and 0.0257. All four intervals are positive. Swapped contrasts are negative at capacity two but unresolved around zero at capacity three. Reset and first-moment-change intervals lie within the fixed margins at ten updates. The inherited contrast exceeds the 0.05 criterion only at capacity three. All failed criteria and secondary 50-update results remain in the archive.

## Reproduce

Use the root requirements-extension.txt environment (Python 3.12.14, pandas 2.2.3, NumPy 2.4.6, PyTorch 2.13.0, CPU). From this directory:

```sh
python audit_natural.py
python run_natural.py --phase confirm --out /tmp/vc_rerun --workers 2
python analyze_natural.py --phase analyze --input /tmp/vc_rerun --out /tmp/vc_reanalysis
```

Use a new output directory. Do not overwrite frozen results or rerun the lock step. To recalculate endpoint selection use `python analyze_natural.py --phase calibrate --input runs_pilot --out /tmp/vc_calibration.json`; the JSON must match calibration.json. The redistributed original CSV is included, so rerunning requires no download.

## Data, records, and inference

- [Dataset attribution](data/ATTRIBUTION.md) identifies UCI Appliances Energy Prediction, DOI 10.24432/C5VC8G, by Luis Candanedo (2017), under CC BY 4.0. Original raw CSV is unmodified; selected columns, chronology, training-only scaling and checksum are in data/split_manifest.json.
- protocol.md fixes data columns, 13 weight candidates, 8 pilot and 40 confirmation seeds, comparisons, margins and stopping. registration.json binds its files before confirmation; this is a local freeze, not public preregistration.
- runs_pilot retains every candidate. runs_confirm retains all formation checkpoints, six ordered directions, six modes, six time points, all three errors/log-error scores, actual update norms and sensor diagnostics.
- results_confirm/result.json distinguishes pooled summaries from the required room-specific endpoint criteria. Positive component effects do not make the unmatched route comparison a matched-endpoint replication.
- audit_natural.py checks independent CSV arithmetic, bootstrap intervals, completeness, starts, scale matching, frozen hashes, time gaps, standardization, room-level gates and input sensitivity. Its 1,859 checks pass. Training means use compensated high-precision addition to avoid treating summation-order roundoff as a scientific mismatch.
- pre_run_validation.json records time separation, training-only scaling, independent float64 ridge arithmetic and identical starts. Excluded smoke seed 34999 is not a confirmation seed.

The 95% intervals resample seeds, averaging six directions within seed. They measure training/noise variability conditional on one building and fixed time blocks, not population uncertainty over homes or periods. Individual intervals are not simultaneous confidence bands. The 24-hour gaps do not guarantee temporal independence. Model training and pulse updates use only the first block; subsequent blocks fit and test fresh readouts. New data and independent calibration are needed for a matched natural-data claim.
