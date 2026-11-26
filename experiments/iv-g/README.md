# Within-batch and saved-checkpoint controls (IV-G)

`results/result.json` contains the frozen estimates. The two seed-summary CSVs give paired results for the within-batch comparison and the saved-checkpoint controls. The original provenance binds the submitted result JSON to the recorded workspace runner. The supplied runner uses the same training, interventions, random streams, data, and analysis with paths relative to this archive and a required output directory.

From the archive root, in the recorded CPU environment:

```sh
python scripts/verify_iv_g.py
python experiments/iv-g/run_review_controls.py --out /tmp/reproduced-iv-g
```

The integrity script recomputes the IV-G estimates recoverable from the archived seed tables. The runner repeats the recorded pipeline using the original follow-up and projection checkpoints. The pilot only compares endpoint measures, on seeds 8000–8015. The within-batch comparison is evaluated on seeds 20000–20039 against the saved acquisition checkpoints. The normalization and movement analyses use the saved alternating comparison from the same cohort. The projection scale check uses seeds 38000–38039. Exact floating-point reproducibility depends on the recorded environment (see `results/provenance.json`).

The fixed-weight exploratory follow-up uses the very same 40 seeds with favored weight 0.80 for both update routes. Its separate plan, full directional evidence and replay command are in [equal-weight-080](equal-weight-080/README.md).
