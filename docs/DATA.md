# Data and statistical status

Most observations are synthetic and generated from the programs in this repository. V-C uses a separately attributed, CC BY 4.0 public sensor dataset from one house; see the natural-data section below. Data-generation parameters, random seeds, model dimensions, training lengths, and software versions are stored in the corresponding `provenance.json` files and experiment programs.

## Reading the core records

- `formation_endpoints.csv` records measurements at the end of formation.
- `protocol_summary.csv` summarizes pulse and continuation responses in the original IV-D/E studies.
- `timecourse.csv` records measurements at specified update counts. In the follow-up it is stored separately for each seed.
- `update_diagnostics.csv` records gradient norms, realized parameter-update norms, and optimizer summaries for each update in the follow-up.
- `formation_checkpoints.pt` contains the model and optimizer dictionaries saved by the follow-up program at formation. The program also regenerates these checkpoints from the recorded seeds.
- `result.json`, `route_effects.csv`, and `levels.csv` contain aggregated outputs. Their underlying per-seed records are included.

For the principal two-route contrasts, target labels are mirrored and averaged within each seed. The 40 seeds are the independent units; treating mirrored directions or time points as independent observations would inflate the sample size. The follow-up contains 27,040 timecourse rows and 571,200 update-diagnostic rows for seeds 20000–20039.

## Fixed analyses and later analyses

The follow-up plan and code hashes were fixed locally before the confirmatory run. `registration.json` is a local record, not evidence of public preregistration. Pilot, smoke, reproduction, and confirmatory directories identify different purposes and are not pooled.

Files named `posthoc_*` and `POSTHOC_*` contain exploratory analyses after the confirmatory run. In particular, the endpoint difference and the difference in change from formation are distinct quantities. The post-hoc endpoint interval at 500 updates and the confirmatory bootstrap interval use different random streams; their slight numerical difference does not indicate a different endpoint mean. Figure intervals likewise use their documented plotting stream.

Matching the mean inverse scaling factor does not match the actual parameter-update norm. The update-norm analysis is included to make this distinction measurable. SGD is a continuation condition after Adam formation. The V-A study tests three-target formation; it does not test the two-target optimizer-state intervention on three targets.

## Failed criteria and superseded runs

Experiment III preserves unsuccessful candidate-coordinate prediction and finite-state criteria. Experiment IV-C retains superseded alternating-precision formation runs in `development_superseded`, with the explanation of why they are not the canonical confirmatory comparison. Calibration and pilot results remain available alongside, rather than merged into, the confirmatory records.

## Integrity

Run `python scripts/verify_data.py` to check every preserved source file against the publication manifest and the follow-up's local registration and result hashes. This is an integrity check, not evidence that every scientific claim is correct. Run `python scripts/check_core_results.py` for a separate arithmetic check of the central reported means.

## v1.2.0 natural data and additional seeds

V-B uses pilot 33000–33007 and confirmation 34000–34039. V-C uses separate pilot 35000–35007 and confirmation 36000–36039. Within either study, both capacities reuse the same 40 confirmation seeds; six directions are averaged within seed. V-C adds one fixed building/time series, not 40 independent houses. Its data/split_manifest.json records chronological blocks and training-only scaling. All endpoint failures and unsuccessful response criteria remain, so component effects must not be described as matched-endpoint natural replication.
