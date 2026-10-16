# Reproducing the results

Start with the quick checks in the [README](../README.md). They recompute central means and regenerate the five figure families without retraining. Run all commands below from the repository root unless a change of directory is shown.

## Environment

The original follow-up records Python 3.11.7, PyTorch 2.13.0, NumPy 2.4.6, CPU execution, deterministic settings, and one thread. The stored 40-seed run reports approximately 671 seconds; runtime on another machine may differ. Other experiments record their own environments in `provenance.json`.

For training, create a separate virtual environment using Python 3.11 and install the recorded packages:

```sh
python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r requirements-recorded-training.txt
```

The original publication validation is in [validation.json](../provenance/validation.json); subsequent checks are recorded in [IV-F](../provenance/ivf_validation.json), [V-B/V-C](../provenance/vbc_validation.json), and [head diagnostic](../provenance/head_validation.json). V-B, V-C, and the head diagnostic used Python 3.12.14, NumPy 2.4.6, PyTorch 2.13.0, CPU with one thread. It is listed separately from the original training environment. Numerical agreement on another platform should be checked from outputs rather than assumed from a package version alone.

## One-seed reproduction of the original intervention

```sh
cd experiments/iv-followup
python run_followup.py --stage reproduce --out results_reproduce_new
cd ../..
python scripts/compare_reproduction.py
```

This reproduces seed 9100 for the original intervention trajectories. Compare the generated rows with the archived `results_reproduce/seed_9100/timecourse.csv`. A separate smoke run uses the designated seed 19799:

```sh
cd experiments/iv-followup
python run_followup.py --stage smoke --out results_smoke_new
cd ../..
```

## Full follow-up

```sh
cd experiments/iv-followup
python run_followup.py --stage confirm --out results_confirm_new
python analyze_followup.py --folder results_confirm_new
python audit_results.py --folder results_confirm_new
cd ../..
```

This runs all 40 prespecified seeds and the recorded intervention conditions. Keep `registration.json`, `protocol.md`, the frozen programs, and `existing_analysis/frozen_models.json` unchanged. The `lock` command creates a local registration; it is not part of reproducing this release.

The original audit writes audit and manifest outputs into its selected result directory. Use a new directory for reruns. The top-level `scripts/verify_data.py` and `scripts/check_core_results.py` leave the archived result files unchanged.

## Original core studies

Each core experiment has its own README and fixed plan. These commands create new outputs:

```sh
python experiments/iv/IV-D/experiment_iv_d.py --stage confirm --device cpu --preregistration experiments/iv/IV-D/preregistration_confirmatory.json --output-dir generated/iv-d
python experiments/iv/IV-E/experiment_iv_e.py --stage confirm --device cpu --preregistration experiments/iv/IV-E/preregistration_confirmatory.json --output-dir generated/iv-e
python experiments/v-a/experiment_v_a.py --stage confirm --device cpu --preregistration experiments/v-a/preregistration_confirmatory.json --output-dir generated/v-a
```

The IV-D and IV-E programs import their sibling programs, so retain the `experiments/iv` folder structure. Historical calibration and maintenance packages are in `experiments/i-*`, `experiments/ii-*`, and `experiments/iii`; their original entry points and dependency files are preserved. Those historical programs are not all rerun during publication validation.

## Optional Japanese figures

The public plotting script defaults to English and has no operating-system font dependency. To also generate Japanese figures, supply a Japanese font installed on your machine:

```sh
python scripts/make_figures.py --languages en ja --japanese-font /path/to/your/Japanese-font.ttf
```

The repository does not redistribute a system font. Font changes can alter layout or PDF metadata; verify regenerated figures visually before using them in a manuscript.

## IV-F and interval provenance

[IV-F instructions](../experiments/iv-extension/README.md) give complete regeneration commands for saved-data analyses and new experiments, plus the independent audit. [Interval audit](INTERVAL_AUDIT.md) gives the three reconstructed-interval calculation and the full bilingual printed-interval verification. These extend, rather than replace, the original verification commands.

## Three-target and natural adaptation

Install `requirements-extension.txt`. From the repository root run the two independent checks:

```sh
python experiments/v-three-adaptation/audit_adaptation.py
python experiments/v-natural/audit_natural.py
```

Each experiment README gives full 40-seed reruns, calibration recomputation and analysis commands. Use new output directories. No source download is required: V-C includes the original UCI CSV, source checksum, CC BY 4.0 attribution and fixed temporal split. Local registration files are preserved; do not rerun lock commands. New-result file integrity is checked by `python scripts/verify_vbc_files.py`.

## Head-specific endpoint diagnostic

[The diagnostic README](../experiments/iv-head-gradients/README.md) documents four input conditions, retained trained heads, paired seed aggregation, and the scope of this exploratory result. It uses the already included checkpoints and regenerated held-out inputs; there is no retraining or download.

```sh
python scripts/verify_head_files.py
python experiments/iv-head-gradients/analyze_gradients.py --archive-root . --out generated/head-gradients
python experiments/iv-head-gradients/audit_gradients.py --archive-root .
python experiments/iv-head-gradients/summarize_existing.py --archive-root . --out generated/existing-contrasts
python experiments/iv-head-gradients/summarize_low_precision.py --archive-root . --out generated/three_low_precision.json
```

Use fresh output destinations. The analytic audit checks archived batch values and 16 head-ratio intervals without automatic differentiation; it writes the deterministic `independent_audit.json` report in the diagnostic folder. The source and input hashes are verified. This supplements the original frozen study audits.
