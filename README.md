# Matched Specialization, Different Adaptation

Code and generated data for **Matched Specialization, Different Adaptation: Second-Moment Structure in Shared Representations**, by Toru Takahashi. Manuscript revision: 17 October 2026; release **v1.5.2**.

Can similar current specialization conceal different responses to further learning? The controlled comparisons show that it can. Adam's retained squared-gradient averages change later reallocation, including through their coordinate pattern at a common average scale. Formation records and a new projection-plus-transplant experiment connect how this state is produced to how it changes adaptation.

The matched comparison extends to three Gaussian targets. Room-sensor measurements retain component effects but fail endpoint matching. The new projected formation also fails endpoint matching. These failures remain in the manuscript, alongside limits on reset equivalence, sign reversal and observation horizon. Faster reallocation is not established as better final performance.

[日本語](README_ja.md) · [Manuscript PDF](paper/preprint.pdf) · [Paper-to-data map](docs/PAPER_TO_DATA.md) · [Reproduction guide](docs/REPRODUCING.md)

## Start with verification

```sh
git clone https://github.com/culturegeek/matched-specialization-different-adaptation.git
cd matched-specialization-different-adaptation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_manifest.py
python scripts/verify_data.py
python scripts/check_core_results.py
python scripts/verify_vbc_files.py
python scripts/verify_head_files.py
python scripts/verify_formation_files.py
python scripts/verify_cancellation_files.py
```

These checks verify stored files and independently recompute the 27 original central means. The [interval audit](docs/INTERVAL_AUDIT.md) binds all 95 printed confidence intervals in each language to exact source keys and rounding. New-study independent audits additionally compare analytic gradients, projection, second-moment updates, paired starts and unchanged native endpoints. See [current validation](provenance/final_checks_validation.json) for their actual scope.

## Evidence needed for the current paper

| Question | Records |
|---|---|
| Matched measurements, common pulses and component exchanges | [IV-D/E](experiments/iv), including the equal-mean history control |
| Mean scale, coordinate pattern and longer continuation | [Follow-up](experiments/iv-followup) |
| Layer/movement interventions and finite regime grid | [Extensions](experiments/iv-extension) |
| How do task gradients accumulate into the retained state? | [Formation logging](experiments/iv-formation-logging/README.md) |
| Does altered formation produce a state that changes a fixed recipient? | [Projection and crossed transplants](experiments/iv-cancellation/README.md) |
| Three Gaussian targets and unequal sensor measurements | [Three-target adaptation](experiments/v-three-adaptation), [natural measurements](experiments/v-natural) |

The manuscript has 17 pages and five evidence appendices. Wider-program experiments I–III, IV-A/C and V-A remain archived with their unsuccessful results and original status, but are not additional evidence chapters the reader must traverse. Endpoint probes and prediction analyses remain available as supporting diagnostics. The current paper map identifies all five figures and eight tables.

## Reproduction and editable sources

Use a fresh output directory for reruns. The original locally frozen code/plans and scientific outputs are preserved. Files named posthoc_* document exploratory analyses. The projection test specified its new seeds and contrasts before execution; local hashes are not public-registry preregistration.

The original follow-up used Python 3.11.7; new formation/projection runs used Python 3.12.14, NumPy 2.4.6 and PyTorch 2.13.0 on single-threaded CPU. Bitwise agreement was observed in the recorded environment and is not guaranteed across releases/platforms. [Formation reproduction](experiments/iv-formation-logging/README.md) documents the strict default and explicit --tolerance option, which records actual errors without altering archived outputs.

Editable TeX, bibliography and styles are in paper/tex, with figures in paper/figures. From paper/tex, build main_arxiv.tex with pdfLaTeX/latexmk for the public English paper, main.tex for the anonymous English paper, or main_ja.tex with LuaLaTeX for Japanese. Figure-generation commands and data sources are in [PAPER_TO_DATA](docs/PAPER_TO_DATA.md).

## Attribution and license

Code is [MIT](LICENSE); generated data, checkpoints, figures and documentation are [CC BY 4.0](LICENSE-DATA). The redistributed UCI Appliances Energy Prediction measurements retain the original author attribution and license. See [license scope](docs/LICENSE_SCOPE.md) and [CITATION.cff](CITATION.cff). Previous release tags remain immutable.

Generative AI assisted experiment design, implementation, analysis, literature work and writing. The author is responsible for the work and release; the paper contains the full statement.

The v1.5.1 checks add the previously recorded secondary horizon to the paper, distinguish individual and combined gradient magnitude, and support explicit audit tolerance. No new experimental cohort was added. See [the fixed verification scope](analyses/final_checks_20261017/README.md).

## Release closure (v1.5.2)

The abstract now states the small Adam-trained model scope and contribution first, and distinguishes the bounded first-moment exchange result from second-moment reversal. The introduction states the scale/pattern/formation contribution explicitly. No new experiments, numerical estimates, or appendices were added.

`SHA256SUMS` covers every file shipped in this release except the manifest itself. Run `python scripts/verify_manifest.py` after extracting the release archive. The release ZIP digest protects the complete package, including the manifest. Runtime-generated files and the local Git database are not part of this manifest.

A separately reported PyTorch 2.14.1 reproduction passed `audit_intervention.py --tolerance 1e-6`, with maximum reported differences of 9.5e-7 for weights, 2.0e-8 for first moments and 1.5e-9 for second moments, and a worst bound fraction of approximately 0.45. These are externally reported results; the raw external log was not supplied. Our own cross-version runs and their exact scopes are recorded in `analyses/final_checks_20261017/`. The older `provenance/reproduction_tolerance_validation.json` is a historical record and is preserved unchanged.
