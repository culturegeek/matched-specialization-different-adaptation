# Paper-to-data map — manuscript v1.5.0

The manuscript asks how matched current specialization can conceal different adaptation. Complete historical experiments remain archived; the paper selects evidence that answers this question. Inference is paired by seed; statuses and failed criteria remain in the source plans/results.

| Paper location | Source |
|---|---|
| Figure 1, Table 2; matched response (§4) | `experiments/iv/IV-B/`, `experiments/iv/IV-D/results_confirm/` |
| Figure 2, Table 4; component exchange (§5.1, Appendix B) | `experiments/iv/IV-E/results_confirm/`, `experiments/iv/IV-E-update-bias-control/results_confirm/` |
| Figure 4; scale and time (§5.2, §6) | `experiments/iv-followup/results_confirm/`, `posthoc_review_results.json` |
| Table 5; layer and movement (§5.2, Appendix C.1) | `experiments/iv-extension/results_mechanism/result.json`, `runs_mechanism/` |
| Figure 5; formation accounting (Appendix C.2) | `experiments/iv-formation-logging/results/`, `runs/`, `audit_logging.py` |
| Figure 3, Table 6; projection and crossed transplant (§5.3, Appendix C.3) | `experiments/iv-cancellation/results/result.json`, `runs/`, `protocol.md`, `protocol_clarification.json`, `independent_audit.json` |
| Paired S differences, observation strata, duration ratios (Appendices C/D) | `experiments/iv-cancellation/saved_results/`, `saved_audit.json` |
| Table 3; setting-by-claim inventory (§6) | `experiments/iv-head-gradients/existing_results/claim_inventory.csv` |
| Table 7; decay/duration and reset (Appendix D) | `experiments/iv-extension/results_beta/result.json`, `results_general/result.json`, `experiments/iv-head-gradients/existing_results/reset_comparison.json` |
| Table 8; Gaussian and sensor targets (Appendix D.1–D.2) | `experiments/v-three-adaptation/results_confirm/result.json`, `experiments/v-natural/results_confirm/result.json` |
| Table 1; prior work | Bibliography in the manuscript |
| Every printed interval, both languages | `analyses/manuscript_intervals/INTERVAL_LEDGER.csv` (92 occurrences per language) |

Endpoint gradient probes are a supporting diagnostic of Appendix C.2: `experiments/iv-head-gradients/`. Their complete four-probe records and correlations remain available. The I–III maintenance/calibration studies, IV-A/C formation/transition details, V-A formation and full exploratory prediction analyses are retained but no longer narrated as separate manuscript appendices. Unsuccessful endpoint predictions, width/decay reset failures, duration-scaling failures, natural-data mismatch and projection mismatch remain in the paper.

Figure generation: `scripts/make_figures.py` regenerates the original figure families; `experiments/iv-formation-logging/make_figures.py` and `experiments/iv-cancellation/make_figures.py` generate the formation figures. Use each program's `--help`. The paper's intervals use their explicitly bound sources; plotting bootstrap streams do not replace those sources.
