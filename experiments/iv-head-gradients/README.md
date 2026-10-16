# Head-specific endpoint gradients (exploratory)

At the saved two-target tanh endpoints, individual head gradients reaching the encoder are larger under acquisition formation when the two routes receive identical high-precision inputs. The two head gradients oppose each other more strongly, and their equally weighted sum is smaller. This endpoint observation does not reconstruct the history accumulated in Adam's second moment. All four probe conditions are retained; the total-gradient ordering changes with input precision and loss weighting.

## Design and results

The plan in `protocol.md` was locally locked before gradient computation, after the earlier results and mechanism questions were known. This is exploratory analysis, not a new confirmatory experiment. Every seed 20000--20039, both routes and both directions are retained (160 checkpoints). The primary probe uses common high-precision inputs, original trained heads, and separate unit MSE losses. Thirty-two 256-example batches are sampled from the original 1536-example held-out test split. Ratios are geometric means of within-direction acquisition/update RMS batch-gradient norm ratios, with directions averaged in log space within seed. Individual 95% intervals use 5000 paired-seed resamples.

`results/batch_gradients.csv` contains all 40960 layer/batch records, including cross terms and weight decay. `checkpoint_gradients.csv` and `seed_ratios.csv` give intermediate aggregation. `result.json` contains every probe condition, head, layer, and interval. No checkpoint or optimizer update is made.

A clarification of the plan's pairing wording: identical sample indices and regenerated base tensors are reused. Each input condition selects its specified high/low-precision tensor; the two routes receive the identical tensor within that condition and favored-target direction. Input conditions intentionally differ in precision.

## Reproduction from the archive root

```sh
python experiments/iv-head-gradients/analyze_gradients.py --archive-root . --out generated/head-gradients
python experiments/iv-head-gradients/audit_gradients.py --archive-root .
python experiments/iv-head-gradients/summarize_existing.py --archive-root . --out generated/existing-contrasts
```

The first and third commands require new output directories. The audit reads the archived results and writes its audit record. It independently computes gradients by explicit NumPy chain rules, without automatic differentiation, and checks all 40960 rows plus 16 seed-ratio intervals. Double-precision analytic and original float32 gradients agree within the stated numerical tolerances. For a near-zero gradient dot product, tolerance scales with the product of vector norms, since cancellation makes a relative-to-dot-product tolerance unstable. An initial audit used the latter tolerance and failed on a 3.5e-9 discrepancy; the corrected check does not change analysis results or frozen code. `independent_audit.json` records the final outcome. The scalar-conversion warning in the original diagnostic does not change values or update weights.

`existing_results/reset_comparison.json` adds exploratory paired reset-minus-own intervals. The beta2=0.9 increases are 0.0075 [0.0059,0.0090] and 0.0308 [0.0282,0.0334]. Reset changes both moments and step count; this is not a v-only contrast. `claim_inventory.csv` lists the original frozen ten-update criteria for 13 distinct settings. Existing margins and all failed criteria are unchanged.

The environment is Python 3.12.14, NumPy 2.4.6, PyTorch 2.13.0, CPU with one thread. Code is MIT; generated results and documentation are CC BY 4.0 under the archive licenses. Original checkpoint hashes, generated input hashes, local plan lock, and numerical checks are retained. No natural-data or formation-time causal conclusion follows from this endpoint diagnostic.

The Appendix D low-precision point summaries (0.137 and 0.235) are traced to the original V-A cells in `existing_results/three_low_precision.json`. Recompute with `python experiments/iv-head-gradients/summarize_low_precision.py --archive-root . --out generated/three_low_precision.json`.
