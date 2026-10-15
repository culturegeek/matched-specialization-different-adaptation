# 実験III-B2 結果：停止規則の独立効果

## 仮説群

- **G1_compute_matched**: PASS
- **G2_adaptive_stopping**: PASS
- **G3_precision_benefit**: PASS
- **G4_information_efficiency**: PASS

## 対応差

|測定量|平均|95%区間|
|---|---:|---:|
|precision_adaptation|18.928522|[18.833308, 19.018799]|
|fixed_adaptation|0.000000|[0.000000, 0.000000]|
|precision_obs_change|0.025950|[-0.057093, 0.104266]|
|precision_accuracy_gain|0.046275|[0.043812, 0.048675]|
|precision_brier_reduction|0.025011|[0.023672, 0.026285]|
|ig_observation_reduction|6.153550|[6.125450, 6.181526]|
|ig_accuracy_change|0.005950|[0.003137, 0.008750]|

停止後も総計算回数は一致している。差は総計算時間ではなく、証拠取得を止める規則から生じた。
