# 対称条件からの分岐PoC 結果

- 初期selector index平均: 0.0393（95% CI [-0.0156, 0.0941]）
- 初期正符号率: 0.625
- balanced最終|probe index|: 0.1588
- fixed最終|probe index|: 0.4558
- stabilization最終|probe index|: 0.8608
- exploration最終|probe index|: 0.0975
- stabilization / fixed: 1.889
- 対応差95% CI: [0.3286, 0.4814]
- 初期符号と安定化後符号の一致率: 0.784

- 最初の25選択と最終符号の一致率: 0.750
- 最初の50選択と最終符号の一致率: 0.892
- 最初の100選択と最終符号の一致率: 0.872

## 探索判定

- PASS: initial_signed_mean_near_zero
- PASS: both_initial_signs_present
- PASS: stabilization_exceeds_fixed_by_20_percent
- PASS: paired_stabilization_minus_fixed_ci_excludes_zero
- PASS: exploration_below_half_stabilization
- PASS: initial_sign_predicts_stabilized_sign