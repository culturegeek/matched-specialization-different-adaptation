# 対称条件からの分岐PoC 結果

- 初期selector index平均: -0.0005（95% CI [-0.0548, 0.0537]）
- 初期正符号率: 0.475
- balanced最終|probe index|: 0.1817
- fixed最終|probe index|: 0.4207
- stabilization最終|probe index|: 0.8683
- exploration最終|probe index|: 0.0580
- stabilization / fixed: 2.064
- 対応差95% CI: [0.3684, 0.5268]
- 初期符号と安定化後符号の一致率: 0.564

- 最初の25選択と最終符号の一致率: 0.700
- 最初の50選択と最終符号の一致率: 0.722
- 最初の100選択と最終符号の一致率: 0.850

## 探索判定

- PASS: initial_signed_mean_near_zero
- PASS: both_initial_signs_present
- PASS: stabilization_exceeds_fixed_by_20_percent
- PASS: paired_stabilization_minus_fixed_ci_excludes_zero
- PASS: exploration_below_half_stabilization
- FAIL: initial_sign_predicts_stabilized_sign