# 弱履歴・独立seed確認結果

- 初期履歴差: 0.2793（95% CI [0.2568, 0.3018]）
- fixed AUC: 0.8727
- stabilization AUC: 1.0080
- stabilization / fixed: 1.155
- 対応AUC差: 0.1353（95% CI [-0.0207, 0.2913]）
- exploration AUC: 0.0504

## 事前判定

- PASS: initial_history_direction_is_reproduced
- FAIL: stabilization_to_fixed_auc_ratio_exceeds_1_20
- FAIL: paired_auc_difference_ci_excludes_zero
- PASS: exploration_auc_below_half_stabilization