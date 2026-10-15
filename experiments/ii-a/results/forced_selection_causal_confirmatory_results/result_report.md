# 初期選択系列操作PoC 結果

- 操作直後の方向効果: 0.0848 (95% CI [0.0719, 0.0977])
- balanced_sampled: 最終方向効果 0.0181, gain 0.214, 方向追随率 0.512
- fixed_propensity: 最終方向効果 0.2371, gain 2.796, 方向追随率 0.662
- stabilization: 最終方向効果 0.2474, gain 2.919, 方向追随率 0.637
- exploration: 最終方向効果 0.0004, gain 0.005, 方向追随率 0.500
- stabilization / fixed: 1.044
- stabilization - fixed 95% CI: [-0.1015, 0.1223]

## 事前判定

- PASS: forced_sequence_creates_weak_directional_effect
- PASS: stabilization_tracks_forced_direction
- PASS: stabilization_amplifies_release_by_50_percent
- FAIL: stabilization_exceeds_fixed_by_20_percent
- FAIL: paired_stabilization_minus_fixed_ci_excludes_zero
- PASS: exploration_below_half_stabilization
- FAIL: forced_direction_tracking_exceeds_80_percent