# 初期選択系列操作PoC 結果

- 操作直後の方向効果: 0.0774 (95% CI [0.0664, 0.0883])
- balanced_sampled: 最終方向効果 0.0130, gain 0.168, 方向追随率 0.500
- fixed_propensity: 最終方向効果 0.2109, gain 2.726, 方向追随率 0.662
- stabilization: 最終方向効果 0.1866, gain 2.412, 方向追随率 0.613
- exploration: 最終方向効果 0.0025, gain 0.033, 方向追随率 0.487
- stabilization / fixed: 0.885
- stabilization - fixed 95% CI: [-0.1089, 0.0603]

## 事前判定

- PASS: forced_sequence_creates_weak_directional_effect
- PASS: stabilization_tracks_forced_direction
- PASS: stabilization_amplifies_release_by_50_percent
- FAIL: stabilization_exceeds_fixed_by_20_percent
- FAIL: paired_stabilization_minus_fixed_ci_excludes_zero
- PASS: exploration_below_half_stabilization
- FAIL: forced_direction_tracking_exceeds_80_percent