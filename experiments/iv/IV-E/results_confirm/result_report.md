# 実験IV-E 結果

- seed数: 40

## Adam形成終点診断
- acquisition_minus_update_m_l2: -0.021547 [-0.027628, -0.016163]
- acquisition_over_update_m_l2: 0.653325 [0.598289, 0.705454]
- acquisition_minus_update_m_hat_l2: -0.021547 [-0.027412, -0.016323]
- acquisition_over_update_m_hat_l2: 0.653325 [0.598384, 0.704113]
- acquisition_minus_update_v_sum: -0.010314 [-0.010788, -0.009853]
- acquisition_over_update_v_sum: 0.557741 [0.533859, 0.581257]
- acquisition_minus_update_v_l2: -0.000659 [-0.000756, -0.000541]
- acquisition_over_update_v_l2: 0.812891 [0.775593, 0.850782]
- acquisition_minus_update_effective_lr_mean: 0.428511 [0.405276, 0.452152]
- acquisition_over_update_effective_lr_mean: 1.950009 [1.918064, 1.982430]

## 10更新後の経路差（取得主導－更新主導）
- acquisition_pulse
  - own: 0.169120 [0.163344, 0.174958]
  - m_other: 0.171381 [0.164610, 0.178562]
  - v_other: -0.117284 [-0.124144, -0.110413]
  - both_other: -0.113833 [-0.120907, -0.106778]
  - reset: 0.001556 [-0.000364, 0.003526]
- update_pulse
  - own: 0.153117 [0.148395, 0.157785]
  - m_other: 0.155793 [0.149690, 0.162084]
  - v_other: -0.105229 [-0.111805, -0.098442]
  - both_other: -0.101705 [-0.108155, -0.094925]
  - reset: 0.010602 [0.008768, 0.012421]
- combined_switch
  - own: 0.196800 [0.190983, 0.202866]
  - m_other: 0.198192 [0.191159, 0.205370]
  - v_other: -0.126463 [-0.134845, -0.118293]
  - both_other: -0.124287 [-0.132503, -0.115769]
  - reset: 0.009681 [0.007847, 0.011663]

## 事前判定
- 形成終点の一致: PASS
- own_state_replicates_route_effect: PASS
- m_swap_preserves_route_effect: PASS
- v_swap_reverses_route_effect: PASS
- v_swap_attenuation_is_large: PASS
- both_swap_reverses_route_effect: PASS
- reset_removes_route_effect: PASS
