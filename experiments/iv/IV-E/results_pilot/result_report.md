# 実験IV-E 結果

- seed数: 16

## Adam形成終点診断
- acquisition_minus_update_m_l2: -0.022905 [-0.032312, -0.015620]
- acquisition_over_update_m_l2: 0.625310 [0.550404, 0.703889]
- acquisition_minus_update_m_hat_l2: -0.022905 [-0.031912, -0.015326]
- acquisition_over_update_m_hat_l2: 0.625310 [0.542775, 0.703384]
- acquisition_minus_update_v_sum: -0.010873 [-0.011615, -0.010194]
- acquisition_over_update_v_sum: 0.571112 [0.542938, 0.598133]
- acquisition_minus_update_v_l2: -0.000664 [-0.000790, -0.000547]
- acquisition_over_update_v_l2: 0.833447 [0.787734, 0.877187]
- acquisition_minus_update_effective_lr_mean: 0.359166 [0.330052, 0.389231]
- acquisition_over_update_effective_lr_mean: 1.823094 [1.778562, 1.866672]

## 10更新後の経路差（取得主導－更新主導）
- acquisition_pulse
  - own: 0.155253 [0.144548, 0.165193]
  - m_other: 0.159517 [0.149865, 0.169085]
  - v_other: -0.111940 [-0.125756, -0.095859]
  - both_other: -0.107192 [-0.122264, -0.089816]
  - reset: -0.000975 [-0.005013, 0.003711]
- update_pulse
  - own: 0.141527 [0.133393, 0.149532]
  - m_other: 0.145819 [0.137520, 0.154081]
  - v_other: -0.101596 [-0.116689, -0.085153]
  - both_other: -0.096932 [-0.112598, -0.079279]
  - reset: 0.009067 [0.005016, 0.013989]
- combined_switch
  - own: 0.182570 [0.172676, 0.192497]
  - m_other: 0.185652 [0.175546, 0.194968]
  - v_other: -0.123632 [-0.141328, -0.103040]
  - both_other: -0.119905 [-0.139143, -0.098081]
  - reset: 0.008084 [0.004216, 0.012417]
