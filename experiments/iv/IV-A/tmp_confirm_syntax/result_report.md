# 実験III-D confirm 結果

## 操作確認

- A追加センサによるA復号誤差改善: 0.1982
- B追加センサによるB復号誤差改善: 0.1625
- A形成枝の開始時 s_ref: 0.1246
- B形成枝の開始時 s_ref: 0.1217

## 主2×2

| 条件 | 最終 s_ref | 枝保持率 | 代替枝到達率 | 制限付き到達時間 | 代替要求損失AUC |
|---|---:|---:|---:|---:|---:|
| acq_alternative__upd_alternative | -0.0937 | 0.000 | 0.500 | 42.5 | 0.8691 |
| acq_alternative__upd_reference | 0.0993 | 1.000 | 0.000 | 50.0 | 0.8796 |
| acq_reference__upd_alternative | -0.0451 | 0.000 | 0.500 | 50.0 | 0.8810 |
| acq_reference__upd_reference | 0.1307 | 1.000 | 0.000 | 50.0 | 0.9274 |

## 取得×更新の交互作用

- final_s_ref: -0.0173 (95%区間 [-0.0173, -0.0173])
- retained: 0.0000 (95%区間 [0.0000, 0.0000])
- transition_event: 0.0000 (95%区間 [0.0000, 0.0000])
- transition_time_restricted: -7.5000 (95%区間 [-7.5000, -7.5000])
- old_loss_auc: 0.0402 (95%区間 [0.0402, 0.0402])
- alternative_loss_auc: 0.0359 (95%区間 [0.0359, 0.0359])

## 事前判定

- PASS: C1_A_sensor_gain
- PASS: C1_B_sensor_gain
- PASS: C1_A_formed_branch
- PASS: C1_B_formed_branch
- PASS: C2_reference_update_retention
- FAIL: C2_alternative_update_transition
- FAIL: C3_alternative_acquisition_accelerates_transition

## 読み方

同じ軌道を、形成済み枝を保つ維持性と、代替枝へ移る再配分可能性の双方から評価した。
代替要求は学習系へ入力しておらず、要求変更を自律的に検知する能力は測っていない。

- 実行時間: 2.0秒
