# 実験III-D confirm 結果

## 操作確認

- A追加センサによるA復号誤差改善: 0.2074
- B追加センサによるB復号誤差改善: 0.2082
- A形成枝の開始時 s_ref: 0.1322
- B形成枝の開始時 s_ref: 0.1329

## 主2×2

| 条件 | 最終 s_ref | 枝保持率 | 代替枝到達率 | 制限付き到達時間 | 代替要求損失AUC |
|---|---:|---:|---:|---:|---:|
| acq_alternative__upd_alternative | -0.1365 | 0.000 | 1.000 | 22.2 | 0.7943 |
| acq_alternative__upd_reference | 0.1011 | 0.988 | 0.000 | 400.0 | 0.8579 |
| acq_reference__upd_alternative | -0.0994 | 0.000 | 1.000 | 40.6 | 0.7724 |
| acq_reference__upd_reference | 0.1391 | 1.000 | 0.000 | 400.0 | 0.9242 |

## 取得×更新の交互作用

- final_s_ref: 0.0009 (95%区間 [-0.0003, 0.0020])
- retained: 0.0125 (95%区間 [0.0000, 0.0375])
- transition_event: 0.0000 (95%区間 [0.0000, 0.0000])
- transition_time_restricted: -18.3750 (95%区間 [-26.0625, -13.1875])
- old_loss_auc: 0.0853 (95%区間 [0.0836, 0.0870])
- alternative_loss_auc: 0.0882 (95%区間 [0.0863, 0.0902])

## 事前判定

- PASS: C1_A_sensor_gain
- PASS: C1_B_sensor_gain
- PASS: C1_A_formed_branch
- PASS: C1_B_formed_branch
- PASS: C2_reference_update_retention
- PASS: C2_alternative_update_transition
- PASS: C3_alternative_acquisition_accelerates_transition

## 読み方

同じ軌道を、形成済み枝を保つ維持性と、代替枝へ移る再配分可能性の双方から評価した。
代替要求は学習系へ入力しておらず、要求変更を自律的に検知する能力は測っていない。

- 実行時間: 205.0秒
