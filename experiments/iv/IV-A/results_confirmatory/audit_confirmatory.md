# 実験III-D 確認結果監査

- 完全性: PASS (640/640 cell rows)

## A/B形成枝の対称性

- A形成枝の取得効果: 16.625 (95%区間 [12.125, 22.125])
- B形成枝の取得効果: 20.125 (95%区間 [12.000, 34.250])
- A－B差: -3.500 (95%区間 [-18.875, 7.250])

## 探索的な一致効果

- 取得先と更新先が一致したときの |s| 増分: 0.0376 (95%区間 [0.0361, 0.0391])
- この量は確認実験前の主要判定に含めておらず、探索的結果として扱う。

## 補助対照

- acq_none__upd_reference: final s_ref=0.1230, transition=0.000, restricted time=400.0
- acq_none__upd_alternative: final s_ref=-0.1203, transition=1.000, restricted time=27.2
- acq_random__upd_reference: final s_ref=0.1311, transition=0.000, restricted time=400.0
- acq_random__upd_alternative: final s_ref=-0.1280, transition=1.000, restricted time=25.8

- 基準側更新で枝保持閾値を下回ったcell: 1
