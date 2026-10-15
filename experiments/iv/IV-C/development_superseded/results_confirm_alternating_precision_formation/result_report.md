# 実験IV-C confirm 結果

## この実験が答える問い

形成済み表現を出発点にしても、更新をA/B均衡に保ったまま、どちらを精密に取得するかだけで表現方向を動かせるか。

## 共通高精度測定

| 出発状態 | A精密 | 均衡取得 | B精密 |
|---|---:|---:|---:|
| A_formed | 0.3537 | -0.0028 | -0.3582 |
| neutral | 0.3533 | -0.0010 | -0.3546 |
| B_formed | 0.3567 | 0.0001 | -0.3539 |

## 中心比較

- 形成済み2状態で平均したA精密－B精密: 0.7112 (95%区間 [0.7065, 0.7158])
- 中立状態のA精密－B精密: 0.7079 (95%区間 [0.7028, 0.7131])

## 事前判定

- PASS: sensor_A
- PASS: sensor_B
- PASS: A_formed_start
- PASS: B_formed_start
- PASS: primary_formed_acquisition_effect
- PASS: A_formed_acquisition_effect
- PASS: B_formed_acquisition_effect
- PASS: A_formed_reaches_B_direction
- PASS: B_formed_reaches_A_direction
- PASS: A_formed_crosses_zero
- PASS: B_formed_crosses_zero
- PASS: neutral_positive_control
- PASS: low_resolution_same_direction

## 適用範囲

この結果が直接示す範囲は、相関する二つの三次元信号を二次元の共有表現へ圧縮して学ぶ人工系である。形成済み状態は更新重みの非対称で作り、継続中の更新重みは全条件で0.5/0.5に固定した。

- 実行時間: 220.5秒
