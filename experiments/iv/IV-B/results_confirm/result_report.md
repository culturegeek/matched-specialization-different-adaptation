# 実験IV-B confirm 結果

## この実験が答える問い

学習前の同じモデルに対し、どちらの情報を精密に取得するかだけを変えると、A/Bを同じ重みで学習しても、形成される共有表現の方向は変わるか。

## 操作確認

- Aを精密にしたときのA情報利得: 0.5479
- Bを精密にしたときのB情報利得: 0.5489

## 形成後の共通高精度測定

| 取得 | 更新 | 最終方向 s | 特化の深さ |
|---|---|---:|---:|
| A_precise | A_heavy | 0.4317 | 0.4317 |
| A_precise | balanced | 0.3385 | 0.3385 |
| A_precise | B_heavy | -0.1762 | 0.1762 |
| balanced | A_heavy | 0.3631 | 0.3631 |
| balanced | balanced | 0.0033 | 0.0499 |
| balanced | B_heavy | -0.3581 | 0.3581 |
| B_precise | A_heavy | 0.1849 | 0.1849 |
| B_precise | balanced | -0.3338 | 0.3338 |
| B_precise | B_heavy | -0.4262 | 0.4262 |

## 中心比較

- 更新を均衡させた取得A精密－B精密: 0.6723 (95%区間 [0.6660, 0.6794])
- 取得を均衡させた更新A重視－B重視: 0.7213 (95%区間 [0.7146, 0.7280])

## 事前判定

- PASS: sensor_A
- PASS: sensor_B
- PASS: positive_control_update
- PASS: primary_acquisition_selects_direction
- PASS: A_precise_forms_A_direction
- PASS: B_precise_forms_B_direction
- PASS: balanced_condition_remains_near_center
- PASS: secondary_low_resolution_same_direction

## 適用範囲

この結果が直接示す範囲は、二つの相関信号を二次元共有表現へ圧縮して学ぶ人工系である。取得と更新の役割が、他のモデルや自然な課題でも同じかは次の実験で確かめる。

- 実行時間: 124.5秒
