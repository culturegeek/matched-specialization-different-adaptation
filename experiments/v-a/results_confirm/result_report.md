# 実験V-A confirm 結果

## この実験が答える問い

三つの推定対象を一つの有限な共有表現で学ぶとき、更新を三対象で均衡させても、どの対象を精密に取得するかによって、形成後に優位となる対象は変わるか。

## 結果

### 表現容量 2

| 精密に取得した対象 | その対象の配分利得 | 優位となったseed割合 | 他二対象間の非対称 |
|---|---:|---:|---:|
| A | 0.3701 [0.3555, 0.3848] | 1.000 | 0.0174 |
| B | 0.3729 [0.3598, 0.3861] | 1.000 | 0.0193 |
| C | 0.3731 [0.3624, 0.3837] | 1.000 | 0.0200 |

均衡条件におけるラベル別平均値の範囲: 0.0067。
ラベル別取得効果の平均値の範囲: 0.0029。
配分平面の第2/第1特異値比: 0.9839 [0.9725, 0.9947]。

### 表現容量 3

| 精密に取得した対象 | その対象の配分利得 | 優位となったseed割合 | 他二対象間の非対称 |
|---|---:|---:|---:|
| A | 0.9232 [0.8938, 0.9502] | 1.000 | 0.0234 |
| B | 0.9030 [0.8765, 0.9305] | 1.000 | 0.0217 |
| C | 0.9222 [0.8996, 0.9434] | 1.000 | 0.0216 |

均衡条件におけるラベル別平均値の範囲: 0.0073。
ラベル別取得効果の平均値の範囲: 0.0202。
配分平面の第2/第1特異値比: 0.9876 [0.9767, 0.9971]。

## 事前判定

27/27判定を通過。

- PASS: sensor_A
- PASS: sensor_B
- PASS: sensor_C
- PASS: capacity_2_acquisition_A
- PASS: capacity_2_update_A
- PASS: capacity_2_other_symmetry_A
- PASS: capacity_2_acquisition_B
- PASS: capacity_2_update_B
- PASS: capacity_2_other_symmetry_B
- PASS: capacity_2_acquisition_C
- PASS: capacity_2_update_C
- PASS: capacity_2_other_symmetry_C
- PASS: capacity_2_balanced_label_symmetry
- PASS: capacity_2_label_exchangeability
- PASS: capacity_2_two_dimensional_allocation_span
- PASS: capacity_3_acquisition_A
- PASS: capacity_3_update_A
- PASS: capacity_3_other_symmetry_A
- PASS: capacity_3_acquisition_B
- PASS: capacity_3_update_B
- PASS: capacity_3_other_symmetry_B
- PASS: capacity_3_acquisition_C
- PASS: capacity_3_update_C
- PASS: capacity_3_other_symmetry_C
- PASS: capacity_3_balanced_label_symmetry
- PASS: capacity_3_label_exchangeability
- PASS: capacity_3_two_dimensional_allocation_span

## 適用範囲

この結果が直接示す範囲は、相関0.35の三つの三次元信号を、二次元または三次元の共有表現へ圧縮して学ぶ人工系である。取得条件と更新重みは研究者が外部から割り当て、形成を600更新まで観測した。

- 実行時間: 234.2秒
