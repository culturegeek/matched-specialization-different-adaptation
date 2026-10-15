# 実験I-C v0.2.1・共同更新の収束性監査

v0.2の400反復後から、継続、Adam状態だけのリセット、softmaxの再加熱、再加熱＋微小摂動へ分岐した。
確率境界で別行動へ移る改善余地も測り、低エントロピーと真の局所安定性を区別した。

最終反復: 1600、seed数: 5

## 結論

全15条件のうち、400反復時点でOracleに達していなかったのは4条件だった。単純な反復継続によって初めてOracleへ移った条件は0件だった。4条件はいずれも、母集団の静的局所最適ではない共同更新の安定点候補として残った。

## 最終時点の比較

| lambda | 分岐 | Oracle到達率 | 平均目的値 | 平均entropy | 境界違反率 | スケジュール分布 |
|---:|:---|---:|---:|---:|---:|:---|
| 0.25 | continue | 0.800 | 0.07585 | 0.000144 | 0.000 | ABBB:4;BABB:1 |
| 0.25 | reheat_x2 | 0.800 | 0.07584 | 0.000013 | 0.000 | ABBB:4;BABB:1 |
| 0.25 | reheat_x4 | 0.800 | 0.07585 | 0.000084 | 0.000 | ABBB:4;BABB:1 |
| 0.25 | reheat_x4_noise | 0.800 | 0.07585 | 0.000085 | 0.000 | ABBB:4;BABB:1 |
| 0.25 | reset_only | 0.800 | 0.07584 | 0.000000 | 0.000 | ABBB:4;BABB:1 |
| 0.50 | continue | 0.800 | 0.09208 | 0.000118 | 0.000 | AABB:4;ABBB:1 |
| 0.50 | reheat_x2 | 0.800 | 0.09208 | 0.000012 | 0.000 | AABB:4;ABBB:1 |
| 0.50 | reheat_x4 | 0.800 | 0.09208 | 0.000081 | 0.000 | AABB:4;ABBB:1 |
| 0.50 | reheat_x4_noise | 0.800 | 0.09208 | 0.000078 | 0.000 | AABB:4;ABBB:1 |
| 0.50 | reset_only | 0.800 | 0.09208 | 0.000000 | 0.000 | AABB:4;ABBB:1 |
| 0.75 | continue | 0.600 | 0.07751 | 0.000132 | 0.000 | AAAB:3;AABA:1;AABB:1 |
| 0.75 | reheat_x2 | 0.600 | 0.07750 | 0.000012 | 0.000 | AAAB:3;AABA:1;AABB:1 |
| 0.75 | reheat_x4 | 0.600 | 0.07750 | 0.000084 | 0.000 | AAAB:3;AABA:1;AABB:1 |
| 0.75 | reheat_x4_noise | 0.600 | 0.07750 | 0.000083 | 0.000 | AAAB:3;AABA:1;AABB:1 |
| 0.75 | reset_only | 0.600 | 0.07750 | 0.000000 | 0.000 | AAAB:3;AABA:1;AABB:1 |

## seed単位の診断分類

- `already_oracle_at_400`: 11
- `joint_dynamics_stable_candidate`: 4

## 判定の読み方

- `already_oracle_at_400`: v0.2の反復数ですでにOracleへ到達していた。
- `continuation_reaches_oracle`: 400反復では足りなかったが、単純継続でOracleへ到達した。
- `optimizer_state_dependence`: Adamの内部状態を捨てるだけでOracleへ到達した。
- `policy_reheating_escape`: logitの縮小による十分な再加熱でのみOracleへ移り、低エントロピー方策の吸引域を越えた。
- `perturbation_escape`: 再加熱だけでは動かず、微小摂動を加えた場合だけOracleへ移った。
- `boundary_saturation_persists`: 非Oracleかつ確率境界に改善方向が残り、飽和捕捉が継続している。
- `static_and_joint_stable_candidate`: 母集団の静的局所最適と共同更新の安定点候補が一致した。
- `joint_dynamics_stable_candidate`: 母集団の静的局所最適ではないが、共同更新では同じ方策へ復帰した。
- `multiple_joint_basins_candidate`: 改善方向は小さいが、再加熱・摂動で異なる方策へ分岐する。

## 限界

この分類は、現行のAdam更新と因子化softmax方策に対する診断である。
二種類の`*_stable_candidate`は数学的な固定点の証明ではなく、より直接的なHessian解析や別最適化法との比較を要する。
本監査は閉ループ方策、志向性、発達的履歴依存、人間の注意を検証していない。

実行時間: 24.9秒
