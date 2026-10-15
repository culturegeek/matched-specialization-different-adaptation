# 実験II v0.2・閉ループ誤差選択PoC 実行結果

## 事前判定

- PASS: history_manipulation
- PASS: open_loop_replicates_erasure
- PASS: stabilization_preserves_history
- PASS: exploration_reduces_history_relative_to_stabilization

## 条件別結果

| 条件 | 最終残存率 | 履歴差AUC | 最終CKA | 最終選択確率差 |
|---|---:|---:|---:|---:|
| all_targets | 0.002 | 0.142 | 0.9876 | 0.000 |
| balanced_sampled | 0.021 | 0.286 | 0.9913 | 0.000 |
| fixed_propensity | 1.042 | 1.934 | 0.2025 | 0.711 |
| stabilization | 1.043 | 1.938 | 0.1981 | 0.719 |
| exploration | 0.007 | 0.075 | 0.9918 | -0.006 |

## 読み方

all_targets は実験II v0.1の負の結果を再現する基準です。balanced_sampled は一度に一方の誤差だけを学ぶこと自体の影響を測ります。
fixed_propensity は後半開始時の選択確率を固定し、継続する外部配分の効果を測ります。stabilization と exploration だけが現在の表現精度から選択確率を更新します。

## 固定配分との差

- 安定化型／固定配分の履歴差AUC比: 1.002
- 安定化型－固定配分の最終残存率差: 0.001
この差が小さい場合、履歴の持続は確認できても、選択確率を継続更新することの上乗せ効果は小さいと解釈します。

## 限界

本実験が選択するのは観測対象そのものではなく、取得済みデータのうちどちらの差異を誤差として学習に入れるかです。
選択指標は正解ラベルを使う線形プローブで測り、系自身が精度推定を学習する機構は含みません。
人間の発達、性格、志向性、準安定レジームを直接検証するものではありません。

- 実行デバイス: CPU
- 実行時間: 957.9秒
