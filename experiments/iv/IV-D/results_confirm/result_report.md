# 実験IV-D confirm 結果

## この実験が答える問い

取得主導と更新主導で表面上ほぼ同じ特化を形成したとき、共通条件と同じ摂動への後続応答も同じになるか。

## 形成終点

| 形成経路 | 方向の深さ | 対象側誤差 | 反対側誤差 |
|---|---:|---:|---:|
| acquisition | 0.3375 | 0.4034 | 0.8144 |
| update | 0.3439 | 0.4000 | 0.8198 |

## 経路差（取得主導－更新主導）

- optimizer=carried
  - neutral / auc_aligned_s_0_200: -0.0025 (95%区間 [-0.0064, 0.0013])
  - acquisition_pulse / pulse_drop: 0.1668 (95%区間 [0.1601, 0.1736])
  - update_pulse / pulse_drop: 0.1510 (95%区間 [0.1456, 0.1562])
  - combined_switch / pulse_drop: 0.1951 (95%区間 [0.1880, 0.2021])
- optimizer=reset
  - neutral / auc_aligned_s_0_200: -0.0018 (95%区間 [-0.0068, 0.0030])
  - acquisition_pulse / pulse_drop: 0.0011 (95%区間 [-0.0010, 0.0035])
  - update_pulse / pulse_drop: 0.0120 (95%区間 [0.0093, 0.0149])
  - combined_switch / pulse_drop: 0.0103 (95%区間 [0.0079, 0.0129])

## 事前判定

- 形成終点の一致: PASS
- carried_neutral_auc: equivalent_within_margin
- reset_neutral_auc: equivalent_within_margin
- carried_acquisition_pulse: different
- carried_update_pulse: different
- carried_combined_pulse: different
- reset_acquisition_pulse: equivalent_within_margin
- reset_update_pulse: equivalent_within_margin
- reset_combined_pulse: equivalent_within_margin
- optimizer状態による経路差の増幅: PASS

## 適用範囲

この結果が直接届く範囲は、相関する二つの三次元信号を二次元共有表現へ圧縮する人工学習系である。形成経路は研究者が固定し、後続応答は最大400更新まで観測した。

- 実行時間: 507.0秒
