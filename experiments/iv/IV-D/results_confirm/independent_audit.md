# 実験IV-D 独立再集計

実験スクリプトとresult.jsonを使わず、確認CSVから再計算した。

- seed: 40
- 形成セル: 160
- 後続条件セル: 1280
- 時系列行: 218880
- 同一checkpoint開始値の最大差: 0.000000000000
- 形成終点の一致: PASS
- optimizer状態の交互作用: PASS

## 後続応答の分類

- carried_neutral_auc: equivalent_within_margin (-0.0025, 95%区間 [-0.0064, 0.0012])
- reset_neutral_auc: equivalent_within_margin (-0.0018, 95%区間 [-0.0068, 0.0030])
- carried_acquisition_pulse: different (0.1668, 95%区間 [0.1603, 0.1732])
- carried_update_pulse: different (0.1510, 95%区間 [0.1455, 0.1564])
- carried_combined_pulse: different (0.1951, 95%区間 [0.1880, 0.2020])
- reset_acquisition_pulse: equivalent_within_margin (0.0011, 95%区間 [-0.0011, 0.0034])
- reset_update_pulse: equivalent_within_margin (0.0120, 95%区間 [0.0093, 0.0149])
- reset_combined_pulse: equivalent_within_margin (0.0103, 95%区間 [0.0079, 0.0129])
