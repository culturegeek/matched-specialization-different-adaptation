# Google Colab再現確認（2026-08-14）

## 受領ファイル

- `experiment_i_minimal_poc_results (2).zip`: quick
- `experiment_i_minimal_poc_results (3).zip`: full

## 実行環境

- Python 3.12.13
- NumPy 2.0.2
- Linux / Google Colab CPU

## 結果

| 条件 | 実行時間 | 解析解とのRMSE | 最大絶対誤差 |
|---|---:|---:|---:|
| quick | 10.4秒 | 0.0086 | 0.0109 |
| full | 100.2秒 | 0.0030 | 0.0042 |

主要数値とSVGは同梱した基準結果と一致した。CSVにはNumPy/BLASの実行環境による浮動小数点末尾の差（おおむね $10^{-16}$ 程度）があるが、集計値、判定、図に影響しない。

この確認により、v0.2の校正パイプラインはGoogle Colab上で再現した。v0.3はこれに標本数スケーリング検査を追加した版である。
