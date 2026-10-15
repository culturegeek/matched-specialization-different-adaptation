# 実験I・測定基盤校正PoC v0.3

解析解を持つ線形ガウス系で、勾配学習、最適読み出し、歪み集計が正しく作動するかを校正します。GPUや大規模言語モデルは使いません。

相関係数 $\rho$ を走査し、実測歪みを解析解と比較します。さらに標本数 $n$ を走査し、RMSEがおおむね $n^{-1/2}$、MSEが $n^{-1}$ のオーダーで縮小するかを確かめます。これは理論の支持実験ではなく、後続実験の測定基盤に対する回帰テストです。

## 初回の推奨手順

1. `colab_runner.ipynb` を Google Colab で開く。
2. セルを上から実行する。
3. 求められたら `minimal_poc.py` をアップロードする。
4. 自動判定、図、CSVをZIPで保存する。

詳しい手順、判定基準、トラブル対応は、同梱のPDF『実験I・最小PoC設計および実行手順書』を参照してください。

## ローカルPCでの実行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python minimal_poc.py --quick --output-dir poc_results
```

Windows PowerShellの場合、仮想環境の有効化は次のようにします。

```powershell
.venv\Scripts\Activate.ps1
```

## 出力

- `result_report.md`: 校正判定と数値の要約
- `tradeoff.svg`: 二つの歪みの関係
- `rho_calibration.svg`: 実測歪みと解析曲線の比較
- `rho_calibration.csv`: $\rho$ ごとの実測値、解析値、誤差
- `rho_calibration_raw.csv`: $\rho$ ・seedごとの $D_A,D_B$
- `sample_size_calibration.svg`: 標本数とRMSEの関係
- `sample_size_calibration.csv`: 標本数ごとのバイアス、RMSE、MSE
- `sample_size_calibration_raw.csv`: 標本数・seedごとの生データ
- `summary_results.csv`: seed平均の結果
- `raw_results.csv`: 全実行の結果
- `metadata.json`: 実行条件と環境

## 主張できないこと

このPoCは、解析的に予測された競合を数値実装が再現するかを確かめるだけです。情報保存トレードオフの一般的な実在、人間の志向性、発達、準安定レジーム、心理志向／客観志向を証明しません。
