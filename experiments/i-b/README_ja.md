# 実験I-B・観測予算／表現容量分離PoC v0.2

このPoCは、情報が失われる二つの段階を分けて測ります。

- 観測予算 `B_A`：世界から取得できるセンサーチャネル数
- 表現容量 `C_R`：取得済み情報から形成できる状態表現の次元

`B_A` を制限して失われた情報は、その後で `C_R` を増やしても回復しません。一方、十分に観測した後の圧縮損失は、`C_R` を増やすことで回復します。この違いを同じ生成データ上の要因計画で測ります。

## この段階で示さないこと

観測チャネルは、生成過程を知るOracleが全組合せから選びます。したがって、学習された注意、履歴依存、志向性、人間の心理志向／客観志向を検証する実験ではありません。目的は、それらを後で検証するために、未観測と観測後圧縮を混同しない測定系を作ることです。

## Google Colabで実行

1. `colab_runner.ipynb` をColabへアップロードします。
2. 最初のセルで `observation_budget_poc.py` をアップロードします。
3. 上から順にセルを実行します。GPUは不要です。
4. まず `--quick` の結果ZIPを保存します。
5. 成功後、必要なら実行セルの `--quick` を `--full` に変更します。

## ローカルで実行

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python observation_budget_poc.py --quick --output-dir observation_budget_results
```

Windowsでは仮想環境の有効化に `.venv\Scripts\activate` を使います。

## 主な出力

- `result_report.md`：自動判定と主要数値
- `factorial_heatmap.svg`：`B_A × C_R` の平均歪み
- `recoverability_curves.svg`：観測後に表現容量を増やしたときの回復曲線
- `distortion_decomposition.csv`：観測制約、表現制約、同時制約の分解
- `mask_overlap.csv`：保存対象A/Bに対するOracle観測マスクの重なりと偶然期待値
- `summary_results.csv`：条件ごとの平均と標準偏差
- `raw_results.csv`：seed、観測マスク、Oracle床を含む生データ
- `metadata.json`：実行環境、条件、自動判定

検証済み結果では、quick/fullの双方で分離判定を通過しています。v0.2では表現容量5を追加し、Oracle観測マスクのJaccard係数を観測予算別の偶然期待値と比較します。この比較は校正ゲートではなく、取得段階の分岐に関する探索的結果です。

負の交互作用には、情報欠落の重複と正規化歪みの飽和が混在します。その値を冗長性の量とは解釈しません。

次段階では、Oracleを逐次的に学習される観測方策へ置き換えます。各時刻に使わなかった観測機会を後から取り戻せない環境にすることで、現在の静的なデータフロー上の不可逆性から、理論稿が扱う時間的不可逆性へ進みます。
