# 実験I-C・時間結合を持つ逐次観測方策PoC v0.2

v0.1では、各時刻にA/Bの一方しか見られない観測機会と、その配分を学習する開ループ方策を校正しました。しかし、時刻ごとの信号が独立であり、目的関数は四つの二値選択へ完全に分離できました。

v0.2はA/Bの両信号にAR(1)時間相関 `phi` を入れます。これにより、ある時刻で何を見たかが、別時刻の値をどこまで推定できるかを変えます。

## 主な問い

- `phi=0` では、v0.1どおり目的関数が時刻ごとに分離できるか
- `phi>0` で加法分解からの残差が生じるか
- Hamming距離1の近傍で抜けられない局所最適が生じるか
- 全16スケジュールからの貪欲改善は、どの局所最適に到達するか
- 学習方策は初期値によってOracle以外のスケジュールに収束するか
- 観測方策に合わせて下流推定器を更新する利得が生じるか

## 構造の測定

全16スケジュールの母集団目的値を、AR(1)ガウス過程の共分散から厳密に計算します。

```text
J(s) = c0 + c1*s1 + c2*s2 + c3*s3 + c4*s4
```

を最小二乗で当て、最大絶対残差とRMSEを測ります。残差がゼロなら時刻別の独立選択です。残差が生じると、一つの時刻の選択価値が他時刻の選択に依存しています。

## Google Colabで実行

1. `colab_runner.ipynb` をGoogle Colabへアップロードします。
2. 最初のセルで `temporal_coupling_poc.py` を選びます。
3. 上から順に実行します。GPUは不要です。
4. まず `--quick` で自動判定と図を確認します。
5. 成功後、実行セルの `--quick` を `--full` に変えると20 seedで再確認できます。

## ローカルで実行

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python temporal_coupling_poc.py --quick --output-dir temporal_coupling_results
```

Windowsでは仮想環境の有効化に `.venv\Scripts\activate` を使います。

## 主な出力

- `result_report.md`：自動判と主要結果
- `landscape_summary.csv`：加法残差、局所最適数、吸引域、Oracle
- `landscape_raw.csv`：全16スケジュールの母集団歪みと加法残差
- `policy_summary.csv`：方法ごとの歪み、Oracle到達率、収束スケジュール
- `policy_raw.csv`：seedごとの方策結果
- `policy_probabilities.csv`：時刻別のA/B観測確率
- `learning_history.csv`：最適化中の目的値とエントロピー
- `temporal_coupling_structure.svg`：時間相関による非加法性と局所最適数
- `policy_oracle_reach.svg`：学習方策のOracle到達率
- `metadata.json`：実行条件と自動判定

## 限界

方策はまだ開ループです。現在までに見た値に応じ、次に何を見るかを変えません。そのため、この実験は時間的結合だけを導入した中間段階です。志向性、心理志向／客観志向、発達的履歴依存、人間の注意を支持する結果ではありません。
