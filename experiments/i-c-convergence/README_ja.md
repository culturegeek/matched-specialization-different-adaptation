# 実験I-C・共同更新の収束性監査 v0.2.1

実験I-C v0.2では、時間相関を加えると観測スケジュールの目的地形が非加法的になり、方策と推定器を共同更新しても、20 seedの一部がOracleとは異なるスケジュールへ収束しました。

v0.2.1は、その非Oracle方策が次のどれに当たるかを調べる診断実験です。

1. 母集団上の静的な局所最適
2. 方策と推定器の共同更新が作る安定点
3. 400反復では足りなかっただけの未収束
4. 0/1に近づいたsoftmax方策から動けないアルゴリズム上の捕捉

閉ループ方策へ進む前に、v0.2で観測された収束先の性質を確定することが目的です。

## 監査設計

v0.2と同じデータ、初期値、AR(1)時間相関 `phi=0.8` を使います。共同更新を400反復行った時点から、同じ方策を五つの条件へ分岐させます。

- `continue`：Adamの状態も保って3200反復まで継続
- `reset_only`：方策は保ち、Adamの内部状態だけをリセット
- `reheat_x2`：logitの差を2分の1に縮めて再加熱
- `reheat_x4`：logitの差を4分の1に縮めて強く再加熱
- `reheat_x4_noise`：強い再加熱に小さなランダム摂動を追加

400、800、1600、3200反復で、目的値、方策エントロピー、logit勾配、収束スケジュールを測ります。

さらに、現在ほぼ選ばれていない行動へ確率を戻したとき、共同目的が一次的に改善するかを `max_boundary_escape_gain` として測ります。これにより、勾配が小さい理由が単なるsoftmax飽和なのか、現在の推定器に対して局所的な改善方向が本当にないのかを区別します。

## Google Colabで実行

1. `colab_runner.ipynb` をGoogle Colabへアップロードします。
2. 最初のセルで、同梱の次の2ファイルを選びます。
   - `temporal_coupling_poc.py`
   - `joint_convergence_audit.py`
3. 上から順に実行します。GPUは不要です。
4. 最初は `--quick` を実行します。
5. 成功後、実行セルを `--full` に変えると、20 seed・3200反復の監査になります。

手元のCPUでは、quickは約25秒、fullは約210秒でした。Colabでは混雑状況により変わります。

## ローカルで実行

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python joint_convergence_audit.py --quick --output-dir joint_convergence_audit_results
```

Windowsでは仮想環境の有効化に `.venv\Scripts\activate` を使います。

## 主な出力

- `result_report.md`：主要結果と診断分類
- `seed_classification.csv`：seed・lambdaごとの最終分類
- `audit_summary.csv`：反復数・分岐条件別の集計
- `audit_raw.csv`：各seedの目的値、entropy、勾配、境界脱出利得
- `schedule_transitions.csv`：最適化中にargmax方策が変わった時点
- `oracle_reach_by_iteration.svg`：Oracle到達率の推移
- `metadata.json`：実行条件と分類件数

## full設定で得られた結果

60条件（20 seed × 3 lambda）のうち、36条件は400反復時点ですでにOracleへ到達していました。残る24条件では、次の結果になりました。

- 22条件：母集団の静的局所最適ではないが、3200反復、Adamリセット、再加熱、摂動後も同じ非Oracle方策へ復帰
- 1条件：母集団の静的局所最適と共同更新の安定点候補が一致
- 1条件：2倍の再加熱では動かず、4倍の再加熱でOracleへ移動

単純な反復継続だけで400反復時点の非Oracle方策からOracleへ移った条件はありませんでした。したがって、v0.2の非Oracle結果を一括して「反復不足」と説明することはできません。

ただし、安定点候補は数学的な固定点の証明ではありません。現行のAdam更新、因子化softmax、有限標本のもとで再現された動力学的性質です。

## 限界

本監査は依然として開ループ方策を扱います。現在までに見た値に応じて次の観測先を変える閉ループ注意は未実装です。また、志向性、心理志向／客観志向、発達的履歴依存、人間の注意を直接支持する実験ではありません。
