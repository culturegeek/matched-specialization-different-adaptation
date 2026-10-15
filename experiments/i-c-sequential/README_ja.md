# 実験I-C・逐次観測方策PoC v0.1

このPoCは、実験I-BのOracle観測選択を、経験から学習される最小の観測方策へ置き換えます。

各時刻にA側またはB側のどちらか一方だけを観測します。その時刻に見なかった値は後から同じ条件で取得できません。方策は、A/Bのどちらをどれだけ保存したいかを表す重み `lambda` に応じ、時刻ごとの観測先を学習します。

## 比較する方策

- `oracle`：16通りの固定スケジュールから検証データで最良を選ぶ上限対照
- `random_fixed`：毎時刻A/Bを等確率で見る固定方策
- `A_fixed` / `B_fixed`：一方だけを見続ける特化対照
- `learned_frozen`：ランダム方策下で作った推定器を固定し、観測方策だけを学習
- `learned_joint`：観測方策と、その方策分布に最適な線形推定器を反復的に更新
- `full_observation`：毎時刻A/Bの両方を見る無制約対照

## この段階で分かること

- 一時刻に一方しか見られないことによる、時間的な機会費用
- 保存要求に応じて学習方策が異なる観測スケジュールへ収束できるか
- ランダム、特化固定、Oracleに対して、学習方策がどの位置にあるか
- 下流推定器を固定した場合と共同更新した場合の差

## この段階では分からないこと

方策は現在の状態を見ず、時刻だけでA/Bを選ぶ開ループ方策です。そのため、状況に応じた注意、心理志向／客観志向のような意味を持つ不変性、発達履歴による志向性の固定を検証しません。

また、検証済みの線形環境では `learned_frozen` と `learned_joint` の差はほぼゼロでした。ランダム方策下で学習した推定器が、選択後の各チャネルを既に十分復号できるためです。このPoCは、共同更新の必要性を示す実験ではありません。

## Google Colabで実行

1. `colab_runner.ipynb` をColabへアップロードします。
2. 最初のセルで `sequential_policy_poc.py` をアップロードします。
3. 上から順にセルを実行します。GPUは不要です。
4. まず `--quick` を実行し、自動判定と図を確認します。
5. 成功後、実行セルの `--quick` を `--full` に変えると20 seedで再確認できます。

## ローカルで実行

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sequential_policy_poc.py --quick --output-dir sequential_policy_results
```

Windowsでは仮想環境の有効化に `.venv\Scripts\activate` を使います。

## 主な出力

- `result_report.md`：自動判定、主要数値、最頻スケジュール
- `policy_performance.svg`：方策ごとの重み付き歪み
- `learned_policy.svg`：時刻ごとのA観測確率
- `summary_results.csv`：方法と `lambda` ごとの平均・標準偏差
- `raw_results.csv`：seedごとの歪みとスケジュール
- `policy_probabilities.csv`：学習された時刻別観測確率
- `learning_history.csv`：方策最適化の履歴
- `metadata.json`：実行環境、条件、自動判定
