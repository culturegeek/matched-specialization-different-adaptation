# 実験II v0.4・初期経験系列への因果介入

初期の選択履歴が表現形成の方向を因果的に変えるか、その差が後続の選択傾向によって持続・増幅するかを検証します。

## 実験の流れ

1. `history_closed_loop_poc.py` は、強いA履歴／B履歴を与え、閉ループ誤差選択で差が維持されるかを測ります。
2. `history_strength_sweep.py` は、初期履歴の強さを走査します。
3. `history_strength_confirmatory.py` は、探索的に得られた弱履歴条件を独立seedで確認します。固定配分に対する閉ループ固有の上乗せは確認されませんでした。
4. `symmetry_breaking_poc.py` は、AとBを等しく学習した対称状態から、偶発差が安定化型で増幅され、探索型で縮小するかを測ります。
5. `forced_selection_causal_poc.py` は、状態から独立に割り当てた鏡像的な初期選択系列により、経験系列から表現差への因果効果を測ります。

## 比較条件

- `balanced_sampled`: A/Bの一方を等確率で学習
- `fixed_propensity`: 開始時の表現精度から選択確率を一度だけ求め、その後は固定
- `stabilization`: 現在より精密に表現できる側を学習対象にしやすくする
- `exploration`: 現在より不精密な側を学習対象にしやすくする

選択確率の更新には、学習用・評価用とは異なる固定バッファを使います。共有表現を固定した線形プローブの復号誤差から現在の表現精度を測ります。このプローブは生成過程の正解を使う上限対照です。

## 実行方法

CPUでも動作します。基本実験と独立確認は、次の順で実行します。

```bash
python history_closed_loop_poc.py --quick --device auto --output-dir closed_loop_results_quick
python history_closed_loop_poc.py --full --device auto --output-dir closed_loop_results_full
python history_strength_sweep.py --device auto --output-dir history_strength_sweep_results
python history_strength_confirmatory.py --device auto --output-dir history_strength_confirmatory_results
python symmetry_breaking_poc.py --device auto --seed-start 200 --seed-count 40 --output-dir symmetry_breaking_results
python symmetry_breaking_poc.py --device auto --seed-start 300 --seed-count 40 --output-dir symmetry_breaking_confirmatory_results
python forced_selection_causal_poc.py --device auto --seed-start 400 --seed-count 40 --output-dir forced_selection_causal_results
python forced_selection_causal_poc.py --device auto --seed-start 500 --seed-count 40 --output-dir forced_selection_causal_confirmatory_results
```

## 主要結果

強い初期履歴では、安定化型と固定配分がほぼ同じ結果になりました。強い履歴の維持は、閉ループ固有の効果ではなく、開始時に決まった配分を適用し続けた結果です。

弱い履歴で探索的に得られた「安定化型が固定配分を上回る」という結果は、独立seedでは確認されませんでした。AUC比は1.155、対応差の95%区間は `[-0.021, 0.291]` でした。

対称条件からの分岐では、独立seedでも安定化型の最終偏差が固定配分の2.064倍となり、対応差の95%区間は `[0.368, 0.527]` でした。探索型の偏差は安定化型の6.7%に抑えられました。一方、初期選択の偏りと内部状態は閉ループ内で相互依存するため、この観察だけでは因果方向を判定できませんでした。

そこで、同一モデルにAを55回・Bを45回選ぶ系列と、その鏡像系列を外部から割り当てました。独立seedでも、操作直後の方向効果は0.0848、95%区間は `[0.072, 0.098]` となりました。その後の固定配分では効果が2.796倍、安定化型では2.919倍へ増幅し、探索型では0.5%まで縮小しました。二つのseed集合を合わせた80組では、操作直後と固定配分の全組、安定化型の79組が予測方向を示しました。

安定化型と固定配分の差は支持されませんでした。80 seedでの対応差は -0.0070、95%区間は `[-0.077, 0.063]` です。したがって、今回確認された因果鎖は「初期経験系列→表現差→選択傾向→差の持続・増幅」であり、選択確率を継続的に更新する追加効果までは確認されていません。

## 出力

- `closed_loop_results_quick/`, `verified_results_full/`: 強い初期履歴条件
- `history_strength_sweep_results/`: 初期履歴強度の探索走査
- `history_strength_confirmatory_results/`: 弱履歴条件の独立確認
- `symmetry_breaking_results/`: 対称条件からの分岐・探索seed
- `symmetry_breaking_confirmatory_results/`: 対称条件からの分岐・独立seed
- `forced_selection_causal_results/`: 初期選択系列への介入・探索seed
- `forced_selection_causal_confirmatory_results/`: 初期選択系列への介入・独立seed
- 各フォルダの `result.json`, `result_report.md`, `raw_timecourse.csv`: 判定、要約、時系列

## 実験が扱う範囲

選ばれるのは、取得済みデータのうちどちらの差異を誤差として学習に入れるかです。観測対象を選ぶ方策、表現精度を系自身が推定する学習機構、心理志向／客観志向などの意味構造、課題を越えた復帰やヒステリシスはまだ導入していません。
