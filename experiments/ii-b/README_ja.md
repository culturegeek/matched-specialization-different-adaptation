# 実験II v0.6・容量×時間 確認実験

共有容量（bottleneck 2 と 3）と時間地平線（8000ステップ）により、
境界特化と内点の定常分布候補を分離する。seed 1000–1039。

## 実行

```bash
python capacity_horizon_poc.py --seed-start 1000 --seed-count 20 --threads 1 --output-dir results
python capacity_horizon_poc.py --seed-start 1020 --seed-count 20 --threads 1 --output-dir results
python capacity_horizon_poc.py --summarize results/raw_1000_1019.csv results/raw_1020_1039.csv --output-dir results
```

生データは逐次書き出される。中断しても `--seed-start` を同じにして再実行すれば、
完了済み seed をスキップして続きから走る（今回コンテナ再起動で実際に使用した）。
`--summarize` は既定で不完全データを拒否する（`--allow-incomplete` で解除）。

## 判定条件

`capacity_horizon_poc.py` の `PREREGISTERED_GATES` に実行前に固定。
改訂履歴（v1〜v2.2）と各撤回理由はファイル冒頭に記録。
v2〜v2.2 は集計と文書のみの改訂で、実行経路のコードは不変
（`provenance.json` の `execution_core_sha256` で検証可能）。

## 結果

`result_v06_review.md` を参照。要点のみ：

- 容量3：安定化型は内点で止まらず境界特化へ向かう（判定A 4件すべて成立）。
  不得意側の誤差は 1-rho^2 = 0.8775 付近に張り付き、得意側だけが 0 へ向かう。
- 容量2：終端 |s| = 0.3902 で明確に内点。得意側誤差は容量の下限 1/3 へ接近して止まる。
  ただし「止まっている」という同等性検定は僅差で不成立（判定B 5件中3件成立）。
- 両容量で状態は正負の二枝へ収束。容量2では L=0.397（上限から遠い）で
  97.5% のアームが ±L 付近に集中し、中間は 2.5%。指標の上限圧縮では説明できない。
- 動的更新と固定配分の質的差は内点でも残る（変動係数 0.093 対 0.526）。
- 探索型は基準線より偏差を小さくする（両容量、区間判定で成立）。

## 出力

- `results/raw_*.csv`: 生データ（5760行、欠損・重複・矛盾なし）
- `results/result.json`: 判定・時系列・診断量
- `results/result_report.md`: 仮説群ごとの判定
- `results/provenance*.json`: 事前登録文・設定・ハッシュ
