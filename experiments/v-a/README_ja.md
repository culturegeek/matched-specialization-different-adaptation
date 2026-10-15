# 実験V-A 再現手順

## 実験の内容

三つの相関信号A、B、Cを、一つの小さな共有表現へ圧縮して学習させます。更新では三信号を同じ重みで扱いながら、どの信号を精密に入力するかだけを変え、形成後に精密入力の対象が相対的に復号しやすくなるかを調べます。

表現容量2と3を比較します。取得を均衡にして更新重みだけを偏らせる条件を陽性対照として含めます。

## 実行

このフォルダの一つ上にある専用環境を使います。

```bash
../.venv-v-a/bin/python experiment_v_a.py --stage smoke --device cpu --output-dir results_smoke
../.venv-v-a/bin/python experiment_v_a.py --stage pilot --device cpu --output-dir results_pilot
../.venv-v-a/bin/python experiment_v_a.py --stage confirm --device cpu \
  --preregistration preregistration_confirmatory.json --output-dir results_confirm
```

確認結果の独立再集計は、確認実験後に固定する `audit_results.py` を用います。

```bash
../.venv-v-a/bin/python audit_results.py \
  --results results_confirm \
  --preregistration preregistration_confirmatory.json
```

## 出力

- `timecourse.csv`：各seed・容量・条件の時系列
- `cell_summary.csv`：各条件の開始時点と形成終点
- `sensor_diagnostics.csv`：取得操作が入力情報量を変えたことの確認
- `result.json`：集約値と事前判定
- `provenance.json`：コードと事前登録のハッシュ、実行環境
- `result_report.md`：人が読める結果要約
- `independent_audit.json` / `independent_audit.md`：生CSVだけから行った独立再集計

確認実験では事前登録した27判定をすべて通過し、独立再集計でも完全性、開始状態、効果量、対称性、二自由度判定、ハッシュを含む33項目をすべて再現しました。研究の問い、方法、結果、適用範囲は、共有フォルダ直下の `../../105_実験V_実施報告.pdf` のV-A部分にまとめています。
