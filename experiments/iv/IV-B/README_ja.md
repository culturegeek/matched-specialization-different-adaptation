# 実験IV-B 再現コード

この実験は、同じ未形成モデルから始めても、AとBのどちらを精密に経験するかによって、形成される共有表現の方向が変わるかを調べる。

## 実行順

```bash
python experiment_iv_b.py --stage smoke --output-dir results_smoke
python experiment_iv_b.py --stage pilot --output-dir results_pilot
```

確認実験は、パイロット後に固定する `preregistration_confirmatory.json` を明示して実行する。

```bash
python experiment_iv_b.py --stage confirm \
  --preregistration preregistration_confirmatory.json \
  --output-dir results_confirm
```

## 主な出力

- `timecourse.csv`: 各seed・各条件の形成過程
- `cell_summary.csv`: 形成後の条件別指標
- `sensor_diagnostics.csv`: 取得精度の操作確認
- `result.json`: 全集計と判定
- `provenance.json`: コード、事前登録、実行環境
- `result_report.md`: 人が読める要約

## 必要環境

- Python 3.11.7
- NumPy 2.4.6
- PyTorch 2.13.0

CPUで再現できる。確認実験の統計単位はseedであり、学習stepや時点を独立標本として数えない。
