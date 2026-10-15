# 実験IV-C 再現コード

この実験は、更新配分によって形成したA/B方向の共有表現を、更新配分を均衡化した後の取得精度だけで反対方向へ動かせるかを調べる。

## 実行順

```bash
python experiment_iv_c.py --stage smoke --output-dir results_smoke
python experiment_iv_c.py --stage pilot --output-dir results_pilot
python experiment_iv_c.py --stage confirm \
  --preregistration preregistration_confirm.json \
  --output-dir results_confirm
python audit_iv_c.py --results results_confirm \
  --script experiment_iv_c.py \
  --preregistration preregistration_confirm.json
```

## 実験条件

- A/B形成済み状態は、A/Bとも雑音標準偏差0.15で取得し、更新重みだけを0.85/0.15または0.15/0.85として500回形成する。
- 形成後の更新重みは全条件0.50/0.50に固定する。
- 取得はA精密、均衡、B精密の三条件で、1000回追跡する。
- 中立checkpointからの同じ三条件を陽性対照として同時に走らせる。
- 主測定は全条件共通のA/B高精度・未使用probe/testで行い、A/B低精度の共通測定を副測定とする。

## 確認結果

- 未使用40 seed（7700--7739）
- 形成済み二状態で平均したA精密取得−B精密取得：0.7161（95%区間0.7108--0.7213）
- A形成→B方向、B形成→A方向とも40/40 seedがゼロ交差
- 平均ゼロ交差：16.9更新、16.8更新
- 事前登録13判定、独立監査13項目とも全PASS

## 主な出力

- `timecourse.csv`: 各seed・出発状態・取得条件の全測定時点
- `cell_summary.csv`: 条件別の開始値、終端値、変化量、ゼロ交差
- `sensor_diagnostics.csv`: 取得精度の操作確認
- `result.json`: 全集計と事前判定
- `independent_audit.json`: 結果JSONを使わない独立再集計
- `provenance.json`: コード、事前登録、実行環境のhash

## 必要環境

- Python 3.11.7
- NumPy 2.4.6
- PyTorch 2.13.0

CPUで再現できる。統計単位はseedであり、学習stepや測定時点を独立標本として数えない。
