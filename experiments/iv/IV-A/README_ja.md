# 実験IV-A：形成済み表現に対する情報取得と更新配分

形成済みの共有表現について、どちらを高精度で観測するかと、どちらの誤差から学ぶかを独立に操作します。同一seedでは、同じ形成済みcheckpoint、optimizer状態、入力系列、ノイズ系列から全条件へ分岐します。

## 主2×2

| | 基準側から更新 | 代替側から更新 |
|---|---|---|
| 基準側を高精度取得 | 一致条件 | 交差条件 |
| 代替側を高精度取得 | 交差条件 | 一致条件 |

追加センサなしと、各更新時点でセンサの向きを無作為に決める条件を補助対照として実行します。

同じ学習軌道を、形成済み状態を保つ「維持性」と、代替側へ移る「再配分可能性」の二面から評価します。代替要求は評価基準としてのみ用い、学習系へ要求変更を入力しません。

主確認後の追加監査では、AとBを同じ低雑音で測る共通入力による再測定と、AとBの取得・更新をともに対称化する中立継続を行います。前者は測定時の雑音が取得効果を作っていないか、後者は形成済みの偏りが取得・更新の非対称性を外した後にどれだけ残るかを調べます。

この実験は当初「実験III-D」として実行したため、固定済みコード、事前計画、結果データ内の実験名とスクリプト名 `experiment_iii_d.py` は変更していません。現在の研究系列では同じ実験を「実験IV-A」と呼びます。

## 実行

実験IIで使用したPython環境から実行します。

```bash
python experiment_iii_d.py --stage smoke --output-dir results_smoke
python experiment_iii_d.py --stage pilot --output-dir results_pilot
python experiment_iii_d.py --stage confirm --formation-steps 500 --trajectory-steps 400 --low-noise 1.2 --high-noise 0.05 --formation-weight 0.85 --branch-threshold 0.08 --output-dir results_confirmatory
python audit_measurement_neutral.py
python audit_measurement_neutral_independent_check.py
```

確認実験は、パイロット後に設定、判定条件、seed範囲を固定してから実行します。

## 出力

- `timecourse.csv`: 全条件の時系列
- `cell_summary.csv`: seed・形成済み状態・条件ごとの要約
- `sensor_diagnostics.csv`: 追加センサの操作確認
- `result.json`: 集計結果
- `result_report.md`: 読みやすい結果概要
- `provenance.json`: 実行環境とスクリプトのハッシュ

追加監査の出力は `results_measurement_neutral_audit/` に保存します。

- `primary_dual_probe_timecourse.csv`: 主2×2を低精度入力と共通低雑音入力の両方で測った時系列
- `neutral_timecourse.csv`: A・Bの取得と更新を対称化した中立継続時系列
- `result.json`: 追加監査の主集計
- `independent_check.json`: 別集計コードによる完全性と主要数値の照合
- `provenance.json`: 実行環境、元実験、追加監査計画、コードのハッシュ
