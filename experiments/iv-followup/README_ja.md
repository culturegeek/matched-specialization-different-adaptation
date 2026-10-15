# 実験IV-E 追加検証（2026-09-03）

二次モーメントの全体的な調整倍率と、パラメータごとの配置の違いを分け、短期応答差の予測と持続時間を調べる。

## 収録物

- `protocol.md`：実行前計画。既存データ解析は探索的、新規40 seedは固定した介入検証。
- `registration.json`：確認実行前の時刻、seed、コード・計画・予測係数のSHA-256。
- `run_followup.py`：形成checkpointの再生成と10条件の介入。既存ファイルを上書きしない。
- `analyze_followup.py`：CSVからのseed内集計と予測評価。
- `audit_results.py`：実験・集計モジュールを読み込まない再集計と整合性検査。
- `posthoc_review_analysis.py`：固定済み確認CSVから、参加比、四条件の再結合、10--500更新の時間プロファイルを探索的に再集計する。
- `POSTHOC_REVIEW_NOTE.md`：レビュー後再集計の証拠地位と、応答差・終点差の区別を記録する。
- `posthoc_review_results.json`：レビュー後再集計の数値出力。確認的判定には用いない。
- `posthoc_update_norm_analysis.py`：固定済み確認CSVから、勾配経路長、実際のパラメータ更新経路長、配分応答の関係を探索的に再集計する。
- `POSTHOC_UPDATE_NORM_NOTE.md`：実測更新量解析の証拠地位、主要結果、監査値を記録する。
- `posthoc_update_norm_results.json`：実測更新量解析の全体集計、相関、交差検証結果。
- `posthoc_update_norm_seed_table.csv`：40 seedごとの倍率、勾配経路、更新経路、配分応答。
- `existing_analysis/`：旧40 seedの探索的予測検証、新規seedへそのまま適用する係数。
- `results_reproduce/`：旧9100 seedの再現検査。
- `results_smoke/`：19799 seedでの動作確認。確認結果へ混ぜない。
- `results_confirm/`：20000–20039の確認40 seed。各seedの形成checkpoint、時系列、更新診断、集計、監査。
- `vendor/`：既存v0.5パッケージから抽出した凍結コードと必要な旧CSV。

数値の正本は`results_confirm`の生CSV・`result.json`、証拠の地位は`protocol.md`と実施記録を参照する。共有用PDFは新しく分割せず、`104_実験IV_実施報告.pdf`へ統合する。

## 再現

Python 3.11 / PyTorch 2.13.0 / NumPy 2.4.6、CPU、単一スレッドで実行。元の実験と同じ環境を使うため、文書用のPython環境とは分けた。

```sh
python run_followup.py --stage reproduce --out results_reproduce_new
python run_followup.py --stage confirm --out results_confirm_new
python analyze_followup.py --folder results_confirm_new
python audit_results.py --folder results_confirm_new
```

監査の既定入力は同梱の`results_confirm`である。別名の再現結果は`--folder`で指定する。新規の確認登録は作り直さず、同梱の登録ハッシュを確認する。

`formation_checkpoints.pt`は本パッケージで生成したモデル／optimizerの数値辞書である。出所不明のPyTorchファイルを代わりに読み込まない。

## 解釈上の区別

平均逆スケーリング項を揃える操作は、実際のパラメータ更新ノルムを揃える操作とは異なる。両方を記録して区別する。SGDは形成後に交換した更新器であり、SGDで形成した状態を比較する実験ではない。均衡継続は、旧IV-Dで行った「元の優位対象を支える条件への復帰」とも異なる。
