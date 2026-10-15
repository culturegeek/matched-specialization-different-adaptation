# 実験III：世界形成の競合源を分離する構成実験

このフォルダは、実験IIIを段階的に実施するためのコードと結果を保存します。
最初の段階 `III-A0` では、後続実験に使う有限状態の資源管理環境、観測チャネル、
Oracle、達成可能な帰結集合、資源拘束水準を校正します。

## 実行順序

NumPy だけを使用します。

```bash
python experiment_iii_a0.py --output-dir results_a0
python experiment_iii_a1.py --output-dir results_a1
python audit_a1_intervention_strength.py
python experiment_iii_a1_confirmatory.py --output-dir results_a1_confirmatory
python experiment_iii_a2.py --output-dir results_a2
python experiment_iii_a2b.py --output-dir results_a2b
python experiment_iii_b1.py --output-dir results_b1
python experiment_iii_b2.py --output-dir results_b2
python experiment_iii_c.py --output-dir results_c
python experiment_iii_c_transition_confirmatory.py --output-dir results_c_transition_confirmatory
python posthoc_c_reaction_coordinate_audit.py --input-dir results_c --output-dir results_c_posthoc_coordinate_audit
```

各確認実験の前に、対応する `preregistration_*.md` を固定しています。`pilot_*.py` は
介入水準・時間幅・測定分解能の選定にのみ用いた探索実験です。
`posthoc_c_reaction_coordinate_audit.py` は結果取得後に設計した事後解析であり、
事前登録判定には含めません。

## 検証規律

- 判定条件は `preregistration_a0.md` とコード中の `PREREGISTERED_GATES` に実行前から固定します。
- 生成過程を完全に列挙し、遷移確率の総和と非負性を検査します。
- Oracle 方策と一様ランダム方策を同じ初期分布・同じ有限地平で比較します。
- 競合環境と、同じ帰結を同時改善できる帰無環境を併設します。
- 「資源非拘束」は名称で仮定せず、追加資源による改善が停止する水準として測定します。
- 出力にはコードハッシュ、設定、完全性検査、全判定結果を保存します。

## 段階

`III-A0`、`III-A1`、`III-A2`、`III-B1`、`III-B2`、`III-C` の順に進みました。
III-A1初回確認では3仮説群、III-C本確認では1仮説群が不成立でした。結果は残したまま、
設計上の原因を監査し、未使用seedによる独立確認を追加しています。
III-Cの候補座標 `q=kappa` は、プール解析では転換を予測しましたが、摂動量 `K` を
条件づけた事後監査では予測連関が保たれず、反応座標候補として支持されませんでした。

## 主要ファイル

- `experiment_iii_v01_report.tex`: 実施報告のTeX原稿
- `results_*/result.json`: 判定条件、数値、完全性検査
- `results_*/provenance.json`: コードと事前登録のハッシュ
- `results_c_posthoc_coordinate_audit/`: 候補座標を `K` で条件づけた事後監査
- `pilot_*.py`: 探索的な設計パイロット
- `experiment_iii_*.py`: 事前判定を固定した確認実験
- `posthoc_c_reaction_coordinate_audit.py`: 事後監査の再現コード
