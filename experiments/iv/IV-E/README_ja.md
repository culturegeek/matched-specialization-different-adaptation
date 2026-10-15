# 実験IV-E：Adam内部状態の成分交換

## 問い

実験IV-Dでは、取得主導と更新主導で形成した状態の短期的な変わりやすさが異なり、Adamを初期化すると大きな差が縮小しました。IV-Eは、Adamのどの内部状態がその差を運んだのかを調べます。

## 実験

二つの形成経路について、モデルの重みは同じまま保ち、Adamの内部状態だけを次の五条件へ分けました。

- `own`：形成時のAdamをそのまま使う
- `m_other`：一次モーメント `m` だけを反対経路から交換する
- `v_other`：二次モーメント `v` だけを交換する
- `both_other`：`m` と `v` の両方を交換する
- `reset`：新しいAdamへ初期化する

各条件へ、取得、更新、両方を組み合わせた三種類の短い反対方向摂動を与えました。パイロット16 seedの後、未使用40 seedで確認しました。

## 結果

一次モーメントを交換しても形成経路による感受性差は保たれました。二次モーメントを交換すると、どちらの形成経路が大きく動くかまで反転しました。複合摂動における取得主導と更新主導の差は、元のAdamで0.1968、`m_other`で0.1982、`v_other`で-0.1265でした。

取得主導経路の二次モーメント総量は更新主導経路の0.558倍で、実効更新幅は1.950倍でした。この人工系で観測された短期的な経路差の主要部分は、過去の二乗勾配を蓄積するAdamの二次モーメントを介していました。

## 再現

```bash
python experiment_iv_e.py --stage pilot --outdir results_pilot
python experiment_iv_e.py --stage confirm --outdir results_confirm
python audit_iv_e.py --results-dir results_confirm --outdir results_confirm
```

数値と判定の正本は `results_confirm/` のCSV、`result.json`、`independent_audit.json`です。実験コード、事前登録JSON、独立監査コードのSHA-256は、それぞれ `e0d244387abe910680cbb8b45b9d87751b9cf8af4452d35785c0ffd87af33744`、`6c4bd8d0c3b62bb797138b9106da1179f1dcd9a699c1a39460a54d173ea1c057`、`157626870ec33f198acc75ea5c0a4e52304ae3c99fc73059ee3a078967b04007`です。

## 適用範囲

この結果は、二対象の信号を二次元表現へ圧縮し、Adamで学習した人工ニューラルネットワークの10更新の応答に関するものです。`v`の全成分を交換したため、総量と重みごとの配置のどちらが反転に必要だったかは分離していません。
