# 実験III-A1 結果：五つの一要因介入

## 仮説群

- **K_task_frame**: PASS
- **M_causal_model**: FAIL
- **Hpred_prediction_horizon**: PASS
- **Heval_evaluation_horizon**: FAIL
- **Risk_sensitivity**: FAIL

## 課題枠組み

|枠組み|選択センサ|対象正答率|非対応正答率|増分|
|---|---|---:|---:|---:|
|demand|demand|0.9400|0.6200|0.3200|
|health|health|0.9400|0.6200|0.3200|

## 方策介入の真の帰結

|条件|現在不足|障害リスク|危険状態率|初期不足|基本目的|
|---|---:|---:|---:|---:|---:|
|baseline|0.346448|0.006215|0.018488|0.270166|0.176332|
|causal_model_misspecified|0.346527|0.006227|0.025782|0.258592|0.176377|
|prediction_horizon_1|0.344957|0.023665|0.070979|0.151189|0.184311|
|evaluation_discount_035|0.345041|0.010117|0.050275|0.157231|0.177579|
|risk_sensitive|0.348945|0.004637|0.008452|0.339382|0.176791|

## 進行判断

次段階の前に不成立要因を監査する: M_causal_model, Heval_evaluation_horizon, Risk_sensitivity

## 主張の範囲

各介入が有限状態環境で識別可能な異なる読み出しを生むかを校正した結果である。人間の対立や志向性の原因を特定したものではない。
