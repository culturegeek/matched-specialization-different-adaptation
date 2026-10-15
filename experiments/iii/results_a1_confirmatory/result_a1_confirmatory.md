# 実験III-A1.1 結果：独立環境集合による確認

## 仮説群

- **K_task_frame**: PASS
- **M_causal_model**: PASS
- **Hpred_prediction_horizon**: PASS
- **Heval_evaluation_horizon**: PASS
- **Risk_sensitivity**: PASS

## 区間推定

|測定量|平均|95%区間|
|---|---:|---:|
|k_target_accuracy|0.941641|[0.936414, 0.946944]|
|k_accuracy_gain|0.323726|[0.316706, 0.331094]|
|m_disagreement|0.136875|[0.130000, 0.143750]|
|m_prediction_l1|0.156991|[0.153478, 0.160695]|
|m_correct_l1|0.000000|[0.000000, 0.000000]|
|hpred_disagreement|0.430875|[0.425250, 0.436500]|
|hpred_objective_gain|0.007955|[0.007736, 0.008184]|
|heval_disagreement|0.295875|[0.290250, 0.301500]|
|heval_early_shortage_gain|0.136465|[0.128344, 0.144842]|
|heval_risk_gain|0.012339|[0.011855, 0.012834]|
|risk_disagreement|0.166500|[0.162375, 0.170500]|
|risk_danger_abs_reduction|0.012737|[0.011987, 0.013530]|
|risk_danger_rel_reduction|0.695503|[0.682488, 0.708799]|
|risk_shortage_change|0.009452|[0.009159, 0.009754]|

## 進行判断

全仮説群が成立した。III-A2へ進む。

区間は事前に定めた人工環境集合内の頑健性を表し、現実環境への母集団推論ではない。
