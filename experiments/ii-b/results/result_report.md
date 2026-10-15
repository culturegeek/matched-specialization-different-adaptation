# 実験II v0.6・容量×時間 結果

## 事前判定（仮説群ごと）


### A_boundary_specialization_capacity3

- PASS: capacity3_terminal_index_above_0_95
- PASS: capacity3_favored_error_near_zero
- PASS: capacity3_neglected_error_at_corr_floor
- PASS: capacity3_no_material_reversal

### B_interior_stationary_distribution_capacity2

- FAIL: capacity2_index_plateaus
- PASS: capacity2_favored_error_stationary
- FAIL: capacity2_neglected_error_stationary
- PASS: capacity2_favored_error_has_floor
- PASS: capacity2_terminal_is_interior

### C_dynamic_vs_fixed_capacity2

- PASS: capacity2_stabilization_exceeds_fixed
- PASS: capacity2_stabilization_concentrated

### D_bipolarity_secondary_diagnosis

- PASS: capacity2_bimodality_persists

### E_exploration_restorative

- PASS: exploration_restorative_both_capacities

合計PASS数では評価しない。A〜Eは別個の仮説群である。