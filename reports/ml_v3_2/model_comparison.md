# Phase 2-F: model comparison (full)

V3_2_MAIN synthetic, Train 2022–2023 / Validation 2024. Test NOT accessed.

| Trial | HK1 PR-AUC | HK1 Recall@Top10% | Pooled PR-AUC | Brier | Fit sec |
|---|---:|---:|---:|---:|---:|
| LR_B5 | 0.225972 | 0.3730 | 0.534498 | 0.057198 | 1.21 |
| RF_A | 0.239971 | 0.3824 | 0.532502 | 0.110479 | 2.41 |
| RF_B | 0.253400 | 0.3981 | 0.538611 | 0.056871 | 3.75 |
| XGB_A | 0.232246 | 0.3668 | 0.537135 | 0.056944 | 1.76 |
| XGB_B | 0.237526 | 0.3730 | 0.539290 | 0.056836 | 1.94 |
| LGBM_A | 0.233284 | 0.3699 | 0.538600 | 0.056818 | 1.26 |
| LGBM_B | 0.232898 | 0.3636 | 0.533101 | 0.057237 | 1.88 |

Preliminary shortlist: RF_B, RF_A, XGB_B.
Recommendation for G analysis: RF_B.
Reason: HK1 PR-AUC improves by >=0.010 absolute vs LR_B5 on Validation.

Threshold and calibration are NOT selected here; only Validation. All metrics are synthetic.
Model size is serialized in-memory for estimation; no model artifacts saved in F.
Final decision and any tuning/calibration belong to Phase G.
