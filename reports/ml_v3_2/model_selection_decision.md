# Phase 2-G: Frozen model selection decision (PRE-TEST)

**Frozen at UTC:** 2026-11-02T16:35:21.284430+00:00
**Status:** PASS. **Test accessed:** NO. **Dataset:** V3_2_MAIN synthetic.

## G1: Model selection

Winner: **RF_B** (v3.2-rf-b-platt-g1) on 58 ordered B5 features.
Rule: prioritize HK1 Validation PR-AUC, HK1 Recall@Top10%, Brier/ECE and inference cost.
HK1 PR-AUC improvement vs LR_B5: **+0.027428 absolute** (>=0.010 rule).
RF_A excluded from final shortlist due to poor pooled Brier 0.110479.
F model-selection evidence: `reports/ml_v3_2/model_comparison.json`.

## G2: Calibration

Platt sigmoid fitted using only Validation 2024 CALIBRATION UIDs (30% of UIDs).
Policy/threshold fitted using remaining Validation 2024 POLICY UIDs (70%).
All HK snapshots from one UID remain in exactly one partition.
Calibration UID count=1453, rows=4183, positives=356.
Policy UID count=3389, rows=9798, positives=884.
Calibration fit never sees policy labels. Model preprocessing and estimator fit only on Train.
Important: full Validation was previously used for F model selection; this is not an unbiased holdout.

## G3: Probability threshold (Validation POLICY only)

Threshold **0.27** maximizes F1 over fixed 0.05..0.80 grid step 0.01.
Policy F1=0.515133, Precision=0.567347, Recall=0.471719.
Policy raw Brier=0.057161; calibrated Brier=0.057117.
Policy raw ECE10=0.006095; calibrated ECE10=0.005361.

## G4: Alert budgets

Top5% and Top10%: `ceil(N*budget)` per term, score descending, UID and term ascending for ties.
Ranking budget is independent of absolute probability threshold.
Policy HK1 Recall@Top10=0.394619; pooled Recall@Top10=0.526018.

## G5: Frozen artifacts

- Model artifact: `models\v3_2\frozen_rf_b_pipeline.joblib`
- SHA256: `5c45deb380e011ff8ecfe94628d19e07db589ea75eed22c14bf08259887ed5c4`
- Source Phase F report SHA256: `d274b6f50eaded63f41793289e4f04b0135856385c5fd885f280e0241715b630`
- Save/load max probability delta: 2.220e-16
- Ordered 58 features and version/config/hashes in `model_selection_decision.json`.
- Model frozen now. Phase H may read Test once; no retuning based on Test.
- Synthetic simulation only; not validated for real CMC students.
