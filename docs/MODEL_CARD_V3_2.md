# MODEL CARD V3.2: Student Dropout Prediction (Synthetic CMC)

**Version:** `v3.2-rf-b-platt-g1`
**State:** FROZEN after Phase G; evaluated once on Test 2025 in Phase H
**Scenario:** `V3_2_MAIN / FULL_SIMULATION`, wholly synthetic research data
**Intended audience:** Academic project reviewers and a controlled educational-support prototype. **Not a real CMC operational/disciplinary model.**

## 1. Purpose, target and permissible use

Predict the **binary event `dropout_next_2_terms`**, meaning simulated dropout within the next two academic terms from the prediction snapshot. One record represents a student's state at the end of an academic term, not an independent student. The model is a *support-prioritization signal*, not a determination of the legal/academic status of a student and not an automatic withdrawal decision. The service only provides data observed by the requested `asof` cutoff and never exposes outcome labels.

**Not intended:** Real-student deployment without consent/appropriate institutional approval, reusing synthetic quality as evidence of real-world validity, assessing disciplinary penalties, ranking protected groups for disadvantageous treatment, or claiming that an unflagged student is safe.

## 2. Data and split protocol

| Partition | Snapshot rows | Positive labels | Distinct simulated students |
|---|---:|---:|---:|
| Train, cohorts 2022–2023 | 34,304 | 3,136 | 11,924 |
| Validation, cohort 2024 | 13,981 | 1,240 | 4,842 |
| Frozen Test, cohort 2025 | 9,284 | 911 | 3,234 |
| **Full Simulation total** | **57,569** | **5,287** | **20,000** |

Each simulated student ID (`record_uid`) stays in one partition; X/y are paired by `(record_uid, snapshot_term_index)`. The serving view `ASOF_2026_DEMO` has 54,609 visible snapshots as of **2026-10-01** and 9,970 right-censored outcomes. Censored records are **not** treated as negative labels or mixed into supervised FULL_SIMULATION training. `V3_2_STRESS` was not mixed into primary Train/Validation/Test.

Source archive SHA-256 from Phase 1 provenance: `5DF81F995347B167C4996F3A83C63C11637F78674834F76DD3D28A0548E6ECD6`. Local Train/Validation checksums are recorded in `b5_training_provisional.json`; Test input checksums are in `reproducibility.json`.

## 3. Feature contract and model

Exactly **58 ordered B5 features**: 20 Academic, 20 Behavior, 4 Finance, 4 Support, 10 Context, registered in `src/features.py`. Excludes identifiers, targets, splits, hidden outcomes and administrative hindsight. The preprocessing pipeline learns median numeric imputation, numeric scaling, most-frequent categorical imputation and one-hot encoding on **Train only**; unexpected categories are handled safely. The 5 categorical features are selected by explicit names to prevent pandas 3.x dtype drift.

The selected estimator is **Random Forest variant RF_B**, with its frozen configuration and seed `221026` recorded in `model_selection_decision.json`, fitted only to Train. A Platt sigmoid calibrator uses 1,453 Validation UIDs (4,183 snapshots); alert-policy selection uses 3,389 different Validation UIDs (9,798 snapshots). These UID groups do not overlap. The whole Validation set had previously contributed to **algorithm** choice in Phase F, so G metrics are still development/selection results, not unbiased holdout evidence.

Candidates tested on the same 58 features in Phase F: Logistic Regression, Random Forest, XGBoost and LightGBM (7 configurations). Phase E independently reproduced B0 prior, B1 GPA, B2 Academic, B3 +Behavior, B4 +Finance, B5 Full Safe and optional B3_V31 ablation.

## 4. Performance on synthetic data

**Validation (Phase F, model selection):**

| Trial | HK1 PR-AUC | HK1 Recall@Top10% | Pooled PR-AUC | Pooled Brier |
|---|---:|---:|---:|---:|
| LR_B5 | 0.225972 | 0.373041 | 0.534498 | 0.057198 |
| RF_B (selected) | 0.253400 | 0.398119 | 0.538611 | 0.056871 |
| XGB_B | 0.237526 | 0.373041 | 0.539290 | 0.056836 |
| LGBM_A | 0.233284 | 0.369906 | 0.538600 | 0.056818 |

Phase E B4 baseline (44 features) separately achieved Validation PR-AUC 0.538306, illustrating that additional features need not improve the result. RF_B was selected under the **pre-Test HK1 prioritization rule**, not because it dominates every pooled metric.

**Frozen Test 2025 (Phase H, one-shot):**

| Group | PR-AUC | ROC-AUC | Brier | F1 at 0.27 | Recall at 0.27 | Recall@Top10% |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | **0.543906** | 0.828460 | 0.062700 | 0.502481 | 0.444566 | 0.514819 |
| HK1 | 0.275367 | 0.771847 | 0.063078 | 0.271540 | 0.209677 | 0.423387 |
| HK2 | 0.573288 | 0.845660 | 0.064130 | 0.559350 | 0.525994 | 0.541284 |
| HK3 | 0.655727 | 0.853432 | 0.060796 | 0.589577 | 0.538690 | 0.553571 |

UID-cluster bootstrap, 400 valid replicates: pooled PR-AUC 95% CI **[0.50336, 0.58204]**; HK1 PR-AUC **[0.22762, 0.33190]**; HK1 Recall@Top10% **[0.36603, 0.47175]**. These are intervals *inside the simulated data-generating system*, not guarantees on real students.

## 5. Frozen alert policy

- **Calibrated probability threshold:** 0.27, chosen by F1 on Validation POLICY UIDs; never readjusted using Test.
- **Top5% / Top10% budgets:** `ceil(N × budget)` per relevant academic term, sorted by calibrated score descending, with deterministic composite-key tie-breaking. **TopK is a batch decision** and is not implied by a single person's threshold.
- HK1 Test: 3,234 snapshots, 248 positives. Threshold 0.27 alerts 135 snapshots, detects **52** positives and misses **196**. Top5% alerts 162, detects **60**, misses **188**. Top10% alerts 324, detects **105** and misses **143**. Threshold Recall (**20.97%**) must not be confused with Top10% Recall (**42.34%**).
- The service should convey calibrated probability as a *research risk estimate*, not a legal decision or guaranteed intervention priority.

## 6. Group audit, failure modes and limitations

Phase I audited 52 groups by academic term, cohort, major, education mode and gender, including HK1 slices. **18 groups had insufficient support**, so detailed metrics were suppressed. One Test cohort (2025) cannot establish across-cohort generalization. HK1 false negative rate is notably high under the fixed threshold. For example, HK1 `education_mode=GM` has FNR at 0.27 of approximately 90%; small-group comparisons and Wilson snapshot intervals do not independently account for repeated observations per UID. Interpret subgroup results cautiously and do not infer absence of bias.

HK1 Brier **0.063078**, ECE10 **0.008335**, mean predicted probability below observed prevalence by **0.007377**. Calibration improvements seen on development Validation are small; no claim of universal calibration. Other failure cases: sparse prior-term features at HK1, missing records at request time, out-of-scope `asof`, distribution shift across cohorts, or dataset/code/environment drift. Fail fast rather than manufacture predictions.

**Core external-validity limitation:** All students, trajectories and outcomes are artificially generated. There is no evidence that this cohort construction represents actual CMC risk factors, dropout mechanisms, demographics or policy effects. Group performance is descriptive, not a fairness certification. Feature importance does not prove causation.

## 7. Reproducibility, security and deployment boundary

- Frozen model: `models/v3_2/frozen_rf_b_pipeline.joblib` (local trusted binary, **Git-ignored**), SHA-256 `5c45deb380e011ff8ecfe94628d19e07db589ea75eed22c14bf08259887ed5c4`.
- Metadata: `models/v3_2/model_metadata.json` (commit permitted), checked against Phase J evidence. Version: `v3.2-rf-b-platt-g1`. Training environment on Windows: Python 3.14.7, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0, XGBoost 3.4.1, LightGBM 4.7.0.
- Reproducibility: independent save/load maximum absolute probability delta **2.22e-16** across 64 Validation rows. API/direct prediction maximum delta **4.16e-17** across 3 demo snapshots.
- Serving contract: `src.inference_v32.predict(uid, term, asof)` via `V32DataAccess`, with `asof <= 2026-10-01`, exactly 58 features, no labels, no implicit training, no V2.2 fallback. Unknown UID/term/snapshot returns `not_available` when appropriate.
- `joblib.load` unpickles Python objects: load **only trusted, hash-verified artifacts**. Cryptographic hashes protect against accidental drift, not a malicious attacker able to modify both file and trusted metadata.
- `reports/ml_v3_2/test_predictions_frozen.csv.gz` contains snapshot-level **Test labels**: keep it local, Git-ignore it, **never** make it available to the Streamlit application, inference API, or public repository.

## 8. Provenance and review pointers

- Phase E: `reports/ml_v3_2/baseline_reproduction.md` and `baseline_comparison.csv`.
- Phase F/G: `model_comparison.md`, `model_selection_decision.md/json`.
- Phase H/I: `frozen_test_metrics.md/json`, `per_term_metrics.csv`, `test_bootstrap_ci.json`, `subgroup_audit.md`.
- Phase J/K: `artifact_reproducibility_J.md/json`, `inference_contract_K.json`.
- Phase L: `phase2_L_verification.json` and `PHASE_2_ML_Training_Handover.md`, to be finalized **only after local tests and Git checks pass**.

Next phase: integrate serving API into the app with safe as-of selection, meaningful explanations and disclaimer; do not retrain on application startup or present synthetic Test outcomes as real-student predictions.
