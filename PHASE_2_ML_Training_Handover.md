# Phase 2: ML Training, Selection, Evaluation & Inference, V3.2

**State:** Implementation A–K verified in local PowerShell logs; Phase L closure **PENDING** until test suite, staging review and push are completed. Do not mark Phase 2 100% before `PHASE_2_L_FINAL_PASS`.

## 1. Git and immutable version identifiers

- Working branch: `Dung` only. **Do not merge/modify `main` as part of this handover.**
- Phase 1 baseline commit: `5c425dce71313e784a77e6b189636cd56f8d2ba6`.
- **Phase 2 implementation commit SHA:** `3dabfefb94356ff17651e7c0153e294e8e88aa03` (filled by `scripts/close_phase2_L.py --stamp-handover` after the implementation commit).
- Handover commit SHA: inspect `git rev-parse HEAD` after second commit and `git log -2 --oneline`; a Git commit cannot embed its own final hash in its own contents.
- Frozen model: `RF_B + Platt`, `v3.2-rf-b-platt-g1`, SHA-256: `5c45deb380e011ff8ecfe94628d19e07db589ea75eed22c14bf08259887ed5c4`.
- Metadata SHA-256: `cf559689b3152e570018ddb266267aea3312bd6d0422eca06b53f1af61385385`.
- Frozen decision: threshold **0.27**, Top5% and Top10% per term, calibration split by UID. **No retuning based on Test.**

## 2. Source and experiment contract

- `V3_2_MAIN/FULL_SIMULATION` synthetic data; target `dropout_next_2_terms`; exactly 58 B5 ordered features.
- Train 2022–23: 34,304 rows / 3,136 positives / 11,924 UID.
- Validation 2024: 13,981 / 1,240 / 4,842 UID.
- Frozen Test 2025: 9,284 / 911 / 3,234 UID.
- `ASOF_2026_DEMO` serving upper cutoff: `2026-10-01`; 9,970 right-censored labels may **not** be reclassified as negatives.
- Source ZIP SHA-256 from Phase 1 provenance: `5DF81F995347B167C4996F3A83C63C11637F78674834F76DD3D28A0548E6ECD6`.
- Tested environment: Python 3.14.7, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0, XGBoost 3.4.1, LightGBM 4.7.0.

## 3. Measured outcomes (unchanged from H/I)

| Test 2025 | Pooled | HK1 | HK2 | HK3 |
|---|---:|---:|---:|---:|
| PR-AUC | 0.543906 | 0.275367 | 0.573288 | 0.655727 |
| ROC-AUC | 0.828460 | 0.771847 | 0.845660 | 0.853432 |
| Brier | 0.062700 | 0.063078 | 0.064130 | 0.060796 |
| Recall@threshold 0.27 | 0.444566 | 0.209677 | 0.525994 | 0.538690 |
| Recall@Top10% | 0.514819 | 0.423387 | 0.541284 | 0.553571 |

HK1 Top10%: 324 alerts, 105/248 positives captured, **143 missed**. Threshold 0.27: 135 alerts, 52/248 positives captured, **196 missed**. These are distinct policies. Pooled PR-AUC 95% UID-cluster CI [0.503358, 0.582043], HK1 Recall@Top10% CI [0.366033, 0.471753], 400 bootstrap replicates.

52 synthetic subgroup slices were audited; 18 with insufficient support. No real-student generalization or fairness claims.

## 4. Source code and read-only verification

- `src/ml_v32.py`: strict FULL_SIMULATION training loader + 58-feature pipeline.
- `src/model_policy_v32.py`: calibration/threshold/TopK utilities.
- `src/model_bundle_v32.py`: trusted hash-pinned joblib loader.
- `src/inference_v32.py`: public `predict(uid, term, asof)` via `V32DataAccess`.
- `scripts/train_v32.py`: **standalone V3.2 trainer**, never use legacy `src/train.py` in Phase 2.
- `scripts/test_phase2_CD.py`, `test_phase2_F.py` ... `test_phase2_K.py`: 93 previous tests; `scripts/test_phase2_L.py`: 13 additional unit tests.
- Verify all tests, A–K reports and Phase 1 E1–E5 **without Test rescore**:
  ```powershell
  .\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --preflight
  .\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_L -v
  .\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --verify
  ```
- Exact implementation protocol, safe stage, real SHA stamping and commit/push in `docs/PHASE_2_L_RUNBOOK.md`.

## 5. Restore artifacts on a different machine

**Git alone is not sufficient** because joblib model and source data are deliberately ignored.

1. `git clone`/checkout `Dung`, provision the documented Python environment (prefer matching scikit-learn 1.9.1), and inspect `git status -sb`.
2. Restore `data/external/v3_2_main/` and `data/processed/v3_2/` from **trusted Phase 1 backups** using `PHASE_1_Data_Integration_V3_2_Handover.md`. Verify source archive checksum and every recorded split/input hash. Do not regenerate new synthetic data as a substitute.
3. Restore **trusted** `models/v3_2/frozen_rf_b_pipeline.joblib` from the original model backup, with SHA-256 `5c45deb380e011ff8ecfe94628d19e07db589ea75eed22c14bf08259887ed5c4`. Similarly restore `b5_reference_pipeline.joblib` from trusted backup if J/H provenance audits are rerun. **Never fetch untrusted joblib**: pickle payloads may execute code.
4. The lightweight `models/v3_2/model_metadata.json` and `reports/ml_v3_2/artifact_reproducibility_J.json` must already be in the branch. Model loader verifies both hash pins, schema, threshold, feature order and sklearn version.
5. Run `python -B -m unittest scripts.test_phase2_K -v`, `python -B scripts/check_phase2_K.py --preflight` (read-only), then safe inference with `src.inference_v32.predict` or `scripts/check_phase2_K.py --run --confirm-api` **only if its report is not already present**, because it is create-once.
6. Never turn on `src/train.py` (legacy V2.2) or silently train a replacement model on application startup. If the trusted artifact is missing, the application should report an explicit configuration error.

## 6. Sensitive artifacts, Git controls and Phase 3 entry

- `models/v3_2/*.joblib`: **Git-ignored**. Not in any code commit.
- `data/external/`, `data/processed/`: **Git-ignored**, retained in private controlled local/backup storage.
- `reports/ml_v3_2/test_predictions_frozen.csv.gz`: contains **per-snapshot true Test labels**. Must be Git-ignored and **never surfaced by app/API**.
- `reports/ml_v3_2/*.json`, `*.md`, and selected **aggregate-only** CSVs may be committed after allowlist review. No raw labels/personal data/keys/`.env`.
- Keep V2.2 model/artifacts and `main` untouched; continue on `Dung` until explicitly reviewing a separate merge.

**Phase 3 integration contract:** expose the existing `predict(uid, term, asof)` API safely in Streamlit; display calibrated risk with data timestamp, synthetic disclaimer, error/unavailable states and optional batch TopK policy; do not show post-outcome labels, full-simulation Test data, or claim an academic sanction. Chatbot/RAG integration is outside Phase 2.

## 7. Evidence inventory

- `reports/ml_v3_2/preflight.md`, `data_contract_verification.json`, `feature_contract_verification.json`.
- `training_smoke_CD.json`, `b5_training_provisional.json`, `baseline_reproduction.json`, `baseline_comparison.csv`.
- `model_comparison.json`, `model_comparison.csv`, `model_selection_decision.json/md`.
- `frozen_test_metrics.json/md`, `per_term_metrics.csv`, `test_bootstrap_ci.json`, `reproducibility.json`.
- `subgroup_audit.json/md`, `hk1_risk_audit.json`, `cohort_shift_audit.csv`.
- `artifact_reproducibility_J.json/md`, `inference_contract_K.json`, `phase2_L_verification.json`.
- `docs/MODEL_CARD_V3_2.md` and `docs/PHASE_2_L_RUNBOOK.md`.

**Acceptance rule:** A–K have been verified through supplied logs; L1–L5 only become PASS after local suite, Git stage review, actual commits and push, ending in `PHASE_2_L_FINAL_PASS`.
