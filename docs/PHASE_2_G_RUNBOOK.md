# Phase 2-G: Selection, calibration & alert policy

**Version:** G 1.0. **Scope:** G1–G5 mandatory. **Does not read Test 2025.** This package was developed using synthetic `V3_2_MAIN` files.

## Inputs & frozen decisions

- Prerequisites: A–F PASS, including full `reports/ml_v3_2/model_comparison.json`, original Train/Validation splits, and `src/ml_v32.py` pandas 3.x hotfix.
- **G1:** select `RF_B` from the *already completed* full F report, because it improved HK1 PR-AUC by >=0.010 absolute over `LR_B5`. `XGB_B` has a slightly higher pooled PR-AUC but the primary decision criterion is HK1. Record Brier, ECE and inference costs. This does not claim RF_B is best on Test.
- **G2:** fit RF_B on full Train 2022–23, 58 B5 features. Split 2024 Validation by `record_uid` with fixed seed 221026: 30% UIDs exclusively for sigmoid/Platt calibration, remaining 70% UIDs for threshold/policy. All 1/2/3 snapshot rows of a UID stay together; no test labels.
- **G3:** maximize F1 using predefined grid 0.05..0.80 step 0.01, on *policy-only* Validation UID subset. Tie-break: higher threshold. Do NOT default to reference threshold 0.21. No change based on Test.
- **G4:** Top5%/Top10% budgets are score-based ranking, `ceil(group_size * fraction)`, sorted descending calibrated score, then `record_uid` and term ascending. Separate TopK from absolute probability threshold.
- **G5:** freeze model, calibration and threshold in `models/v3_2/frozen_rf_b_pipeline.joblib`. Report immutable evidence in `reports/ml_v3_2/model_selection_decision.md` and `.json`, with file hashes and model version. Further J tasks will expand bundle metadata, without retraining based on Test.

**Caveat:** Model selection in F used the entire Validation. Thus the `Validation POLICY` subset still reflects earlier selection, and its metrics are development metrics, **not independent generalization estimates**. Test is opened only after G is frozen. No claim on real CMC students.

## Windows PowerShell (run during your scheduled work hours)

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_G -v
.\.venv\Scripts\python.exe -B scripts/freeze_phase2_G.py --mode smoke
.\.venv\Scripts\python.exe -B scripts/freeze_phase2_G.py --mode full --confirm-freeze
Get-Content reports/ml_v3_2/model_selection_decision.md
Get-Content reports/ml_v3_2/model_selection_decision.json | Select-Object -Last 40
Get-Item models/v3_2/frozen_rf_b_pipeline.joblib | Select-Object Name,Length
git status -sb
```

Expected: `Ran 12 tests / OK`, `PHASE_2_G_SMOKE_PASS`, `PHASE_2_G_FULL_PASS; frozen=RF_B; ... Test accessed=NO`.

## Safety/recovery

- Frozen artifact is Git-ignored; new lightweight code and reports are untracked until committed intentionally.
- Script will **refuse overwriting** any existing model/decision report. Do not delete a completed freeze to retune after seeing Test. If a run fails before freezing, inspect the error before retrying.
- Do not call legacy `src/train.py`. Do not move forward to H until the generated `model_selection_decision.md` is inspected and G1–G5 are PASS locally.
- For future inference, load `bundle['pipeline']`, get `predict_proba(X)[:,1]`, then call `apply_platt(bundle['calibrator'], raw_p)` from `src/model_policy_v32.py`, and apply `bundle['threshold']` or TopK policy. Load only trusted joblib artifacts; never use Test labels or raw generator ledgers in inference.
- If sigmoid worsens validation Brier/ECE, report it without changing the algorithm after viewing the Test. Any follow-up experiment must be a separately identified research run, not overwrite the locked model.
