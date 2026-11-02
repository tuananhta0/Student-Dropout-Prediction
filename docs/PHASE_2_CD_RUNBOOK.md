# Phase 2 C4–C5 & D1–D5 | Implementation runbook

Source: `Student-Dropout-Prediction-Dung` Phase 1 (`5c425dc`), contract `src/features.py` (58 B5).
Environment target: Windows PowerShell, Python 3.14.7, sklearn 1.9.1; tested on independent Python 3.13.5, sklearn 1.8.0.

## Files shipped

- `src/ml_v32.py`: Train/Val-only loader, keyed X/y alignment, hard schema guards, as-of timestamp verification from original `features/prediction_snapshots.csv.gz`, Train-only B5 LogisticRegression Pipeline.
- `scripts/train_v32.py`: independent trainer CLI; default **smoke** fit, not final benchmark; validates V3_2_MAIN manifest; hashes X/y and protects V2.2 artifact hashes.
- `scripts/test_phase2_CD.py`: 12 unit tests including forbidden column, changed order, duplicate composite keys, shuffled y, future timestamp, missing labels, numeric dtype drift, unseen categories, and train-only preprocessing.

## Run only these two commands now

```powershell
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_CD -v
.\.venv\Scripts\python.exe -B scripts/train_v32.py --mode smoke
```

Expected: `Ran 12 tests ... OK`, then `PHASE_2_CD_SMOKE_PASS; train_fitted=1,500; val_scored=100`. Report is written to `reports/ml_v3_2/training_smoke_CD.json`.

## Gate C4/C5 and D1–D5 evidence

| Check | Evidence |
|---|---|
| C4 | Unit test checks numeric imputer fitted solely from Train; Pipeline.fit is called only for Train; Val is predict_proba only. |
| C5 | 12 fail-fast/mutation tests; missing, out of order, forbidden field, key/timestamp errors fail closed. |
| D1 | Independent `scripts/train_v32.py`; does not import legacy `src/train.py` or NLP intent trainer. |
| D2 | `V32TrainingLoader` reads Train/Val full simulation only, uses key joins; declines `TEST_2025`; manifest must be `V3_2_MAIN`. |
| D3 | `Pipeline(ColumnTransformer(numeric imputer/scaler; categorical imputer/onehot), LR)` and unknown categories accepted. |
| D4 | JSON report stores feature order, seed, packages, input gzip hashes, training counts and type. |
| D5 | Optional full fit writes only into `models/v3_2`; legacy models protected by before/after SHA; local `.gitignore` includes `/models/**/*.joblib`. Confirm ignore with `git check-ignore -v --no-index models/v3_2/b5_reference_pipeline.joblib`. |

## Do not run full fit until Phase 2 E

A future explicit opt-in exists:

```powershell
.\.venv\Scripts\python.exe -B scripts/train_v32.py --mode full --confirm-full-fit
```

Full mode trains B5 on Train and calculates **Validation-only** reference metrics. It writes `models/v3_2/b5_reference_pipeline.joblib` (never overwritten) and `reports/ml_v3_2/b5_training_provisional.json`. That is still NOT Phase E complete: B0–B4, the baseline comparison table, and the reproduction report are separate tasks.

**No Test file is read by this package.** The baseline ZIP's generator published Test scores before this phase, but this implementation never imports or evaluates Test. Phase H is the first permitted Test execution.

## Known boundaries

- `C3` verifies source timestamps as stored, not every ledger edge. Independent ledger/temporal audits remain in Phase 1 evidence.
- CSV numeric/categorical dtype contracts follow this source dataset. Missing/unknown categories are managed by training-fitted preprocessing.
- Synthetic data is not a real CMC cohort, and scores must not be presented as actual school outcomes.
- The completed work is local to Phase 2; commit to `Dung` only after validating this package in your Windows environment.
