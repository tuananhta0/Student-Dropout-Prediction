# PHASE 2-J — Immutable artifact & reproducibility

## Contract

- **Never re-fit, re-score Test, re-calibrate, change threshold or modify Phase G/H/I files.**
- Frozen model: `models/v3_2/frozen_rf_b_pipeline.joblib`, verified against the SHA-256 in `reports/ml_v3_2/model_selection_decision.json`.
- Generates `models/v3_2/model_metadata.json`, `reports/ml_v3_2/artifact_reproducibility_J.json`, `reports/ml_v3_2/artifact_reproducibility_J.md`. Create-once; no overwrite.
- Adds a **trusted-only** verification loader: `src/model_bundle_v32.py`. The loader refuses missing, modified, or version-incompatible models; it will **not** train or use V2.2 fallback.
- It records the original V3.2 ZIP SHA reported in Phase 1 but does **not** re-hash the original ZIP in J; original ZIP can be large or unavailable on the runtime host.

## Run after A–I passed, from root of Windows repo

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction

# Extract the ZIP, after verifying no target exists.
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_J -v
.\.venv\Scripts\python.exe -B scripts/package_phase2_J.py --preflight
.\.venv\Scripts\python.exe -B scripts/package_phase2_J.py --build --confirm-bundle

Get-Content -Encoding UTF8 reports/ml_v3_2/artifact_reproducibility_J.md
Get-Content reports/ml_v3_2/artifact_reproducibility_J.json
git status -sb
```

Expected messages:

- `Ran 12 tests` / `OK`.
- `PHASE_2_J_PREFLIGHT_PASS; ... Test NOT opened`.
- `PHASE_2_J_PASS; ... reload_delta=...; Test rescored=NO; artifact unchanged=YES`.

## Integrity and portability

1. `joblib` is **pickle-based**. Load only locally produced, trusted model files. SHA pinning is an accidental-drift check, not protection from an attacker able to replace model and metadata together.
2. Code and lightweight JSON/Markdown should be committed on `Dung` only after L checks. Binary `*.joblib`, large datasets, raw labels, and Test prediction row files must **not** be committed. Check with `git ls-files`, `git check-ignore` and `git status`.
3. This model was trained with the exact Python/sklearn/numpy/pandas versions in `model_selection_decision.json`. The verified loader checks the exact scikit-learn version. Copy a trusted frozen model from local backup together with the recorded metadata and J report; verify SHA before loading.
4. For a fresh machine, restore the exact synthetic V3.2 dataset from the Phase 1-approved archive (SHA from `preflight.md`) and copy the trusted model artifact, then install the recorded environment. If original frozen model is unavailable, recreate it by repeating the documented **Phase A–G training process in a new controlled experiment** and verify Validation metrics, hashes, and decisions; do not claim it is necessarily bitwise identical across library/platform versions. Do not re-run H Test selection.
5. If the bundle is missing, explicitly return `not_available` / error. Do not automatically train on launch and never silently use `models/dropout_model.joblib` V2.2.
6. Do not serve `reports/ml_v3_2/test_predictions_frozen.csv.gz` or any label-bearing research file in UI/inference.

## Scope boundary

Phase J provides metadata and trusted loading; actual `predict(uid, term, asof)` using Phase 1 `V32DataAccess` is **Phase K**, not declared PASS here. The fixed demo cutoff is `2026-10-01`.
