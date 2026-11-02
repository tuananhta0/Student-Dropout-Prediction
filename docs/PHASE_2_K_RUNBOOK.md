# Phase 2-K: Inference API contract (K1-K6)

**Source scenario:** `V3_2_MAIN` synthetic; **model:** frozen RF_B + Platt `v3.2-rf-b-platt-g1`; **demo ceiling:** `2026-10-01`.

## Files

- `src/inference_v32.py`: API `predict(record_uid, term, asof=None)` and injectable `V32InferenceService`.
- `scripts/test_phase2_K.py`: standalone mutation unit tests (no dataset/model required).
- `scripts/check_phase2_K.py`: frozen hash preflight and real as-of end-to-end verification; only uses **cohort 2022** demo snapshots. Never reads Test split or labels.

## Commands (PowerShell, repo root)

```powershell
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_K -v
.\.venv\Scripts\python.exe -B scripts/check_phase2_K.py --preflight
.\.venv\Scripts\python.exe -B scripts/check_phase2_K.py --run --confirm-api
Get-Content reports/ml_v3_2/inference_contract_K.json
```

Phase 1 E1–E5 regression cases were previously accepted. To repeat their **data access** checks separately (they are not part of model inference):

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_v32_edge_cases_e1_e5
```

Check end-to-end probability against the directly loaded verified pipeline on the **same** 58-feature snapshot with numeric max delta `<=1e-10`. No new training occurs. The check refuses overwriting the evidence report.

## Serving API (Phase 3 integration)

```python
from src.inference_v32 import predict
result = predict("SIMV31_000001", 1, "2026-10-01")
# result['status'] == 'ok' or 'not_available'
# if 'ok': result['probability'] and result['policy']
```

`uid` is authoritative `record_uid`, not public `student_id`. `term` is integer 1–3, not a display label. `asof` is an ISO calendar date on/before `2026-10-01`; `None` resolves to demo ceiling. A requested snapshot unavailable at the requested cutoff returns `not_available` **without a guessed probability**. Invalid dates, bad schema, altered artifact, model not found or future timestamps cause explicit errors.

Only the *ordered 58 B5* features are passed to the estimator. Existing null numeric/categorical entries stay missing for the frozen Train-fitted imputer. Numeric Python values convert to float64; two source boolean flags (`warning_current`, `academic_recovery_flag`) convert to 0.0/1.0 as declared by their frozen numeric feature roles. Strings do not silently convert to numbers. Categorical values stay strings; previously unseen categories use the frozen OneHotEncoder's `handle_unknown='ignore'`.

Policy fields include frozen probability threshold **0.27** and independent budgets **5%, 10%**. A TopK alert cannot be computed from one UID: the caller needs a term-specific, eligible population ranking; Phase 3 may implement batch ranking without altering the frozen model. A single-record response reports only the threshold alert flag.

This module does **not** import Train scripts, legacy models, Test/evaluation artifacts or labels; `V32DataAccess` is the sole data lookup. The loader `src/model_bundle_v32.py` verifies SHA before deserializing trusted joblib. It reads the lightweight Phase J hash report to check metadata integrity, not Test predictions. Model missing/untrusted fails closed; no automatic retraining or V2.2 fallback.

**Limits:** Source snapshots are synthetic. Risk alerts are suggestions for outreach prioritization, not disciplinary or real CMC student decisions. Keep raw labels/Phase H predictions out of UI. No UI refactor required in Phase 2.
