# Phase 2-E: Reproduce B0-B5 on Validation (source-frozen)

**Scope:** E1-E5 mandatory; optional E6 V31 feature ablation. Reads Train and Validation only. Does **not** read or evaluate Test. Does not retrain/overwrite V2.2 model, does not modify B5 artifact, does not modify the synthetic source files.

## Pre-requisites
- Phase 2 C/D tests 13/13 PASS, B5 full training PASS.
- Existing `src/features.py`, patched `src/ml_v32.py`, `models/v3_2/b5_reference_pipeline.joblib`.
- Local source `data/external/v3_2_main` and original V3.2 generator ZIP.

## Command (Windows PowerShell)

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction
.\.venv\Scripts\python.exe -B scripts/reproduce_phase2_E.py `
    --generator-zip "C:\Code\AI\new\CMC_Synthetic_Dataset_V3_2_RESEARCH_CANDIDATE.zip" `
    --include-v31-ablation
```

If the generator ZIP is stored elsewhere, replace `--generator-zip` with its actual path. The script checks fixed SHA256 of source code and benchmark JSON inside the source ZIP; it rejects unknown versions to prevent silent benchmark drift.

## Expected artifacts
- `reports/ml_v3_2/baseline_reproduction.json`: provenance, reference deltas, Train/Validation results, threshold and per-model score details.
- `reports/ml_v3_2/baseline_comparison.csv`: B0-B5 (+ optional ablation), pooled Validation metrics.
- `reports/ml_v3_2/baseline_per_term_validation.csv`: HK1, HK2, HK3 breakdown, Recall@Top10%.
- `reports/ml_v3_2/baseline_reproduction.md`: compact report for review.

Successful run ends in `PHASE_2_E_PASS; ... Test accessed=NO`. If any source hash is unexpected, the script stops before training. If metrics differ by >= 0.0001 absolute from source, reports use `INVESTIGATE`, script exits with nonzero status. This tolerance is an engineering reproducibility check, **not** a performance threshold. Do not alter source dataset, Test, benchmark, labels, or thresholds to force a PASS.

Note: Validation F1-selected threshold and thresholded metrics are in-sample to threshold selection, so they should not be presented as unbiased future performance. Model-selection (F/G) comes later. All reported results are research-only synthetic.
