# PHASE 2-H: FROZEN TEST 2025 (one-shot)

## Scope / acceptance

- H1–H6 mandatory; H7 optional where administrative-action metadata exists.
- Uses `V3_2_MAIN/splits/full_simulation/TEST_2025` only, not ASOF demo or stress.
- Test is opened/scored only after Phase G freeze checksum, source provenance, selection source and B5 artifact SHA are verified.
- **Never** fit, recalibrate, select a threshold or choose between model candidates based on Test results.
- The chosen RF_B + Platt model, 58 ordered features, version `v3.2-rf-b-platt-g1`, probability threshold `0.27` and per-HK Top5%/Top10% are fixed by Phase G.

## Installation

Run from `C:\Code\AI\CMC_Dropout_Prediction` (on branch `Dung`).

```powershell
$targets = @("scripts/evaluate_phase2_H.py", "scripts/test_phase2_H.py", "docs/PHASE_2_H_RUNBOOK.md")
$existing = @($targets | Where-Object { Test-Path $_ })
if ($existing.Count -gt 0) { throw "Already exists: $($existing -join ', ')" }
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\PHASE_2_H_Frozen_Test_Package.zip" -DestinationPath .
```

## Unit tests (never load Test)

```powershell
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_H -v
```

Expected: `Ran 12 tests`, `OK`.

## Preflight (does NOT open Test X/y)

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_phase2_H.py --preflight
```

Expected: `PHASE_2_H_PREFLIGHT_PASS; freeze SHA verified; threshold=0.27; Test data NOT opened`.

The check requires: Phase G freeze PASS; frozen model SHA, B5 artifact SHA, Phase F report SHA, Phase G script hashes, original Train/Validation split hashes and synthetic manifest; Test files exist. It rejects previously written Phase H report files.

## One-shot frozen Test evaluation

**Only after unit tests and preflight PASS, and only once:**

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_phase2_H.py --run --confirm-frozen-test
```

Expected final line starts with `PHASE_2_H_PASS` and contains `B5 reference=MATCH; model=FROZEN; Test retune=NO`.

The script fails without overwriting output when any Phase H report already exists. Do not delete the reports to re-run and cherry-pick better scores. If a runtime or file-write failure occurs, document the technical fault and the re-execution explicitly, never change trained model, threshold or calibration.

## Reports written (all under `reports/ml_v3_2/`)

- `frozen_test_metrics.json`: pooled and HK1/HK2/HK3 AP (PR-AUC), ROC-AUC, Brier, ECE, prevalence, Precision/Recall/F1, confusion matrix, Top5/Top10, B5 generator cross-check, CI.
- `per_term_metrics.csv`: compact pooled/HK1–HK3 comparison.
- `test_bootstrap_ci.json`: 95% percentile intervals, 400 **student-UID cluster** bootstrap resamples (all HK snapshots of each selected UID).
- `test_predictions_frozen.csv.gz`: keyed `(record_uid, snapshot_term_index)` and y_true/raw/calibrated probabilities and frozen model ID. **For internal research only. Never serve this file to the UI or inference API.**
- `reproducibility.json`: G decision source/hash, Test X/y hash, frozen artifact hash, runtime versions.
- `frozen_test_metrics.md`: presentation summary.

## Interpretation warnings

- Top10% on HK1 selects `ceil(3234*0.1)=324` snapshot warnings, sorted by calibrated risk descending, UID/term ascending as tie-break. It is **not** probability threshold `0.27`.
- H5 compares **B5 LR reference** Test results to original generator numbers. It does not compare RF_B to the generator's LR reference as though they are the same model.
- B5 Test source references: pooled AP `0.5404592363`, HK1 AP `0.2824202049`, HK1 Recall@Top10 `0.3790322581`.
- Full simulation is deliberately synthetic and contains future labels by construction. It is **not** the as-of 2026 demo serving view, which has right-censored targets.
- No Test result is admissible for new model selection or hyperparameter changes within this experiment. Fairness and robustness analysis comes in Phase I, without changing G.

## Readout

```powershell
Get-Content -Encoding UTF8 reports/ml_v3_2/frozen_test_metrics.md
Import-Csv reports/ml_v3_2/per_term_metrics.csv | Format-Table group,n,positives,pr_auc,brier,recall_at_threshold,top10_recall
Get-Content reports/ml_v3_2/test_bootstrap_ci.json
Get-Content reports/ml_v3_2/reproducibility.json | Select-Object -Last 25
git status -sb
```

Do not commit or push trained `.joblib` binary artifacts; these remain local, Git-ignored. Phase J will add artifact restore and metadata checks. Leave Phase 3 UI unchanged.
