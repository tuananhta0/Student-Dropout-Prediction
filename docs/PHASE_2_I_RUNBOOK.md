# Phase 2-I: Synthetic subgroup / risk audit

## Scope

- **I1–I4 mandatory:** cohort, term, major, education mode, gender subgroup support; explicit `INSUFFICIENT_SUPPORT`; HK1 false negatives/calibration/top5/top10 budgets; descriptive distribution shift and limitations.
- **I5 optional:** stress experiments are *not run* and `V3_2_STRESS` is never mixed into full simulation.
- Uses persisted `test_predictions_frozen.csv.gz` from H, frozen report, frozen G policy, and source metadata. **Does not load/fit models or re-score Test.** It does *read existing Test features for grouping*, not for model selection.
- `master/students.csv.gz` is read *only* for `gender` and UID; never used for model training, inference or UI. Do not put row-level labels/probabilities in the fairness reports.
- Group support: at least 100 snapshots, 20 positives, 20 negatives and 50 distinct UIDs. Otherwise `INSUFFICIENT_SUPPORT` and subgroup performance numbers suppressed. This is a **reporting convention**, not proof of unbiased operation.
- Wilson subgroup proportion intervals treat snapshots as independent (a limitation). For primary confidence bounds, rely on H's 400 UID-cluster bootstrap replicates.
- Full-simulation Test cohort is **only 2025**, so subgroup Test metrics cannot meaningfully compare cohort 2022/2023/2024. The cohort shift CSV reports covariate means across Train, Validation and Test without using those data to retune the model.

## Install (branch `Dung`)

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction
$targets = @('scripts/audit_phase2_I.py','scripts/test_phase2_I.py','docs/PHASE_2_I_RUNBOOK.md')
$existing = @($targets | Where-Object { Test-Path $_ })
if ($existing.Count -gt 0) { throw "Already exists: $($existing -join ', ')" }
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\PHASE_2_I_Synthetic_Audit_Package.zip" -DestinationPath .
```

## Tests and audit

```powershell
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_I -v
.\.venv\Scripts\python.exe -B scripts/audit_phase2_I.py --preflight
.\.venv\Scripts\python.exe -B scripts/audit_phase2_I.py --run --confirm-audit
```

Expected `PHASE_2_I_PREFLIGHT_PASS` and `PHASE_2_I_PASS`, **never** `PHASE_2_H_PASS` again.

## Outputs

- `reports/ml_v3_2/subgroup_audit.json` aggregated details, source/model fingerprints, cautions;
- `reports/ml_v3_2/subgroup_audit.md` readable summary;
- `reports/ml_v3_2/subgroup_metrics.csv` aggregated groups (not per student);
- `reports/ml_v3_2/hk1_risk_audit.json` HK1 threshold and operational alert-budget misses;
- `reports/ml_v3_2/cohort_shift_audit.csv` descriptive mean drift by cohort and split.

```powershell
Get-Content -Encoding UTF8 reports/ml_v3_2/subgroup_audit.md
Import-Csv reports/ml_v3_2/subgroup_metrics.csv |
  Format-Table group_by,group_value,n,positives,status,pr_auc,false_negative_rate_threshold
Get-Content reports/ml_v3_2/hk1_risk_audit.json
git status -sb
```

Do not repeat H/Test scoring or change G threshold=0.27 based on any I result. Synthetic data cannot establish real-world CMC fairness.
