# V3.2.1 MAIN dataset / Phase 2

Base: `9c1396e` (stabilization). This change keeps the existing 64-feature trained pipeline, its metadata and scored CSV immutable. The CSV is a research artifact; the application creates a separate in-memory point-in-time view. No training is required or performed by the installer, audit, smoke test or app.

## Setup (Python 3.11, existing requirements)

```powershell
python install_dataset.py "$HOME\Downloads\CMC_Synthetic_Dataset_V3_2_1_CMC_IDS_RESEARCH_CANDIDATE.zip"
python audit_dataset.py
python -B -X utf8 -m unittest discover -s tests -v
$env:CMC_DATA_MODE = 'ASOF_2026_DEMO'
python phase2_smoke.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Install extracts only the explicitly contracted files from MAIN into `dataset_full/artifacts/runs/V3_2_MAIN` (about 38 MiB compressed on disk), verifies their manifest SHA256, shape and schema, then renames the staged directory into place. It refuses to overwrite an existing destination. If `dataset_full` already exists, keep it and pass `--destination data/phase2-main` and set `$env:CMC_DATASET_ROOT = "$PWD\data\phase2-main"` in each server/audit shell. Both directories are gitignored. No BASE/STRESS/hidden latents are installed. The full original archive must be kept separately for later research. Do not commit source data, generated audit outputs or private datasets.

## View semantics

- `ASOF_2026_DEMO` (default): manifest's fixed date 2026-10-01, 54,609 historical prediction snapshots. 44,639 labels observed, 9,970 right-censored; the latter must not be treated as negatives. Label columns are never part of the API view.
- `FULL_SIMULATION`: manifest's fixed date 2028-12-31, 57,569 prediction snapshots and 57,569 observed labels in the separate research file. Start with `$env:CMC_DATA_MODE = 'FULL_SIMULATION'`; switch back by assigning `ASOF_2026_DEMO`. Both views are explicitly synthetic. The pre-trained model was fitted with FULL_SIMULATION labels; using the demo runtime view is **not** an independent historically deployable validation. A strict cutoff-trained model and rolling-origin evaluation belong to Phase 4.
- For each snapshot, display terms are joined one-to-one, with `available_at <= snapshot_cutoff`. `verified_certificates` is derived from latest verified event by requirement as of that snapshot. `current_state` is the status at the snapshot; `state_as_of` separately means status at the selected global view date (also supplied on history rows as view context). For source transitions, `effective_at` is the timestamp; audit verifies linked events were available by then. Class/faculty risk directories contain only `ENROLLED` students at the view date; historical student lookup remains available. GPA and finances in summaries are last scored snapshots, not a live ledger. `dropout_rate_pct` uses status at the view date, not future labels. High-risk percentages in the school summary use all students with snapshots as denominator and only enrolled students as numerator. Course summaries use published attempts visible through each student's latest scored cutoff. Historical term queries also apply that term's own cutoff.
- `term_academics` supplies work/study hours, attempted credits, retake counts, registration bounds/status and cumulative failed course count. They are display context, not added model features. Model feature columns are preserved; audit checks source alignment and the four course aggregates. GPA projection still uses old heuristic; changing its formula is Phase 3/4, not part of data audit.

## Audit and acceptance

`python audit_dataset.py` writes `data/phase2_audit/phase2_audit.json` and separate review CSVs. A nonzero exit or exception means do not commit; the exception categories below are intentionally review-only, not silently approved. The report captures source missing counts, source FK/schema checks, credit/GPA ledger, state timeline, published grade cutoff, admission scales, source/score alignment, both label horizons, observed/censored counts, cohort split membership and calendar ranges. The generator's rule is replicated for the current all-GPA-eligible synthetic course catalog with retakes replacing failed grades; it is not a general implementation for real academic records.

Measured on the supplied MAIN archive:

| Check | Result |
|---|---:|
| Students | 20,000 |
| Term academic rows | 149,196 |
| Course attempts (all source) | 729,416 |
| Demo snapshots / enrolled | 54,609 / 15,796 |
| Demo visible course attempts | 273,045 |
| Full snapshots / enrolled | 57,569 / 7,754 |
| Full visible course attempts | 287,845 |
| Registration exceptions (unapproved) | 4,425 |
| Exam eligibility discrepancies | 2 |
| Missing admission score, excluding direct admission | 269 |

Registration exceptions (4,338 final remainder, 87 low load) require institutional approval evidence. Admission methods and scales are synthetic source assumptions; official historical applicability is not certified. Exam discrepancy rows require a reason or correction, not automatically a different eligibility rule. `registration_exceptions.csv`, `exam_eligibility_discrepancies.csv` and `admission_missing_non_direct.csv` expose the exact rows for review. No source records have been edited. Cohort splits are student-disjoint but their calendar ranges touch/overlap; chronological label availability evaluation is outstanding. RAG and authentication are outside this patch; keep the app on localhost with synthetic data only.
