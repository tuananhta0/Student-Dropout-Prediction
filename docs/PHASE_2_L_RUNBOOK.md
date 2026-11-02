# Phase 2-L: Final Acceptance, Safe Git Commit & Handover

**Scope:** Only L1–L5 from `PHASE_2_ML_Training_Acceptance_Checklist.md`. No additional ML experiments, no Test scoring, no model edits. The existing one-shot Test is never re-run.

## L1–L3. Evidence and tests

In Windows PowerShell (already checked out on `Dung`):

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction

# 1. Verify evidence, model metadata SHA, and V2.2 baseline hashes.
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --preflight

# 2. Protect row-level Test predictions (only .gitignore is modified).
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --protect-predictions

git check-ignore -v --no-index reports/ml_v3_2/test_predictions_frozen.csv.gz

# 3. Additional L safety tests.
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_L -v

# 4. All previous C–K tests (93 tests), Phase 1 E1–E5,
#    and cross-check A–K reports/hashes. Creates one new L report, once.
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --verify

Get-Content reports/ml_v3_2/phase2_L_verification.json
```

Expected success markers:

```text
PHASE_2_L_PREFLIGHT_PASS
PHASE_2_L_PRIVACY_PASS
Ran 13 tests ... OK
PHASE_2_L_VERIFY_PASS; tests=93; Phase1 E1-E5=PASS; model=UNCHANGED; Test rescored=NO
```

L verifier writes only `reports/ml_v3_2/phase2_L_verification.json`, status `READY_FOR_COMMIT`, not `PHASE_2_COMPLETE`. If it fails, **do not stage/commit**. Resolve the specific error without touching model, Test or frozen Phase G–K reports.

## L4. Safe Git staging, first commit

The `--stage-safe` command uses an exact allowlist. It stages the 4 V3.2 `src` modules, Phase 2 scripts/tests, runbooks, Model Card, model metadata JSON, summary reports, `.gitignore` and L verification evidence. It **does not stage**:

- `models/v3_2/*.joblib` or legacy binary models.
- `data/external/`, `data/processed/` or synthetic ZIP archives.
- `reports/ml_v3_2/test_predictions_frozen.csv.gz` (per-record labels).
- `.env`, private keys or unapproved scripts.
- The handover until it has a real commit SHA.

```powershell
git status -sb
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --stage-safe
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --stage-check

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only

# Carefully inspect the staged paths before committing.
git commit -m "Phase 2 V3.2 ML training evaluation and inference"
```

**Do not use `git add .` or `git add -A` for this phase.** The stage checker refuses paths outside its allowlist. If some file is already staged and rejected, unstage only that path, inspect it and rerun `--stage-check`.

## L5. Stamp the implementation SHA and second commit

The handover references the **real first implementation commit**. A file cannot contain its own Git commit SHA without changing that SHA, so record the Phase 2 implementation commit in the handover and the final handover commit in Git/console.

```powershell
# Fills one placeholder in the handover using the just-created HEAD.
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --stamp-handover

Select-String -Path PHASE_2_ML_Training_Handover.md -Pattern "implementation commit SHA"

# Stage only allowlisted content, including the now-stamped handover.
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --stage-safe
.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --stage-check

git diff --cached --name-only
git commit -m "Phase 2 V3.2 handover and implementation SHA"

# Push ONLY feature branch; no merge to main.
git push origin Dung

.\.venv\Scripts\python.exe -B scripts/close_phase2_L.py --post-push-check

git log -3 --oneline
git status -sb
```

Expected end marker:

```text
PHASE_2_L_FINAL_PASS; branch=Dung; HEAD=<real SHA>; origin/Dung=MATCH; tree=CLEAN; Test labels=NOT TRACKED
PHASE_2_COMPLETE: A-L mandatory gates PASS, subject to synthetic-only limitations.
```

**Note:** The verifier's `READY_FOR_COMMIT` status is intentionally never retroactively rewritten in the first commit after the push; the second Git commit and final console PASS are the evidence of L4–L5. For other researchers, record both actual SHA values from `git log -3 --oneline` in the Phase 3 kickoff.

## What reviewers should check

- `docs/MODEL_CARD_V3_2.md`: intended use, all splits, frozen Test metrics, HK1 failure analysis, synthetic and fairness limitations.
- `PHASE_2_ML_Training_Handover.md`: actual implementation commit SHA, artifact restoration procedure, Phase 3 contract.
- `reports/ml_v3_2/`: A–K evidence, plus L test confirmation without row-level Test labels.
- `git ls-files`/`git diff --cached --name-only`: no surprise binary models, raw labels, future data, source ZIP or secrets.

### Optional checklist sections

C6, E6, F6, H7, I5 and any other `[+]` items are documented as optional; not completing all optional work does **not** block Phase 2. E6 (B3_V31) has already been reproduced; H7 was reported only where supporting metadata exists. Never claim an optional item PASS without evidence.
