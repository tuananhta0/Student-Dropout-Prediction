# Dung feature freeze and final acceptance

Historical verified baseline: 9f7e1653325fdf4c13e6c13ef3e3c45c66c796f9.
Results below are historical evidence, not results of this new commit.

## Verified baseline

- Python: 156 tests passed; JavaScript: 7 tests passed.
- RAG offline: 77 regression cases passed the quality gate.
- Paced cloud benchmark: 24/24 known-answer cases generated successfully;
  six outside-scope questions were refused; 22 cases used embeddings.
- HTTP integration: 17/17; browser integration: 50/50.
- MAIN/STRESS audits passed all blocking checks.
- Runtime archive SHA256 and manifest verified.
- Model: cmc-research-v1, AP 0.62910, ROC-AUC 0.82926,
  recall 0.57742, precision 0.50768, ECE 0.03894.
- HK5/HK6 have only positive observed labels: do not report separate AUC/AP.
- Model and data are synthetic research artifacts, not validated on real students.

## Reproduction

Run the standard cloud benchmark with --cloud and --cases.
Pacing is automatic and does not relax the runtime rate limiter.
Save every report with its actual commit, including failed reports.
Do not manually replace provenance or reuse old results as new evidence.

## Data exceptions

MAIN: 4331 registration exceptions; 273 missing non-direct admission scores.
STRESS: 1045 registration exceptions; 77 missing non-direct admission scores.
MAIN registration exceptions were classified as
FINAL_REMAINDER_SIMULATION_EXCEPTION in the baseline policy review.

Run review_policy separately for STRESS to verify its classification.
These exceptions are simulation assumptions, not university approvals.
Preserve missing admission scores and distinguish source missing values
from scores excluded after comparison with reviewed admission floors.

## Manual acceptance record

Complete independently; do not pre-fill PASS.

- Reviewer/member code:
- Date and time:
- Git commit and clean-tree evidence:
- Machine, OS, Python version:
- Fresh clone and runtime SHA256:
- Student desktop/mobile visual review:
- Advisor desktop/mobile visual review:
- Admin desktop/mobile visual review:
- Independent hidden question set and its SHA256:
- Semantic review CSV and JSON results:
- Credit/admission exception review:
- Remaining failures and limitations:
- Reviewer confirmation:

A developer regression set is not a blind evaluation set.
Do not publish API keys, session tokens or private credentials.
Never push or merge main as part of Dung acceptance.
