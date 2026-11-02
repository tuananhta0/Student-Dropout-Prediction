# PHASE 2 - PREFLIGHT REPORT

## Git
Branch: Dung
HEAD: 5c425dce71313e784a77e6b189636cd56f8d2ba6
Remote: origin/Dung
Working tree before report: CLEAN

## Environment
Python: Python 3.14.7
Pip check: No broken requirements found.

scikit-learn: 1.9.1
numpy: 2.5.4
pandas: 3.1.0
xgboost: 3.4.1
lightgbm: 4.7.0
joblib: 1.6.0
imbalanced-learn: 0.14.2

## Dataset
Version: V3.2
Scenario: V3_2_MAIN
Type: SYNTHETIC RESEARCH CANDIDATE
Target: dropout_next_2_terms
Features: B5_FULL_SAFE (58)
Source SHA256 (Phase 1 provenance):
5DF81F995347B167C4996F3A83C63C11637F78674834F76DD3D28A0548E6ECD6

## Input verification
Required files checked: 14
Missing files: 0
File presence: PASS
Content validation: PHASE 2 SECTION B

## Scope
- Train and evaluate V3.2 ML models.
- Use frozen 2022-2023 / 2024 / 2025 splits.
- Preserve V2.2 models and data.
- Do not run legacy src/train.py.
- Do not use Test for model selection.
- Do not modify source dataset.
- Streamlit refactoring belongs to Phase 3.

## Acceptance
A1: PASS
A2: PASS
A3: PASS (file presence, Phase 1 provenance)
A4: PASS (file presence, clean Git)
A5: PASS (preflight report generated)

Training started: NO
Phase 2 Gate A: PASS
