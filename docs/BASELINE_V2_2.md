# Student Dropout Prediction - V2.2 Baseline

## Source
- Branch at baseline: Dung
- Source commit: 9477d3430923e3d8bdfb546cae62e323a03f5e0b
- Dataset: Synthetic V2.2
- Model: Random Forest
- Artifact: models/dropout_model.joblib
- Python runtime: 3.14.7
- scikit-learn runtime: 1.9.1

## Evaluation
- Method: Load existing model and run inference
- No model retraining
- Decision threshold: 0.5

| Metric | Validation | Test |
|---|---:|---:|
| Samples | 2391 | 1604 |
| Positive labels | 62 | 49 |
| ROC-AUC | 0.676304 | 0.674283 |
| PR-AUC | 0.169902 | 0.142092 |
| Brier Score | 0.104896 | 0.096093 |
| Precision | 0.218391 | 0.189189 |
| Recall | 0.306452 | 0.285714 |
| F1 | 0.255034 | 0.227642 |

## Confusion Matrix

Rows: Actual [0, 1]
Columns: Predicted [0, 1]

Validation:
[[2261, 68], [43, 19]]

Test:
[[1495, 60], [35, 14]]

## Historical Validation
- reports/validation_results.json: 52/52 PASS
- Historical Python version: 3.13.5
- Historical scikit-learn version: 1.8.0

## Limitations
- Evaluation uses a synthetic dataset.
- Test recall at threshold 0.5 is approximately 28.57%.
- Baseline metrics are not directly comparable to V3.2 metrics.
- This report records existing model behavior before migration.
