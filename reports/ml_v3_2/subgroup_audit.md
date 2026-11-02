# Phase 2-I: synthetic subgroup and stability audit

**Status:** PASS (audit completeness only). **Test rescored:** NO. **Frozen model/threshold:** unchanged.
**Frozen model:** v3.2-rf-b-platt-g1. **Threshold:** 0.27.

## HK1 operational risk
| Policy | Alerts | Correctly flagged positives | Missed positives | Recall |
|---|---:|---:|---:|---:|
| threshold | 135 | 52 | 196 | 0.2097 |
| top5 | 162 | 60 | 188 | 0.2419 |
| top10 | 324 | 105 | 143 | 0.4234 |

HK1 calibration: Brier 0.063078, ECE10 0.008335, mean predicted probability minus observed prevalence -0.007377.

## Subgroups

Sufficient groups: 34. Insufficient support: 18.

| Group | Value | N | Positive | Prevalence | Support | PR-AUC | FNR@0.27 |
|---|---|---:|---:|---:|---|---:|---:|
| cohort_year | 2025 | 9284 | 911 | 0.0981 | SUFFICIENT | 0.5439 | 0.5554 |
| snapshot_term_index | 1 | 3234 | 248 | 0.0767 | SUFFICIENT | 0.2754 | 0.7903 |
| snapshot_term_index | 2 | 3090 | 327 | 0.1058 | SUFFICIENT | 0.5733 | 0.4740 |
| snapshot_term_index | 3 | 2960 | 336 | 0.1135 | SUFFICIENT | 0.6557 | 0.4613 |
| major_code | AI | 728 | 71 | 0.0975 | SUFFICIENT | 0.4885 | 0.6479 |
| major_code | BA | 902 | 80 | 0.0887 | SUFFICIENT | 0.4913 | 0.5750 |
| major_code | CB | 297 | 25 | 0.0842 | SUFFICIENT | 0.4485 | 0.5600 |
| major_code | CL | 412 | 48 | 0.1165 | SUFFICIENT | 0.4999 | 0.5208 |
| major_code | CS | 821 | 83 | 0.1011 | SUFFICIENT | 0.5848 | 0.4819 |
| major_code | DA | 281 | 23 | 0.0819 | SUFFICIENT | 0.6710 | 0.4348 |
| major_code | EC | 192 | 21 | 0.1094 | SUFFICIENT | 0.4257 | 0.8095 |
| major_code | EM | 529 | 45 | 0.0851 | SUFFICIENT | 0.5612 | 0.5333 |
| major_code | GA | 162 | 16 | 0.0988 | INSUFFICIENT_SUPPORT | N/A | N/A |
| major_code | GD | 361 | 25 | 0.0693 | SUFFICIENT | 0.6701 | 0.4400 |
| major_code | IB | 296 | 34 | 0.1149 | SUFFICIENT | 0.4943 | 0.7059 |
| major_code | IT | 1616 | 147 | 0.0910 | SUFFICIENT | 0.5303 | 0.5510 |
| major_code | KL | 391 | 43 | 0.1100 | SUFFICIENT | 0.7533 | 0.3953 |
| major_code | LS | 338 | 30 | 0.0888 | SUFFICIENT | 0.5782 | 0.5667 |
| major_code | MC | 215 | 30 | 0.1395 | SUFFICIENT | 0.5515 | 0.6333 |
| major_code | MK | 318 | 43 | 0.1352 | SUFFICIENT | 0.5919 | 0.6279 |
| major_code | NS | 388 | 40 | 0.1031 | SUFFICIENT | 0.6292 | 0.4750 |
| major_code | PR | 351 | 38 | 0.1083 | SUFFICIENT | 0.5603 | 0.4737 |
| major_code | SE | 686 | 69 | 0.1006 | SUFFICIENT | 0.4558 | 0.6667 |
| education_mode | GM | 3212 | 283 | 0.0881 | SUFFICIENT | 0.5167 | 0.5901 |
| education_mode | SM | 6072 | 628 | 0.1034 | SUFFICIENT | 0.5572 | 0.5398 |
| gender | Khác | 60 | 7 | 0.1167 | INSUFFICIENT_SUPPORT | N/A | N/A |
| gender | Nam | 5493 | 507 | 0.0923 | SUFFICIENT | 0.5450 | 0.5483 |
| gender | Nữ | 3731 | 397 | 0.1064 | SUFFICIENT | 0.5422 | 0.5668 |
| HK1/major_code | AI | 256 | 22 | 0.0859 | SUFFICIENT | 0.1951 | 0.9091 |
| HK1/major_code | BA | 315 | 25 | 0.0794 | SUFFICIENT | 0.3559 | 0.7600 |
| HK1/major_code | CB | 103 | 7 | 0.0680 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | CL | 144 | 14 | 0.0972 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | CS | 286 | 23 | 0.0804 | SUFFICIENT | 0.3457 | 0.6522 |
| HK1/major_code | DA | 96 | 4 | 0.0417 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | EC | 68 | 8 | 0.1176 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | EM | 185 | 14 | 0.0757 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | GA | 55 | 1 | 0.0182 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | GD | 126 | 9 | 0.0714 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | IB | 103 | 8 | 0.0777 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | IT | 560 | 38 | 0.0679 | SUFFICIENT | 0.2670 | 0.8421 |
| HK1/major_code | KL | 136 | 9 | 0.0662 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | LS | 118 | 10 | 0.0847 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | MC | 75 | 7 | 0.0933 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | MK | 111 | 9 | 0.0811 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | NS | 136 | 11 | 0.0809 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | PR | 121 | 10 | 0.0826 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/major_code | SE | 240 | 19 | 0.0792 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/education_mode | GM | 1112 | 72 | 0.0647 | SUFFICIENT | 0.2084 | 0.9028 |
| HK1/education_mode | SM | 2122 | 176 | 0.0829 | SUFFICIENT | 0.3039 | 0.7443 |
| HK1/gender | Khác | 22 | 1 | 0.0455 | INSUFFICIENT_SUPPORT | N/A | N/A |
| HK1/gender | Nam | 1913 | 141 | 0.0737 | SUFFICIENT | 0.2800 | 0.8085 |
| HK1/gender | Nữ | 1299 | 106 | 0.0816 | SUFFICIENT | 0.2726 | 0.7736 |

## Shift and limitations

Covariate-shift descriptors by train/validation/test cohort are in `cohort_shift_audit.csv`.
Standardized mean differences are descriptive, not proof of deployment transferability.
Test 2025 contains a single cohort, so cohort-comparison Test metrics cannot be inferred.
Subgroup Wilson intervals treat snapshots as independent; one UID may contribute multiple snapshots.
For overall UID-cluster intervals see `test_bootstrap_ci.json` (Phase H).
Insufficient-support group metrics are intentionally suppressed.
No claim of fairness or actual CMC performance; no threshold/model retuning.
Never serve labels or historical Test predictions in UI.
