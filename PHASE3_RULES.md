# Phase 3: Academic rules and policy review

Baseline: `42decf3`, V3.2.1 MAIN, QĐ 22/2023/QĐ-ĐHCMC-HĐT.
This project uses synthetic data and is not the university's official academic record.
The attached regulation applies from academic year 2023–2024; applying it to all synthetic cohorts is a project assumption, not historical policy validation.

## Rules and calculations

- Article 12.2e (printed page 9): minimum ceil(2/3 × standard load), maximum floor(3/2 × standard load).
  Standard load is curriculum credits / designed semesters in this source, not a hardcoded 15.
  For a hypothetical standard of 15 the limits are 10–22. Source flags below these limits are reviews, not approved exceptions.
- Articles 13–15 (printed pages 11–13): final exam <=2 is F; round the weighted course score to one decimal before grading. P requires >=5; I/X/R/P are not GPA grades.
  For non-GPA score 4.0–4.9, the supplied text does not explicitly specify a grade; return unknown.
  A retest (thi lại) is capped at C; a new course attempt (học lại/cải thiện) retains the highest official grade. These are not the same operation.
- Articles 15 and 17 (pages 13–15): GPA uses best official letter grade per GPA-eligible course × credits, excluding certificate and non-GPA courses. F contributes zero grade points, but its credits are included. Repeat a course without duplicating its credits. GPA is rounded to two decimals.
  The exact source ledger uses recorded letter grades; it does not pretend to repair the source grade-rounding discrepancy.
- Article 18.1–2 (page 15), Article 3.5 (page 3): warning if GPA <0.8 in HK1, <1.0 thereafter, failed credits >50% registered, or outstanding credits >24 excluding certificate courses. Flag withdrawal review for streak >2 or total >3 or counted duration >18 (Bachelor)/22 (Engineer); never declare an official decision. Total >3 is a screening flag, and counted duration respects source's counted semesters instead of elapsed calendar time.
- Article 25.1a (page 21): unexcused hours >20% or total absence >30% of a course. The exact boundary is allowed by this reason alone. Recorded eligibility can depend on other reasons.
- Articles 19.6 and 20 (pages 16–18): required courses, credits, GPA >=2, English (SM B1/GM B2), PE and national defense, other program outcomes, financial and legal/disciplinary clearance. A verified synthetic English event is only a proxy for the standard; absent events and unrecorded clearance are unknown. Zero debt is not proof of all financial obligations.

The dashboard, advisor class view and chatbot share `academic_rules.projection`. The two values are deterministic *scenarios*, not a validated GPA forecast: next-term GPA equal to recorded cumulative GPA or 0.50 higher (capped at 4), at most 15 new GPA-bearing credits, limited by remaining graduation credits. Retakes/improvements require the actual course grade history; no duplicate credits. GPA scenario and dropout probability are explicitly separate. The previous speculative behavioral What-If answer is suspended until Phase 4 evaluation; the trained model and `predict_what_if` remain unchanged. No causal effect is claimed.

## Reproducible reviews

Run:
```
python -B -X utf8 audit_dataset.py
python -B -X utf8 review_policy.py
python -B -X utf8 -m unittest discover -s tests -v
```
Reports under ignored `data/phase3_review/`:

- `registration.csv`: 4,338 final remainder registrations (registered <= remaining before term, remaining < default minimum); 87 low loads have no adequate documented exception. All pending approval.
- `exam.csv`: 2 recorded ineligible attempts with exactly 9/45 unexcused hours and total <=30%; attendance alone cannot justify them. Do not override recorded status.
- `grade_rounding.csv`: 37,938 attempts whose grade letter differs after rounding half-up to one decimal as required by Article 13. Dataset maps the unrounded score to grade. Changing these requires regenerating GPA, credits, warnings, features, labels as appropriate and retraining model in Phase 4. Until then, recorded scores/grades are preserved, and the course view flags policy disagreement.
- `admission_missing.csv`: 269 unrecorded scores, all explicitly `SYNTHETIC_SOURCE_MISSING`. 756 direct admissions are not applicable, not imputed. Preserve NaN.
- `admission_method_summary.csv`, `admission_method_mismatch.csv`, `admission_floor_review.csv`: method compatibility by cohort (0 mismatches); 11,224 comparable observed scores with selected published floors, 4 below (2024 EC, THPT 30-scale). These are review flags, not evidence of actual individual admission. Scores, special admission evidence, subject combinations, preference and priority points are not sufficient to validate trúng tuyển individually. Normalized score is a synthetic ML feature, not the official cross-method conversion.
- `review.json`: aggregate counts; all 149,196 cumulative records reconcile to best-letter ledger (maximum error ~5e-9). Phase 2 audit still checks internal consistency, not full regulatory compliance.

## Admission policy evidence

Official university pages (consulted 2026-11-22):
- 2022: https://fad.cmc-u.edu.vn/phuong-thuc-tuyen-sinh-2022/ (THPT/HocBa/direct/HSA/VNUHCM/TSA groups; observed methods are a subset; HSA 80, VNUHCM 650, THPT 17 and HocBa 18 minimum published).
- 2023: https://cmcu.edu.vn/phuong-thuc-tuyen-sinh-2023/ (three methods, 30-scale; IT HocBa 20). https://cmcu.edu.vn/diem-san-va-du-bao-diem-chuan-xet-tuyen-truong-dai-hoc-cmc-nam-2023/ (THPT floor range 18–20).
- 2024: https://fict.cmc-u.edu.vn/cau-hoi-thuong-gap/ (CMC100, 200, 303, 410, 409; source uses only first three). https://cmcu.edu.vn/diem-san-nop-ho-so-xet-tuyen-truong-dai-hoc-cmc-theo-phuong-thuc-xet-diem-thi-tot-nghiep-thpt-nam-2024/ (18–20, IT/CS/EC floor 20).
- 2025: https://cmcu.edu.vn/truong-dai-hoc-cmc-cong-bo-4-phuong-thuc-xet-tuyen-nam-2025/ ; https://cmcu.edu.vn/thong-bao-quy-doi-diem-tuong-duong-giua-cac-phuong-thuc-xet-tuyen-truong-dai-hoc-cmc-nam-2025/ (THPT and HocBa 40-scale, CMC-TEST 80; IT floors 24/26/48 respectively).

## Prediction freshness

Runtime adds `prediction_valid_until` from the known two-term horizon (no outcome labels added) and `prediction_status` (STALE at or after horizon end). Expired forecasts remain historical probabilities; excluded from high-risk intervention counts. Demo: 8,857 expired active-student predictions, 6,939 within horizon, 453 within horizon and >=0.21. FULL simulation at 2028-12-31: all active forecasts expired. VALID means only within horizon, not guaranteed current. Current academic semester can be newer than ML snapshot; separating academic and ML views and constructing new features is still Phase 4. No ML artifact is modified. Keep localhost only; auth/RBAC remains Phase 6.

## Local acceptance

Run both modes with `phase2_smoke.py` on pinned environment, then `phase3_smoke.py`.
