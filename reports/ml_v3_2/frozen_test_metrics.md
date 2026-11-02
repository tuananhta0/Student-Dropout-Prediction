# Phase 2-H: Frozen Test 2025 (synthetic, one-shot)

**Status:** PASS. **Frozen before Test:** 2026-11-02T16:35:21.284430+00:00.
**Model:** v3.2-rf-b-platt-g1 / RF_B + Platt; **frozen threshold:** 0.27.
**Evaluated:** 2026-11-02T16:47:59.887161+00:00; **Test 2025:** 9,284 snapshots, 911 positives.

| Set | PR-AUC | ROC-AUC | Brier | F1@threshold | Recall@threshold | Recall@Top10% |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 0.543906 | 0.828460 | 0.062700 | 0.502481 | 0.444566 | 0.514819 |
| HK1 | 0.275367 | 0.771847 | 0.063078 | 0.271540 | 0.209677 | 0.423387 |
| HK2 | 0.573288 | 0.845660 | 0.064130 | 0.559350 | 0.525994 | 0.541284 |
| HK3 | 0.655727 | 0.853432 | 0.060796 | 0.589577 | 0.538690 | 0.553571 |

HK1 Top10% budget: **324** / 3234 snapshots; recall threshold is a separate quantity.
B5 Test reference comparison: **MATCH**.
Bootstrap 95% cluster CI by `record_uid` in `test_bootstrap_ci.json`.
No refit, retuning, recalibration or Test-based model selection.
Synthetic only; no real CMC deployment claim.
