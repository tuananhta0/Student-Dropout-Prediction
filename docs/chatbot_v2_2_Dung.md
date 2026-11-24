# Chatbot V2.2: focused local queries

Apply after the uncommitted V2.1 candidate on 284dac26cb11797cce14c634100d93efe6eece1e, not to main.
The V2.1 binary tracked diff SHA256 was 59cfa522edad09777e7eaee716d2e2ffe11fc9965c3109ecb170bf509fc15939.

Scope: narrow local policy calculations and lookups, personal metric expansion,
current dropout risk and its limits, and selected scoped aggregate queries.
Model, corpus, dataset, credentials, and production rate limits unchanged.

- Risk counts include high and very high by the deployed threshold; only
  enrolled students with valid predictions and valid probabilities enter the
  denominator. Coverage is reported separately.
- State counts are at the demo cutoff, among students with academic records;
  not incidence rates, full intake population rates, or causal estimates.
- Advisor data must pass scope checks, including assigned class restrictions.
  A dean label in the question bank does not grant new privileges.
- Historical semester averages use the selected semester, not future grades.
- Unsupported feedback collection and course-level prediction are not fabricated.
- New policy answer patterns are based on the reviewed corpus; PDF remains authoritative.
- Not every query is supported. In particular, comparisons, causal analysis,
  course statistics, complex filters, multi-intent and multi-turn need further work.

Observed V2.1 baseline/candidate comparison: 1800 cases each, offline;
263 shorter, 1537 unchanged, 0 longer; 402/832 labeled questions have an article
intersection. This is not semantic accuracy. Candidate average 99.33 words
(using the original tool's definition), 269 answers over 200 words.

Run existing `audit_chatbot_1800.py` with the same CSV and runtime, then:

    python -m scripts.compare_chat_benchmarks data/kiem_tra_rag/v2_candidate.csv data/kiem_tra_rag/v2_2_candidate.csv --output data/kiem_tra_rag/v2_2_comparison.json

Check changed HTTP statuses, lost article overlaps and actual answers manually.
Existing tests and new fixture tests do not substitute for the real runtime
1800-case audit, a paced cloud benchmark, or independent semantic review.
