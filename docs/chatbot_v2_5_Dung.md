# Chatbot V2.3B–V2.5 integrated candidate

Base: real Git commit `284dac26cb11797cce14c634100d93efe6eece1e`, with the exact uncommitted V2.3A diff SHA256 `41ca08ab647c7b0263bacaa9a0f24224451a75ae092304c5590a90cbe798bf75`. Do not apply to main or apply earlier patches again.

## Scope

- Public clause facets and local character n-gram TF-IDF, alongside existing BM25 and optional Gemini embedding. The local index is lexical, not a trained Vietnamese semantic embedding. Facets come from reviewed evidence, not question IDs. Existing refusal and explicit Điều/Khoản paths remain.
- Short answer with visible references and expandable evidence; preserve multi-topic evidence. Numerical rules for GPA classification, score conversion, failed-credit thresholds. Score conversion from 10-point grades needs both Điều 13 and 17; the two old regression expectations were corrected accordingly.
- Role-scoped additional aggregates: history by student's semester index, intake cohort, major/class, mode/degree; course-score distributions, failure counts/rates and ranked groups. Course stats count published attempts, not unique students, and historical views enforce the snapshot cutoff. Only supported query plans run, with explicit missing-data and scope boundaries. Not arbitrary natural-language SQL; unsupported dimensions, causal estimates, event histories and comparisons remain limited.
- Vietnamese abbreviations, bounded public follow-ups and student metric follow-ups; independent public questions can be answered separately offline. No raw question or private record is sent to Gemini. Follow-up state is session-local and only contains known topic or metric identifiers. Existing auth checks stay active.
- Actual model, dataset, thresholds, PDF corpus, runtime limits and main unchanged.

## Evaluation

The original 1800-question bank and all original outputs stay unchanged. `prepare_v25_review` creates a separate review CSV with original and corrected source fields. Four corrections were verified against the original PDF images (page numbers are PDF page numbers, not printed page numbers):

| ID | Correction | PDF page |
|---|---|---|
| SV-0034 | Điều 9.1b: retains student ID after authorized leave | 11 |
| SV-0092 | Điều 26.2d: passing thesis retained 20 years | 27 |
| SV-0093 | Điều 26.2d: non-passing thesis retained 5 years | 27 |
| SV-0504 | Điều 13.2: online defense conditions | 15 |

Other labels are NOT_REVIEWED. A matching article is not evidence of semantic accuracy or complete coverage. Do not add incorrect citations merely to match the bank. The seeded stratified cloud sample selects 48 single-article public policy questions across expected intent groups (regardless of result) and 6 fixed refusal controls. It is not independent, and the sample is direct RAG rather than authenticated API integration. Existing cloud generation gate stays unchanged; failure is reported, not suppressed. No unsupported query is treated as correct merely because method is python or HTTP 200.

Before commit: run the bundled PowerShell against the real demo release. Review paired losses and status changes, compare all changed high-risk calculations against original PDFs and source tables, and verify no cross-scope data. Do not commit a candidate with failed automation. Browser smoke is not human visual sign-off. Fresh clone remains a member task.

Independent review: another team member prepares a new hidden JSON list of public questions `[question, article_or_null]` before seeing answers, freezes its hash, runs the benchmark and grades the output CSV against original PDF (all four 0/1 columns). For private and analytics questions, independently compare each response to authorized source tables, including denominator, time and scope. Record reviewer/date/commit, misses and corrective actions. Do not send any private records to an external model. The original 1800 bank, regression and developer sample are not blind sets.

## Limitations

This is an integrated implementation, not a certificate of completion: actual 1800 API run, Gemini sample, human review, final push/CI and other-machine acceptance must be evidenced. The 50 reviewed corpus chunks cannot answer unavailable operational details; no predictions of intervention effects. Synthetic performance does not establish effectiveness on real students.
