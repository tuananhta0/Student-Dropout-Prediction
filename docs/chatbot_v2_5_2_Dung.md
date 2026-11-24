# V2.5.2 acceptance and limits

Preserves the original 1800-question bank, V2.5 stratified 54-case sample,
model, dataset, PDF and cloud budget. Local development results are not
independent semantic scores. Precise public clause hints now precede broad
retrieval keywords; deterministic answers and authorization checks retain
priority. Gemini receives only fixed public aspects and reviewed evidence.

## Observed local results

- V2.5.1: 37/48 known-answer sample retrieval hits, 36/48 top1.
- V2.5.2: 47/48 hits and top1, six of six refusal controls.
- Original sample gate remains FAIL (requires every expected article displayed).
- SV-0510 (`lan 2 bi tru bao nhieu diem`) lacks the type of violation. Asking
  for clarification is appropriate; do not manufacture context or silently
  change its expected article. A human reviewer must confirm this exception.
- Direct retrieval on the same 781 labeled policy questions: 570 to 584
  overlaps, 534 to 550 top1, no lost overlaps. Not the full API benchmark.
- 198 Python and 7 JS tests; 77 regression cases passed locally.

## Remaining acceptance

Run the provided V2.5.2 script on Windows using the actual demo runtime and
private Gemini configuration. It preserves V2.5 and V2.5.1 reports, runs
full 1800-case API audit, original offline/cloud sample and live HTTP/browser.
The browser failure reason is not yet known. Port 8000 was previously tested
successfully; a separate local smoke browser may use --direct-local if a
proxy is suspected. This never changes system proxy or security policies.
The browser report prints the failure stage and detail. If the port is busy,
choose a free port or stop only the server you started yourself.

Independent reviewer: freeze at least 30 new questions before seeing answers,
run benchmark_rag with --cases, and review each response against the original
PDF. Mark the four semantic fields 0/1 and explain every failure. Record the
reviewer identity, independently check scoped data answers, permission changes,
visual behavior and the ambiguity exception. Do not prefill acceptance.

The closing script requires current candidate provenance, successful automated
checks and real reviewer confirmation. It may accept a separately documented
ambiguity exception for the original sample only; it never rewrites the raw
quality gate. Commit and push Dung only. Fresh-clone on another machine must
follow the exact new commit; do not claim that result before it is performed.
