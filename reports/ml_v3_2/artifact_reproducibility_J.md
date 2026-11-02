# Phase 2-J: Frozen artifact & reproducibility

**Status:** PASS (only if the companion JSON and model SHA validate).

- Model: `v3.2-rf-b-platt-g1` / SHA256 `5c45deb380e011ff8ecfe94628d19e07db589ea75eed22c14bf08259887ed5c4`
- Metadata: `models/v3_2/model_metadata.json`
- Frozen threshold: `0.27`; budgets: `5%`, `10%`
- Train cutoff (source snapshots): `2025-02-03`; as-of demo ceiling: `2026-10-01`
- Save/load maximum abs delta (64 Validation feature rows): `2.220e-16` (limit `1e-10`)
- Model, calibration, feature list and threshold were **not** modified. Test was **not** rescored.
- Original V2.2 artifacts remain separate; no auto-train or silent fallback.
- Dataset archive hash is recorded from Phase 1 provenance, not re-hashed in J.
- WARNING: `joblib.load` is pickle-based. Never load untrusted model files.
- Synthetic only; outputs cannot be used as real CMC student decisions.
