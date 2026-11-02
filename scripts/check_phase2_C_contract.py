r"""Phase 2 C1-C3: read-only whitelist, metadata and temporal leakage check.

Run from repository root:
  .\.venv\Scripts\python.exe -B .\scripts\check_phase2_C_contract.py

C4 and C5 require the new trainer/loader in Phase 2-D; this script does not
pretend to certify them. It never trains a model or reads target labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import B5_FEATURES, B5_FEATURE_COUNT  # noqa: E402

SRC = ROOT / "data" / "external" / "v3_2_main"
OUT = ROOT / "reports" / "ml_v3_2" / "feature_contract_verification.json"
KEY = ["record_uid", "snapshot_term_index"]
SPLITS = ["TRAIN_2022_23", "VALIDATION_2024", "TEST_2025"]
TIMES = ["snapshot_cutoff", "feature_source_latest_at"]
FORBIDDEN = {
    "record_uid", "student_id", "cohort_year", "split", "label_observed",
    "dropout_next_2_terms", "censor_reason", "exit_reason",
    "administrative_action_eligible_at_cutoff", "weeks_observed",
    "snapshot_cutoff", "feature_source_latest_at", "event_id", "scenario_id",
}


def ensure(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def verify() -> dict:
    ensure(B5_FEATURE_COUNT == 58 and len(B5_FEATURES) == 58, "B5 contract count != 58")
    ensure(len(set(B5_FEATURES)) == 58, "B5 contract duplicated")
    ensure(not (set(B5_FEATURES) & FORBIDDEN), "Forbidden field in B5 whitelist")
    registry = pd.read_csv(SRC / "docs" / "feature_registry.csv")
    ensure(registry.feature_name.is_unique, "Duplicate registry names")
    reg = registry.set_index("feature_name")
    ensure(set(B5_FEATURES).issubset(set(reg.index)), "Feature not in registry")
    roles = reg.loc[B5_FEATURES, "role"]
    ensure(roles.eq("FEATURE_ALLOWED_ASOF").all(),
           f"Non-as-of feature: {roles[roles.ne('FEATURE_ALLOWED_ASOF')].to_dict()}")
    # is_allowed_at_t1=False means undefined at HK1 for lagged features; it is
    # NOT a blanket ban for HK2/HK3 and must be handled as missing at HK1.
    lagged_t1 = reg.loc[B5_FEATURES].index[~reg.loc[B5_FEATURES, "is_allowed_at_t1"].astype(bool)].tolist()

    snaps = pd.read_csv(SRC / "features" / "prediction_snapshots.csv.gz",
                        usecols=KEY + TIMES)
    ensure(len(snaps) == 57569, "Source snapshot count mismatch")
    ensure(not snaps.duplicated(KEY).any(), "Source duplicated composite key")
    ensure(not snaps[TIMES].isna().any().any(), "Missing source timestamps")
    source_cutoff = pd.to_datetime(snaps["snapshot_cutoff"], errors="raise", utc=True)
    source_latest = pd.to_datetime(snaps["feature_source_latest_at"], errors="raise", utc=True)
    ensure((source_latest <= source_cutoff).all(), "Source has FUTURE feature availability")

    checks = {}
    total = 0
    for split in SPLITS:
        X = pd.read_csv(SRC / "splits" / "full_simulation" / f"X_{split}.csv.gz")
        ensure(X.columns.is_unique, f"{split}: duplicate header")
        ensure(not X.duplicated(KEY).any(), f"{split}: duplicate composite key")
        ensure(set(B5_FEATURES).issubset(X.columns), f"{split}: missing B5 features")
        extras = sorted(set(X.columns) - set(B5_FEATURES))
        ensure(extras == sorted(["record_uid", "cohort_year", "administrative_action_eligible_at_cutoff", "weeks_observed"]),
               f"{split}: unexpected non-B5 inputs {extras}")
        selected = X.loc[:, B5_FEATURES]
        ensure(list(selected.columns) == B5_FEATURES, f"{split}: feature order wrong")
        # Match every train/val/test key to the timeline snapshot; never rely
        # on row position or on X alone to inspect the latest source time.
        merged = X[KEY].merge(snaps, on=KEY, how="left", validate="one_to_one", indicator=True)
        ensure(merged["_merge"].eq("both").all(), f"{split}: no source timestamp for some keys")
        ensure(pd.to_datetime(merged[TIMES[1]], utc=True).le(pd.to_datetime(merged[TIMES[0]], utc=True)).all(),
               f"{split}: future feature_source_latest_at")
        # The HK1-specific unavailable features should be null, not filled from
        # future HK2/HK3 observations.
        first = X["snapshot_term_index"].eq(1)
        if lagged_t1 and first.any():
            violations = [f for f in lagged_t1 if X.loc[first, f].notna().any()]
            ensure(not violations, f"{split}: HK1 future/lagged values populated: {violations}")
        total += len(X)
        checks[split] = {"rows": int(len(X)), "b5_count": int(selected.shape[1]),
                         "metadata_removed": extras,
                         "all_snapshot_times_asof": True,
                         "hk1_lagged_features_null": True}
        print(f"PASS C1-C3 {split}: {len(X):,} rows, 58 B5 features, temporal as-of PASS")

    ensure(total == 57569, "Total split rows mismatch")
    return {"status": "PASS", "scope": "C1,C2,C3 (static data contract)",
            "C1": "PASS", "C2": "PASS", "C3": "PASS",
            "C4": "PENDING trainer Pipeline, fit on Train only",
            "C5": "PENDING mutation and fail-fast tests on new loader",
            "total_rows": total, "source_snapshots": len(snaps),
            "t1_lagged_features": lagged_t1, "splits": checks,
            "notes": ["No target labels loaded", "No model fitted or scored", "Dataset not modified",
                      "C3 verifies stored source timestamps; full lineage depends on Phase 1 source contract"]}


def main() -> int:
    try:
        r = verify()
    except Exception as e:
        print(f"PHASE_2_C_STATIC_FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PHASE_2_C_STATIC_PASS; report: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
