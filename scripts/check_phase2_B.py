r"""Phase 2-B: read-only data-contract audit for CMC V3.2 FULL_SIMULATION.

Usage from repository root (PowerShell):
  .\.venv\Scripts\python.exe -B .\scripts\check_phase2_B.py

Reads training-data inputs and writes one audit report to
  reports/ml_v3_2/data_contract_verification.json
Does NOT fit models or modify source datasets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "external" / "v3_2_main"
OUT = ROOT / "reports" / "ml_v3_2" / "data_contract_verification.json"
KEY = ["record_uid", "snapshot_term_index"]
TARGET = "dropout_next_2_terms"
EXPECTED = {
    "TRAIN_2022_23": {"rows": 34304, "uid": 11924, "positive": 3136, "cohorts": [2022, 2023]},
    "VALIDATION_2024": {"rows": 13981, "uid": 4842, "positive": 1240, "cohorts": [2024]},
    "TEST_2025": {"rows": 9284, "uid": 3234, "positive": 911, "cohorts": [2025]},
}


def ensure(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def verify() -> dict:
    # B1, B2, B3, B5: verify frozen splits and align by key, NEVER row position.
    report = {
        "phase": "2-B",
        "status": "FAIL",
        "scenario": "V3_2_MAIN",
        "observation_view": "FULL_SIMULATION",
        "split_summaries": {},
        "notes": ["No model was trained or scored", "Test labels inspected for structural integrity only"],
    }
    all_uid: dict[str, set[str]] = {}
    all_parts: list[pd.DataFrame] = []
    for split, exp in EXPECTED.items():
        src = DATA / "splits" / "full_simulation"
        X = pd.read_csv(src / f"X_{split}.csv.gz")
        y = pd.read_csv(src / f"y_{split}.csv.gz")
        ensure(X.shape == (exp["rows"], 62), f"{split}: unexpected X shape {X.shape}")
        ensure(y.shape == (exp["rows"], 3), f"{split}: unexpected y shape {y.shape}")
        ensure(set(y.columns) == set(KEY + [TARGET]), f"{split}: unexpected y columns")
        ensure(all(c in X.columns and c in y.columns for c in KEY), f"{split}: missing composite key")
        ensure(X.columns.is_unique and y.columns.is_unique, f"{split}: duplicate column names")
        ensure(not X[KEY].isna().any().any() and not y[KEY].isna().any().any(), f"{split}: null key")
        ensure(not X.duplicated(KEY).any(), f"{split}: duplicate X composite key")
        ensure(not y.duplicated(KEY).any(), f"{split}: duplicate y composite key")
        ensure(y[TARGET].notna().all(), f"{split}: missing target")
        ensure(set(y[TARGET].unique()).issubset({0, 1}), f"{split}: target not binary")
        ensure(X.record_uid.nunique() == exp["uid"], f"{split}: wrong UID count")
        ensure(set(X.cohort_year.unique()) == set(exp["cohorts"]), f"{split}: cohort mismatch")
        ensure(set(X.snapshot_term_index.unique()).issubset({1, 2, 3}), f"{split}: invalid term")
        ensure(int(y[TARGET].sum()) == exp["positive"], f"{split}: positive count mismatch")
        # Sorting/shuffling y cannot change the matched label of any X key.
        joined = X[KEY].merge(y[KEY + [TARGET]], on=KEY, how="left", validate="one_to_one", indicator=True, sort=False)
        ensure(len(joined) == len(X) and joined["_merge"].eq("both").all(), f"{split}: X/y mismatch")
        reversed_y = y.iloc[::-1].reset_index(drop=True)
        joined_reverse = X[KEY].merge(reversed_y[KEY + [TARGET]], on=KEY, how="left", validate="one_to_one", sort=False)
        ensure(joined[TARGET].reset_index(drop=True).equals(joined_reverse[TARGET].reset_index(drop=True)), f"{split}: shuffle changed labels")
        all_uid[split] = set(X.record_uid)
        all_parts.append(joined[KEY + [TARGET]].assign(split=split))
        missing = {k: int(v) for k, v in X.isna().sum().items() if v}
        per_term = {str(term): {"snapshots": int(len(g)), "positives": int(g[TARGET].sum())} for term, g in joined.groupby("snapshot_term_index")}
        report["split_summaries"][split] = {
            "rows": len(X), "unique_students": X.record_uid.nunique(), "positives": int(y[TARGET].sum()),
            "x_columns": int(X.shape[1]), "y_columns": int(y.shape[1]), "cohorts": sorted(map(int, X.cohort_year.unique())),
            "x_column_dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
            "missing_x": missing, "per_term": per_term,
            "keys_unique": True, "keyed_alignment_invariant_to_y_shuffle": True,
        }
        print(f"PASS {split}: {len(X):,} snapshots, {X.record_uid.nunique():,} UID, {int(y[TARGET].sum()):,} positives")

    splits = list(all_uid)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            ensure(not (all_uid[a] & all_uid[b]), f"UID overlap between {a} and {b}")
    report["total_snapshots"] = sum(p["rows"] for p in report["split_summaries"].values())
    report["total_positive"] = sum(p["positives"] for p in report["split_summaries"].values())
    report["total_students"] = sum(p["unique_students"] for p in report["split_summaries"].values())
    ensure((report["total_snapshots"], report["total_students"], report["total_positive"]) == (57569, 20000, 5287), "global totals mismatch")

    # B4: source FULL_SIMULATION labels have been observed; as-of right censoring remains unknown.
    full = pd.read_csv(DATA / "labels" / "snapshot_labels_full_simulation.csv.gz")
    demo = pd.read_csv(DATA / "labels" / "snapshot_labels_asof_2026_demo.csv.gz")
    ensure(len(full) == 57569, "FULL_SIMULATION label count")
    ensure(full["label_observed"].eq(True).all(), "FULL_SIMULATION has unobserved labels")
    ensure(full["observation_view_id"].eq("FULL_SIMULATION").all(), "Unexpected full view")
    ensure(full["scenario_id"].eq("V3_2_MAIN").all(), "Unexpected full scenario")
    ensure(not full.duplicated(KEY).any(), "Duplicate full labels")
    concat = pd.concat(all_parts, ignore_index=True)
    chk = concat.merge(full[KEY + [TARGET, "split"]], on=KEY, how="outer", validate="one_to_one", suffixes=("_xy", "_full"), indicator=True)
    ensure(chk["_merge"].eq("both").all(), "Split keys differ from full label table")
    ensure(chk[f"{TARGET}_xy"].equals(chk[f"{TARGET}_full"]), "Split label differs from full truth")
    ensure(chk["split_xy"].equals(chk["split_full"]), "Full labels split mismatch")
    ensure(len(demo) == 54609, "ASOF label count unexpected")
    ensure(demo["observation_view_id"].eq("ASOF_2026_DEMO").all(), "Unexpected demo view")
    censored = demo.loc[demo["label_observed"].eq(False)]
    ensure(len(censored) == 9970, "Right-censored count mismatch")
    ensure(censored[TARGET].isna().all(), "Censored targets must be null, never 0")
    report["label_views"] = {"full_simulation_observed": int(full.label_observed.sum()),
                             "asof_demo_rows": len(demo), "asof_demo_right_censored": len(censored),
                             "right_censored_labels_null": True}
    report["gates"] = {k: "PASS" for k in ["B1", "B2", "B3", "B4", "B5"]}
    report["status"] = "PASS"
    return report


def main() -> int:
    try:
        report = verify()
    except Exception as exc:
        print(f"PHASE_2_B_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE_2_B_PASS; report: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
