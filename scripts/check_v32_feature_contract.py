"""Audit the V3.2 B5 feature whitelist and frozen cohort partitions.

Read-only: no model fitting and no writes to original data.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.features import B5_FEATURES, B5_FEATURE_COUNT

KEY = ["record_uid", "snapshot_term_index"]
SPLITS = {
    "TRAIN_2022_23": (34304, {2022, 2023}),
    "VALIDATION_2024": (13981, {2024}),
    "TEST_2025": (9284, {2025}),
}
META_X = {"record_uid", "cohort_year", "administrative_action_eligible_at_cutoff", "weeks_observed"}
FORBIDDEN = {
    "student_id", "snapshot_cutoff", "feature_source_latest_at", "split",
    "dropout_next_2_terms", "label_observed", "label_observation_end",
    "observation_as_of", "censor_reason", "first_dropout_event_id",
    "exit_reason_for_evaluation", "synthetic_shock_events",
    "weekly_state_hidden",
}


def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def main(root: Path):
    registry = pd.read_csv(root / "docs/feature_registry.csv")
    snapshots = pd.read_csv(root / "features/prediction_snapshots.csv.gz", low_memory=False)
    labels = pd.read_csv(root / "labels/snapshot_labels_full_simulation.csv.gz", low_memory=False)

    print("\n=== FEATURE REGISTRY ===")
    check(len(registry) == 65 and registry["feature_name"].is_unique,
          "65 unique registered snapshot columns")
    check(set(registry["feature_name"]) == set(snapshots.columns),
          "Feature registry matches snapshot schema")
    roles = registry.set_index("feature_name")["role"].to_dict()
    allowed = {name for name, role in roles.items() if role == "FEATURE_ALLOWED_ASOF"}
    check(len(allowed) == 60, "60 feature-allowed-asof registry entries")
    check(len(B5_FEATURES) == B5_FEATURE_COUNT and len(set(B5_FEATURES)) == 58,
          "B5 has 58 unique features")
    check(set(B5_FEATURES).issubset(allowed),
          "All 58 B5 features approved by registry")
    check(allowed - set(B5_FEATURES) == {
        "administrative_action_eligible_at_cutoff", "weeks_observed"
    }, "B5 excludes two allowed-but-nonbaseline features")
    check(not set(B5_FEATURES).intersection(FORBIDDEN | META_X),
          "No outcomes, identity or split context in B5")

    print("\n=== TRAIN / VALIDATION / TEST SCHEMAS ===")
    check(len(snapshots) == len(labels) == 57569,
          "57,569 snapshot and full-simulation label rows")
    seen_uids = {}
    total = 0
    for name, (expected_count, years) in SPLITS.items():
        x = pd.read_csv(root / f"splits/full_simulation/X_{name}.csv.gz", low_memory=False)
        y = pd.read_csv(root / f"splits/full_simulation/y_{name}.csv.gz", low_memory=False)
        check(len(x) == len(y) == expected_count,
              f"{name}: expected X/y row count")
        check(len(x.columns) == 62 and set(x.columns) == set(B5_FEATURES) | META_X,
              f"{name}: 62-column X schema equals 58 B5 + 4 metadata")
        check(list(y.columns) == KEY + ["dropout_next_2_terms"],
              f"{name}: label file contains only key and outcome")
        check(not set(x).intersection(FORBIDDEN),
              f"{name}: no future/label columns in X")
        check(set(x["cohort_year"].unique()) == years,
              f"{name}: correct cohort assignment")
        check(not x.duplicated(KEY).any() and not y.duplicated(KEY).any(),
              f"{name}: no duplicate composite keys")
        check(x[KEY].equals(y[KEY]),
              f"{name}: X and y key order identical")

        ref = snapshots.set_index(KEY, drop=False).loc[
            pd.MultiIndex.from_frame(x[KEY])
        ].reset_index(drop=True)
        check(x[KEY].equals(ref[KEY]), f"{name}: all X keys exist in snapshots")
        check(all(x[c].equals(ref[c]) for c in x.columns if c not in KEY),
              f"{name}: all X values match snapshots")
        y_ref = labels.set_index(KEY).loc[
            pd.MultiIndex.from_frame(y[KEY])
        ].reset_index(drop=True)
        check(y["dropout_next_2_terms"].equals(y_ref["dropout_next_2_terms"]),
              f"{name}: outcomes match source labels")
        check(y_ref["label_observed"].eq(True).all() and y_ref["split"].eq(name).all(),
              f"{name}: labels are fully observed and correctly partitioned")
        for uid in x["record_uid"].unique():
            if uid in seen_uids:
                raise AssertionError(f"FAIL: UID crosses splits: {uid}")
            seen_uids[uid] = name
        total += len(x)
    check(total == 57569, "All 57,569 full-simulation snapshots partitioned")
    print("\nV3.2 FEATURE CONTRACT PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    main(args.data.resolve())
