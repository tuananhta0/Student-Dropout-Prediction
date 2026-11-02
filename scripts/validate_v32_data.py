import argparse
from pathlib import Path

import pandas as pd

KEY = ["record_uid", "snapshot_term_index"]

SPLITS = {
    "TRAIN_2022_23": ({2022, 2023}, 34304),
    "VALIDATION_2024": ({2024}, 13981),
    "TEST_2025": ({2025}, 9284),
}


def check(condition, description):
    if not bool(condition):
        raise AssertionError(f"FAIL: {description}")
    print(f"PASS: {description}")


def read(root, relative):
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def dates(series, description):
    result = pd.to_datetime(series, errors="coerce")
    check(result.notna().all(), f"Valid dates: {description}")
    return result


def main(root):
    students = read(root, "master/students.csv.gz")
    snapshots = read(root, "features/prediction_snapshots.csv.gz")
    labels = read(root, "labels/snapshot_labels_full_simulation.csv.gz")
    demo = read(root, "labels/snapshot_labels_asof_2026_demo.csv.gz")
    academics = read(root, "master/term_academics.csv.gz")

    print("\n=== SCHEMA AND IDENTITY ===")
    check(students["schema_version"].eq("3.2.0").all(),
          "Student schema version 3.2.0")
    check(academics["schema_version"].eq("3.2.0").all(),
          "Academic schema version 3.2.0")
    check(students["scenario_id"].eq("V3_2_MAIN").all(),
          "Student scenario V3_2_MAIN")
    check(labels["scenario_id"].eq("V3_2_MAIN").all(),
          "Label scenario V3_2_MAIN")
    check(students["student_id"].is_unique,
          "Student ID unique")
    check(snapshots["record_uid"].isin(students["record_uid"]).all(),
          "Snapshot UID foreign keys")

    print("\n=== TEMPORAL VALIDITY ===")
    cutoff = dates(snapshots["snapshot_cutoff"], "snapshot_cutoff")
    latest = dates(snapshots["feature_source_latest_at"],
                   "feature_source_latest_at")
    check((latest <= cutoff).all(),
          "No snapshot feature after cutoff")

    available = dates(academics["available_at"],
                      "academic available_at")
    academic_cutoff = dates(academics["snapshot_cutoff"],
                            "academic snapshot_cutoff")
    check((available <= academic_cutoff).all(),
          "No academic record available after its cutoff")

    print("\n=== LABEL ALIGNMENT ===")
    aligned = snapshots[
        KEY + ["snapshot_cutoff", "cohort_year"]
    ].merge(
        labels[KEY + ["snapshot_cutoff", "cohort_year"]],
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_feature", "_label"),
    )
    check(aligned["_merge"].eq("both").all(),
          "All snapshot-label keys match")
    check(
        aligned["snapshot_cutoff_feature"].eq(
            aligned["snapshot_cutoff_label"]
        ).all(),
        "Snapshot-label cutoffs match",
    )
    check(
        aligned["cohort_year_feature"].eq(
            aligned["cohort_year_label"]
        ).all(),
        "Snapshot-label cohorts match",
    )

    print("\n=== COHORT SPLITS ===")
    check(labels["observation_view_id"].eq(
        "FULL_SIMULATION").all(), "Full simulation view")
    check(labels["label_observed"].eq(True).all(),
          "Full simulation labels observed")
    check(labels.groupby("record_uid")["split"].nunique().eq(1).all(),
          "No UID crossing splits")

    for name, (years, count) in SPLITS.items():
        part = labels[labels["split"].eq(name)]
        check(len(part) == count, f"{name}: {count} rows")
        check(set(part["cohort_year"].unique()) == years,
              f"{name}: correct cohort years")

        x = read(root, f"splits/full_simulation/X_{name}.csv.gz")
        y = read(root, f"splits/full_simulation/y_{name}.csv.gz")

        check(len(x) == len(y) == count,
              f"{name}: X/y row counts")
        check(x[KEY].equals(y[KEY]),
              f"{name}: X/y key order")

        check(
            x[KEY].merge(
                part[KEY], on=KEY, how="outer",
                validate="one_to_one", indicator=True
            )["_merge"].eq("both").all(),
            f"{name}: keys agree with labels",
        )

    check(labels["split"].isin(SPLITS).all(),
          "No unexpected split")

    print("\n=== AS-OF 2026 DEMO ===")
    check(len(demo) == 54609, "54,609 as-of label rows")
    check(demo["observation_view_id"].eq(
        "ASOF_2026_DEMO").all(), "Demo view identity")
    check(demo["observation_as_of"].eq(
        "2026-10-01").all(), "Demo observation cutoff")

    demo_cutoffs = dates(demo["snapshot_cutoff"],
                         "demo snapshot_cutoff")
    check((demo_cutoffs <= pd.Timestamp("2026-10-01")).all(),
          "No future demo snapshots")

    censored = demo["label_observed"].eq(False)
    check(censored.sum() == 9970,
          "9,970 right-censored demo labels")
    check(demo.loc[censored, "dropout_next_2_terms"].isna().all(),
          "Censored outcomes are missing")

    check(
        demo[KEY].merge(
            labels[KEY], on=KEY, how="left",
            validate="one_to_one", indicator=True
        )["_merge"].eq("both").all(),
        "Demo is subset of full simulation",
    )

    print("\nV3.2 SCHEMA, SPLIT AND TIME CHECKS PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    main(args.data.resolve())
