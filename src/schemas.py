"""Core V3.2 schema checks; no model or app side effects."""

import argparse
from pathlib import Path

import pandas as pd

VERSION = "3.2.0"
SCENARIO = "V3_2_MAIN"
SNAPSHOT_KEY = ["record_uid", "snapshot_term_index"]


def require_columns(frame, columns, table):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")


def require_unique_key(frame, keys, table):
    require_columns(frame, keys, table)
    if frame[keys].isna().any(axis=1).any():
        raise ValueError(f"{table}: null key {keys}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{table}: duplicate key {keys}")


def require_range(frame, field, low, high, table):
    require_columns(frame, [field], table)
    values = pd.to_numeric(frame[field], errors="coerce")

    invalid = frame[field].notna() & values.isna()
    outside = values.notna() & ~values.between(low, high)

    if (invalid | outside).any():
        raise ValueError(
            f"{table}.{field}: invalid values or "
            f"values outside [{low}, {high}]"
        )


def require_dates_not_after(frame, earlier, later, table):
    require_columns(frame, [earlier, later], table)

    a = pd.to_datetime(frame[earlier], errors="coerce")
    b = pd.to_datetime(frame[later], errors="coerce")

    if a.isna().any() or b.isna().any() or (a > b).any():
        raise ValueError(
            f"{table}: invalid dates or "
            f"{earlier} after {later}"
        )


def validate_students(frame):
    table = "students"

    require_unique_key(frame, ["record_uid"], table)
    require_unique_key(frame, ["student_id"], table)
    require_columns(
        frame, ["schema_version", "scenario_id"], table
    )

    if not frame["schema_version"].eq(VERSION).all():
        raise ValueError(f"{table}: schema version mismatch")

    if not frame["scenario_id"].eq(SCENARIO).all():
        raise ValueError(f"{table}: scenario mismatch")

    require_range(frame, "cohort_year", 2022, 2025, table)
    require_range(frame, "scholarship_percent", 0, 100, table)
    require_range(frame, "enrollment_age", 0, 120, table)


def validate_snapshots(frame, students):
    table = "prediction_snapshots"

    require_unique_key(frame, SNAPSHOT_KEY, table)
    require_columns(
        frame, ["student_id", "cohort_year"], table
    )

    require_range(frame, "snapshot_term_index", 1, 3, table)

    for field in ("gpa4_term", "gpa4_cumulative"):
        require_range(frame, field, 0, 4, table)

    for field in (
        "attendance_rate_term",
        "scholarship_percent_at_cutoff",
    ):
        require_range(frame, field, 0, 100, table)

    require_dates_not_after(
        frame,
        "feature_source_latest_at",
        "snapshot_cutoff",
        table,
    )

    matched = frame[
        ["record_uid", "student_id", "cohort_year"]
    ].merge(
        students[
            ["record_uid", "student_id", "cohort_year"]
        ],
        on="record_uid",
        how="left",
        validate="many_to_one",
        suffixes=("_snapshot", "_student"),
        indicator=True,
    )

    if not matched["_merge"].eq("both").all():
        raise ValueError(
            f"{table}: missing student foreign key"
        )

    for field in ("student_id", "cohort_year"):
        if not matched[
            f"{field}_snapshot"
        ].eq(
            matched[f"{field}_student"]
        ).all():
            raise ValueError(
                f"{table}: {field} does not match students"
            )


def validate_term_academics(frame, students):
    table = "term_academics"

    require_unique_key(
        frame, ["record_uid", "term_index"], table
    )
    require_columns(frame, ["schema_version"], table)

    if not frame["schema_version"].eq(VERSION).all():
        raise ValueError(
            f"{table}: schema version mismatch"
        )

    if not frame["record_uid"].isin(
        students["record_uid"]
    ).all():
        raise ValueError(
            f"{table}: missing student foreign key"
        )

    require_range(frame, "term_index", 1, 20, table)

    for field in ("gpa4_term", "gpa4_cumulative"):
        require_range(frame, field, 0, 4, table)

    require_range(frame, "gpa10_term", 0, 10, table)
    require_range(
        frame, "attendance_rate_term", 0, 100, table
    )

    require_dates_not_after(
        frame, "available_at", "snapshot_cutoff", table
    )


def validate_core_tables(root):
    def load(relative):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path, low_memory=False)

    students = load("master/students.csv.gz")
    snapshots = load(
        "features/prediction_snapshots.csv.gz"
    )
    academics = load("master/term_academics.csv.gz")

    validate_students(students)
    print("PASS: students schema")

    validate_snapshots(snapshots, students)
    print(
        "PASS: snapshot schema, keys, ranges, "
        "dates and student mapping"
    )

    validate_term_academics(academics, students)
    print(
        "PASS: academic schema, keys, ranges, "
        "dates and student FK"
    )

    print("V3.2 CORE SCHEMAS PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True
    )
    args = parser.parse_args()

    validate_core_tables(args.data.resolve())
