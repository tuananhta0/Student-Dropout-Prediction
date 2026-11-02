"""Read-only C3 audit: temporal validity and as-of visibility of V3.2 tables.

This audit validates source timestamps; it does NOT authorize serving every
column of the checked tables. Serving must still enforce its own allowlist.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Every master table carrying a dated field. The other master tables
# (courses, curricula, curriculum_courses) are static reference data.
DATE_FIELDS = {
    "students": ("enrollment_date",),
    "term_academics": ("available_at", "snapshot_cutoff"),
    "course_attempts": ("score_published_at", "available_at"),
    "engagement_weekly": ("observed_at", "available_at"),
    "finance_terms": ("payment_due_date", "available_at"),
    "finance_transactions": ("event_at", "available_at"),
    "student_events": ("event_at", "effective_at", "available_at"),
    "enrollment_terms": ("available_at",),
    "state_transitions": ("effective_at",),
    "support_events": ("event_at", "available_at"),
    "academic_requirement_events": ("available_at",),
    "course_assessment_components": ("available_at",),
    "graduation_clearance": ("available_at",),
    "term_calendar": ("start_date", "end_date", "grade_release_date"),
}

# Explicit temporal constraints. Do not compare finance payment_due_date
# with available_at: a due date is not necessarily a publication date.
ORDER_RULES = {
    "term_academics": (("available_at", "snapshot_cutoff"),),
    "course_attempts": (("score_published_at", "available_at"),),
    "engagement_weekly": (("observed_at", "available_at"),),
    "finance_transactions": (("event_at", "available_at"),),
    "student_events": (
        ("event_at", "effective_at"),
        ("effective_at", "available_at"),
    ),
    "support_events": (("event_at", "available_at"),),
    "term_calendar": (
        ("start_date", "end_date"),
        ("end_date", "grade_release_date"),
    ),
}

# Earliest known visibility timestamp for each dated operational table.
# Calendar has future planned dates, not an event publication timestamp,
# so it must NOT be treated as safe-to-serve simply because it has a date.
VISIBILITY_FIELD = {
    "students": "enrollment_date",
    "term_academics": "snapshot_cutoff",
    "course_attempts": "available_at",
    "engagement_weekly": "available_at",
    "finance_terms": "available_at",
    "finance_transactions": "available_at",
    "student_events": "available_at",
    "enrollment_terms": "available_at",
    "state_transitions": "effective_at",
    "support_events": "available_at",
    "academic_requirement_events": "available_at",
    "course_assessment_components": "available_at",
    "graduation_clearance": "available_at",
}

STATIC_TABLES = {"courses", "curricula", "curriculum_courses"}
CHUNK_SIZE = 200_000


def audit(root: Path) -> None:
    manifest_path = root / "manifests" / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("generator_version") == "3.2.0"
        and manifest.get("scenario_id") == "V3_2_MAIN"
        and manifest.get("is_synthetic") is True
    ):
        raise ValueError("Unexpected V3.2 manifest identity")

    demo_asof = pd.Timestamp(manifest["demo_as_of"])
    expected_tables = set(DATE_FIELDS) | STATIC_TABLES
    actual_tables = {
        name for name in manifest["table_artifacts"]
        if (root / "master" / manifest["table_artifacts"][name]["file"]).is_file()
    }
    if expected_tables != actual_tables:
        raise ValueError(
            f"Unexpected master table set: "
            f"missing={sorted(expected_tables - actual_tables)}, "
            f"unexpected={sorted(actual_tables - expected_tables)}"
        )

    print("=== C3 MASTER TIMESTAMP AUDIT ===", flush=True)
    total_rows = 0
    future_rows = {}

    for table, columns in DATE_FIELDS.items():
        info = manifest["table_artifacts"][table]
        filename = info["file"]
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe manifest path for {table}")
        source = root / "master" / filename

        rows = 0
        future = 0
        checked = 0
        for part in pd.read_csv(source, usecols=list(columns),
                                chunksize=CHUNK_SIZE, low_memory=False):
            rows += len(part)
            parsed = {}
            for column in columns:
                values = pd.to_datetime(part[column], errors="coerce")
                if values.isna().any():
                    raise ValueError(
                        f"{table}.{column}: null or invalid dates "
                        f"(rows in chunk={int(values.isna().sum())})"
                    )
                parsed[column] = values

            for earlier, later in ORDER_RULES.get(table, ()):
                invalid = parsed[earlier] > parsed[later]
                if invalid.any():
                    raise ValueError(
                        f"{table}: {int(invalid.sum())} rows violate "
                        f"{earlier} <= {later}"
                    )
                checked += len(part)

            if table in VISIBILITY_FIELD:
                visibility = parsed[VISIBILITY_FIELD[table]]
                future += int((visibility > demo_asof).sum())

        if rows != int(info["rows"]):
            raise ValueError(f"{table}: rows {rows} != manifest {info['rows']}")
        total_rows += rows
        if table in VISIBILITY_FIELD:
            future_rows[table] = future
        print(
            f"PASS: {table}: {rows:,} rows; "
            f"dates valid; order valid; "
            f"after demo cutoff={future if table in VISIBILITY_FIELD else 'N/A'}",
            flush=True,
        )

    for table in sorted(STATIC_TABLES):
        info = manifest["table_artifacts"][table]
        source = root / "master" / info["file"]
        rows = sum(len(c) for c in pd.read_csv(
            source, usecols=[0], chunksize=CHUNK_SIZE
        ))
        if rows != int(info["rows"]):
            raise ValueError(f"{table}: unexpected number of rows")
        total_rows += rows
        print(f"PASS: {table}: {rows:,} static-reference rows", flush=True)

    # Check that the ML feature availability timestamp never exceeds cutoff.
    snap_info = manifest["table_artifacts"]["prediction_snapshots"]
    snapshot_path = root / "features" / snap_info["file"]
    rows = 0
    demo_rows = 0
    for part in pd.read_csv(
        snapshot_path,
        usecols=["snapshot_cutoff", "feature_source_latest_at"],
        chunksize=CHUNK_SIZE,
    ):
        rows += len(part)
        cutoff = pd.to_datetime(part["snapshot_cutoff"], errors="coerce")
        latest = pd.to_datetime(part["feature_source_latest_at"], errors="coerce")
        if cutoff.isna().any() or latest.isna().any() or (latest > cutoff).any():
            raise ValueError("Prediction snapshot has invalid/future feature dates")
        demo_rows += int((cutoff <= demo_asof).sum())
    if rows != 57569 or rows != int(snap_info["rows"]):
        raise ValueError("Unexpected prediction snapshot count")
    if demo_rows != 54609:
        raise ValueError(f"Unexpected demo snapshot count: {demo_rows}")

    print(f"PASS: 57,569 prediction snapshots; 0 future features", flush=True)
    print(f"PASS: 54,609 snapshots eligible as of {demo_asof.date()}", flush=True)
    print("PASS: Historical source tables classified by visibility timestamp", flush=True)
    print("NOTE: Future calendar entries are legitimate reference schedules, NOT historical events")
    print("NOTE: Finance original duplicate IDs remain documented under C2")
    print("NOTE: Source timestamp audit does not replace per-API as-of filtering")
    print("\nV3.2 C3 TEMPORAL AUDIT PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    audit(args.data.resolve())
