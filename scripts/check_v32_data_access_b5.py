"""Independent cross-UID and temporal tests for V3.2 access."""

import ast
from pathlib import Path

import pandas as pd

from src.data_access import V32DataAccess

KEY = ["record_uid", "snapshot_term_index"]
DATES = ("2024-01-01", "2026-10-01")
OUTCOMES = {
    "dropout_next_2_terms", "label_observed",
    "censor_reason", "first_dropout_event_id",
    "exit_reason_for_evaluation", "split",
}


def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def expected_terms(frame, uid, asof):
    rows = frame.loc[frame["record_uid"].eq(uid)]
    cutoff = pd.Timestamp(asof)
    visible = (
        pd.to_datetime(rows["available_at"]).le(cutoff)
        & pd.to_datetime(rows["snapshot_cutoff"]).le(cutoff)
    )
    return sorted(rows.loc[visible, "term_index"].tolist())


def expected_weeks(frame, uid, asof):
    rows = frame.loc[
        frame["record_uid"].eq(uid)
        & frame["term_index"].eq(1)
    ]
    cutoff = pd.Timestamp(asof)
    visible = (
        pd.to_datetime(rows["observed_at"]).le(cutoff)
        & pd.to_datetime(rows["available_at"]).le(cutoff)
    )
    return sorted(rows.loc[visible, "teaching_week"].tolist())


def main():
    repo = V32DataAccess()
    root = repo.source_root
    students = repo._students_by_uid

    uid22 = str(students.loc[
        students["cohort_year"].eq(2022)
    ].index[0])
    uid25 = str(students.loc[
        students["cohort_year"].eq(2025)
    ].index[0])

    uids = (uid22, uid25)

    print("\n=== 1. DEMO SNAPSHOT IDENTITY ===")

    demo = pd.read_csv(
        root / "labels/snapshot_labels_asof_2026_demo.csv.gz",
        usecols=KEY,
    )
    served = repo._snapshots_by_key.reset_index(
        drop=True
    )[KEY]

    matched = served.merge(
        demo, on=KEY, how="outer",
        indicator=True, validate="one_to_one",
    )

    check(
        len(served) == len(demo) == 54609
        and matched["_merge"].eq("both").all(),
        "54,609 serving keys match ASOF_2026_DEMO",
    )

    print("\n=== 2. STUDENT AND SNAPSHOT ISOLATION ===")

    raw_snapshots = pd.read_csv(
        root / "features/prediction_snapshots.csv.gz",
        usecols=KEY + ["snapshot_cutoff"],
    )
    raw_cutoffs = pd.to_datetime(
        raw_snapshots["snapshot_cutoff"]
    )

    for uid in uids:
        profile = repo.get_student(uid)
        check(
            profile is not None
            and profile["record_uid"] == uid,
            f"Correct profile: {uid}",
        )

        for asof in DATES:
            expected = sorted(raw_snapshots.loc[
                raw_snapshots["record_uid"].eq(uid)
                & raw_cutoffs.le(pd.Timestamp(asof)),
                "snapshot_term_index",
            ].tolist())

            actual = repo.list_snapshots(uid, asof=asof)

            check(
                expected == sorted(
                    x["snapshot_term_index"] for x in actual
                )
                and all(
                    x["record_uid"] == uid
                    and not OUTCOMES.intersection(x)
                    for x in actual
                ),
                f"Snapshot UID, asof, labels: {uid}, {asof}",
            )

            for row in actual:
                term = int(row["snapshot_term_index"])
                found = repo.get_snapshot(
                    uid, term, asof=asof
                )
                check(
                    found is not None
                    and found["record_uid"] == uid
                    and found["snapshot_term_index"] == term,
                    f"Composite key: {uid}, HK{term}",
                )

    student_id = repo.get_student(uid22)["student_id"]

    check(
        repo.get_student(student_id) is None
        and repo.list_snapshots(student_id) == []
        and repo.get_snapshot(student_id, 1) is None
        and repo.get_term_history(student_id) == []
        and repo.get_weekly_history(student_id, 1) == [],
        "student_id cannot replace record_uid",
    )

    print("\n=== 3. ACADEMIC HISTORY AS-OF ===")

    academics = pd.read_csv(
        root / "master/term_academics.csv.gz",
        usecols=[
            "record_uid", "term_index",
            "available_at", "snapshot_cutoff",
        ],
    )

    for uid in uids:
        for asof in DATES:
            expected = expected_terms(academics, uid, asof)
            actual = repo.get_term_history(uid, asof=asof)

            check(
                expected == [
                    row["term_index"] for row in actual
                ]
                and all(
                    row["record_uid"] == uid
                    and not OUTCOMES.intersection(row)
                    for row in actual
                ),
                f"Academic history: {uid}, {asof}",
            )

    print("\n=== 4. WEEKLY HISTORY AS-OF ===")

    parts = []
    source = root / "master/engagement_weekly.csv.gz"

    for chunk in pd.read_csv(
        source,
        usecols=[
            "record_uid", "term_index",
            "teaching_week", "observed_at", "available_at",
        ],
        chunksize=150000,
    ):
        selected = chunk.loc[
            chunk["record_uid"].isin(uids)
            & chunk["term_index"].eq(1)
        ]

        if not selected.empty:
            parts.append(selected)

    weekly_source = pd.concat(parts, ignore_index=True)

    for uid in uids:
        for asof in DATES:
            expected = expected_weeks(
                weekly_source, uid, asof
            )
            actual = repo.get_weekly_history(
                uid, 1, asof=asof
            )

            check(
                expected == [
                    row["teaching_week"] for row in actual
                ]
                and all(
                    row["record_uid"] == uid
                    and row["term_index"] == 1
                    and not OUTCOMES.intersection(row)
                    for row in actual
                ),
                f"Weekly history: {uid}, HK1, {asof}",
            )

    print("\n=== 5. ACCESS BOUNDARIES ===")

    for name in (
        "snapshot_labels_full_simulation",
        "weekly_state_hidden",
    ):
        try:
            repo._table_path(name)
        except ValueError:
            continue
        raise AssertionError(
            f"FAIL: Unregistered table accessible: {name}"
        )

    check(True, "Label and hidden tables are not registered")

    source_code = Path(
        "src/data_access.py"
    ).read_text(encoding="utf-8-sig")

    tree = ast.parse(source_code)

    check(
        not any(
            isinstance(node, ast.Attribute)
            and node.attr == "iloc"
            for node in ast.walk(tree)
        ),
        "No positional .iloc access in V3.2 Data Access",
    )

    print("\nV3.2 DATA ACCESS B5 PASS")


if __name__ == "__main__":
    main()
