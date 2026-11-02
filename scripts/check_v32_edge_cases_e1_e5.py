"""Phase 1 E1-E5: read-only, dynamically selected V3.2 student edge cases.

Run from repository root:
    python -m scripts.check_v32_edge_cases_e1_e5

The label table is loaded only by this audit script, never by Data Access.
"""

from pathlib import Path

import pandas as pd

from src.data_access import PROFILE_FIELDS, V32DataAccess

DEMO = pd.Timestamp("2026-10-01")
KEY = ["record_uid", "snapshot_term_index"]
FORBIDDEN = {
    "dropout_next_2_terms",
    "label_observed",
    "label_observation_end",
    "censor_reason",
    "first_dropout_event_id",
    "exit_reason_for_evaluation",
    "split",
    "observation_view_id",
}
EXIT_TYPES = {
    "VOLUNTARY_WITHDRAWAL",
    "FORCED_ACADEMIC_WITHDRAWAL",
    "CONFIRMED_INSTITUTIONAL_ABANDONMENT",
}


def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def require_safe_profile(repo, uid):
    profile = repo.get_student(uid)
    check(
        profile is not None
        and profile["record_uid"] == uid
        and set(profile) == set(PROFILE_FIELDS),
        f"Whitelisted profile for {uid}",
    )
    return profile


def require_safe_snapshot(repo, uid, term):
    row = repo.get_snapshot(uid, int(term))
    check(
        row is not None
        and row["record_uid"] == uid
        and row["snapshot_term_index"] == int(term)
        and not (FORBIDDEN & set(row)),
        f"Safe snapshot for {uid}, HK{term}",
    )
    return row


def main():
    repo = V32DataAccess()
    root = Path(repo.source_root)

    students = pd.read_csv(
        root / "master/students.csv.gz",
        usecols=["record_uid", "cohort_year", "enrollment_date"],
    )
    snapshots = pd.read_csv(
        root / "features/prediction_snapshots.csv.gz",
        usecols=KEY + ["snapshot_cutoff", "admission_score_normalized"],
    )
    labels = pd.read_csv(
        root / "labels/snapshot_labels_asof_2026_demo.csv.gz",
        usecols=KEY + [
            "snapshot_cutoff", "label_observed", "dropout_next_2_terms",
            "label_observation_end", "observation_as_of", "censor_reason",
        ],
    )
    events = pd.read_csv(
        root / "master/student_events.csv.gz",
        usecols=[
            "record_uid", "event_type", "effective_at", "available_at"
        ],
    )

    check(len(students) == 20000, "20,000 source students")
    check(len(snapshots) == 57569, "57,569 source snapshots")
    check(len(labels) == 54609, "54,609 demo labels")

    snapshot_dates = pd.to_datetime(snapshots["snapshot_cutoff"])
    demo_snaps = snapshots.loc[snapshot_dates.le(DEMO)].copy()
    check(len(demo_snaps) == 54609, "Demo cutoff excludes future snapshots")

    print("\n=== E1: STUDENT WITH THREE SNAPSHOTS ===")
    counts = demo_snaps.groupby("record_uid").size()
    three_uids = counts[counts.eq(3)].index
    check(len(three_uids) > 0, "Found student with 3 observable snapshots")
    uid = str(three_uids[0])
    require_safe_profile(repo, uid)
    rows = repo.list_snapshots(uid)
    check(
        [r["snapshot_term_index"] for r in rows] == [1, 2, 3]
        and all(not (FORBIDDEN & set(r)) for r in rows),
        "E1: HK1, HK2, HK3 ordered; outcomes not exposed",
    )
    for term in (1, 2, 3):
        require_safe_snapshot(repo, uid, term)
    first_cutoff = rows[0]["snapshot_cutoff"]
    before_first = (
        pd.Timestamp(first_cutoff) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    check(
        repo.list_snapshots(uid, asof=before_first) == []
        and len(repo.list_snapshots(uid, asof=first_cutoff)) == 1,
        "E1: First snapshot becomes available only at cutoff",
    )
    print(f"E1 EXAMPLE UID: {uid}")

    print("\n=== E2: EARLY WITHDRAWAL ===")
    per_uid = demo_snaps.groupby("record_uid").agg(
        n=("snapshot_term_index", "size"),
        last_cutoff=("snapshot_cutoff", "max"),
    ).reset_index()
    withdrawn = events.loc[
        events["event_type"].isin(EXIT_TYPES)
        & pd.to_datetime(events["effective_at"]).le(DEMO)
        & pd.to_datetime(events["available_at"]).le(DEMO)
    ].merge(per_uid, on="record_uid", how="inner", validate="many_to_one")
    withdrawn = withdrawn.loc[
        withdrawn["n"].between(1, 2)
        & pd.to_datetime(withdrawn["last_cutoff"]).le(
            pd.to_datetime(withdrawn["effective_at"])
        )
    ].sort_values(["effective_at", "record_uid"])
    check(not withdrawn.empty, "Found documented early withdrawal")
    exit_row = withdrawn.iloc[0]
    uid = str(exit_row["record_uid"])
    effective_at = str(exit_row["effective_at"])
    history = repo.list_snapshots(uid)
    at_exit = repo.list_snapshots(uid, asof=effective_at)
    check(
        len(history) == len(at_exit) == int(exit_row["n"])
        and len(history) < 3
        and all(pd.Timestamp(r["snapshot_cutoff"]) <= pd.Timestamp(effective_at)
                for r in history),
        "E2: Early-exit UID has no later prediction snapshots",
    )
    check(
        repo.get_student(uid, asof=effective_at) is not None,
        "E2: Previous student profile remains available historically",
    )
    print(f"E2 EXAMPLE UID: {uid}; event={exit_row['event_type']} at {effective_at}")

    print("\n=== E3: GENUINE MISSING FEATURE ===")
    missing = demo_snaps.loc[
        demo_snaps["admission_score_normalized"].isna()
    ].sort_values(KEY)
    check(not missing.empty, "Found observed snapshot with missing admission feature")
    item = missing.iloc[0]
    uid = str(item["record_uid"])
    term = int(item["snapshot_term_index"])
    row = require_safe_snapshot(repo, uid, term)
    check(
        "admission_score_normalized" in row
        and row["admission_score_normalized"] is None,
        "E3: Missing feature preserved as None, not invented or imputed",
    )
    print(f"E3 EXAMPLE UID: {uid}; term={term}")

    print("\n=== E4: RIGHT-CENSORED DEMO LABEL ===")
    censored = labels.loc[
        labels["label_observed"].eq(False)
        & labels["censor_reason"].eq("RIGHT_CENSORED_ASOF")
        & labels["dropout_next_2_terms"].isna()
        & pd.to_datetime(labels["label_observation_end"]).gt(DEMO)
        & labels["observation_as_of"].eq("2026-10-01")
    ].sort_values(KEY)
    check(len(censored) == 9970, "9,970 genuinely right-censored demo labels")
    item = censored.iloc[0]
    uid = str(item["record_uid"])
    term = int(item["snapshot_term_index"])
    row = require_safe_snapshot(repo, uid, term)
    check(
        pd.Timestamp(row["snapshot_cutoff"]) <= DEMO
        and not (FORBIDDEN & set(row)),
        "E4: Censored outcome and observation metadata absent from serving",
    )
    print(f"E4 EXAMPLE UID: {uid}; term={term}")

    print("\n=== E5: COHORT 2025 WITHOUT FUTURE TERM ===")
    cohort25 = set(students.loc[
        students["cohort_year"].eq(2025), "record_uid"
    ])
    all_counts = snapshots.groupby("record_uid").size()
    demo_counts = demo_snaps.groupby("record_uid").size()
    candidates = sorted(
        uid for uid in cohort25
        if all_counts.get(uid, 0) == 3 and demo_counts.get(uid, 0) == 2
    )
    check(bool(candidates), "Found cohort-2025 UID with 2 observed / 3 simulated snapshots")
    uid = candidates[0]
    profile = require_safe_profile(repo, uid)
    check(profile["cohort_year"] == 2025, "E5: Student belongs to cohort 2025")
    check(
        repo.get_student(uid, asof="2024-12-31") is None
        and repo.list_snapshots(uid, asof="2024-12-31") == [],
        "E5: Student and snapshots invisible before enrollment",
    )
    visible = repo.list_snapshots(uid)
    check(
        [s["snapshot_term_index"] for s in visible] == [1, 2]
        and repo.get_snapshot(uid, 3) is None,
        "E5: HK3 exists in simulation but is hidden from 2026 demo",
    )
    print(f"E5 EXAMPLE UID: {uid}")

    print("\nV3.2 E1-E5 EDGE CASES PASS")


if __name__ == "__main__":
    main()
