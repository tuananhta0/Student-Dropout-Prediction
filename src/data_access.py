"""Read-only data access for CMC Synthetic Dataset V3.2."""

import json
from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd

from src.schemas import validate_students


PROFILE_FIELDS = (
    "record_uid",
    "student_id",
    "cohort_year",
    "enrollment_date",
    "campus",
    "major_code",
    "major_name",
    "specialization",
    "class_code",
    "degree_type",
    "education_mode",
    "curriculum_year",
    "curriculum_id",
    "credits_required",
    "designed_semesters",
    "max_semesters",
)


class V32DataAccess:
    """Access approved V3.2 tables through a local data catalog."""

    def __init__(self, catalog_path=None):
        repo = Path(__file__).resolve().parents[1]

        if catalog_path is None:
            catalog_path = (
                repo / "data/processed/v3_2/dataset_catalog.json"
            )

        catalog_path = Path(catalog_path).resolve()

        with catalog_path.open("r", encoding="utf-8") as stream:
            self.catalog = json.load(stream)

        if self.catalog.get("dataset_version") != "3.2.0":
            raise ValueError("Unsupported dataset version")

        if self.catalog.get("scenario_id") != "V3_2_MAIN":
            raise ValueError("Unsupported dataset scenario")

        if self.catalog.get("synthetic") is not True:
            raise ValueError("Expected synthetic dataset")

        self.source_root = Path(
            self.catalog["source_root"]
        ).resolve()

        if not self.source_root.is_dir():
            raise FileNotFoundError(self.source_root)

    def _table_path(self, name):
        """Resolve only registered tables inside the source root."""
        tables = self.catalog["tables"]

        if name not in tables:
            raise ValueError(f"Table is not registered: {name}")

        relative = Path(tables[name]["file"])
        path = (self.source_root / relative).resolve()

        if not path.is_relative_to(self.source_root):
            raise ValueError("Unsafe table path")

        if not path.is_file():
            raise FileNotFoundError(path)

        return path

    @cached_property
    def _students_by_uid(self):
        """Load and validate students once per repository instance."""
        path = self._table_path("students")

        frame = pd.read_csv(
            path,
            dtype={
                "record_uid": "string",
                "student_id": "string",
            },
            low_memory=False,
        )

        validate_students(frame)

        expected_rows = int(
            self.catalog["tables"]["students"]["rows"]
        )

        if len(frame) != expected_rows or len(frame) != 20000:
            raise ValueError("Unexpected student count")

        indexed = frame.set_index(
            "record_uid",
            drop=False,
        )
        if not indexed.index.is_unique:
            raise ValueError("Duplicate record_uid index")
        return indexed

    def get_student(
        self, uid: str, asof=None
    ) -> dict | None:
        """
        Return approved student profile fields.

        A student is not visible before enrollment.
        Never expose generator metadata or raw master rows.
        """
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("uid must be a non-empty string")

        cutoff = self._effective_asof(asof)
        uid = uid.strip()

        try:
            record = self._students_by_uid.loc[uid]
        except KeyError:
            return None

        enrolled = pd.to_datetime(
            record["enrollment_date"], errors="coerce"
        )

        if pd.isna(enrolled):
            raise ValueError("Invalid enrollment_date")

        if enrolled > cutoff:
            return None

        missing = set(PROFILE_FIELDS) - set(record.index)

        if missing:
            raise ValueError(
                f"Missing profile fields: {sorted(missing)}"
            )

        approved = record.loc[list(PROFILE_FIELDS)]

        return self._row_to_dict(approved)


    @staticmethod
    def _row_to_dict(row):
        result = {}
        for key, value in row.items():
            if pd.isna(value):
                result[key] = None
            elif hasattr(value, "item"):
                result[key] = value.item()
            else:
                result[key] = value
        return result

    def _effective_asof(self, asof=None):
        """Restrict serving to the observed demo period."""
        maximum = pd.Timestamp(self.catalog["demo_as_of"])

        if asof is None:
            return maximum

        if not isinstance(asof, str):
            raise ValueError("asof must be YYYY-MM-DD")

        import re
        if not re.fullmatch("[0-9]{4}-[0-9]{2}-[0-9]{2}", asof):
            raise ValueError("asof must be YYYY-MM-DD")

        try:
            selected = pd.Timestamp(asof)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid asof date") from exc

        if selected > maximum:
            raise ValueError(
                "asof exceeds ASOF_2026_DEMO observation date"
            )

        return selected

    @cached_property
    def _snapshots_by_key(self):
        """Load validated features, excluding future demo snapshots."""
        path = self._table_path("prediction_snapshots")

        frame = pd.read_csv(
            path,
            dtype={
                "record_uid": "string",
                "student_id": "string",
            },
            low_memory=False,
        )

        from src.schemas import validate_snapshots
        validate_snapshots(
            frame,
            self._students_by_uid.reset_index(drop=True),
        )

        expected = int(
            self.catalog["tables"]["prediction_snapshots"]["rows"]
        )
        if len(frame) != expected or len(frame) != 57569:
            raise ValueError("Unexpected snapshot count")

        forbidden = {
            "dropout_next_2_terms",
            "label_observed",
            "censor_reason",
            "first_dropout_event_id",
            "exit_reason_for_evaluation",
            "split",
        }

        if forbidden.intersection(frame.columns):
            raise ValueError("Forbidden outcome columns in features")

        maximum = self._effective_asof()
        cutoff = pd.to_datetime(frame["snapshot_cutoff"])

        # Keep only observations available in the demo view.
        frame = frame.loc[cutoff <= maximum].copy()

        if len(frame) != 54609:
            raise ValueError("Unexpected demo snapshot count")

        indexed = frame.set_index(
            ["record_uid", "snapshot_term_index"],
            drop=False,
        ).sort_index()

        if not indexed.index.is_unique:
            raise ValueError("Duplicate snapshot composite keys")

        return indexed

    def list_snapshots(self, uid: str, asof=None) -> list[dict]:
        """Return snapshots available by asof, ordered by term."""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("uid must be non-empty")

        cutoff = self._effective_asof(asof)

        try:
            rows = self._snapshots_by_key.loc[uid.strip()]
        except KeyError:
            return []

        valid = rows.loc[
            pd.to_datetime(rows["snapshot_cutoff"]) <= cutoff
        ]

        valid = valid.sort_index()

        return [
            self._row_to_dict(row)
            for _, row in valid.iterrows()
        ]

    def get_snapshot(
        self, uid: str, term: int, asof=None
    ) -> dict | None:
        """Retrieve exactly one UID + term, never by row position."""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("uid must be non-empty")

        if type(term) is not int or term not in (1, 2, 3):
            raise ValueError("term must be 1, 2 or 3")

        cutoff = self._effective_asof(asof)

        try:
            row = self._snapshots_by_key.loc[(uid.strip(), term)]
        except KeyError:
            return None

        if pd.Timestamp(row["snapshot_cutoff"]) > cutoff:
            return None

        return self._row_to_dict(row)



    @cached_property
    def _term_academics_by_key(self):
        """Load approved academic history fields, indexed by UID + term."""
        from src.schemas import validate_term_academics

        fields = [
            "record_uid",
            "term_index",
            "calendar_term_id",
            "term_status",
            "registration_rule_status",
            "credits_registered_term",
            "credits_passed_term",
            "credits_failed_term",
            "credits_earned_cumulative",
            "credits_owed_cumulative",
            "courses_failed_term",
            "retake_count_cumulative",
            "gpa4_term",
            "gpa10_term",
            "gpa4_cumulative",
            "attendance_rate_term",
            "warning_current",
            "warnings_total_to_date",
            "warning_streak",
            "available_at",
            "snapshot_cutoff",
            "policy_id",
            "schema_version",
        ]

        frame = pd.read_csv(
            self._table_path("term_academics"),
            usecols=fields,
            dtype={"record_uid": "string"},
            low_memory=False,
        )

        validate_term_academics(
            frame,
            self._students_by_uid.reset_index(drop=True),
        )

        expected = int(
            self.catalog["tables"]["term_academics"]["rows"]
        )

        if len(frame) != expected:
            raise ValueError("Academic history row count mismatch")

        indexed = frame.set_index(
            ["record_uid", "term_index"],
            drop=False,
        ).sort_index()

        if not indexed.index.is_unique:
            raise ValueError("Duplicate academic history keys")

        return indexed

    def get_term_history(self, uid: str, asof=None) -> list[dict]:
        """Return only academic terms available at the requested date."""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("uid must be a non-empty string")

        cutoff = self._effective_asof(asof)

        try:
            rows = self._term_academics_by_key.loc[uid.strip()]
        except KeyError:
            return []

        available = pd.to_datetime(
            rows["available_at"], errors="raise"
        )
        published = pd.to_datetime(
            rows["snapshot_cutoff"], errors="raise"
        )

        visible = rows.loc[
            (available <= cutoff) & (published <= cutoff)
        ].sort_index()

        return [
            self._row_to_dict(row)
            for _, row in visible.iterrows()
        ]



    @lru_cache(maxsize=64)
    def _weekly_rows_for_uid(self, uid):
        """Read weekly data in chunks and cache one UID only."""
        columns = [
            "record_uid",
            "term_index",
            "teaching_week",
            "lms_login_count",
            "lms_active_days",
            "assignments_due",
            "assignments_submitted",
            "assignments_submitted_on_time",
            "course_content_views",
            "attendance_rate_week",
            "concern_reported_week",
            "observed_at",
            "available_at",
            "calendar_term_id",
        ]

        chunks = pd.read_csv(
            self._table_path("engagement_weekly"),
            usecols=columns,
            dtype={
                "record_uid": "string",
                "calendar_term_id": "string",
            },
            chunksize=100_000,
            low_memory=False,
        )

        found = []
        total = 0

        for chunk in chunks:
            total += len(chunk)

            selected = chunk.loc[
                chunk["record_uid"].eq(uid)
            ].copy()

            if not selected.empty:
                found.append(selected)

        expected = int(
            self.catalog["tables"]["engagement_weekly"]["rows"]
        )

        if total != expected or total != 1790352:
            raise ValueError("Weekly dataset row count mismatch")

        if not found:
            return pd.DataFrame(columns=columns)

        frame = pd.concat(found, ignore_index=True)

        keys = ["record_uid", "term_index", "teaching_week"]

        if frame[keys].isna().any(axis=1).any():
            raise ValueError("Null weekly key")

        if frame.duplicated(keys).any():
            raise ValueError("Duplicate weekly composite key")

        if not frame["teaching_week"].between(1, 12).all():
            raise ValueError("Invalid teaching week")

        if not frame["term_index"].between(1, 20).all():
            raise ValueError("Invalid academic term")

        if not frame["attendance_rate_week"].between(
            0, 100
        ).all():
            raise ValueError("Invalid weekly attendance rate")

        observed = pd.to_datetime(
            frame["observed_at"], errors="coerce"
        )
        available = pd.to_datetime(
            frame["available_at"], errors="coerce"
        )

        if (
            observed.isna().any()
            or available.isna().any()
            or (observed > available).any()
        ):
            raise ValueError("Invalid weekly timestamps")

        return frame

    def get_weekly_history(
        self, uid: str, term: int, asof=None
    ) -> list[dict]:
        """Return published weekly activity for one UID and term."""
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("uid must be a non-empty string")

        if type(term) is not int or not 1 <= term <= 20:
            raise ValueError("term must be an integer from 1 to 20")

        cutoff = self._effective_asof(asof)
        uid = uid.strip()

        if uid not in self._students_by_uid.index:
            return []

        rows = self._weekly_rows_for_uid(uid)

        if rows.empty:
            return []

        selected = rows.loc[
            rows["term_index"].eq(term)
        ].copy()

        if selected.empty:
            return []

        observed = pd.to_datetime(
            selected["observed_at"], errors="raise"
        )
        available = pd.to_datetime(
            selected["available_at"], errors="raise"
        )

        visible = selected.loc[
            (observed <= cutoff) & (available <= cutoff)
        ].sort_values("teaching_week")

        return [
            self._row_to_dict(row)
            for _, row in visible.iterrows()
        ]


def smoke_test():
    repository = V32DataAccess()

    students = repository._students_by_uid

    assert len(students) == 20000
    print("PASS: Catalog and 20,000 student records loaded")

    uid = str(students.index[0])

    student = repository.get_student(uid)

    assert student is not None
    assert student["record_uid"] == uid
    assert "student_id" in student

    print("PASS: get_student returns correct record_uid")

    missing = repository.get_student("__NOT_A_REAL_UID__")

    assert missing is None
    print("PASS: Unknown UID returns None")


    print("V3.2 DATA ACCESS B1 PASS")

    snapshots = repository.list_snapshots(uid)
    assert len(snapshots) == 3
    assert [s["snapshot_term_index"] for s in snapshots] == [1, 2, 3]
    print("PASS: Three snapshots, correctly ordered")

    first = repository.get_snapshot(uid, 1)
    assert first is not None
    assert first["record_uid"] == uid
    assert first["snapshot_term_index"] == 1
    print("PASS: Composite-key snapshot lookup")

    first_date = pd.Timestamp(first["snapshot_cutoff"])
    earlier = (first_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert repository.list_snapshots(uid, asof=earlier) == []
    assert repository.get_snapshot(uid, 1, asof=earlier) is None
    print("PASS: Historical cutoff hides future snapshots")

    uid_2025 = str(
        students.loc[students["cohort_year"].eq(2025)].index[0]
    )
    assert repository.get_snapshot(uid_2025, 3) is None
    print("PASS: 2027 snapshot hidden from 2026 demo")

    try:
        repository.list_snapshots(uid, asof="2027-03-01")
    except ValueError:
        pass
    else:
        raise AssertionError("Future asof was not rejected")

    assert repository.list_snapshots("__NOT_A_REAL_UID__") == []
    print("PASS: Future asof rejected; unknown UID safe")

    print("V3.2 DATA ACCESS B2 PASS")

    history = repository.get_term_history(uid)

    assert len(history) == 8
    assert [h["term_index"] for h in history] == list(range(1, 9))
    print("PASS: Eight observed academic terms ordered correctly")

    first_release = history[0]["available_at"]

    at_first = repository.get_term_history(
        uid, asof=first_release
    )
    assert len(at_first) == 1
    print("PASS: First released term is visible")

    before_first = (
        pd.Timestamp(first_release) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    assert repository.get_term_history(
        uid, asof=before_first
    ) == []
    print("PASS: Unreleased academic terms are hidden")

    assert all(
        pd.Timestamp(h["available_at"]) <= pd.Timestamp("2026-10-01")
        and pd.Timestamp(h["snapshot_cutoff"]) <= pd.Timestamp("2026-10-01")
        for h in history
    )
    print("PASS: No future academic records in demo view")

    assert "registration_rule_status" in history[0]
    assert "part_time_work_hours_week" not in history[0]
    print("PASS: Academic fields restricted to approved whitelist")

    assert repository.get_term_history(
        "__NOT_A_REAL_UID__"
    ) == []
    print("PASS: Unknown UID returns empty academic history")


    print("V3.2 DATA ACCESS B3 PASS")

    weekly = repository.get_weekly_history(uid, 1)

    assert len(weekly) == 12
    assert [w["teaching_week"] for w in weekly] == list(
        range(1, 13)
    )
    print("PASS: Twelve weekly records ordered correctly")

    assert all(
        w["record_uid"] == uid and w["term_index"] == 1
        for w in weekly
    )
    print("PASS: Correct UID and academic term")

    first_date = weekly[0]["available_at"]

    at_first = repository.get_weekly_history(
        uid, 1, asof=first_date
    )
    assert len(at_first) == 1
    print("PASS: Only first available week is visible")

    before_first = (
        pd.Timestamp(first_date) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    assert repository.get_weekly_history(
        uid, 1, asof=before_first
    ) == []
    print("PASS: Unpublished weekly records are hidden")

    assert repository.get_weekly_history(uid, 9) == []
    print("PASS: Future term 9 is hidden in 2026 demo")

    assert repository.get_weekly_history(
        "__NOT_A_REAL_UID__", 1
    ) == []
    print("PASS: Unknown UID returns empty history")

    assert "lms_login_count" in weekly[0]
    assert "attendance_rate_week" in weekly[0]
    assert "dropout_next_2_terms" not in weekly[0]
    print("PASS: Weekly fields contain no dropout labels")

    try:
        repository.get_weekly_history(uid, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid term was accepted")

    print("PASS: Invalid term rejected")
    print("V3.2 DATA ACCESS B4 PASS")


if __name__ == "__main__":
    smoke_test()
