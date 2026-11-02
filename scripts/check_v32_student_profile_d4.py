"""Smoke tests for the V3.2 student profile serving contract."""

import pandas as pd

from src.data_access import V32DataAccess, PROFILE_FIELDS


def check(condition, message):
    if not bool(condition):
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def main():
    repo = V32DataAccess()
    students = repo._students_by_uid

    check(
        len(PROFILE_FIELDS) == 16
        and len(set(PROFILE_FIELDS)) == 16,
        "Exactly 16 approved profile fields",
    )

    forbidden = {
        "gender",
        "birth_year",
        "home_province",
        "living_status",
        "admission_score",
        "admission_score_normalized",
        "admission_score_missing_reason",
        "is_counterfactual_program",
        "sampling_design",
        "source_type",
        "schema_version",
        "scenario_id",
        "dropout_next_2_terms",
        "exit_reason_for_evaluation",
    }

    check(
        not forbidden.intersection(PROFILE_FIELDS),
        "Sensitive, audit and outcome fields excluded",
    )

    for year in (2022, 2023, 2024, 2025):
        uid = str(
            students.loc[
                students["cohort_year"].eq(year)
            ].index[0]
        )

        profile = repo.get_student(uid)

        check(
            profile is not None
            and profile["record_uid"] == uid
            and set(profile) == set(PROFILE_FIELDS)
            and not forbidden.intersection(profile),
            f"Cohort {year}: correct whitelisted profile",
        )

        enrolled = profile["enrollment_date"]

        before = (
            pd.Timestamp(enrolled)
            - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        check(
            repo.get_student(uid, asof=before) is None
            and repo.get_student(uid, asof=enrolled)
                == profile,
            f"Cohort {year}: enrollment cutoff enforced",
        )

    uid_2025 = str(
        students.loc[
            students["cohort_year"].eq(2025)
        ].index[0]
    )

    check(
        repo.get_student(
            uid_2025, asof="2024-01-01"
        ) is None,
        "2025 student hidden in 2024 historical view",
    )

    check(
        repo.get_student("__NOT_A_REAL_UID__") is None,
        "Unknown UID returns None",
    )

    try:
        repo.get_student(uid_2025, asof="2027-01-01")
    except ValueError:
        print("PASS: Future asof rejected")
    else:
        raise AssertionError(
            "FAIL: Future asof was accepted"
        )

    print("\nV3.2 STUDENT PROFILE D4 PASS")


if __name__ == "__main__":
    main()
