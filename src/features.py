"""Frozen B5 feature contract for CMC Synthetic Dataset V3.2.

These are feature columns only. Never include identity, splits or outcomes.
"""

ACADEMIC = [
    "gpa4_term", "gpa4_cumulative", "gpa4_previous", "gpa_delta_1term",
    "gpa_mean_last3", "gpa_slope_last3", "credits_registered_term",
    "credits_failed_term", "failed_credit_ratio", "credits_owed_cumulative",
    "credits_owed_delta", "credits_earned_cumulative",
    "curriculum_progress_ratio", "courses_failed_term",
    "courses_failed_total_to_date", "retake_count_to_date",
    "warning_current", "warnings_total_to_date", "warning_streak",
    "counted_duration_semesters",
]

BEHAVIOR = [
    "attendance_rate_term", "unexcused_absence_rate_term", "attendance_delta_1term",
    "lms_logins_4weeks", "lms_active_days_4weeks",
    "days_since_last_lms_activity", "assignment_on_time_rate_term",
    "assignment_completion_rate_term", "lms_activity_delta_last4w",
    "lms_decline_ratio_4w", "assignment_miss_streak_weeks",
    "process_score10_mean_term", "process_failure_course_rate",
    "lms_slope_last6w", "attendance_slope_last4w",
    "assignment_missed_streak", "longest_inactive_streak_6w",
    "observed_concern_reports_term", "lms_first6_mean", "lms_last6_mean",
]

FINANCE = [
    "tuition_debt_to_due_ratio", "tuition_debt_end_vnd",
    "tuition_overdue_days_end", "tuition_overdue_terms_to_date",
]

SUPPORT = [
    "advisor_contacts_to_date", "tutoring_sessions_to_date",
    "academic_recovery_flag", "leave_semesters_to_date",
]

CONTEXT = [
    "major_code", "degree_type", "education_mode", "credits_required",
    "admission_method", "admission_score_normalized", "enrollment_age",
    "living_status_at_cutoff", "scholarship_percent_at_cutoff",
    "snapshot_term_index",
]

B5_FEATURES = ACADEMIC + BEHAVIOR + FINANCE + SUPPORT + CONTEXT
B5_FEATURE_COUNT = 58

if len(B5_FEATURES) != B5_FEATURE_COUNT or len(set(B5_FEATURES)) != B5_FEATURE_COUNT:
    raise ValueError("B5 feature contract must contain 58 unique columns")
