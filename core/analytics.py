"""Thống kê học vụ và độ bao phủ của dự báo."""
import pandas as pd

def build_analytics_summary(df_all: pd.DataFrame, df_ca: pd.DataFrame = None, threshold=0.21, snapshot_count=None) -> dict:
    latest = df_all.sort_values('snapshot_term_index').drop_duplicates('record_uid', keep='last').copy()
    latest['warn_bool'] = latest['warning_current'].astype(str).str.lower() == 'true'
    latest['high_risk_bool'] = (latest['risk_probability'] >= threshold) & latest['state_as_of'].eq('ENROLLED') & latest['prediction_status'].eq('VALID')
    latest['dropout_bool'] = latest['state_as_of'].eq('DROPOUT')

    active = latest.state_as_of.eq('ENROLLED')
    valid = active & latest.prediction_status.eq('VALID')
    summary = {
        "total_students": int(len(latest)),
        "active_students": int(latest['state_as_of'].eq('ENROLLED').sum()),
        "state_counts": {str(k): int(v) for k, v in latest.state_as_of.value_counts().items()},
        "data_as_of": str(latest['data_as_of'].iloc[0]),
        "data_mode": str(latest['data_mode'].iloc[0]),
        "total_academic_terms": int(len(df_all)),
        "total_snapshots": int(snapshot_count) if snapshot_count is not None else None,
        "valid_predictions": int(valid.sum()),
        "coverage": float(valid.sum()/active.sum()) if active.any() else None,
        "active_tuition_debt_vnd": float(latest.loc[active,'tuition_debt_end_vnd'].sum()),
        "total_classes": int(latest['class_code'].nunique()),
        "total_majors": int(latest['major_code'].nunique()),
        "total_tuition_debt_vnd": float(latest['tuition_debt_end_vnd'].sum()),
        "overdue_30d_students": int((latest.get('tuition_overdue_days_end', pd.Series(0, index=latest.index)).fillna(0) > 30).sum()),
        "warning_streak_2_students": int((latest['warning_streak'].fillna(0) >= 2).sum()),
        "owed_near_24_students": int(((latest['credits_owed_cumulative'].fillna(0) >= 20) & (latest['credits_owed_cumulative'].fillna(0) <= 24)).sum()),
        "owed_over_24_students": int((latest['credits_owed_cumulative'].fillna(0) > 24).sum()),
        "unexcused_over_20_students": int((latest.get('unexcused_absence_rate_term', pd.Series(0, index=latest.index)).fillna(0) > 20).sum()),
        "attendance_under_80_students": int((latest['attendance_rate_term'].fillna(100) < 80).sum()),
        "high_risk_no_advisor": int(((latest['high_risk_bool']) & (latest.get('advisor_contacts_to_date', pd.Series(0, index=latest.index)).fillna(0) == 0)).sum()),
        "high_risk_no_tutoring": int(((latest['high_risk_bool']) & (latest.get('tutoring_sessions_to_date', pd.Series(0, index=latest.index)).fillna(0) == 0)).sum()),
        "under_min_credits_students": int((latest['credits_registered_term'] < latest['registration_min_credits']).sum())
    }

    # Thống kê theo khóa tuyển sinh.
    cohorts = {}
    for c_year, grp in latest.groupby('cohort_year'):
        hk1 = df_all[(df_all['cohort_year'] == c_year) & (df_all['snapshot_term_index'] == 1)]
        att_u80_hk1 = round(float((hk1['attendance_rate_term'] < 80).mean() * 100), 2) if len(hk1) else 0.0
        cohorts[str(int(c_year))] = {
            "n_students": int(len(grp)),
            "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
            "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
            "high_risk_count": int(grp['high_risk_bool'].sum()),
            "high_risk_pct": round(float(grp['high_risk_bool'].mean() * 100), 2),
            "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2),
            "hk1_att_under_80_pct": att_u80_hk1,
            "leave_students": int((grp.get('leave_semesters_to_date', pd.Series(0, index=grp.index)).fillna(0) > 0).sum())
        }
    summary["by_cohort"] = cohorts

    # Thống kê theo hệ đào tạo.
    modes = {}
    for m_code, grp in latest.groupby('education_mode'):
        modes[str(m_code)] = {
            "n_students": int(len(grp)),
            "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
            "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
            "high_risk_pct": round(float(grp['high_risk_bool'].mean() * 100), 2),
            "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2)
        }
    summary["by_education_mode"] = modes

    # Thống kê theo bậc đào tạo.
    degrees = {}
    for d_code, grp in latest.groupby('degree_type'):
        degrees[str(d_code)] = {
            "n_students": int(len(grp)),
            "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
            "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
            "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2)
        }
    summary["by_degree_type"] = degrees

    # Thống kê theo nơi ở.
    living = {}
    if 'living_status' in latest.columns:
        for l_val, grp in latest.groupby('living_status'):
            living[str(l_val)] = {
                "n_students": int(len(grp)),
                "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
                "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
                "high_risk_pct": round(float(grp['high_risk_bool'].mean() * 100), 2),
                "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2)
            }
    summary["by_living_status"] = living

    # Thống kê theo phương thức tuyển sinh.
    adm = {}
    if 'admission_method' in latest.columns:
        for a_val, grp in latest.groupby('admission_method'):
            adm[str(a_val)] = {
                "n_students": int(len(grp)),
                "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
                "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
                "high_risk_pct": round(float(grp['high_risk_bool'].mean() * 100), 2),
                "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2)
            }
    summary["by_admission_method"] = adm

    # Không suy đoán nguyên nhân thôi học.
    exit_reasons = {}
    summary["exit_reasons"] = exit_reasons

    # Phổ điểm và tỷ lệ trượt theo học phần.
    grade_dist = {}
    top_failed_courses = []
    ineligible_count = 0
    dead_point_pct = 0.0
    if df_ca is not None and not df_ca.empty:
        total_ca = max(1, len(df_ca))
        if 'grade_letter' in df_ca.columns:
            for g_let, cnt in df_ca['grade_letter'].value_counts().items():
                grade_dist[str(g_let)] = round(float(cnt / total_ca * 100), 2)
        if 'exam_eligible' in df_ca.columns:
            ineligible_count = int((~df_ca['exam_eligible'].astype(bool)).sum())
        if 'exam_final_score10' in df_ca.columns:
            dead_point_pct = round(float((df_ca['exam_final_score10'].fillna(10) <= 2.0).mean() * 100), 2)
        if 'is_passed' in df_ca.columns and 'course_id' in df_ca.columns:
            c_grp = df_ca.groupby('course_id', as_index=False).agg(
                attempts=('course_id', 'count'),
                fail_rate=('is_passed', lambda s: float((~s.astype(bool)).mean() * 100)),
                fail_count=('is_passed', lambda s: int((~s.astype(bool)).sum()))
            )
            c_grp = c_grp[c_grp['attempts'] >= 30].sort_values('fail_rate', ascending=False).head(15)
            for _, cr in c_grp.iterrows():
                top_failed_courses.append({
                    "course_id": str(cr['course_id']),
                    "attempts": int(cr['attempts']),
                    "fail_count": int(cr['fail_count']),
                    "fail_rate_pct": round(float(cr['fail_rate']), 1)
                })
    summary["grade_distribution_pct"] = grade_dist
    summary["ineligible_exam_attempts"] = ineligible_count
    summary["exam_dead_point_pct"] = dead_point_pct
    summary["top_failed_courses"] = top_failed_courses
    return summary
