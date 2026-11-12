import os
import json
import joblib
import pypdf
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score, precision_score, recall_score

DATASET_ROOT = Path("dataset_full") if (Path("dataset_full") / "artifacts").exists() else Path("..")
MAIN_RUN = DATASET_ROOT / "artifacts" / "runs" / "V3_2_MAIN"
OUTPUT_DIR = Path("artifacts_model")
OUTPUT_DIR.mkdir(exist_ok=True)

def find_file(rel_path: str, fallback_name: str = "") -> Path:
    candidates = [
        DATASET_ROOT / rel_path,
        MAIN_RUN / rel_path,
        Path("data") / (fallback_name or Path(rel_path).name),
        Path(rel_path)
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    if DATASET_ROOT.exists():
        matches = list(DATASET_ROOT.rglob(fallback_name or Path(rel_path).name))
        if matches:
            return matches[0]
    return Path("")

#tong hop thong ke toan truong va khoa luu vao analytics_summary.json
def build_analytics_summary(df_all: pd.DataFrame, df_ca: pd.DataFrame = None) -> dict:
    latest = df_all.sort_values('snapshot_term_index').drop_duplicates('record_uid', keep='last').copy()
    latest['warn_bool'] = latest['warning_current'].astype(str).str.lower() == 'true'
    latest['high_risk_bool'] = latest['risk_probability'] >= 0.21
    latest['dropout_bool'] = latest.get('dropout_next_2_terms', pd.Series(0, index=latest.index)).astype(float) > 0

    summary = {
        "total_students": int(len(latest)),
        "total_snapshots": int(len(df_all)),
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
        "under_min_credits_students": int((latest['credits_registered_term'].fillna(15) < 10).sum())
    }

    #thong ke theo khoa / khoi tuyen sinh (2022-2025)
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

    #thong ke theo he dao tao GM / SM
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

    #thong ke theo bac dao tao Bachelor / Engineer
    degrees = {}
    for d_code, grp in latest.groupby('degree_type'):
        degrees[str(d_code)] = {
            "n_students": int(len(grp)),
            "gpa_mean": round(float(grp['gpa4_cumulative'].mean()), 2),
            "warning_rate_pct": round(float(grp['warn_bool'].mean() * 100), 2),
            "dropout_rate_pct": round(float(grp['dropout_bool'].mean() * 100), 2)
        }
    summary["by_degree_type"] = degrees

    #thong ke theo noi o (living_status)
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

    #thong ke theo phuong thuc tuyen sinh (admission_method)
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

    #thong ke nguyen nhan thoi hoc
    exit_reasons = {}
    if 'exit_reason_for_evaluation' in latest.columns:
        ex_sub = latest[latest['exit_reason_for_evaluation'].notna() & (latest['exit_reason_for_evaluation'].astype(str) != 'nan') & (latest['exit_reason_for_evaluation'].astype(str) != '')]
        for r_val, cnt in ex_sub['exit_reason_for_evaluation'].value_counts().items():
            exit_reasons[str(r_val)] = int(cnt)
    summary["exit_reasons"] = exit_reasons

    #pho diem toan truong va ty le truot mon tu course_attempts
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

def main():
    ps_path = find_file("features/prediction_snapshots.csv.gz", "prediction_snapshots.csv.gz")
    if not ps_path.exists():
        #neu chay tren may chi co artifacts_model (khong co dataset_full)
        scored_path = OUTPUT_DIR / "student_snapshots_scored.csv.gz"
        if scored_path.exists():
            print("Đang tạo analytics_summary.json từ student_snapshots_scored.csv.gz...")
            df_scored = pd.read_csv(scored_path, low_memory=False)
            summary = build_analytics_summary(df_scored, None)
            with open(OUTPUT_DIR / "analytics_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print("Đã cập nhật xong artifacts_model/analytics_summary.json!")
            return
        raise FileNotFoundError("Không tìm thấy dữ liệu nguồn.")

    print("Đang nạp dữ liệu từ dataset_full...")
    lf_path = find_file("labels/snapshot_labels_full_simulation.csv.gz", "snapshot_labels_full_simulation.csv.gz")
    fr_path = find_file("docs/feature_registry.csv", "feature_registry.csv")
    stu_path = find_file("master/students.csv.gz", "students.csv.gz")
    ca_path = find_file("master/course_attempts.csv.gz", "course_attempts.csv.gz")
    req_path = find_file("master/academic_requirement_events.csv.gz", "academic_requirement_events.csv.gz")

    df_X = pd.read_csv(ps_path, low_memory=False)
    df_y = pd.read_csv(lf_path, low_memory=False)
    df_fr = pd.read_csv(fr_path)
    df_stu = pd.read_csv(stu_path, low_memory=False)
    df_ca = pd.read_csv(ca_path, low_memory=False)
    df_req = pd.read_csv(req_path)

    ca_agg = df_ca.groupby(['record_uid', 'term_index'], as_index=False).agg(
        ca_score10_min=('course_final_score10', 'min'),
        ca_exam10_min=('exam_final_score10', 'min'),
        ca_ineligible_exams=('exam_eligible', lambda s: int((~s.astype(bool)).sum())),
        ca_unexcused_hours_sum=('unexcused_absent_hours', 'sum')
    )
    req_map = df_req.groupby('record_uid')['requirement_type'].apply(lambda s: ", ".join(sorted(set(s)))).to_dict()

    df = pd.merge(
        df_X,
        df_y[['record_uid', 'snapshot_term_index', 'dropout_next_2_terms', 'exit_reason_for_evaluation']],
        on=['record_uid', 'snapshot_term_index']
    )
    df = pd.merge(
        df,
        ca_agg,
        left_on=['record_uid', 'snapshot_term_index'],
        right_on=['record_uid', 'term_index'],
        how='left'
    ).drop(columns=['term_index'], errors='ignore')

    allowed = df_fr[df_fr['role'] == 'FEATURE_ALLOWED_ASOF']['feature_name'].tolist()
    extra_master_cols = ['ca_score10_min', 'ca_exam10_min', 'ca_ineligible_exams', 'ca_unexcused_hours_sum']
    exclude = ['record_uid', 'student_id', 'cohort_year', 'snapshot_cutoff', 'feature_source_latest_at']
    feature_cols = [c for c in (allowed + extra_master_cols) if c in df.columns and c not in exclude]

    train_df = df[df['cohort_year'].isin([2022, 2023])].copy()
    val_df = df[df['cohort_year'] == 2024].copy()
    test_df = df[df['cohort_year'] == 2025].copy()

    X_train, y_train = train_df[feature_cols].copy(), train_df['dropout_next_2_terms'].astype(int)
    X_val, y_val = val_df[feature_cols].copy(), val_df['dropout_next_2_terms'].astype(int)
    X_test, y_test = test_df[feature_cols].copy(), test_df['dropout_next_2_terms'].astype(int)

    cat_cols = X_train.select_dtypes(include=['object', 'str', 'bool']).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)
        X_test[c] = X_test[c].astype(str)
        df[c] = df[c].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ]
    )

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=221026, n_jobs=-1)
    gb_clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=5, random_state=221026)

    ensemble_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', VotingClassifier(estimators=[('rf', rf_clf), ('gb', gb_clf)], voting='soft', weights=[0.55, 0.45]))
    ])
    ensemble_pipeline.fit(X_train, y_train)

    val_probs = ensemble_pipeline.predict_proba(X_val)[:, 1]
    test_probs = ensemble_pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.21
    test_preds = (test_probs >= threshold).astype(int)

    metrics = {
        "val_pr_auc": round(float(average_precision_score(y_val, val_probs)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "test_brier": round(float(brier_score_loss(y_test, test_probs)), 4),
        "test_precision_at_021": round(float(precision_score(y_test, test_preds)), 4),
        "test_recall_at_021": round(float(recall_score(y_test, test_preds)), 4),
        "test_f1_at_021": round(float(f1_score(y_test, test_preds)), 4),
        "threshold": threshold,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "num_medians": {c: float(train_df[c].median()) for c in num_cols}
    }

    df['risk_probability'] = ensemble_pipeline.predict_proba(df[feature_cols])[:, 1]
    df['risk_level'] = pd.cut(
        df['risk_probability'],
        bins=[-0.01, 0.12, 0.21, 0.45, 1.01],
        labels=['Thấp (An toàn)', 'Trung bình (Theo dõi)', 'Cao (Cảnh báo)', 'Rất cao (Can thiệp khẩn)']
    ).astype(str)

    stu_extra_cols = [c for c in ['record_uid', 'major_name', 'gender', 'home_province', 'enrollment_date', 'class_code', 'living_status', 'admission_method', 'admission_score_ratio', 'current_state'] if c in df_stu.columns]
    df_stu_sub = df_stu[stu_extra_cols].drop_duplicates('record_uid')
    drop_existing = [c for c in stu_extra_cols if c != 'record_uid' and c in df.columns]
    df_full_index = pd.merge(df.drop(columns=drop_existing, errors='ignore'), df_stu_sub, on='record_uid', how='left')
    df_full_index['verified_certificates'] = df_full_index['record_uid'].map(lambda u: req_map.get(u, "Chưa hoàn thành"))

    joblib.dump(ensemble_pipeline, OUTPUT_DIR / "dropout_pipeline.joblib")
    with open(OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    df_full_index.to_csv(OUTPUT_DIR / "student_snapshots_scored.csv.gz", index=False, compression="gzip")

    summary = build_analytics_summary(df_full_index, df_ca)
    with open(OUTPUT_DIR / "analytics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Hoàn tất! PR-AUC={metrics['test_pr_auc']} | ROC-AUC={metrics['test_roc_auc']}")

if __name__ == "__main__":
    main()