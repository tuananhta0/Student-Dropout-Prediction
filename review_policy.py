"""Review actual MAIN source against QĐ22 and published admission methods.
No mutation of dataset, feature values or model artifacts.
"""
from pathlib import Path
import json
import numpy as np
from data_pipeline import Dataset, ROOT, gpa_ledger

METHODS = {
    2022: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang','CMC402_HSA_150','CMC402_VNUHCM_1200'},
    2023: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang'},
    2024: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang','CMC410','CMC409'},
    2025: {'CMC100_THPT_40','CMC200_HocBa_40','CMC303_TuyenThang','CMC401_CMCTEST_80'},
}


def main():
    source = Dataset()
    output = ROOT / 'data/phase3_review'
    output.mkdir(parents=True, exist_ok=True)
    t = source.table('term_academics').sort_values(['record_uid','term_index']).copy()
    s = source.table('students')
    a = source.table('course_attempts')
    report = {}
    t = t.merge(s[['record_uid','designed_semesters']],on='record_uid',validate='many_to_one')
    low = np.ceil(2*t.credits_required/(3*t.designed_semesters))
    high = np.floor(3*t.credits_required/(2*t.designed_semesters))
    assert (low == t.registration_min_credits).all() and (high == t.registration_max_credits).all()
    t['remaining_before'] = t.credits_required-t.groupby('record_uid').credits_earned_cumulative.shift().fillna(0)
    ex = t.loc[(t.credits_registered_term < low)|(t.credits_registered_term > high)].copy()
    final = (ex.remaining_before < ex.registration_min_credits) & (ex.credits_registered_term <= ex.remaining_before)
    ex['review_status'] = np.where(final,'FINAL_REMAINDER_PENDING_APPROVAL','LOW_LOAD_PENDING_APPROVAL')
    ex[['record_uid','term_index','remaining_before','credits_registered_term','registration_min_credits','registration_max_credits','review_status']].to_csv(output/'registration.csv',index=False)
    report['registration'] = ex.review_status.value_counts().to_dict()
    barred = (5*a.unexcused_absent_hours > a.scheduled_hours) | (10*(a.unexcused_absent_hours+a.excused_absent_hours)>3*a.scheduled_hours)
    exam = a.loc[barred.eq(a.exam_eligible)].copy()
    exam['review_status'] = 'ATTENDANCE_DOES_NOT_EXPLAIN_RECORDED_STATUS'
    exam.to_csv(output/'exam.csv',index=False)
    report['exam_reviews'] = len(exam)
    # Source scores have two decimal places. Round half-up to one before grading.
    hundredths = np.rint(a.course_final_score10*100).astype(int)
    rounded = ((hundredths+5)//10)/10
    expected = np.select([~a.exam_eligible|a.exam_final_score10.le(2),rounded>=9,rounded>=8.5,rounded>=8,rounded>=7,rounded>=6.5,rounded>=5.5,rounded>=5,rounded>=4],['F','A+','A','B+','B','C+','C','D+','D'],default='F')
    grades = a.loc[expected!=a.grade_letter].copy()
    grades['expected_letter_after_rounding'] = expected[expected!=a.grade_letter]
    grades.to_csv(output/'grade_rounding.csv',index=False)
    report['grade_rounding_reviews'] = len(grades)
    ledger = gpa_ledger(source).merge(t,left_on=['record_uid','snapshot_term_index'],right_on=['record_uid','term_index'],validate='one_to_one')
    error = np.abs(ledger.gpa_quality_points/ledger.gpa_credit_count-ledger.gpa4_cumulative)
    report['ledger_max_error'] = float(error.max())
    assert len(ledger)==len(t) and error.le(1e-7).all()
    compatible = s.apply(lambda row: row.admission_method in METHODS.get(row.cohort_year,set()),axis=1)
    report['admission_method_mismatch'] = int((~compatible).sum())
    assert compatible.all()
    s.loc[~compatible].to_csv(output/'admission_method_mismatch.csv',index=False)
    direct = s.admission_method.eq('CMC303_TuyenThang')
    missing = s.admission_score.isna() & ~direct
    assert s.loc[missing,'admission_score_missing_reason'].eq('SYNTHETIC_SOURCE_MISSING').all()
    report['missing_non_direct'] = int(missing.sum())
    report['direct_not_applicable'] = int((direct & s.admission_score.isna()).sum())
    s.loc[missing,['record_uid','cohort_year','admission_method','admission_score_missing_reason']].to_csv(output/'admission_missing.csv',index=False)
    s.groupby(['cohort_year','admission_method']).admission_score.agg(['size','count','min','max']).to_csv(output/'admission_method_summary.csv')
    # Only explicitly published, comparable floors. A review is not a rejection decision.
    floor = np.full(len(s), np.nan)
    for year, method, major, minimum in [
        (2022,'CMC100_THPT_30',None,17),(2022,'CMC200_HocBa_30',None,18),
        (2022,'CMC402_HSA_150',None,80),(2022,'CMC402_VNUHCM_1200',None,650),
        (2023,'CMC100_THPT_30',None,18),(2023,'CMC200_HocBa_30','IT',20),
        (2024,'CMC100_THPT_30',None,18),(2024,'CMC100_THPT_30','IT',20),
        (2024,'CMC100_THPT_30','CS',20),(2024,'CMC100_THPT_30','EC',20),
        (2025,'CMC100_THPT_40','IT',24),(2025,'CMC200_HocBa_40','IT',26),
        (2025,'CMC401_CMCTEST_80','IT',48),
    ]:
        mask = s.cohort_year.eq(year) & s.admission_method.eq(method)
        if major is not None:
            mask &= s.major_code.eq(major)
        floor[mask] = minimum
    comparable = s.admission_score.notna() & np.isfinite(floor)
    below = comparable & s.admission_score.lt(floor)
    out = s.loc[below,['record_uid','cohort_year','major_code','admission_method','admission_score']].copy()
    out['published_floor'] = floor[below]
    out.to_csv(output/'admission_floor_review.csv',index=False)
    report['admission_scores_with_comparable_floor'] = int(comparable.sum())
    report['admission_below_published_floor'] = int(below.sum())
    report['scope'] = 'Methods, scales and selected published comparable floors; no individual admission or exception approval inferred.'
    (output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2,default=int))


if __name__ == '__main__':
    main()
