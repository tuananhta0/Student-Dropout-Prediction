"""Đối chiếu dữ liệu MAIN với QĐ 22 và phương thức tuyển sinh đã công bố.
Không thay đổi dữ liệu, đặc trưng hoặc artifact mô hình.
"""
from pathlib import Path
import json
import numpy as np
from core.data_pipeline import Dataset, ROOT, gpa_ledger, admission_evidence

METHODS = {
    2022: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang','CMC402_HSA_150','CMC402_VNUHCM_1200'},
    2023: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang'},
    2024: {'CMC100_THPT_30','CMC200_HocBa_30','CMC303_TuyenThang','CMC410','CMC409'},
    2025: {'CMC100_THPT_40','CMC200_HocBa_40','CMC303_TuyenThang','CMC401_CMCTEST_80'},
}


def main():
    source = Dataset()
    output = ROOT / 'data/doi_chieu_quy_che'
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
    ex['review_status'] = np.where(final,'FINAL_REMAINDER_SIMULATION_EXCEPTION','LOW_LOAD_UNVERIFIED')
    ex[['record_uid','term_index','remaining_before','credits_registered_term','registration_min_credits','registration_max_credits','review_status']].to_csv(output/'registration.csv',index=False)
    report['registration'] = ex.review_status.value_counts().to_dict()
    u = np.rint(a.unexcused_absent_hours*10000).astype('int64'); e = np.rint(a.excused_absent_hours*10000).astype('int64'); h = np.rint(a.scheduled_hours*10000).astype('int64')
    barred = (5*u > h) | (10*(u+e)>3*h)
    exam = a.loc[barred.eq(a.exam_eligible)].copy()
    exam['review_status'] = 'ATTENDANCE_DOES_NOT_EXPLAIN_RECORDED_STATUS'
    exam.to_csv(output/'exam.csv',index=False)
    report['exam_reviews'] = len(exam)
    # Làm tròn điểm gốc đến một chữ số thập phân trước khi quy đổi.
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
    assert s.loc[missing,'admission_score_missing_reason'].isin(['SYNTHETIC_SOURCE_MISSING','BELOW_REVIEWED_FLOOR_EXCLUDED']).all()
    report['missing_non_direct'] = int(missing.sum())
    report['direct_not_applicable'] = int((direct & s.admission_score.isna()).sum())
    s.loc[missing,['record_uid','cohort_year','admission_method','admission_score_missing_reason']].to_csv(output/'admission_missing.csv',index=False)
    s.groupby(['cohort_year','admission_method']).admission_score.agg(['size','count','min','max']).to_csv(output/'admission_method_summary.csv')
    # Chỉ đối chiếu ngưỡng công bố có thể so sánh, không kết luận trượt tuyển sinh.
    evidence = admission_evidence(s)
    floor = evidence.published_floor
    comparable = s.admission_score.notna() & floor.notna()
    below = evidence.admission_evidence_status.eq('BELOW_REVIEWED_FLOOR')
    evidence.to_csv(output/'admission_evidence.csv',index=False)
    out = s.loc[below,['record_uid','cohort_year','major_code','admission_method','admission_score']].copy()
    out['published_floor'] = floor[below]
    out.to_csv(output/'admission_floor_review.csv',index=False)
    report['admission_scores_with_comparable_floor'] = int(comparable.sum())
    report['admission_below_published_floor'] = int(below.sum())
    report['scope'] = 'Đối chiếu phương thức, thang điểm và các ngưỡng công bố có thể so sánh. Không suy ra quyết định của trường.'
    report['xu_ly'] = {'dang_ky_cuoi_chuong_trinh':'Ngoại lệ giả định mô phỏng; không quy kết vi phạm tự động',
        'dang_ky_thap':'Thiếu căn cứ; không tự tạo tín chỉ hoặc quyết định',
        'dieu_kien_thi':'Không kết luận cấm thi chỉ từ chuyên cần; giữ kết quả lịch sử',
        'tuyen_sinh_thieu':'Giữ giá trị thiếu, không điền điểm giả',
        'tuyen_sinh_duoi_nguong':'Đánh dấu không dùng để kết luận đủ điều kiện tuyển sinh',
        'lam_tron':'Đã tái sinh; kiểm tra lại lịch sử và nhãn' if source.manifest.get('research_revision') else 'Chờ tái tạo đồng bộ điểm, lịch sử học lại, GPA, feature và nhãn trước khi huấn luyện'}
    (output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    print('Kết quả đối chiếu quy chế (chi tiết trong data/doi_chieu_quy_che/review.json):')
    for label,key in [('Ngoại lệ đăng ký','registration'),('Điều kiện thi cần đối chiếu','exam_reviews'),('Điểm chữ cần hiệu chỉnh','grade_rounding_reviews'),('Thiếu điểm tuyển sinh','missing_non_direct'),('Dưới ngưỡng đã đối chiếu','admission_below_published_floor')]:
        print(f'{label}: {report[key]}')
    print('Không thay đổi dữ liệu nguồn hoặc quyết định học vụ.')


if __name__ == '__main__':
    main()
