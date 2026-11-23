"""Kiểm tra dữ liệu V3.2.1 và dựng hồ sơ theo thời điểm, không huấn luyện model."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['record_uid', 'snapshot_term_index']
ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('artifacts/runs/V3_2_MAIN')
MODES = {'ASOF_2026_DEMO': 'demo_as_of', 'FULL_SIMULATION': 'simulation_as_of'}
EXIT_TYPES = {'VOLUNTARY_WITHDRAWAL', 'FORCED_ACADEMIC_WITHDRAWAL',
              'CONFIRMED_INSTITUTIONAL_ABANDONMENT', 'DISCIPLINARY_WITHDRAWAL'}
RUNTIME_TERM = ['part_time_work_hours_week', 'study_hours_week',
                'credits_attempted_cumulative', 'courses_failed_cumulative',
                'retake_count_cumulative', 'registration_rule_status',
                'registration_min_credits', 'registration_max_credits']
EXTRA_FEATURES = ['ca_score10_min', 'ca_exam10_min', 'ca_ineligible_exams', 'ca_unexcused_hours_sum']
TABLES = {
    'students': ('master/students.csv.gz', ['record_uid', 'student_id', 'scenario_id', 'cohort_year', 'curriculum_id', 'admission_method', 'admission_score', 'admission_score_normalized', 'admission_score_missing_reason']),
    'courses': ('master/courses.csv.gz', ['course_id', 'course_name', 'is_gpa_eligible', 'is_certificate_course']),
    'curricula': ('master/curricula.csv.gz', ['curriculum_id', 'degree_type', 'credits_required', 'designed_semesters', 'max_semesters']),
    'curriculum_courses': ('master/curriculum_courses.csv.gz', ['curriculum_id', 'course_id']),
    'term_calendar': ('master/term_calendar.csv.gz', ['calendar_term_id', 'cohort_year', 'term_index', 'grade_release_date']),
    'enrollment_terms': ('master/enrollment_terms.csv.gz', ['record_uid', 'term_index', 'calendar_term_id', 'available_at']),
    'course_attempts': ('master/course_attempts.csv.gz', ['attempt_id', 'record_uid', 'term_index', 'calendar_term_id', 'course_id', 'attempt_no', 'registered_credits', 'process_score10', 'exam_final_score10', 'course_final_score10', 'grade_letter', 'grade_point4', 'is_passed', 'exam_eligible', 'scheduled_hours', 'unexcused_absent_hours', 'excused_absent_hours', 'available_at', 'score_published_at']),
    'term_academics': ('master/term_academics.csv.gz', ['record_uid', 'term_index', 'calendar_term_id', 'available_at', 'snapshot_cutoff', 'credits_registered_term', 'credits_passed_term', 'credits_failed_term', 'credits_earned_cumulative', 'credits_owed_cumulative', 'gpa4_term', 'gpa4_cumulative', 'warning_current', *RUNTIME_TERM]),
    'academic_requirement_events': ('master/academic_requirement_events.csv.gz', ['record_uid', 'requirement_type', 'completion_status', 'available_at']),
    'state_transitions': ('master/state_transitions.csv.gz', ['record_uid', 'event_id', 'from_status', 'to_status', 'effective_at']),
    'student_events': ('master/student_events.csv.gz', ['record_uid', 'event_id', 'event_type', 'term_index', 'effective_at', 'available_at']),
    'prediction_snapshots': ('features/prediction_snapshots.csv.gz', [*KEY, 'snapshot_cutoff', 'feature_source_latest_at', 'student_id', 'cohort_year']),
    'snapshot_labels_FULL_SIMULATION': ('labels/snapshot_labels_full_simulation.csv.gz', [*KEY, 'snapshot_cutoff', 'scenario_id', 'observation_view_id', 'dropout_next_2_terms', 'label_observed', 'label_observation_end', 'observation_as_of', 'split', 'first_dropout_event_id']),
    'snapshot_labels_ASOF_2026_DEMO': ('labels/snapshot_labels_asof_2026_demo.csv.gz', [*KEY, 'snapshot_cutoff', 'scenario_id', 'observation_view_id', 'dropout_next_2_terms', 'label_observed', 'label_observation_end', 'observation_as_of', 'split', 'first_dropout_event_id']),
    'graduation_clearance': ('master/graduation_clearance.csv.gz', ['record_uid','term_index','available_at','status','clear_credits','clear_gpa4','clear_no_outstanding_academic_debt','clear_pe','clear_defense','clear_english','clear_finance','clear_discipline']),
    'feature_registry': ('docs/feature_registry.csv', ['feature_name', 'role']),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(df, keys, name):
    require(not df[keys].isna().any().any() and not df.duplicated(keys).any(),
            f'{name}: khóa rỗng hoặc trùng {keys}')


def dates(values):
    result = pd.to_datetime(values, format='%Y-%m-%d', errors='raise')
    require(result.notna().all(), 'Thiếu thời điểm bắt buộc')
    return result


def main_run():
    return Path(os.environ.get('CMC_DATASET_ROOT', ROOT / 'dataset_full')).resolve() / MAIN


class Dataset:
    def __init__(self, run=None):
        self.run = Path(run) if run is not None else main_run()
        manifest = self.run / 'manifests/generation_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Thiếu dữ liệu MAIN: {manifest}; chạy python -m scripts.install_dataset <archive>')
        self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
        require(self.manifest.get('scenario_id') == 'V3_2_MAIN' and
                (self.manifest.get('identity_migration_version') == '3.2.1' or
                 self.manifest.get('research_revision') == 'cmc-policy-2026-01') and
                self.manifest.get('n_students') == 20000 and self.manifest.get('is_synthetic') is True,
                'Yêu cầu dữ liệu mô phỏng V3.2.1 MAIN, không dùng BASE/STRESS')
        self.cache = {}

    def table(self, name):
        if name not in self.cache:
            rel, columns = TABLES[name]
            path = self.run / rel
            contract = self.manifest['documentation_artifacts' if name == 'feature_registry' else 'table_artifacts'][name]
            require(path.is_file() and digest(path) == contract['sha256'], f'SHA256 nguồn không hợp lệ: {rel}')
            df = pd.read_csv(path, low_memory=False)
            require(set(columns) <= set(df.columns), f'{rel}: thiếu các trường {set(columns)-set(df.columns)}')
            require(len(df) == contract['rows'] and len(df.columns) == contract['columns'], f'{rel}: không khớp kích thước')
            self.cache[name] = df
        return self.cache[name]

    def cutoff(self, mode):
        if mode == 'DEMO':
            # Mốc minh họa sau khi khóa 2025 có kết quả HK3; không phải ngày hiện tại.
            return pd.Timestamp('2027-02-03')
        require(mode in MODES, f'CMC_DATA_MODE không hợp lệ: {mode}')
        return pd.Timestamp(self.manifest[MODES[mode]])


def merge_exact(left, right, keys, name):
    unique(right, keys, name)
    result = left.merge(right, on=keys, how='left', validate='one_to_one', indicator=True)
    require(result['_merge'].eq('both').all(), f'{name}: thiếu bản ghi tham chiếu')
    return result.drop(columns='_merge')


def course_features(snapshots, attempts):
    """Chỉ dùng điểm đã khả dụng và công bố tại thời điểm chốt."""
    keys = snapshots[[*KEY, 'snapshot_cutoff']].rename(columns={'snapshot_term_index': 'term_index'})
    unique(keys, ['record_uid', 'term_index'], 'snapshot keys')
    a = attempts.merge(keys, on=['record_uid', 'term_index'], validate='many_to_one')
    a = a[(dates(a.available_at) <= dates(a.snapshot_cutoff)) &
          (dates(a.score_published_at) <= dates(a.snapshot_cutoff))]
    return a.groupby(['record_uid', 'term_index'], as_index=False).agg(
        ca_score10_min=('course_final_score10', 'min'), ca_exam10_min=('exam_final_score10', 'min'),
        ca_ineligible_exams=('exam_eligible', lambda s: int((~s).sum())),
        ca_unexcused_hours_sum=('unexcused_absent_hours', 'sum')).rename(columns={'term_index': 'snapshot_term_index'})


def asof_state(frame, states, when):
    unique(states, ['record_uid', 'effective_at'], 'state transitions')
    left = frame[['record_uid', when]].copy()
    left['_order'] = np.arange(len(left))
    left['_date'] = dates(left[when])
    right = states[['record_uid', 'effective_at', 'to_status']].copy()
    right['_effective'] = dates(right.effective_at)
    result = pd.merge_asof(left.sort_values('_date'), right.sort_values('_effective'),
                           left_on='_date', right_on='_effective', by='record_uid', direction='backward')
    require(result.to_status.notna().all(), 'Thiếu trạng thái sinh viên tại mốc chốt')
    return result.sort_values('_order').to_status.to_numpy()


def certificates(frame, events):
    # Lấy trạng thái mới nhất của từng yêu cầu tại thời điểm chốt.
    unique(events, ['record_uid', 'requirement_type', 'available_at'], 'requirement events')
    results = [[] for _ in range(len(frame))]
    for kind, sub in events.groupby('requirement_type'):
        left = frame[['record_uid', 'snapshot_cutoff']].copy()
        left['_order'] = np.arange(len(left))
        left['_date'] = dates(left.snapshot_cutoff)
        right = sub[['record_uid', 'available_at', 'completion_status']].copy()
        right['_available'] = dates(right.available_at)
        m = pd.merge_asof(left.sort_values('_date'), right.sort_values('_available'),
                          left_on='_date', right_on='_available', by='record_uid', direction='backward')
        for idx in m.loc[m.completion_status.eq('VERIFIED_SYNTHETICALLY'), '_order']:
            results[idx].append(kind)
    return [', '.join(sorted(r)) or 'Chưa hoàn thành' for r in results]


def enrich(frame, source):
    frame = frame.copy()
    terms = source.table('term_academics')
    right = terms[['record_uid', 'term_index', 'available_at', *RUNTIME_TERM]].rename(columns={'term_index': 'snapshot_term_index', 'available_at': '_term_available'})
    frame = merge_exact(frame.drop(columns=RUNTIME_TERM, errors='ignore'), right, KEY, 'term academics')
    require((dates(frame._term_available) <= dates(frame.snapshot_cutoff)).all(), 'Hồ sơ học vụ thuộc tương lai')
    frame = frame.drop(columns='_term_available')
    frame['living_status'] = frame['living_status_at_cutoff']
    frame['verified_certificates'] = certificates(frame, source.table('academic_requirement_events'))
    frame['current_state'] = asof_state(frame, source.table('state_transitions'), 'snapshot_cutoff')
    return frame


def gpa_ledger(source):
    """Tổng điểm trọng số tích lũy từ điểm chính thức cao nhất đã công bố."""
    from core.academic_rules import GRADE_POINTS
    a = source.table('course_attempts').merge(
        source.table('courses')[['course_id','is_gpa_eligible','is_certificate_course']],
        on='course_id', validate='many_to_one')
    terms = source.table('term_academics')[['record_uid','term_index','available_at']]
    a = a.merge(terms.rename(columns={'available_at':'term_available'}), on=['record_uid','term_index'], validate='many_to_one')
    require((dates(a.available_at) <= dates(a.term_available)).all() and
            (dates(a.score_published_at) <= dates(a.term_available)).all(), 'Điểm chưa công bố trong sổ GPA')
    a = a.loc[a.is_gpa_eligible & ~a.is_certificate_course & a.grade_letter.isin(GRADE_POINTS)].copy()
    a = a.sort_values(['record_uid','course_id','term_index','attempt_no'])
    groups = ['record_uid','course_id']
    require(a.groupby(groups).registered_credits.nunique().le(1).all(), 'Tín chỉ học phần thay đổi')
    a['points'] = a.grade_letter.map(GRADE_POINTS) * a.registered_credits
    a['best'] = a.groupby(groups).points.cummax()
    a['delta'] = a['best'] - a.groupby(groups)['best'].shift().fillna(0)
    a['credits'] = a.registered_credits.where(~a.duplicated(groups), 0)
    net = a.groupby(['record_uid','term_index'])[['delta','credits']].sum()
    ledger = terms.set_index(['record_uid','term_index']).sort_index().join(net).fillna({'delta':0,'credits':0})
    ledger[['gpa_quality_points','gpa_credit_count']] = ledger.groupby(level=0)[['delta','credits']].cumsum()
    return ledger.reset_index().rename(columns={'term_index':'snapshot_term_index'})[[*KEY,'gpa_quality_points','gpa_credit_count']]


def mandatory_course_ledger(source):
    students = source.table('students')[['record_uid','curriculum_id']]
    required = source.table('curriculum_courses').query("required_or_elective == 'REQUIRED'")[['curriculum_id','course_id']]
    counts = required.groupby('curriculum_id').size()
    attempts = source.table('course_attempts')
    a = attempts.loc[attempts.is_passed, ['record_uid','term_index','course_id']].merge(students,on='record_uid',validate='many_to_one')
    a = a.merge(required,on=['curriculum_id','course_id'],validate='many_to_one')
    first = a.sort_values('term_index').drop_duplicates(['record_uid','course_id'])
    increments = first.groupby(['record_uid','term_index']).size().rename('passed')
    terms = source.table('term_academics')[['record_uid','term_index']].merge(students,on='record_uid',validate='many_to_one')
    terms = terms.set_index(['record_uid','term_index']).sort_index().join(increments)
    terms['mandatory_courses_remaining'] = terms.curriculum_id.map(counts).fillna(0) - terms.passed.fillna(0).groupby(level=0).cumsum()
    return terms.reset_index().rename(columns={'term_index':'snapshot_term_index'})[[*KEY,'mandatory_courses_remaining']]



def new_course_ledger(source):
    """Tín chỉ học phần tính GPA chưa từng đăng ký trong chương trình mô phỏng."""
    cc = source.table('curriculum_courses').merge(source.table('courses')[['course_id','credits','is_gpa_eligible','is_certificate_course']],on='course_id',validate='many_to_one')
    eligible = cc.loc[cc.is_gpa_eligible & ~cc.is_certificate_course & cc.credit_counted_to_graduation,['curriculum_id','course_id','credits']]
    total = eligible.groupby('curriculum_id').credits.sum()
    students=source.table('students')[['record_uid','curriculum_id']]
    attempts=source.table('course_attempts')[['record_uid','course_id','term_index']].merge(students,on='record_uid',validate='many_to_one')
    first=attempts.merge(eligible,on=['curriculum_id','course_id'],validate='many_to_one').sort_values('term_index').drop_duplicates(['record_uid','course_id'])
    used=first.groupby(['record_uid','term_index']).credits.sum().rename('used')
    terms=source.table('term_academics')[['record_uid','term_index']].merge(students,on='record_uid',validate='many_to_one').set_index(['record_uid','term_index']).sort_index().join(used)
    terms['new_course_credits_available']=terms.curriculum_id.map(total)-terms.used.fillna(0).groupby(level=0).cumsum()
    require(terms.new_course_credits_available.notna().all() and terms.new_course_credits_available.ge(0).all(),'Số tín chỉ môn mới không hợp lệ')
    return terms.reset_index().rename(columns={'term_index':'snapshot_term_index'})[[*KEY,'new_course_credits_available']]

def runtime_view(scored, source, mode='ASOF_2026_DEMO'):
    """Giữ nguyên đặc trưng ML và xác suất; chỉ bổ sung trường hiển thị."""
    snapshots = source.table('prediction_snapshots')
    unique(scored, KEY, 'scored'); unique(snapshots, KEY, 'snapshots')
    require(len(scored) == len(snapshots), 'Quần thể dữ liệu chấm điểm khác nguồn')
    common = snapshots.columns.intersection(scored.columns).difference(KEY)
    check = scored[[*KEY, *common]].merge(snapshots[[*KEY, *common]], on=KEY, validate='one_to_one', suffixes=('_old', '_src'))
    require(len(check) == len(scored), 'Khóa snapshot khác artifact model')
    for c in common:
        a, b = check[c+'_old'], check[c+'_src']
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
        else:
            ok = a.fillna('').astype(str).eq(b.fillna('').astype(str)).all()
        require(ok, f'Trường chấm điểm không khớp nguồn: {c}')
    cutoff = source.cutoff(mode)
    frame = scored.loc[dates(scored.snapshot_cutoff) <= cutoff].copy()
    frame = frame.drop(columns=['dropout_next_2_terms', 'exit_reason_for_evaluation', 'label_observed', 'first_dropout_event_id'], errors='ignore')
    # Không đưa nhãn đánh giá vào dữ liệu runtime.
    frame = enrich(frame, source)
    frame = merge_exact(frame, gpa_ledger(source), KEY, 'GPA ledger')
    frame = merge_exact(frame, mandatory_course_ledger(source), KEY, 'mandatory courses')
    horizon = source.table('snapshot_labels_FULL_SIMULATION')[[*KEY,'label_observation_end']].rename(columns={'label_observation_end':'prediction_valid_until'})
    frame = merge_exact(frame, horizon, KEY, 'prediction horizon')
    frame['prediction_status'] = np.where(dates(frame.prediction_valid_until) <= cutoff, 'STALE', 'VALID')
    frame['data_mode'] = mode
    frame['data_as_of'] = cutoff.date().isoformat()
    latest = frame.sort_values('snapshot_term_index').drop_duplicates('record_uid', keep='last').copy()
    latest['state_as_of'] = asof_state(latest, source.table('state_transitions'), 'data_as_of')
    frame['state_as_of'] = frame.record_uid.map(latest.set_index('record_uid').state_as_of)
    return frame, latest


def visible_attempts(attempts, latest):
    keys = latest[['record_uid', 'snapshot_cutoff', 'snapshot_term_index']]
    a = attempts.merge(keys, on='record_uid', validate='many_to_one')
    return a.loc[(a.term_index <= a.snapshot_term_index) &
                 (dates(a.available_at) <= dates(a.snapshot_cutoff)) &
                 (dates(a.score_published_at) <= dates(a.snapshot_cutoff))].drop(columns=['snapshot_cutoff', 'snapshot_term_index'])


def graduation_asof(frame, clearance, when='snapshot_cutoff'):
    """Ghép kết quả xét tốt nghiệp đã có tại thời điểm tra cứu."""
    unique(clearance, ['record_uid', 'available_at'], 'xét tốt nghiệp')
    left = frame[['record_uid', when]].copy()
    left['_order'] = np.arange(len(left))
    left['_date'] = dates(left[when])
    fields = [c for c in clearance if c.startswith('clear_')]
    right = clearance[['record_uid','available_at','status',*fields]].copy()
    right['_date_clear'] = dates(right.pop('available_at'))
    right = right.rename(columns={'status': 'graduation_status'})
    joined = pd.merge_asof(left.sort_values('_date'), right.sort_values('_date_clear'),
                           left_on='_date',right_on='_date_clear',by='record_uid',direction='backward')
    joined = joined.sort_values('_order')
    result = frame.copy()
    for c in ['graduation_status',*fields]:
        result[c] = joined[c].to_numpy()
    result['graduation_review_date'] = joined['_date_clear'].dt.strftime('%Y-%m-%d').to_numpy()
    return result


def academic_view(ml, source, mode):
    """Hồ sơ học vụ độc lập với tập đặc trưng ML; không dùng nhãn tương lai."""
    cutoff = source.cutoff(mode)
    terms = source.table('term_academics')
    frame = terms.loc[(dates(terms.available_at) <= cutoff) & (dates(terms.snapshot_cutoff) <= cutoff)].copy()
    frame = frame.rename(columns={'term_index':'snapshot_term_index', 'duration_counted_to_date':'counted_duration_semesters'})
    unique(frame, KEY, 'học kỳ')
    students = source.table('students')
    static = ['record_uid','student_id','cohort_year','major_code','major_name','class_code','degree_type','education_mode','gender','home_province','enrollment_date','living_status','admission_method','designed_semesters','max_semesters','admission_score_missing_reason','admission_score']
    frame = frame.merge(students[static+['scholarship_percent']],on='record_uid',validate='many_to_one')
    evidence = admission_evidence(students)[['record_uid','admission_evidence_status']]
    frame = frame.merge(evidence,on='record_uid',validate='many_to_one')  # Điểm tuyển sinh chỉ dùng trong dịch vụ đã kiểm tra quyền.
    frame = frame.sort_values(['record_uid','snapshot_term_index'])
    remaining = frame.credits_required - frame.groupby('record_uid').credits_earned_cumulative.shift().fillna(0)
    out_of_range = frame.credits_registered_term.lt(frame.registration_min_credits) | frame.credits_registered_term.gt(frame.registration_max_credits)
    final_remainder = out_of_range & remaining.lt(frame.registration_min_credits) & frame.credits_registered_term.le(remaining)
    frame['registration_assessment'] = np.select([final_remainder,out_of_range],['FINAL_REMAINDER_SIMULATION_EXCEPTION','LOW_LOAD_UNVERIFIED'],default='WITHIN_RANGE')
    frame['scholarship_percent_at_cutoff'] = frame['scholarship_percent']
    frame['failed_credit_ratio'] = frame.credits_failed_term.div(frame.credits_registered_term.replace(0,np.nan))
    frame['verified_certificates'] = certificates(frame, source.table('academic_requirement_events'))
    frame['current_state'] = asof_state(frame, source.table('state_transitions'), 'snapshot_cutoff')
    frame = merge_exact(frame,gpa_ledger(source),KEY,'GPA')
    frame = merge_exact(frame,mandatory_course_ledger(source),KEY,'học phần bắt buộc')
    frame = merge_exact(frame,new_course_ledger(source),KEY,'tín chỉ môn mới')
    # Ghép xác suất lịch sử tại hoặc trước ngày học vụ, không thay đổi feature của model.
    risk = ml[['record_uid','snapshot_term_index','snapshot_cutoff','risk_probability','risk_level','prediction_valid_until']].rename(columns={'snapshot_term_index':'prediction_term','snapshot_cutoff':'prediction_cutoff'})
    frame['_date'] = dates(frame.snapshot_cutoff)
    risk['_prediction_date'] = dates(risk.prediction_cutoff)
    frame = pd.merge_asof(frame.sort_values('_date'),risk.sort_values('_prediction_date'),left_on='_date',right_on='_prediction_date',by='record_uid',direction='backward').drop(columns=['_date','_prediction_date'])
    frame['prediction_status'] = np.select([frame.risk_probability.isna(), (pd.to_datetime(frame.prediction_valid_until) <= cutoff) | frame.prediction_term.ne(frame.snapshot_term_index)],['UNKNOWN','STALE'],default='VALID')
    frame['data_mode'] = mode
    frame['data_as_of'] = cutoff.date().isoformat()
    frame = graduation_asof(frame, source.table('graduation_clearance'))
    latest = frame.sort_values('snapshot_term_index').drop_duplicates('record_uid',keep='last').copy()
    latest['state_as_of'] = asof_state(latest,source.table('state_transitions'),'data_as_of')
    frame['state_as_of'] = frame.record_uid.map(latest.set_index('record_uid').state_as_of)
    # Điều kiện tốt nghiệp có thể được công bố sau ngày chốt bảng điểm.
    latest = graduation_asof(latest, source.table('graduation_clearance'), 'data_as_of')
    latest['verified_certificates'] = certificates(latest.assign(snapshot_cutoff=latest.data_as_of),source.table('academic_requirement_events'))
    # Cập nhật đúng dòng cuối của lịch sử để API hồ sơ và thống kê nhất quán.
    fields = ['verified_certificates','graduation_status','graduation_review_date',*[c for c in latest if c.startswith('clear_')]]
    frame = frame.set_index(KEY); update = latest.set_index(KEY)
    frame.loc[update.index,fields] = update[fields]
    return frame.reset_index(),latest


def admission_evidence(s):
    """Phân biệt điểm thiếu, tuyển thẳng, điểm dưới ngưỡng và dữ liệu chưa đủ căn cứ."""
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
    direct = s.admission_method.eq('CMC303_TuyenThang')
    status = np.select([direct, s.admission_score.isna(), below, comparable],
        ['DIRECT_NOT_APPLICABLE','MISSING','BELOW_REVIEWED_FLOOR','MEETS_REVIEWED_FLOOR'],default='NO_COMPARABLE_FLOOR')
    return pd.DataFrame({'record_uid':s.record_uid,'admission_evidence_status':status,'published_floor':floor},index=s.index)
