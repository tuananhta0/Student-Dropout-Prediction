"""Kiểm tra dữ liệu MAIN, dừng khi có lỗi cấu trúc và tách các mục cần đối chiếu."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_pipeline import (Dataset, ROOT, KEY, MODES, EXIT_TYPES, EXTRA_FEATURES,
                           dates, unique, runtime_view, course_features, visible_attempts)


def audit(source, scored_path, output):
    output.mkdir(parents=True, exist_ok=True)
    checks, reviews, stats = {}, {}, {}

    def check(name, bad):
        n = int(np.asarray(bad).sum()) if not np.isscalar(bad) else int(bad)
        checks[name] = n
        print(f'[{"ĐẠT" if n == 0 else "LỖI"}] {name}: {n}')

    def review(name, rows):
        reviews[name] = len(rows)
        rows.to_csv(output / (name + '.csv'), index=False)
        print(f'[CẦN ĐỐI CHIẾU] {name}: {len(rows)}')

    s = source.table('students'); c = source.table('courses'); curr = source.table('curricula')
    cc = source.table('curriculum_courses'); cal = source.table('term_calendar')
    e = source.table('enrollment_terms'); t = source.table('term_academics')
    a = source.table('course_attempts'); req = source.table('academic_requirement_events')
    states = source.table('state_transitions'); events = source.table('student_events')
    x = source.table('prediction_snapshots'); registry = source.table('feature_registry')
    grad_clearance = source.table('graduation_clearance')
    termkey = ['record_uid', 'term_index']
    for df, keys, name in [(s,['record_uid'],'students'), (s,['student_id'],'student IDs'),
                           (c,['course_id'],'courses'), (curr,['curriculum_id'],'curricula'),
                           (cc,['curriculum_id','course_id'],'curriculum courses'),
                           (cal,['calendar_term_id'],'calendar'), (cal,['cohort_year','term_index'],'calendar horizon'),
                           (e,termkey,'enrollment'), (t,termkey,'academics'), (a,['attempt_id'],'attempts'),
                           (a,['record_uid','course_id','attempt_no'],'attempt sequence'),
                           (req,['record_uid','requirement_type','available_at'],'requirements'),
                           (states,['record_uid','effective_at'],'states'), (events,['event_id'],'events'),
                           (x,KEY,'snapshots'), (registry,['feature_name'],'registry')]:
        unique(df,keys,name)
    unique(grad_clearance,['record_uid','available_at'],'xét tốt nghiệp')
    check('graduation_student_fk',~grad_clearance.record_uid.isin(s.record_uid))
    # Xét tốt nghiệp có thể tiếp diễn sau học kỳ cuối đã đăng ký: đối chiếu lịch, không buộc có bảng điểm kỳ đó.
    gc = grad_clearance.merge(s[['record_uid','cohort_year']],on='record_uid',validate='many_to_one').merge(cal[['cohort_year','term_index','grade_release_date']],on=['cohort_year','term_index'],how='left',validate='many_to_one')
    check('graduation_calendar_fk',gc.grade_release_date.isna())
    check('graduation_flags_boolean',sum(not pd.api.types.is_bool_dtype(grad_clearance[col]) or grad_clearance[col].isna().any() for col in grad_clearance if col.startswith('clear_')))
    check('graduation_available_before_term',gc.grade_release_date.notna() & gc.available_at.lt(gc.grade_release_date))
    check('scenario_main', ~s.scenario_id.eq('V3_2_MAIN'))
    for name, df in [('enrollment',e),('academics',t),('attempts',a),('requirements',req),('states',states),('events',events),('snapshots',x)]:
        check(name+'_student_fk', ~df.record_uid.isin(s.record_uid))
    check('students_curriculum_fk', ~s.curriculum_id.isin(curr.curriculum_id))
    check('curriculum_course_fk', ~cc.course_id.isin(c.course_id) | ~cc.curriculum_id.isin(curr.curriculum_id))
    check('attempt_course_fk', ~a.course_id.isin(c.course_id))
    for name, df in [('enrollment',e),('academics',t),('attempts',a)]:
        check(name+'_calendar_fk', ~df.calendar_term_id.isin(cal.calendar_term_id))
    for name, df in [('academics',t),('attempts',a)]:
        check(name+'_enrollment_fk', ~pd.MultiIndex.from_frame(df[termkey]).isin(pd.MultiIndex.from_frame(e[termkey])))
    check('attempt_curriculum_fk', ~pd.MultiIndex.from_frame(a[['record_uid','course_id']].merge(s[['record_uid','curriculum_id']],on='record_uid',validate='many_to_one')[['curriculum_id','course_id']]).isin(pd.MultiIndex.from_frame(cc[['curriculum_id','course_id']])))
    check('snapshot_latest_feature_after_cutoff', dates(x.feature_source_latest_at)>dates(x.snapshot_cutoff))
    check('snapshot_student_identity', ~x.student_id.eq(x.record_uid.map(s.set_index('record_uid').student_id)))
    check('snapshot_term_fk', ~pd.MultiIndex.from_frame(x.rename(columns={'snapshot_term_index':'term_index'})[termkey]).isin(pd.MultiIndex.from_frame(t[termkey])))
    alignment = x.merge(t.rename(columns={'term_index':'snapshot_term_index'}), on=KEY, validate='one_to_one', suffixes=('_x','_term'))
    check('snapshot_term_available_after_cutoff', dates(alignment.available_at) > dates(alignment.snapshot_cutoff_x))
    for field in ('gpa4_term','gpa4_cumulative','credits_registered_term','credits_failed_term','credits_earned_cumulative','credits_owed_cumulative','attendance_rate_term','unexcused_absence_rate_term','warnings_total_to_date','warning_streak'):
        check('snapshot_term_match_'+field, ~np.isclose(alignment[field+'_x'],alignment[field+'_term'],equal_nan=True))
    del alignment
    check('attempt_published_after_available', dates(a.score_published_at)>dates(a.available_at))
    at = a.merge(t[termkey+['available_at']].rename(columns={'available_at':'term_available'}),on=termkey,validate='many_to_one')
    check('term_grade_after_cutoff', (dates(at.available_at)>dates(at.term_available)) | (dates(at.score_published_at)>dates(at.term_available)))
    del at
    seq = states.sort_values(['record_uid','effective_at'])
    expected = seq.groupby('record_uid').to_status.shift().fillna('NOT_ENROLLED')
    check('state_chain', ~seq.from_status.eq(expected))
    linked = states[states.event_id.ne('ENROLLMENT_SYNTHETIC')].merge(events[['record_uid','event_id','effective_at','available_at']],on=['record_uid','event_id'],validate='many_to_one',how='left',suffixes=('_state','_event'))
    check('state_event_times', linked.effective_at_event.isna() | linked.effective_at_state.ne(linked.effective_at_event) | linked.effective_at_state.lt(linked.available_at))

    # Tính lại sổ điểm và tín chỉ độc lập theo dữ liệu mô phỏng.
    check('gpa_eligibility_contract', ~c.is_gpa_eligible | c.is_certificate_course)
    a = a.sort_values(['record_uid','course_id','attempt_no']).copy()
    check('attempt_no_sequence', a.attempt_no.ne(a.groupby(['record_uid','course_id']).cumcount()+1))
    check('course_pass_then_retake', a.groupby(['record_uid','course_id']).is_passed.shift().eq(True))
    check('course_credit_positive', a.registered_credits.le(0))
    check('grade_range', ~a.grade_point4.between(0,4) | ~a.course_final_score10.between(0,10))
    check('failed_grade', (~a.is_passed & a.grade_point4.ne(0)) | (a.is_passed & a.grade_point4.lt(1)))
    a['earned'] = a.registered_credits.where(a.is_passed,0)
    a['failed'] = a.registered_credits.where(~a.is_passed,0)
    a['points'] = a.grade_point4*a.registered_credits
    a['owed_delta'] = a.failed.where(a.attempt_no.eq(1),0)-a.earned.where(a.attempt_no.gt(1),0)
    a['retake'] = a.attempt_no.gt(1)
    a['failed_count'] = (~a.is_passed).astype(int)
    sums = a.groupby(termkey).agg(reg=('registered_credits','sum'), passed=('earned','sum'), failed=('failed','sum'), points=('points','sum'), owed=('owed_delta','sum'), retakes=('retake','sum'), fail_count=('failed_count','sum')).sort_index()
    balances = sums[['reg','passed','points','owed','retakes','fail_count']].groupby(level=0).cumsum().add_suffix('_cum')
    ledger = t.set_index(termkey).join(sums).join(balances)
    for src, target in [('reg','credits_registered_term'),('passed','credits_passed_term'),('failed','credits_failed_term'),('reg_cum','credits_attempted_cumulative'),('passed_cum','credits_earned_cumulative'),('owed_cum','credits_owed_cumulative'),('retakes_cum','retake_count_cumulative'),('fail_count_cum','failed_attempts_total_to_date')]:
        check('ledger_'+target, ~np.isclose(ledger[src],ledger[target],atol=1e-7))
    check('ledger_term_gpa', ~np.isclose(ledger.points/ledger.reg,ledger.gpa4_term,atol=1e-7))
    check('ledger_cumulative_gpa', ~np.isclose(ledger.points_cum/(ledger.passed_cum+ledger.owed_cum),ledger.gpa4_cumulative,atol=1e-7))
    warning = (t.gpa4_term < np.where(t.term_index.eq(1),.8,1.0)) | (t.credits_failed_term > .5*t.credits_registered_term) | (t.credits_owed_cumulative>24)
    check('art18_warning', warning.ne(t.warning_current))
    check('registration_flags', ((t.credits_registered_term<t.registration_min_credits)|(t.credits_registered_term>t.registration_max_credits)).ne(t.registration_rule_status.ne('WITHIN_ART12_DEFAULT_RANGE')))
    review('registration_exceptions',t.loc[t.registration_rule_status.ne('WITHIN_ART12_DEFAULT_RANGE'),termkey+['registration_rule_status','registration_exception_reason','credits_registered_term','registration_min_credits','registration_max_credits']])
    absent = (a.unexcused_absent_hours > .2*a.scheduled_hours) | ((a.unexcused_absent_hours+a.excused_absent_hours) > .3*a.scheduled_hours)
    review('exam_eligibility_discrepancies',a.loc[absent.eq(a.exam_eligible), ['attempt_id',*termkey,'scheduled_hours','unexcused_absent_hours','excused_absent_hours','exam_eligible']])
    # Thang điểm tuyển sinh của dữ liệu mô phỏng không thay thế quy định chính thức.
    scale = s.admission_method.str.extract(r'_(30|40|80|150|1200)$')[0].astype(float)
    direct = s.admission_method.eq('CMC303_TuyenThang')
    check('admission_known_method', scale.isna() & ~direct)
    check('admission_bounds', s.admission_score.notna() & (s.admission_score.lt(0) | s.admission_score.gt(scale) | direct))
    check('admission_normalization', ~np.isclose(s.admission_score/scale,s.admission_score_normalized,atol=1e-5,equal_nan=True))
    check('admission_missing_reason', s.admission_score.isna().ne(s.admission_score_missing_reason.notna()))
    review('admission_missing_non_direct',s.loc[s.admission_score.isna() & ~direct,['record_uid','admission_method','admission_score_missing_reason']])
    bachelor = curr.degree_type.eq('Bachelor'); engineer = curr.degree_type.eq('Engineer')
    check('curriculum_credit_duration', ~(bachelor & curr.credits_required.between(120,135) & curr.designed_semesters.eq(9) & curr.max_semesters.eq(18) | engineer & curr.credits_required.between(150,164) & curr.designed_semesters.eq(11) & curr.max_semesters.eq(22)))

    scored = pd.read_csv(scored_path,low_memory=False)
    meta = json.loads(scored_path.with_name('model_metadata.json').read_text(encoding='utf-8'))
    allowed = set(registry.loc[registry.role.eq('FEATURE_ALLOWED_ASOF'),'feature_name']) | set(EXTRA_FEATURES)
    check('model_disallowed_features',len(set(meta['feature_cols'])-allowed))
    ca = course_features(x,a).merge(scored[KEY+EXTRA_FEATURES],on=KEY,validate='one_to_one',suffixes=('_src','_scored'))
    check('course_feature_missing',len(x)-len(ca))
    for col in EXTRA_FEATURES:
        check('asof_'+col,~np.isclose(ca[col+'_src'],ca[col+'_scored'],equal_nan=True))
    for mode in MODES:
        frame, latest = runtime_view(scored, source, mode)
        vis = visible_attempts(a,latest)
        labels = source.table('snapshot_labels_'+mode)
        unique(labels,KEY,mode)
        cutoff = source.cutoff(mode)
        check(mode+'_label_keys', len(labels.merge(frame[KEY],on=KEY,validate='one_to_one'))!=len(frame) or len(labels)!=len(frame))
        check(mode+'_label_view', labels.observation_view_id.ne(mode) | labels.scenario_id.ne('V3_2_MAIN') | dates(labels.observation_as_of).ne(cutoff))
        check(mode+'_censor_null', labels.label_observed.ne(labels.dropout_next_2_terms.notna()))
        check(mode+'_label_domain', labels.label_observed & ~labels.dropout_next_2_terms.isin([0,1]))
        # Tính lại nhãn theo học kỳ và thời điểm sự kiện khả dụng.
        lk = labels[KEY+['snapshot_cutoff','cohort_year','label_observation_end','label_observed','dropout_next_2_terms','first_dropout_event_id','split']].copy()
        ev = lk[KEY].merge(events,on='record_uid',validate='many_to_many')
        ev = ev[dates(ev.available_at)<=cutoff]
        exit_ev = ev[ev.event_type.isin(EXIT_TYPES) & ev.term_index.gt(ev.snapshot_term_index) & ev.term_index.le(ev.snapshot_term_index+2)].sort_values('available_at').drop_duplicates(KEY)
        grad = ev[ev.event_type.eq('GRADUATED') & ev.term_index.ge(ev.snapshot_term_index) & ev.term_index.le(ev.snapshot_term_index+2)].drop_duplicates(KEY)
        lk = lk.merge(exit_ev[KEY+['event_id']],on=KEY,how='left',validate='one_to_one').merge(grad[KEY].assign(graduated=True),on=KEY,how='left',validate='one_to_one')
        positive = lk.event_id.notna(); observed = positive | lk.graduated.eq(True) | dates(lk.label_observation_end).le(cutoff)
        check(mode+'_label_recomputed', observed.ne(lk.label_observed) | (observed & positive.astype(int).ne(lk.dropout_next_2_terms)))
        check(mode+'_positive_event_id', positive & lk.event_id.ne(lk.first_dropout_event_id))
        # Dùng lịch học mô phỏng để kiểm chứng cửa sổ dự báo.
        horizon = lk.assign(term_index=lk.snapshot_term_index+2).merge(cal[['cohort_year','term_index','grade_release_date']],on=['cohort_year','term_index'],how='left',validate='many_to_one')
        check(mode+'_known_calendar_horizon', horizon.grade_release_date.notna() & horizon.grade_release_date.ne(horizon.label_observation_end))
        expected_split = np.select([lk.cohort_year.le(2023),lk.cohort_year.eq(2024),lk.cohort_year.eq(2025)],['TRAIN_2022_23','VALIDATION_2024','TEST_2025'],default='INVALID')
        check(mode+'_split', lk.split.ne(expected_split))
        check(mode+'_student_split_overlap',lk.groupby('record_uid').split.nunique().gt(1))
        stats[mode] = {'snapshots':len(frame),'students':len(latest),'visible_attempts':len(vis), 'observed':int(observed.sum()),'censored':int((~observed).sum()),'positive':int(positive.sum()),'states':{k:int(v) for k,v in latest.state_as_of.value_counts().items()}, 'cutoff_ranges_by_split':lk.groupby('split').snapshot_cutoff.agg(['min','max']).to_dict('index')}
        if mode == 'FULL_SIMULATION':
            joined=labels.merge(scored[KEY+['dropout_next_2_terms']],on=KEY,validate='one_to_one',suffixes=('_src','_scored'))
            check('full_scored_label_match',joined.dropout_next_2_terms_src.ne(joined.dropout_next_2_terms_scored))
    stats['source_missing_counts']={name:{k:int(v) for k,v in df.isna().sum().items() if v} for name,df in source.cache.items()}
    report={'dataset_version':'3.2.1','scenario':'V3_2_MAIN','blocking_errors':checks,'review_required':reviews,'views':stats,'limitations':source.manifest['known_limitations'], 'temporal_split_note':'Các tập không trùng sinh viên nhưng khoảng thời gian có thể chồng lấn; cần thực nghiệm rolling-origin.', 'policy_note':'Không suy ra quyết định học vụ hoặc trúng tuyển chính thức từ dữ liệu mô phỏng.'}
    (output/'kiem_tra_du_lieu.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Báo cáo kiểm tra dữ liệu:',output/'kiem_tra_du_lieu.json')
    return not any(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/kiem_tra_du_lieu')
    p.add_argument('--scored',type=Path,default=ROOT/'artifacts_model/student_snapshots_scored.csv.gz')
    args=p.parse_args()
    raise SystemExit(0 if audit(Dataset(),args.scored,args.output) else 1)
