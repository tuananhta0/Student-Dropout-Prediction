"""Kiểm tra GPA, dữ liệu xét tốt nghiệp và kịch bản học lại trên toàn hệ thống."""
from app import engine, get_student_me, retrieval
from core.intent_router import route
from core.security import Principal
from core.academic_rules import graduation_conditions


def main():
    active = engine.active_latest_df
    assert active.loc[active.prediction_status.ne('VALID'),'intervention_risk'].isna().all()
    assert engine.df.loc[engine.df.prediction_term.ne(engine.df.snapshot_term_index),'prediction_status'].ne('VALID').all()
    data = get_student_me('BIT250093',Principal('BIT250093','student'))
    row = data['latest']
    assert all('eligibility_review' in c and 'grade_policy_review' in c for c in data['courses'])
    if engine.data_mode == 'DEMO' and not engine.research_revision:
        p = data['gpa_projection']
        assert row['snapshot_term_index']==3 and p['next_term']==4
        assert p['gpa_credit_count']==42 and p['gpa_quality_points']==114
        assert p['pred_cum_imp']==2.84 and len(data['courses'])==15
        result = engine.calculate_retake(engine.get_student_records('BIT250093').iloc[-1]['record_uid'], {data['courses'][0]['course_id']:'A'})
        assert result['gpa_credits']==42 and result['gpa_after']>=result['gpa_before']
        reply=route(engine,retrieval,Principal('BIT250093','student'),'Kịch bản học lại '+data['courses'][0]['course_id']+' đạt A',{})['reply']
        assert 'không cộng trùng' in reply
        print('BIT250093: HK3 → HK4; GPA tích lũy 2,71; kịch bản học lại:',result)
    if engine.research_revision and len(active):
        sample=active.iloc[0]
        result=engine.calculate_retake(sample.record_uid,{},0,0)
        assert abs(result['gpa_before']-result['gpa_after'])<1e-9
        candidates=active[active.prediction_status.eq('VALID') & active.tuition_debt_end_vnd.gt(0)]
        if not candidates.empty:
            row=candidates.iloc[0]; outcome=engine.calculate_payment_scenario(row.record_uid,float(row.tuition_debt_end_vnd))
            assert abs(outcome['before']['risk_percent']-row.risk_probability*100)<.011
            print('Kịch bản thanh toán:',outcome)
    clearance = engine.latest_df[engine.latest_df.graduation_status.notna()]
    assert len(clearance)>0
    for _, r in clearance.head(10).iterrows():
        c=graduation_conditions(r.to_dict())
        assert 'academic_debt' in c
    print('Hồ sơ có kết quả xét tốt nghiệp:',len(clearance))
    print('Dự báo còn hiệu lực:',int(active.prediction_status.eq('VALID').sum()))
    print('[ĐẠT] Quy chế, GPA, học lại và thời hạn dự báo')

if __name__ == '__main__':
    main()
