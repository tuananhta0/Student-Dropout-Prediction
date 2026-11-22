"""Kiểm tra GPA, dữ liệu xét tốt nghiệp và kịch bản học lại trên toàn hệ thống."""
from app import engine, get_student_me
from core.academic_rules import graduation_conditions


def main():
    active = engine.active_latest_df
    assert active.loc[active.prediction_status.ne('VALID'),'intervention_risk'].isna().all()
    assert engine.df.loc[engine.df.prediction_term.ne(engine.df.snapshot_term_index),'prediction_status'].ne('VALID').all()
    data = get_student_me('BIT250093')
    row = data['latest']
    assert all('eligibility_review' in c and 'grade_policy_review' in c for c in data['courses'])
    if engine.data_mode == 'DEMO':
        p = data['gpa_projection']
        assert row['snapshot_term_index']==3 and p['next_term']==4
        assert p['gpa_credit_count']==42 and p['gpa_quality_points']==114
        assert p['pred_cum_imp']==2.84 and len(data['courses'])==15
        result = engine.calculate_retake(row['record_uid'], {data['courses'][0]['course_id']:'A'})
        assert result['gpa_credits']==42 and result['gpa_after']>=result['gpa_before']
        reply=engine.answer_query('Kịch bản học lại '+data['courses'][0]['course_id']+' đạt A',logged_in_user='BIT250093')['reply']
        assert 'không cộng trùng' in reply
        print('BIT250093: HK3 → HK4; GPA tích lũy 2,71; kịch bản học lại:',result)
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
