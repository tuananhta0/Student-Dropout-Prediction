"""Integration checks on installed MAIN with the real pinned inference runtime."""
from app import engine, get_student_me

assert {'gpa_credit_count','gpa_quality_points','prediction_valid_until','prediction_status'} <= set(engine.df.columns)
assert not {'dropout_next_2_terms','exit_reason_for_evaluation'} & set(engine.df.columns)
active = engine.active_latest_df
assert active.loc[active.prediction_status.eq('STALE'),'intervention_risk'].isna().all()
n = int((active.intervention_risk >= engine.meta['threshold']).sum())
assert n == (453 if engine.data_mode == 'ASOF_2026_DEMO' else 0)
data = get_student_me('BIT250093')
p = data['gpa_projection']
assert p['gpa_credit_count'] == (27 if engine.data_mode == 'ASOF_2026_DEMO' else 42)
assert p['gpa_quality_points'] == (64.5 if engine.data_mode == 'ASOF_2026_DEMO' else 114)
assert all('eligibility_review' in c and 'grade_policy_review' in c for c in data['courses'])
print(engine.data_mode, active.prediction_status.value_counts().to_dict(), 'intervention_count:',n)
print('BIT250093 GPA scenario:',p)
print('[PASS] Phase 3 academic rules + runtime')
