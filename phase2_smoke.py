"""End-to-end model compatibility and runtime smoke test on the user's pinned Python."""
import numpy as np
from app import engine, get_student_me, data_view
from data_pipeline import KEY

view = data_view()
print(view)
expected = {'ASOF_2026_DEMO': (54609, 15796), 'FULL_SIMULATION': (57569, 7754)}
assert (len(engine.df), len(engine.active_latest_df)) == expected[view['mode']]
assert len(engine.latest_df) == 20000
assert not {'dropout_next_2_terms','exit_reason_for_evaluation'} & set(engine.df.columns)
assert engine.df.snapshot_cutoff.le(view['as_of']).all()
assert engine.df_attempts is not None and not engine.df_attempts.empty
result = get_student_me('BIT250093')
latest = result['latest']
assert result['courses'] and all(c['term'] <= latest['snapshot_term_index'] for c in result['courses'])
assert latest['part_time_work_hours_week'] > 0
assert 'ENGLISH' not in latest['verified_certificates']
sample = engine.df.sample(256, random_state=42)
X = sample[engine.meta['feature_cols']].copy()
for c in engine.meta['cat_cols']:
    X[c] = X[c].astype(str)
probs = engine.pipeline.predict_proba(X)[:, 1]
assert np.allclose(probs, sample.risk_probability, atol=1e-10, rtol=0)
print('BIT250093:', {k:latest[k] for k in ['snapshot_cutoff','state_as_of','verified_certificates','part_time_work_hours_week','study_hours_week','risk_probability']})
print('Courses:', len(result['courses']))
print('Max probability delta:', np.max(np.abs(probs-sample.risk_probability)))
print('[PASS] Phase 2 runtime + real model inference')
