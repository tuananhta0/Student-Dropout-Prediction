"""Kiểm tra tích hợp dữ liệu và khả năng suy luận của mô hình thật."""
import numpy as np
from core.prediction import model_inputs
from app import engine, get_student_me, data_view, get_advisor_classes, get_admin_faculties, get_advisor_class_detail


def main():
    view = data_view()
    expected = {'DEMO': (57569, 14642), 'ASOF_2026_DEMO': (54609,15796), 'FULL_SIMULATION':(57569,7754)}
    if not engine.research_revision:
        assert (len(engine.ml_df),len(engine.active_latest_df)) == expected[view['mode']]
    else:
        assert engine.ml_df.snapshot_term_index.max() > 3
    assert len(engine.latest_df) == 20000
    assert not {'dropout_next_2_terms','exit_reason_for_evaluation'} & set(engine.df.columns)
    assert engine.df.snapshot_cutoff.le(view['as_of']).all()
    assert engine.df_attempts.available_at.le(view['as_of']).all()
    assert engine.df_attempts.score_published_at.le(view['as_of']).all()
    sample = engine.ml_df.sample(256,random_state=42)
    X = model_inputs(sample,engine.meta)
    probs = engine.pipeline.predict_proba(X)[:,1]
    assert np.allclose(probs,sample.risk_probability,atol=1e-10,rtol=0)
    result = get_student_me('BIT250093')
    assert result['courses'] and all(c['term']<=result['latest']['snapshot_term_index'] for c in result['courses'])
    if engine.research_revision:
        valid=engine.active_latest_df[engine.active_latest_df.prediction_status.eq('VALID')]
        assert valid.prediction_term.isin(engine.meta['supported_terms']).all()
        assert valid.prediction_cutoff.ge(engine.meta['model_available_as_of']).all()
        print('Độ bao phủ dự báo:',len(valid),'/',len(engine.active_latest_df))
    assert get_advisor_classes()['classes'] and get_admin_faculties()['faculties']
    rows = get_advisor_class_detail('IT25-09')['students']
    statuses = [r['prediction_status']=='VALID' for r in rows]
    assert statuses == sorted(statuses,reverse=True)
    print(f"Mốc dữ liệu: {view['as_of']}; hồ sơ học kỳ: {len(engine.df)}; snapshot ML: {len(engine.ml_df)}")
    print('BIT250093: học kỳ',result['latest']['snapshot_term_index'],'; lượt học phần:',len(result['courses']))
    print('Sai số xác suất tối đa:',np.max(np.abs(probs-sample.risk_probability)))
    print('[ĐẠT] Dữ liệu, mô hình thật, hồ sơ và thống kê')

if __name__ == '__main__':
    main()
