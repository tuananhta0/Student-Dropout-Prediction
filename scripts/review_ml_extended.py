"""Đánh giá phân tầng ML, censoring, calibration và nguồn MAIN/STRESS; không train.

Trả trạng thái NOT_INSTALLED cho STRESS bị thiếu; chỉ gắn PASS khi thực sự
đọc, kiểm SHA256 theo manifest và xác minh không trộn ID với MAIN.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss
from core.data_pipeline import Dataset, MAIN, KEY, digest, require
from core.runtime_config import configure_runtime
from scripts.release_model import verify_pair
from scripts.report_provenance import provenance
from scripts.train_model import TARGET, StressDataset, split_at_freeze
from scripts.review_experiment import bootstrap_students


def group_stats(df, threshold, min_n=30):
    y=df[TARGET].to_numpy(dtype=int)
    p=df.risk_probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Xác suất không hợp lệ')
    result={'n':len(df),'positive':int(y.sum()),'prevalence':float(y.mean()) if len(y) else None,
            'coverage_both_labels':bool(len(y)>=min_n and len(np.unique(y))==2)}
    if result['coverage_both_labels']:
        yp=p>=threshold
        result.update(ap=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),
                      recall=float(recall_score(y,yp,zero_division=0)),
                      precision=float(precision_score(y,yp,zero_division=0)),
                      brier=float(brier_score_loss(y,p)),alert_rate=float(yp.mean()))
    else:
        result['note']='Không tính AP/ROC trên nhóm thiếu hai lớp hoặc n quá nhỏ; không kết luận hiệu năng'
    return result


def groups(df, col, threshold, min_n=30):
    if col not in df:
        return {'status':'MISSING_COLUMN'}
    return {str(k):group_stats(g,threshold,min_n) for k,g in df.groupby(col,dropna=False)}


def censor_by_term(source, name):
    table=source.table(name)
    if not table.label_observed.isin([True,False,0,1]).all():
        raise ValueError(name+': nhãn label_observed không phải Boolean')
    rows={}
    for term,g in table.groupby('snapshot_term_index'):
        seen=g[g.label_observed.eq(True)]
        rows[str(int(term))]={'total':len(g),'observed':len(seen),'censored':len(g)-len(seen),
            'observed_fraction':len(seen)/len(g),
            'positive_observed':int(seen[TARGET].eq(1).sum()),
            'negative_observed':int(seen[TARGET].eq(0).sum()),
            'single_class_observed':seen[TARGET].nunique()<2}
    return rows


def calibration_bins(df, edges=None):
    if edges is None:edges=np.linspace(0,1,11)
    bin_idx=np.minimum(np.searchsorted(edges,df.risk_probability.to_numpy(),side='right')-1,len(edges)-2)
    result=[]
    for i in range(len(edges)-1):
        g=df.iloc[np.flatnonzero(bin_idx==i)]
        result.append({'min':float(edges[i]),'max':float(edges[i+1]),'n':len(g),
                       'mean_predicted':float(g.risk_probability.mean()) if len(g) else None,
                       'actual_positive_rate':float(g[TARGET].mean()) if len(g) else None})
    ece=sum(b['n']*abs(b['mean_predicted']-b['actual_positive_rate']) for b in result if b['n'])/max(1,len(df))
    return {'bins':result,'ece':ece,'note':'ECE của dữ liệu mô phỏng không chứng minh xác suất chuẩn trên dữ liệu thực'}


def verify_stress(source, require_stress=False):
    path=source.run.with_name('V3_2_STRESS')
    if not (path/'manifests/generation_manifest.json').is_file():
        if require_stress: raise FileNotFoundError('Thiếu V3_2_STRESS: '+str(path))
        return {'status':'NOT_INSTALLED','path_hint':'Cần dataset hiệu chỉnh MAIN/STRESS cùng bản sinh'}
    stress=StressDataset(path)
    main=source.table('students')
    other=stress.table('students')
    # record_uid chi la ID noi bo tung scenario, nen co the trung giua
    # MAIN/STRESS. Khong merge hai tap chi theo record_uid.
    for frame, scenario in ((main, 'V3_2_MAIN'), (other, 'V3_2_STRESS')):
        require({'scenario_id', 'record_uid', 'student_id'} <= set(frame.columns),
                f'{scenario}: thieu cot dinh danh')
        require(frame.scenario_id.eq(scenario).all(),
                f'{scenario}: scenario_id khong khop manifest')
        for key in ('record_uid', 'student_id'):
            require(frame[key].notna().all() and frame[key].is_unique,
                    f'{scenario}: {key} rong hoac trung trong cung scenario')
    local_uid_overlap = len(set(main.record_uid) & set(other.record_uid))
    student_id_overlap = len(set(main.student_id) & set(other.student_id))
    require(student_id_overlap == 0, 'MAIN/STRESS trung student_id (co nguy co ro ri du lieu)')
    # Dataset.table kiểm tra đúng SHA và kích thước của từng table khi được mở.
    for name in ('prediction_snapshots','snapshot_labels_FULL_SIMULATION','snapshot_labels_ASOF_2026_DEMO'):
        stress.table(name)
    return {'status':'PASS','n_students':len(other),
            'stress_manifest_sha256':digest(path/'manifests/generation_manifest.json'),
            'identity_scope':'scenario_id + record_uid',
            'record_uid_overlap_local':local_uid_overlap,
            'student_id_overlap':student_id_overlap,
            'checked_tables':['students','prediction_snapshots','snapshot_labels_FULL_SIMULATION','snapshot_labels_ASOF_2026_DEMO'],
            'full_stress_audit':'NOT_RUN (chạy scripts.audit_dataset --stress-run để audit toàn bộ)'}


def review(source, model_dir, n_bootstrap=0, require_stress=False):
    meta=verify_pair(source.run.parents[2],model_dir)
    scored=pd.read_csv(model_dir/'student_snapshots_scored.csv.gz',low_memory=False)
    _,_,test=split_at_freeze(scored,meta['model_available_as_of'])
    test=test[test.snapshot_term_index.isin(meta['supported_terms'])].copy()
    require(len(test)>0,'Không có tập test theo model metadata')
    require(test[TARGET].isin([0,1]).all(),'Test chứa nhãn chưa quan sát hoặc không hợp lệ')
    t=float(meta['threshold'])
    result={'model_version':meta['version'],'threshold':t,'model_manifest_sha256':digest(model_dir/'manifest.json'),
        'dataset_manifest_sha256':meta.get('dataset_manifest_sha256'),
        'test_overall':group_stats(test,t),
        'by_term':groups(test,'snapshot_term_index',t),
        'by_major':groups(test,'major_code',t),
        'by_cohort':groups(test,'cohort_year',t),
        'by_degree_type':groups(test,'degree_type',t),
        'by_education_mode':groups(test,'education_mode',t),
        'calibration':calibration_bins(test),
        'censoring':{key:censor_by_term(source,key) for key in ('snapshot_labels_FULL_SIMULATION','snapshot_labels_ASOF_2026_DEMO')},
        'stress':verify_stress(source,require_stress),
        'bootstrap_students':bootstrap_students(test,t,n_bootstrap),
        'notes':['HK chỉ một lớp: không báo AUC/AP như chất lượng mô hình.',
                 'Không so sánh trực tiếp tỷ lệ nhãn giữa học kỳ đã quan sát và học kỳ bị censor.',
                 'Chỉ thống kê mô phỏng: không chứng minh fairness hay khả năng khái quát sang dữ liệu thật.',
                 'MAIN/STRESS phải được audit nguồn riêng; check ở đây chỉ là subset integrity.'],
    }
    result['provenance']=provenance(model_version=meta['version'],model_manifest_sha256=result['model_manifest_sha256'],dataset_manifest_sha256=result['dataset_manifest_sha256'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--require-stress',action='store_true')
    p.add_argument('--output',type=Path,default=Path('data/kiem_tra_ml/phan_tang_va_censoring.json'))
    args=p.parse_args()
    from core.data_pipeline import ROOT
    model_dir=Path(configure_runtime())
    source=Dataset()
    result=review(source,model_dir,args.bootstrap,args.require_stress)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'model_version':result['model_version'],'test_overall':result['test_overall'],
         'by_term':result['by_term'],'calibration_ece':result['calibration']['ece'],
         'stress_status':result['stress']['status'],'output':str(args.output)},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
