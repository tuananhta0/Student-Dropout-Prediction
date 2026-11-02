"""Phase 2-I synthetic-only frozen-Test diagnostic audit.

Consumes *saved* one-shot Phase H predictions. Never loads/fits a model,
re-scores Test, chooses threshold, or writes back to Phase H artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
KEY = ['record_uid', 'snapshot_term_index']
MODEL_VERSION = 'v3.2-rf-b-platt-g1'
THRESHOLD = 0.27
MIN_N = 100
MIN_POS = 20
MIN_NEG = 20
MIN_UID = 50
OUTPUTS = ('subgroup_audit.json', 'subgroup_audit.md', 'subgroup_metrics.csv',
           'hk1_risk_audit.json', 'cohort_shift_audit.csv')
DRIFT_NUMERIC = ('gpa4_term', 'attendance_rate_term', 'lms_logins_4weeks',
                 'tuition_debt_to_due_ratio', 'failed_credit_ratio')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    require(path.is_file(), f'Required report missing: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def wilson(success, total, z=1.959963984540054):
    """Wilson score 95% CI for an event frequency; returns None if unsupported."""
    if total < 1:
        return None
    require(0 <= success <= total, 'Invalid Wilson inputs')
    p = success / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    delta = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0.,center-delta),min(1.,center+delta)]


def topk_indices(scores, uids, terms, budget):
    scores = np.asarray(scores, dtype=float)
    uids = np.asarray(uids, dtype=str)
    terms = np.asarray(terms, dtype=int)
    require(len(scores)==len(uids)==len(terms) and len(scores)>0, 'Bad topK input shape')
    require(np.isfinite(scores).all() and ((0<=scores)&(scores<=1)).all(), 'Invalid scores')
    require(budget in (.05,.10), 'Only locked 5%/10% budgets allowed')
    order = np.lexsort((terms, uids, -scores))
    return order[:math.ceil(len(scores)*budget)]


def support(n, positives, negatives, uids):
    return 'SUFFICIENT' if (n>=MIN_N and positives>=MIN_POS and negatives>=MIN_NEG and uids>=MIN_UID) else 'INSUFFICIENT_SUPPORT'


def ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    if not len(y): return None
    bins=np.minimum((p*n_bins).astype(int), n_bins-1)
    return float(sum(np.sum(bins==i)/len(y) * abs(float(np.mean(p[bins==i])) - float(np.mean(y[bins==i])))
                     for i in range(n_bins) if np.any(bins==i)))


def metrics_of_group(g: pd.DataFrame, group_by, group_value):
    y=g.y_true.to_numpy(dtype=int)
    p=g.calibrated_probability.to_numpy(dtype=float)
    n=len(y); pos=int(y.sum()); neg=n-pos; unique_uids=g.record_uid.nunique()
    status=support(n,pos,neg,unique_uids)
    preds=p >= THRESHOLD
    tp=int(np.sum(preds & (y==1)))
    fp=int(np.sum(preds & (y==0)))
    fn=pos-tp; tn=neg-fp
    result={
        'group_by':str(group_by), 'group_value':str(group_value),
        'status':status, 'n':n, 'uids':int(unique_uids), 'positives':pos,
        'negatives':neg, 'prevalence':float(pos/n),
        'prevalence_wilson95':wilson(pos,n),
        'mean_probability':float(p.mean()),'overprediction_mean_minus_prevalence':float(p.mean()-pos/n),
        'alerts_threshold':int(preds.sum()),'false_positives_threshold':fp,
        'false_negatives_threshold':fn,
        'brier':float(brier_score_loss(y,p)),
        'ece10':ece(y,p),
    }
    # Performance estimates for small groups are NOT presented as fairness evidence.
    if status == 'SUFFICIENT':
        result.update({
            'pr_auc':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p)),
            'recall_threshold':float(tp/pos),
            'recall_wilson95':wilson(tp,pos),
            'false_negative_rate_threshold':float(fn/pos),
            'false_positive_rate_threshold':float(fp/neg),
            'fpr_wilson95':wilson(fp,neg),
        })
    else:
        result.update({'pr_auc':None, 'roc_auc':None, 'recall_threshold':None,
                       'recall_wilson95':None,'false_negative_rate_threshold':None,
                       'false_positive_rate_threshold':None,'fpr_wilson95':None})
    return result


def assert_unique_keys(df, name):
    require(set(KEY).issubset(df.columns), f'{name}: missing keys')
    require(not df[KEY].isna().any().any(), f'{name}: null key')
    require(not df.duplicated(KEY).any(), f'{name}: duplicate composite keys')


def merge_audit(pred, x, students):
    """Join by keys, NEVER by row position. Gender used only for retrospective audit."""
    require('gender' in students.columns,'Master students must include gender for I1')
    for name,frame in [('predictions',pred),('X_TEST',x)]:
        assert_unique_keys(frame,name)
    require(students.record_uid.notna().all() and students.record_uid.is_unique,
            'Student master contains missing/duplicated UIDs')
    require(len(pred)==9284 and pred.y_true.sum()==911 and pred.record_uid.nunique()==3234,
            'Frozen Test prediction shape/labels changed')
    expected={'model_id':'RF_B','model_version':MODEL_VERSION}
    for col,value in expected.items():
        require(col in pred.columns and pred[col].eq(value).all(),f'Frozen prediction {col} mismatch')
    require(pred.y_true.isin((0,1)).all(), 'Non-binary labels')
    require(pred.calibrated_probability.notna().all() and
            pred.calibrated_probability.between(0,1).all(), 'Invalid calibrated probability')
    require(pred.snapshot_term_index.isin((1,2,3)).all(), 'Unexpected snapshot terms')
    require(set(['record_uid','major_code','education_mode','cohort_year']).issubset(x.columns),
            'Missing group data in X_TEST')
    require('gender' in students.columns,'Master students must include gender for I1')
    right=x[KEY+['cohort_year','major_code','education_mode']].copy()
    students=students[['record_uid','gender']]
    joined=pred.merge(right, on=KEY, how='left', validate='one_to_one',indicator=True)
    require(joined._merge.eq('both').all() and len(joined)==len(pred), 'Missing X_TEST matches')
    joined=joined.drop(columns='_merge').merge(students,on='record_uid',how='left',validate='many_to_one',indicator=True)
    require(joined._merge.eq('both').all() and len(joined)==len(pred), 'Missing master students matches')
    joined=joined.drop(columns='_merge')
    require(joined.cohort_year.eq(2025).all(), 'Test cohort year changed')
    for col in ('major_code','education_mode','gender'):
        joined[col]=joined[col].fillna('MISSING').astype(str)
    return joined


def check_frozen_evidence(reports:Path, modeldir:Path, root:Path):
    """Read provenance and H reports; do not load estimator or re-score predictions."""
    h=read_json(reports/'frozen_test_metrics.json')
    g=read_json(reports/'model_selection_decision.json')
    provenance=read_json(reports/'reproducibility.json')
    require(h.get('status')=='PASS' and h.get('phase')=='2-H', 'Missing H success evidence')
    require(g.get('status')=='PASS' and g.get('test_accessed') is False, 'G freeze evidence changed')
    require(h.get('model_version')==g.get('model_version')==MODEL_VERSION,
            'Frozen model version mismatch')
    require(h.get('threshold')==g.get('threshold',{}).get('threshold')==THRESHOLD,
            'Frozen threshold mismatch')
    require(h['pooled']['n']==9284 and h['pooled']['positive']==911,'H report totals changed')
    require(provenance.get('model_artifact_sha256')==g.get('model_sha256'),
            'H/G artifact hashes differ')
    model=modeldir/'frozen_rf_b_pipeline.joblib'
    require(model.is_file() and sha(model)==g['model_sha256'], 'Frozen model artifact modified')
    testx=root/'splits/full_simulation/X_TEST_2025.csv.gz'
    require(testx.is_file() and sha(testx)==provenance['source_test_sha256']['X_TEST_2025.csv.gz'],
            'Test X provenance checksum mismatch')
    require(provenance.get('model_selection_decision_sha256')==sha(reports/'model_selection_decision.json'),
            'G decision modified after H')
    pred=reports/'test_predictions_frozen.csv.gz'
    require(pred.is_file(), 'Frozen H predictions missing')
    master=root/'master/students.csv.gz'
    require(master.is_file(), 'Missing master/students.csv.gz; restore Phase 1 V3.2 master table')
    return h,g,provenance,pred,testx,master


def validate_h_metrics(joined,h):
    p=joined.calibrated_probability.to_numpy(dtype=float)
    y=joined.y_true.to_numpy(dtype=int)
    require(abs(average_precision_score(y,p)-h['pooled']['pr_auc'])<1e-12,
            'H PR-AUC inconsistent with frozen predictions')
    require(abs(brier_score_loss(y,p)-h['pooled']['brier'])<1e-12,
            'H Brier inconsistent with frozen predictions')
    hk1=joined.loc[joined.snapshot_term_index.eq(1)]
    require(len(hk1)==3234 and int(hk1.y_true.sum())==248, 'HK1 partition mismatch')
    order=topk_indices(hk1.calibrated_probability, hk1.record_uid, hk1.snapshot_term_index, .10)
    recall=float(hk1.y_true.to_numpy(dtype=int)[order].sum()/hk1.y_true.sum())
    require(abs(recall-h['by_term']['1']['top10']['recall_at_k'])<1e-12,
            'H HK1 Top10 recall inconsistent with frozen predictions')


def risk_budgets(hk1):
    y=hk1.y_true.to_numpy(dtype=int)
    p=hk1.calibrated_probability.to_numpy(dtype=float)
    uid=hk1.record_uid.to_numpy(dtype=str)
    t=hk1.snapshot_term_index.to_numpy(dtype=int)
    outcome={}
    for budget in (.05,.10):
        idx=topk_indices(p,uid,t,budget)
        tp=int(y[idx].sum())
        outcome[f'top{int(budget*100)}']={'budget':budget,'alerts':len(idx),'true_positives':tp,
                                         'false_negatives':int(y.sum()-tp),
                                         'recall':float(tp/y.sum()),
                                         'miss_rate':float(1-tp/y.sum())}
    pred=p>=THRESHOLD
    tp=int(np.sum(pred & (y==1)))
    outcome['threshold']={'threshold':THRESHOLD,'alerts':int(pred.sum()),
                          'true_positives':tp,'false_negatives':int(y.sum()-tp),
                          'recall':float(tp/y.sum()),'miss_rate':float(1-tp/y.sum())}
    outcome['hk1_n']=len(hk1); outcome['hk1_positives']=int(y.sum())
    outcome['calibration']={'brier':float(brier_score_loss(y,p)), 'ece10':ece(y,p),
                            'mean_probability':float(p.mean()), 'prevalence':float(y.mean()),
                            'overprediction_mean_minus_prevalence':float(p.mean()-y.mean())}
    return outcome


def drift_table(root):
    """Descriptive covariate shift only, NEVER used for model selection/tuning."""
    base=root/'splits/full_simulation'
    specs=[('TRAIN_2022_23','X_TRAIN_2022_23.csv.gz'),
           ('VALIDATION_2024','X_VALIDATION_2024.csv.gz'),
           ('TEST_2025','X_TEST_2025.csv.gz')]
    tables=[]
    for label,name in specs:
        path=base/name
        require(path.is_file(), f'Drift input missing: {name}')
        frame=pd.read_csv(path, usecols=['cohort_year']+list(DRIFT_NUMERIC),low_memory=False)
        for year,sub in frame.groupby('cohort_year',dropna=False):
            for feature in DRIFT_NUMERIC:
                values=pd.to_numeric(sub[feature],errors='coerce')
                tables.append({'split':label,'cohort_year':int(year),'feature':feature,
                               'n':len(sub),'nonmissing':int(values.notna().sum()),
                               'missing_rate':float(values.isna().mean()),
                               'mean':float(values.mean()) if values.notna().any() else None,
                               'std':float(values.std(ddof=0)) if values.notna().any() else None})
    ref=pd.DataFrame(tables)
    train=ref.loc[ref['split']=='TRAIN_2022_23'].groupby('feature',as_index=True)
    # Re-compute pooled reference moments using n and population variance, no raw labels.
    means={}
    for feature,group in train:
        n=group.nonmissing.to_numpy(float); mu=group['mean'].to_numpy(float); sd=group['std'].to_numpy(float)
        w=float(n.sum()); mean=float(np.sum(n*mu)/w); var=float(np.sum(n*(sd**2+(mu-mean)**2))/w)
        means[feature]=(mean,math.sqrt(var))
    for r in tables:
        m,sd=means[r['feature']]
        r['train_reference_mean']=m
        r['train_reference_std']=sd
        r['standardized_mean_difference_vs_train']=((r['mean']-m)/sd if sd>1e-9 and r['mean'] is not None else None)
    return tables


def compute_audit(joined,h,root, pred_sha, model_sha):
    validate_h_metrics(joined,h)
    groups=[]
    for field in ('cohort_year','snapshot_term_index','major_code','education_mode','gender'):
        for label,subset in joined.groupby(field,dropna=False,sort=True):
            groups.append(metrics_of_group(subset,field,label))
    hk1=joined.loc[joined.snapshot_term_index.eq(1)].copy()
    for field in ('major_code','education_mode','gender'):
        for label,subset in hk1.groupby(field,dropna=False,sort=True):
            groups.append(metrics_of_group(subset,f'HK1/{field}',label))
    risk=risk_budgets(hk1)
    drift=drift_table(root)
    insufficient=[f'{r["group_by"]}={r["group_value"]}' for r in groups
                  if r['status']=='INSUFFICIENT_SUPPORT']
    return {'status':'PASS','phase':'2-I','dataset':'V3_2_MAIN/FULL_SIMULATION synthetic',
            'model_version':MODEL_VERSION,'threshold_unchanged':THRESHOLD,
            'frozen_predictions_sha256':pred_sha,'frozen_model_sha256':model_sha,
            'test_rescored':False,'model_fitted':False,'threshold_tuned':False,
            'support_rule':{'minimum_n':MIN_N,'minimum_positives':MIN_POS,'minimum_negatives':MIN_NEG,'minimum_uids':MIN_UID},
            'support_status_counts':{'SUFFICIENT':sum(r['status']=='SUFFICIENT' for r in groups),
                                     'INSUFFICIENT_SUPPORT':len(insufficient)},
            'insufficient_support_groups':insufficient,
            'subgroups':groups,'hk1_risk_budgets':risk,
            'shift_summary':{'purpose':'descriptive only','measured_numeric':list(DRIFT_NUMERIC),
                             'flag_large_abs_standardized_difference':0.2,
                             'max_abs_standardized_mean_difference':max(abs(r['standardized_mean_difference_vs_train'])
                               for r in drift if r['standardized_mean_difference_vs_train'] is not None)},
            'limitations':['Synthetic generator, not measured population of CMC students',
                           'Test consists only of 2025 cohort: no 2022/23/24 Test-cohort fairness comparison',
                           'Subgroup uncertainty uses Wilson snapshot-level intervals; repeated snapshots per UID are correlated',
                           'Overall H confidence intervals use UID cluster bootstrap; subgroup Wilson intervals are descriptive, not cluster-aware',
                           'Prior Validation model selection prevents an unbiased fairness selection claim',
                           'Protected-group audit uses retrospective metadata, not serving features',
                           'No causality inferred from feature correlations or importance',
                           'All Test subgroup analyses are diagnostic, not justification to change the frozen model']}, drift


def save_report(reportdir, report, drift, pred_source):
    reportdir.mkdir(parents=True,exist_ok=True)
    for name in OUTPUTS:
        require(not (reportdir/name).exists(),f'Phase I output already exists: {name}. Do not overwrite')
    staging=reportdir/'.phase2_i_staging'
    require(not staging.exists(), 'Phase I staging already exists; inspect manually')
    staging.mkdir()
    try:
        report['created_utc']=datetime.now(timezone.utc).isoformat()
        (staging/'subgroup_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (staging/'hk1_risk_audit.json').write_text(json.dumps(report['hk1_risk_budgets'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        pd.DataFrame(report['subgroups']).to_csv(staging/'subgroup_metrics.csv',index=False,encoding='utf-8')
        pd.DataFrame(drift).to_csv(staging/'cohort_shift_audit.csv',index=False,encoding='utf-8')
        risk=report['hk1_risk_budgets']
        lines=[
            '# Phase 2-I: synthetic subgroup and stability audit','',
            '**Status:** PASS (audit completeness only). **Test rescored:** NO. **Frozen model/threshold:** unchanged.',
            f'**Frozen model:** {MODEL_VERSION}. **Threshold:** {THRESHOLD}.','',
            '## HK1 operational risk',
            '| Policy | Alerts | Correctly flagged positives | Missed positives | Recall |',
            '|---|---:|---:|---:|---:|',
        ]
        for label in ('threshold','top5','top10'):
            r=risk[label]
            lines.append(f'| {label} | {r["alerts"]} | {r["true_positives"]} | {r["false_negatives"]} | {r["recall"]:.4f} |')
        lines+=['',f'HK1 calibration: Brier {risk["calibration"]["brier"]:.6f}, ECE10 {risk["calibration"]["ece10"]:.6f}, '
                 f'mean predicted probability minus observed prevalence {risk["calibration"]["overprediction_mean_minus_prevalence"]:+.6f}.',
                '', '## Subgroups', '',
                f'Sufficient groups: {report["support_status_counts"]["SUFFICIENT"]}. '
                f'Insufficient support: {report["support_status_counts"]["INSUFFICIENT_SUPPORT"]}.',
                '', '| Group | Value | N | Positive | Prevalence | Support | PR-AUC | FNR@0.27 |',
                '|---|---|---:|---:|---:|---|---:|---:|']
        for r in report['subgroups']:
            fmt=lambda x: 'N/A' if x is None else f'{x:.4f}'
            lines.append(f'| {r["group_by"]} | {r["group_value"]} | {r["n"]} | {r["positives"]} | '
                         f'{r["prevalence"]:.4f} | {r["status"]} | {fmt(r["pr_auc"])} | '
                         f'{fmt(r["false_negative_rate_threshold"])} |')
        lines+=['','## Shift and limitations','',
                'Covariate-shift descriptors by train/validation/test cohort are in `cohort_shift_audit.csv`.',
                'Standardized mean differences are descriptive, not proof of deployment transferability.',
                'Test 2025 contains a single cohort, so cohort-comparison Test metrics cannot be inferred.',
                'Subgroup Wilson intervals treat snapshots as independent; one UID may contribute multiple snapshots.',
                'For overall UID-cluster intervals see `test_bootstrap_ci.json` (Phase H).',
                'Insufficient-support group metrics are intentionally suppressed.',
                'No claim of fairness or actual CMC performance; no threshold/model retuning.',
                'Never serve labels or historical Test predictions in UI.']
        (staging/'subgroup_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        require(set(f.name for f in staging.iterdir())==set(OUTPUTS), 'Incomplete I reports')
        for name in OUTPUTS:
            require(not (reportdir/name).exists(), 'I report already exists')
        for name in OUTPUTS:
            (staging/name).replace(reportdir/name)
    finally:
        if staging.exists():
            for file in staging.iterdir(): file.unlink()
            staging.rmdir()


def main():
    ap=argparse.ArgumentParser(description='Phase 2-I: read-only analysis of frozen H predictions')
    act=ap.add_mutually_exclusive_group(required=True)
    act.add_argument('--preflight',action='store_true')
    act.add_argument('--run',action='store_true')
    ap.add_argument('--confirm-audit',action='store_true')
    args=ap.parse_args()
    if args.run and not args.confirm_audit:
        ap.error('--run needs --confirm-audit')
    root=PROJECT/'data/external/v3_2_main'
    reports=PROJECT/'reports/ml_v3_2'
    modeldir=PROJECT/'models/v3_2'
    h,g,provenance,predfile,testx,master=check_frozen_evidence(reports,modeldir,root)
    if args.preflight:
        print('PHASE_2_I_PREFLIGHT_PASS; H/G SHA verified; no model loaded; Test not rescored')
        return
    for name in OUTPUTS:
        require(not (reports/name).exists(),f'I report already exists: {name}')
    pred_sha=sha(predfile)
    pred=pd.read_csv(predfile,dtype={'record_uid':'string'},low_memory=False)
    x=pd.read_csv(testx,dtype={'record_uid':'string'},usecols=KEY+['cohort_year','major_code','education_mode'],low_memory=False)
    students=pd.read_csv(master,dtype={'record_uid':'string'},usecols=['record_uid','gender'],low_memory=False)
    merged=merge_audit(pred,x,students)
    report,drift=compute_audit(merged,h,root,pred_sha,g['model_sha256'])
    save_report(reports,report,drift,predfile)
    risk=report['hk1_risk_budgets']
    print(f'PHASE_2_I_PASS; subgroups={len(report["subgroups"])}; '
          f'insufficient={report["support_status_counts"]["INSUFFICIENT_SUPPORT"]}; '
          f'HK1 Top10 recall={risk["top10"]["recall"]:.6f}; '
          'model=FROZEN; Test rescored=NO')


if __name__=='__main__':
    main()
