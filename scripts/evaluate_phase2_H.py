"""Phase 2-H: ONE-SHOT frozen evaluation on FULL_SIMULATION TEST_2025.

Never fits or selects a model/threshold. The preflight does not read Test data.
Execution is guarded by immutable Phase G evidence and refuses pre-existing H reports.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.features import B5_FEATURES
from src.ml_v32 import join_x_y, validate_feature_frame, validate_timestamps
from src.model_policy_v32 import apply_platt, check_probs, expected_calibration_error, ranked_budget

SEED = 221026
TEST = "TEST_2025"
EXPECTED = {'n': 9284, 'positive': 911, 'uids': 3234,
            'by_term': {1:(3234,248), 2:(3090,327), 3:(2960,336)}}
REFERENCE_B5 = {'pooled_pr_auc': 0.5404592363,
                'hk1_pr_auc': 0.2824202049,
                'hk1_recall_at_top10': 0.3790322581}
REPS = 400
OUTPUT_NAMES = (
    'frozen_test_metrics.json', 'frozen_test_metrics.md',
    'per_term_metrics.csv', 'test_predictions_frozen.csv.gz',
    'test_bootstrap_ci.json', 'reproducibility.json'
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, explanation: str) -> None:
    if not bool(cond):
        raise ValueError(explanation)


def read_json(path: Path) -> dict:
    require(path.is_file(), f'Missing required evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_pretest(root: Path, report_dir: Path, model_dir: Path) -> tuple[dict, Path, Path]:
    """No Test contents accessed here; only names checked for presence."""
    gpath = report_dir / 'model_selection_decision.json'
    fpath = report_dir / 'model_comparison.json'
    b5report = report_dir / 'b5_training_provisional.json'
    g = read_json(gpath)
    require(g.get('status') == 'PASS' and g.get('mode') == 'full' and
            g.get('phase') == '2-G' and g.get('test_accessed') is False,
            'Phase G full freeze is not verified')
    require(g.get('dataset') == 'V3_2_MAIN/FULL_SIMULATION synthetic',
            'Wrong source scenario for frozen model')
    require(g.get('model_id') == 'RF_B' and g.get('model_version') == 'v3.2-rf-b-platt-g1',
            'Frozen model selection changed')
    require(g.get('target') == 'dropout_next_2_terms' and
            g.get('features_ordered') == B5_FEATURES and g.get('seed') == SEED,
            'Feature / target / seed changed after G')
    threshold = g.get('threshold',{}).get('threshold')
    require(threshold == .27 and g.get('alert_policy',{}).get('top_budgets') == [.05,.10],
            'Frozen threshold or top-k policy changed')
    require(g.get('model_artifact','').replace('\\','/') ==
            'models/v3_2/frozen_rf_b_pipeline.joblib', 'Unexpected frozen model artifact')
    frozen = model_dir/'frozen_rf_b_pipeline.joblib'
    require(frozen.is_file() and sha256(frozen) == g.get('model_sha256'),
            'Frozen RF_B model checksum mismatch: STOP before Test')
    require(fpath.is_file() and sha256(fpath) == g.get('source_f_report_sha256'),
            'Frozen F-selection source report changed')
    for relpath, expected in g.get('source_code_sha256', {}).items():
        require(relpath in {'scripts/compare_phase2_F.py', 'scripts/freeze_phase2_G.py',
                            'src/model_policy_v32.py'}, 'Unexpected G source-code hash path')
        require(sha256(PROJECT/relpath) == expected,
                f'Phase G selection source code changed: {relpath}')
    source_hashes = g.get('source_input_sha256',{})
    require(set(source_hashes) == {f'{xy}_{s}.csv.gz' for xy in ('X','y')
                                   for s in ('TRAIN_2022_23','VALIDATION_2024')},
            'Incomplete original train/validation provenance')
    for name, expected in source_hashes.items():
        path = root/'splits/full_simulation'/name
        require(path.is_file() and sha256(path) == expected,
                f'Original Train/Validation source changed: {name}')
    b5 = read_json(b5report)
    require(b5.get('status') == 'PASS' and b5.get('test_accessed') is False and
            b5.get('train_rows_fitted') == 34304 and
            b5.get('validation_rows_transformed') == 13981,
            'B5 reference fitting evidence unavailable')
    b5model = model_dir/'b5_reference_pipeline.joblib'
    require(b5model.is_file() and sha256(b5model) == b5.get('model_sha256'),
            'B5 reference artifact checksum mismatch: STOP before Test')
    require(root.name == 'v3_2_main' and (root/'manifests/generation_manifest.json').is_file(),
            'Expected Phase 1 v3_2_main source directory')
    manifest = read_json(root/'manifests/generation_manifest.json')
    require(manifest.get('generator_version') == '3.2.0' and
            manifest.get('scenario_id') == 'V3_2_MAIN' and
            manifest.get('is_synthetic') is True, 'Synthetic V3_2_MAIN manifest mismatch')
    for x in ('X','y'):
        require((root/'splits/full_simulation'/f'{x}_{TEST}.csv.gz').is_file(),
                f'Missing FULL_SIMULATION {x}_{TEST} file')
    for output in OUTPUT_NAMES:
        require(not (report_dir/output).exists(),
                f'Frozen Test already reported ({output}); rerun prohibited without audit')
    return g, frozen, b5model


def binary_metrics(y, p, uid, terms, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = check_probs(p)
    uid = np.asarray(uid, dtype=str)
    terms = np.asarray(terms, dtype=int)
    require(len(y) == len(p) == len(uid) == len(terms) and len(y) > 0,
            'Metric input lengths wrong')
    require(set(np.unique(y)).issubset({0,1}), 'Nonbinary Test labels')
    require(0 < threshold < 1, 'Bad threshold')
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    n = int(len(y))
    positives = int(y.sum())
    return {
        'n': n, 'positive': positives, 'prevalence': positives/n,
        'pr_auc': float(average_precision_score(y, p)) if positives else None,
        'pr_auc_lift': float(average_precision_score(y, p)/(positives/n)) if positives else None,
        'roc_auc': float(roc_auc_score(y, p)) if 0 < positives < n else None,
        'brier': float(brier_score_loss(y,p)),
        'ece10': expected_calibration_error(y,p),
        'threshold': threshold,
        'f1_at_threshold': float(f1_score(y, pred, zero_division=0)),
        'precision_at_threshold': float(precision_score(y,pred,zero_division=0)),
        'recall_at_threshold': float(recall_score(y,pred,zero_division=0)),
        'confusion_matrix': [[tn,fp],[fn,tp]],
        'alerts_at_threshold': int(pred.sum()),
        'top5': ranked_budget(y,p,uid,terms,.05),
        'top10': ranked_budget(y,p,uid,terms,.10)
    }


def stratify_by_term(y, p, uid, terms, threshold: float) -> dict[str, dict]:
    t = np.asarray(terms, dtype=int)
    require(set(t) == {1,2,3}, 'Expected HK1/HK2/HK3')
    return {str(term): binary_metrics(np.asarray(y)[t==term], np.asarray(p)[t==term],
                                      np.asarray(uid)[t==term],t[t==term],threshold)
            for term in (1,2,3)}


def bootstrap_uid(y, p, uid, term, threshold: float, reps: int=REPS, seed: int=SEED+1) -> dict:
    """Student-cluster bootstrap: resample all term snapshots for each sampled UID.

    Replicate IDs are unique even if same UID is resampled multiple times.
    """
    y = np.asarray(y,dtype=int)
    p = np.asarray(p,dtype=float)
    u = np.asarray(uid,dtype=str)
    t = np.asarray(term,dtype=int)
    unique, reverse = np.unique(u,return_inverse=True)
    require(reps>=100 and len(unique)>50, 'Bootstrap sample size too small')
    groups = [np.where(reverse==i)[0] for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    vals = {'pooled_pr_auc':[], 'pooled_brier':[], 'pooled_f1_at_threshold':[],
            'hk1_pr_auc':[], 'hk1_recall_at_top10':[]}
    for _ in range(reps):
        draws = rng.integers(len(unique),size=len(unique))
        idx = np.concatenate([groups[i] for i in draws])
        # Draw-slot provides stable unique ranking tie-break even for repeat UIDs.
        slot = np.repeat(np.arange(len(draws)), [len(groups[i]) for i in draws])
        by, bp, bt = y[idx], p[idx], t[idx]
        hk = bt == 1
        if len(np.unique(by)) != 2 or len(np.unique(by[hk])) != 2:
            continue
        vals['pooled_pr_auc'].append(float(average_precision_score(by,bp)))
        vals['pooled_brier'].append(float(np.mean((bp-by)**2)))
        vals['pooled_f1_at_threshold'].append(float(f1_score(by,bp>=threshold,zero_division=0)))
        vals['hk1_pr_auc'].append(float(average_precision_score(by[hk],bp[hk])))
        hk_y, hk_p, hk_slot = by[hk],bp[hk],slot[hk]
        rank = np.lexsort((hk_slot,u[idx][hk],-hk_p))
        topk = math.ceil(len(hk_y)*.10)
        vals['hk1_recall_at_top10'].append(float(hk_y[rank[:topk]].sum()/hk_y.sum()))
    require(len(vals['pooled_pr_auc'])>=reps*.95, 'Insufficient valid bootstrap replicates')
    return {'method':'cluster_percentile_95pct', 'cluster':'record_uid',
            'sampling':'UID with replacement; all HK snapshots preserved per draw',
            'seed':seed,'requested_reps':reps,'valid_reps':len(vals['pooled_pr_auc']),
            'intervals':{k:{'ci_2_5':float(np.quantile(v,.025)),
                             'ci_97_5':float(np.quantile(v,.975))}
                         for k,v in vals.items()}}


def get_test_partition(root: Path):
    base = root/'splits/full_simulation'
    x = pd.read_csv(base/f'X_{TEST}.csv.gz', dtype={'record_uid':'string'},low_memory=False)
    y = pd.read_csv(base/f'y_{TEST}.csv.gz', dtype={'record_uid':'string'},low_memory=False)
    features, labels, keys = join_x_y(x,y)
    require(len(labels)==EXPECTED['n'] and int(labels.sum())==EXPECTED['positive'] and
            keys['record_uid'].nunique()==EXPECTED['uids'], 'Frozen Test totals mismatch')
    require(set(x['cohort_year'].unique()) == {2025}, 'Frozen Test cohort not 2025')
    term = keys['snapshot_term_index'].to_numpy(dtype=int)
    for k,(n,pos) in EXPECTED['by_term'].items():
        mask = term == k
        require(int(mask.sum()) == n and int(labels[mask].sum())==pos,
                f'Frozen Test HK{k} expected rows/positives mismatch')
    return features, labels.to_numpy(dtype=int), keys, x


def verify_reference_metrics(ref: dict) -> dict:
    actual = {'pooled_pr_auc':ref['pooled']['pr_auc'],
              'hk1_pr_auc':ref['by_term']['1']['pr_auc'],
              'hk1_recall_at_top10':ref['by_term']['1']['top10']['recall_at_k']}
    tolerances={'pooled_pr_auc':.005,'hk1_pr_auc':.005,'hk1_recall_at_top10':.015}
    checks = {k:{'expected':v, 'observed':float(actual[k]),
                 'abs_delta':float(abs(actual[k]-v)),'tolerance':tolerances[k],
                 'match':abs(actual[k]-v)<=tolerances[k]}
              for k,v in REFERENCE_B5.items()}
    return {'status':'MATCH' if all(x['match'] for x in checks.values()) else 'INVESTIGATE',
            'reference':'V3_2_MAIN generator B5 (not RF_B) TEST benchmark',
            'checks':checks}


def write_atomic_csv(path:Path, rows:list[dict]):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description='Frozen Test 2025, Phase 2-H, one shot')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true',help='No Test contents opened')
    mode.add_argument('--run',action='store_true',help='Score Test exactly once after freeze')
    p.add_argument('--confirm-frozen-test',action='store_true')
    p.add_argument('--data-root',type=Path,default=PROJECT/'data/external/v3_2_main')
    p.add_argument('--report-dir',type=Path,default=PROJECT/'reports/ml_v3_2')
    p.add_argument('--model-dir',type=Path,default=PROJECT/'models/v3_2')
    args=p.parse_args()
    if args.run and not args.confirm_frozen_test:
        p.error('--run requires --confirm-frozen-test')
    root,reportdir,modeldir=(args.data_root.resolve(),args.report_dir.resolve(),args.model_dir.resolve())
    require(root== (PROJECT/'data/external/v3_2_main').resolve(), 'Do not redirect to another dataset')
    require(reportdir==(PROJECT/'reports/ml_v3_2').resolve(), 'Do not redirect Test reports')
    require(modeldir==(PROJECT/'models/v3_2').resolve(), 'Do not redirect frozen artifacts')
    g,frozen,b5model=verify_pretest(root,reportdir,modeldir)
    if args.preflight:
        print('PHASE_2_H_PREFLIGHT_PASS; freeze SHA verified; threshold=0.27; Test data NOT opened')
        return
    # IMPORTANT: do not score any Test before the immutable preflight above.
    bundle=joblib.load(frozen)   # trusted locally frozen artifact; SHA verified above
    baseline=joblib.load(b5model)  # trusted locally produced B5 reference
    for name in ('pipeline','calibrator','feature_columns','target','model_id',
                 'model_version','threshold','alert_policy'):
        require(name in bundle, f'Missing frozen bundle key: {name}')
    require(bundle['feature_columns']==B5_FEATURES and bundle['model_id']=='RF_B' and
            bundle['model_version']==g['model_version'] and bundle['target']==g['target'] and
            bundle['threshold']==g['threshold']['threshold'], 'Frozen model metadata mismatch')
    require(baseline.get('feature_columns') == B5_FEATURES and
            baseline.get('target') == 'dropout_next_2_terms', 'B5 reference schema mismatch')
    X,y,keys,meta=get_test_partition(root)
    validate_feature_frame(X)
    # Source temporal contract validated once for Test; this is independent of y.
    snap=pd.read_csv(root/'features/prediction_snapshots.csv.gz',
                     usecols=['record_uid','snapshot_term_index','snapshot_cutoff','feature_source_latest_at'],
                     dtype={'record_uid':'string'},low_memory=False)
    validate_timestamps(snap,keys)
    uid=keys['record_uid'].astype(str).to_numpy()
    term=keys['snapshot_term_index'].to_numpy(dtype=int)
    raw=bundle['pipeline'].predict_proba(X.loc[:,B5_FEATURES])[:,1]
    score=apply_platt(bundle['calibrator'],raw)
    basep=baseline['pipeline'].predict_proba(X.loc[:,B5_FEATURES])[:,1]
    require(np.isfinite(score).all() and (score>=0).all() and (score<=1).all(),
            'Frozen RF_B Test scores invalid')
    threshold=float(g['threshold']['threshold'])
    pooled=binary_metrics(y,score,uid,term,threshold)
    terms=stratify_by_term(y,score,uid,term,threshold)
    b5={'pooled':binary_metrics(y,basep,uid,term,.21),
        'by_term':stratify_by_term(y,basep,uid,term,.21)}
    reference=verify_reference_metrics(b5)
    # H5 mismatch must be investigated, not tuned away.
    require(reference['status']=='MATCH',
            f'Generator B5 TEST reproduction mismatch: {reference["checks"]}')
    ci=bootstrap_uid(y,score,uid,term,threshold)
    by_eligible={}
    if 'administrative_action_eligible_at_cutoff' in meta.columns:
        for label,value in [('eligible',True),('not_eligible',False)]:
            mask=meta['administrative_action_eligible_at_cutoff'].eq(value).fillna(False).to_numpy(dtype=bool)
            if mask.any():
                by_eligible[label]=binary_metrics(y[mask],score[mask],uid[mask],term[mask],threshold)
    created=datetime.now(timezone.utc).isoformat()
    report={'status':'PASS','phase':'2-H','dataset':'V3_2_MAIN/FULL_SIMULATION synthetic',
            'test_split':TEST,'test_accessed':True,'frozen_at_utc':g['created_utc'],
            'evaluated_at_utc':created,'model_id':g['model_id'],
            'model_version':g['model_version'],'threshold':threshold,
            'threshold_selection':'FIXED in Phase G, no Test tuning',
            'alert_policy':g['alert_policy'], 'pooled':pooled,'by_term':terms,
            'hk1_policy':{'threshold_recall':terms['1']['recall_at_threshold'],
                          'top10_recall':terms['1']['top10']['recall_at_k'],
                          'top10_budget_n':terms['1']['top10']['k'],
                          'not_interchangeable':True},
            'b5_reference_test':{'pooled':b5['pooled'],'by_term':b5['by_term'],
                                 'generator_reference_check':reference},
            'cluster_bootstrap_ci':ci,'eligible_subset_optional':by_eligible,
            'test_evaluation_once':True,
            'notes':['No model or calibration fitting; no threshold adjustment',
                     'Test contains synthetic future outcomes for research-only evaluation',
                     'Test predictions with true labels must not be loaded in serving UI']}
    # compute hashes of Test and frozen provenance after scoring; new output files only.
    provenance={'status':'PASS','phase':'2-H','created_at_utc':created,
                'model_artifact_sha256':sha256(frozen),
                'model_selection_decision_sha256':sha256(reportdir/'model_selection_decision.json'),
                'model_comparison_sha256':sha256(reportdir/'model_comparison.json'),
                'b5_reference_artifact_sha256':sha256(b5model),
                'source_test_sha256':{f'{xy}_{TEST}.csv.gz':sha256(root/'splits/full_simulation'/f'{xy}_{TEST}.csv.gz')
                                      for xy in ('X','y')},
                'features_ordered':B5_FEATURES,'model_version':g['model_version'],
                'threshold_frozen':threshold,'bootstrap_seed':SEED+1,
                'bootstrap_reps':REPS,
                'environment':{'python':platform.python_version(),**{
                    name:version(name) for name in ('numpy','pandas','scikit-learn','joblib')}},
                'script_sha256':sha256(PROJECT/'scripts/evaluate_phase2_H.py')}
    per_term_rows=[]
    for name,entry in [('POOLED',pooled),('HK1',terms['1']),('HK2',terms['2']),('HK3',terms['3'])]:
        per_term_rows.append({'group':name,'n':entry['n'],'positives':entry['positive'],
                              'prevalence':entry['prevalence'],'pr_auc':entry['pr_auc'],
                              'roc_auc':entry['roc_auc'],'brier':entry['brier'],
                              'ece10':entry['ece10'],'threshold':threshold,
                              'f1_at_threshold':entry['f1_at_threshold'],
                              'precision_at_threshold':entry['precision_at_threshold'],
                              'recall_at_threshold':entry['recall_at_threshold'],
                              'top5_k':entry['top5']['k'],
                              'top5_precision':entry['top5']['precision_at_k'],
                              'top5_recall':entry['top5']['recall_at_k'],
                              'top10_k':entry['top10']['k'],
                              'top10_precision':entry['top10']['precision_at_k'],
                              'top10_recall':entry['top10']['recall_at_k']})
    # Temporary outputs become official only after every score/invariant is checked.
    prediction=pd.DataFrame({
        'record_uid':uid,'snapshot_term_index':term,'y_true':y,
        'raw_probability':raw,'calibrated_probability':score,
        'pred_at_frozen_threshold':(score>=threshold).astype(int),
        'model_id':bundle['model_id'],'model_version':bundle['model_version']})
    reportdir.mkdir(parents=True,exist_ok=True)
    staging=reportdir/'.phase2_h_staging'
    require(not staging.exists(),'Leftover H temp output; inspect manually before proceeding')
    staging.mkdir()
    try:
        for filename,obj in [('frozen_test_metrics.json',report),('test_bootstrap_ci.json',ci),
                             ('reproducibility.json',provenance)]:
            (staging/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        write_atomic_csv(staging/'per_term_metrics.csv',per_term_rows)
        with gzip.open(staging/'test_predictions_frozen.csv.gz','wt',encoding='utf-8',newline='') as fp:
            prediction.to_csv(fp,index=False)
        lines=['# Phase 2-H: Frozen Test 2025 (synthetic, one-shot)', '',
               f'**Status:** PASS. **Frozen before Test:** {g["created_utc"]}.',
               f'**Model:** {g["model_version"]} / RF_B + Platt; **frozen threshold:** {threshold:.2f}.',
               f'**Evaluated:** {created}; **Test 2025:** {pooled["n"]:,} snapshots, {pooled["positive"]} positives.', '',
               '| Set | PR-AUC | ROC-AUC | Brier | F1@threshold | Recall@threshold | Recall@Top10% |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name,entry in [('Pooled',pooled),('HK1',terms['1']),('HK2',terms['2']),('HK3',terms['3'])]:
            lines.append(f'| {name} | {entry["pr_auc"]:.6f} | {entry["roc_auc"]:.6f} | '
                         f'{entry["brier"]:.6f} | {entry["f1_at_threshold"]:.6f} | '
                         f'{entry["recall_at_threshold"]:.6f} | {entry["top10"]["recall_at_k"]:.6f} |')
        lines += ['',f'HK1 Top10% budget: **{terms["1"]["top10"]["k"]}** / '
                  f'{terms["1"]["n"]} snapshots; recall threshold is a separate quantity.',
                  f'B5 Test reference comparison: **{reference["status"]}**.',
                  'Bootstrap 95% cluster CI by `record_uid` in `test_bootstrap_ci.json`.',
                  'No refit, retuning, recalibration or Test-based model selection.',
                  'Synthetic only; no real CMC deployment claim.']
        (staging/'frozen_test_metrics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        require(set(p.name for p in staging.iterdir())==set(OUTPUT_NAMES),
                'Output staging incomplete')
        for name in OUTPUT_NAMES:
            require(not (reportdir/name).exists(),'H result exists: never overwrite')
        for name in OUTPUT_NAMES:
            (staging/name).replace(reportdir/name)
    finally:
        if staging.exists():
            for f in staging.iterdir():
                f.unlink()
            staging.rmdir()
    print(f'PHASE_2_H_PASS; Test={pooled["n"]:,}; positive={pooled["positive"]}; '
          f'HK1 PR-AUC={terms["1"]["pr_auc"]:.6f}; '
          f'HK1 Recall@Top10={terms["1"]["top10"]["recall_at_k"]:.6f}; '
          'B5 reference=MATCH; model=FROZEN; Test retune=NO')


if __name__=='__main__':
    main()
