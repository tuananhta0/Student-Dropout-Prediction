"""Phase 2-G: pre-Test freeze of RF_B calibration, F1 threshold and Top-k policy.

All fitting is Train only except Platt calibration on UID-disjoint Validation
calibration subset. Threshold is selected on remaining Validation policy UIDs.
Never opens Test or ASOF files, never overwrites existing freeze artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.features import B5_FEATURES
from src.ml_v32 import V32TrainingLoader, validate_feature_frame
from src.model_policy_v32 import (SEED, apply_platt, best_f1_threshold,
                                  expected_calibration_error, fit_platt,
                                  ranked_budget, split_uids)
from scripts.compare_phase2_F import TRIALS, build_trial

WINNER = 'RF_B'
MODEL_VERSION = 'v3.2-rf-b-platt-g1'
CALIBRATION_FRACTION = .30


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            d.update(block)
    return d.hexdigest()


def assess(y, p, uids, terms):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(p, dtype=float)
    base = {'n': int(len(y)), 'positive': int(y.sum()), 'prevalence': float(y.mean()),
            'pr_auc': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
            'brier': float(brier_score_loss(y, scores)),
            'ece_10bin': expected_calibration_error(y, scores),
            'top5': ranked_budget(y, scores, uids, terms, .05),
            'top10': ranked_budget(y, scores, uids, terms, .10)}
    return base


def validate_previous_F(f_report, root: Path):
    if not f_report.is_file():
        raise FileNotFoundError('Missing Phase F full report')
    doc = json.loads(f_report.read_text(encoding='utf-8-sig'))
    expected = {'phase': '2-F', 'status': 'PASS', 'mode': 'full',
                'test_accessed': False, 'scenario': 'V3_2_MAIN',
                'target': 'dropout_next_2_terms', 'seed': SEED,
                'feature_count': 58, 'train_rows': 34304, 'val_rows': 13981,
                'trial_budget_predefined': 7}
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ValueError(f'Phase F evidence mismatch: {key}')
    if doc.get('features_ordered') != B5_FEATURES:
        raise ValueError('Feature contract has changed from Phase F')
    if doc.get('trial_configs') != [{'trial': name, 'family': family, 'params': params} for name, family, params in TRIALS]:
        raise ValueError('F trial registry changed')
    for file, previous in doc['input_hashes'].items():
        if file not in {f'{xy}_{split}.csv.gz' for xy in ('X','y')
                        for split in ('TRAIN_2022_23','VALIDATION_2024')}:
            raise ValueError(f'Unexpected F hash target: {file}')
        now = sha256(root / 'splits/full_simulation' / file)
        if now != previous:
            raise ValueError(f'Data changed since F: {file}')
    results = doc.get('results', [])
    if len(results) != 7 or {r['trial'] for r in results} != {t[0] for t in TRIALS}:
        raise ValueError('Incomplete F trials')
    winner = next(r for r in results if r['trial'] == WINNER)
    lr = next(r for r in results if r['trial'] == 'LR_B5')
    hk1_gain = (winner['metrics']['by_term']['1']['pr_auc'] -
                lr['metrics']['by_term']['1']['pr_auc'])
    if hk1_gain < .01:
        raise ValueError('RF_B failed predeclared HK1 improvement vs LR_B5')
    if doc['F_recommendation']['preliminary_recommendation'] != WINNER:
        raise ValueError('Phase F recommendation is not RF_B')
    return doc, winner, lr, float(hk1_gain)


def main():
    p = argparse.ArgumentParser(description='Freeze G1-G5 before Test 2025')
    p.add_argument('--mode', choices=('smoke','full'), default='smoke')
    p.add_argument('--confirm-freeze', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--data-root', type=Path, default=PROJECT / 'data/external/v3_2_main')
    p.add_argument('--report-dir', type=Path, default=PROJECT / 'reports/ml_v3_2')
    p.add_argument('--model-dir', type=Path, default=PROJECT / 'models/v3_2')
    args = p.parse_args()
    if args.mode == 'full' and not args.confirm_freeze:
        p.error('Full freeze requires --confirm-freeze')
    if args.threads < 1 or args.threads > 16:
        p.error('1 <= threads <= 16 required')
    root = args.data_root.resolve()
    reportdir = args.report_dir.resolve()
    modeldir = args.model_dir.resolve()
    if reportdir.is_relative_to(root) or modeldir.is_relative_to(root):
        p.error('Output must not be inside dataset root')
    if not modeldir.is_relative_to((PROJECT / 'models/v3_2').resolve()):
        p.error('Model must be under models/v3_2')
    if reportdir == root:
        p.error('Invalid report destination')
    frozen_model = modeldir / 'frozen_rf_b_pipeline.joblib'
    freeze_json = reportdir / 'model_selection_decision.json'
    freeze_md = reportdir / 'model_selection_decision.md'
    smoke_json = reportdir / 'g_smoke_selection.json'
    targets = [smoke_json] if args.mode == 'smoke' else [frozen_model, freeze_json, freeze_md]
    if any(path.exists() for path in targets):
        raise FileExistsError('Existing output: ' + ', '.join(str(x) for x in targets if x.exists()))
    previous, winner, lr, hk1_gain = validate_previous_F(reportdir/'model_comparison.json', root)
    loader = V32TrainingLoader(root)
    train, val = loader.load('TRAIN_2022_23'), loader.load('VALIDATION_2024')
    validate_feature_frame(val.features, reference=train.features)
    if set(train.keys.record_uid) & set(val.keys.record_uid):
        raise ValueError('Train and Validation UID overlap')
    cal_mask, policy_mask = split_uids(val.keys.record_uid.to_numpy(dtype=str),
                                      CALIBRATION_FRACTION, SEED)
    yval = val.labels.to_numpy(dtype=int)
    uids = val.keys.record_uid.to_numpy(dtype=str)
    terms = val.keys.snapshot_term_index.to_numpy(dtype=int)
    if set(np.unique(yval[cal_mask])) != {0,1} or set(np.unique(yval[policy_mask])) != {0,1}:
        raise ValueError('Both validation groups need both classes')
    trial = next(x for x in TRIALS if x[0] == WINNER)
    pipeline = build_trial(trial, threads=args.threads)
    if args.mode == 'smoke':
        rng = np.random.default_rng(SEED)
        train_idx = np.sort(rng.choice(len(train.labels), 1600, replace=False))
        Xtrain, ytrain = train.features.iloc[train_idx], train.labels.iloc[train_idx]
        if len(np.unique(ytrain)) != 2:
            raise ValueError('Smoke sample missing class')
        # smoke only; do not pretend resulting scores are legitimate G metrics
        cal_idx = np.where(cal_mask)[0][:300]
        policy_idx = np.where(policy_mask)[0][:500]
    else:
        Xtrain, ytrain = train.features, train.labels
        cal_idx = np.where(cal_mask)[0]
        policy_idx = np.where(policy_mask)[0]
    if len(np.unique(yval[cal_idx])) != 2 or len(np.unique(yval[policy_idx])) != 2:
        raise ValueError('Selected calibration/policy rows must contain both classes')
    pipeline.fit(Xtrain.loc[:, B5_FEATURES], ytrain)
    cal_raw = pipeline.predict_proba(val.features.iloc[cal_idx].loc[:, B5_FEATURES])[:,1]
    policy_raw = pipeline.predict_proba(val.features.iloc[policy_idx].loc[:, B5_FEATURES])[:,1]
    calibrator = fit_platt(cal_raw, yval[cal_idx])
    cal_p = apply_platt(calibrator, cal_raw)
    policy_p = apply_platt(calibrator, policy_raw)
    policy_y, policy_u, policy_t = yval[policy_idx], uids[policy_idx], terms[policy_idx]
    threshold = best_f1_threshold(policy_y, policy_p)
    policy_measures = assess(policy_y, policy_p, policy_u, policy_t)
    raw_measures = assess(policy_y, policy_raw, policy_u, policy_t)
    by_term = {}
    for term in (1,2,3):
        mask = policy_t == term
        if not mask.any():
            raise ValueError(f'Missing term in policy set: {term}')
        by_term[str(term)] = assess(policy_y[mask], policy_p[mask], policy_u[mask], policy_t[mask])
    report = {
        'status': 'PASS', 'phase': '2-G', 'mode': args.mode,
        'model_id': WINNER, 'model_version': MODEL_VERSION, 'seed': SEED,
        'test_accessed': False, 'dataset': 'V3_2_MAIN/FULL_SIMULATION synthetic',
        'target': 'dropout_next_2_terms', 'features_ordered': B5_FEATURES,
        'train_rows': len(Xtrain), 'train_positives': int(np.sum(ytrain)),
        'validation_rows': len(val.labels), 'validation_positives': int(yval.sum()),
        'calibration': {'method': 'Platt sigmoid on clipped logit(raw_probability)',
                        'fit_source': 'Validation 2024 calibration UID subset only',
                        'fraction_uids': CALIBRATION_FRACTION,
                        'uid_count': int(len(set(uids[cal_idx]))), 'rows': len(cal_idx),
                        'positives': int(yval[cal_idx].sum()),
                        'slope': float(calibrator.coef_[0,0]),
                        'intercept': float(calibrator.intercept_[0]),
                        'ece_10bin': expected_calibration_error(yval[cal_idx], cal_p),
                        'brier': float(brier_score_loss(yval[cal_idx], cal_p))},
        'policy_subset': {'uid_count': int(len(set(policy_u))), 'rows': len(policy_idx),
                          'positives': int(policy_y.sum()),
                          'raw_metrics': raw_measures, 'calibrated_metrics': policy_measures,
                          'calibrated_by_term': by_term},
        'threshold': threshold,
        'alert_policy': {'threshold_kind': 'calibrated_probability',
                         'threshold_selection': 'maximize pooled policy-UID Validation F1 on 0.05..0.80 step 0.01; tie higher threshold',
                         'top_budgets': [.05, .10], 'count': 'ceil(N * budget)',
                         'ranking': 'calibrated_probability DESC, record_uid ASC, snapshot_term_index ASC',
                         'scope': 'per snapshot term for HK1/HK2/HK3, optionally pooled',
                         'topk_independent_of_threshold': True},
        'selection': {'basis': 'F full Validation; predeclared >=0.010 HK1 PR-AUC improvement over LR_B5',
                      'hk1_pr_auc_gain_absolute_vs_lr': hk1_gain,
                      'rf_b_f_metric': winner['metrics'], 'lr_b5_f_metric': lr['metrics'],
                      'tie_break': 'prefer simpler model absent meaningful HK1 gain',
                      'limitation': 'Selection uses full Validation; policy metrics remain development-set estimates, not an independent test'},
        'source_f_report_sha256': sha256(reportdir/'model_comparison.json'),
        'source_input_sha256': previous['input_hashes'],
        'hyperparameters': trial[2],
        'environment': {'python': platform.python_version(), **{
            name: version(name) for name in ('numpy', 'pandas','scikit-learn','xgboost','lightgbm','joblib')}},
        'source_code_sha256': {'scripts/compare_phase2_F.py': sha256(PROJECT/'scripts/compare_phase2_F.py'),
                                'scripts/freeze_phase2_G.py': sha256(PROJECT/'scripts/freeze_phase2_G.py'),
                                'src/model_policy_v32.py': sha256(PROJECT/'src/model_policy_v32.py')},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'notes': 'Synthetic-only; frozen model/policy must NOT be revised based on Test 2025',
    }
    reportdir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'smoke':
        smoke_json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'PHASE_2_G_SMOKE_PASS; smoke_train={len(Xtrain)}; '
              f'cal={len(cal_idx)}; policy={len(policy_idx)}; Test accessed=NO')
        return
    # Freeze with metadata within joblib. J will expand metadata contract later.
    modeldir.mkdir(parents=True, exist_ok=True)
    bundle = {'pipeline': pipeline, 'calibrator': calibrator,
              'feature_columns': list(B5_FEATURES), 'target': 'dropout_next_2_terms',
              'model_id': WINNER, 'model_version': MODEL_VERSION,
              'threshold': threshold['threshold'], 'alert_policy': report['alert_policy'],
              'train_split': 'TRAIN_2022_23',
              'calibration_split': 'VALIDATION_2024/UID_CAL',
              'policy_split': 'VALIDATION_2024/UID_POLICY',
              'source': 'V3_2_MAIN_FULL_SIMULATION', 'seed': SEED}
    tmp = modeldir / 'frozen_rf_b_pipeline.joblib.tmp'
    if tmp.exists():
        raise FileExistsError(f'Leftover model temp file: {tmp}')
    try:
        joblib.dump(bundle, tmp)
        loaded = joblib.load(tmp)  # trusted artifact we just created
        sample = val.features.iloc[policy_idx[:37]].loc[:, B5_FEATURES]
        expected = apply_platt(calibrator, pipeline.predict_proba(sample)[:,1])
        restored = apply_platt(loaded['calibrator'], loaded['pipeline'].predict_proba(sample)[:,1])
        delta = float(np.max(np.abs(expected - restored)))
        if delta > 1e-10:
            raise RuntimeError(f'Save/load score drift: {delta}')
        tmp.replace(frozen_model)
    finally:
        if tmp.exists():
            tmp.unlink()
    report['model_artifact'] = str(frozen_model.relative_to(PROJECT))
    report['model_sha256'] = sha256(frozen_model)
    report['save_load_delta_max'] = delta
    freeze_json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = [
        '# Phase 2-G: Frozen model selection decision (PRE-TEST)', '',
        f'**Frozen at UTC:** {report["created_utc"]}  ',
        f'**Status:** PASS. **Test accessed:** NO. **Dataset:** V3_2_MAIN synthetic.', '',
        '## G1: Model selection', '',
        f'Winner: **{WINNER}** ({MODEL_VERSION}) on 58 ordered B5 features.',
        'Rule: prioritize HK1 Validation PR-AUC, HK1 Recall@Top10%, Brier/ECE and inference cost.',
        f'HK1 PR-AUC improvement vs LR_B5: **+{hk1_gain:.6f} absolute** (>=0.010 rule).',
        'RF_A excluded from final shortlist due to poor pooled Brier 0.110479.',
        'F model-selection evidence: `reports/ml_v3_2/model_comparison.json`.', '',
        '## G2: Calibration', '',
        'Platt sigmoid fitted using only Validation 2024 CALIBRATION UIDs (30% of UIDs).',
        'Policy/threshold fitted using remaining Validation 2024 POLICY UIDs (70%).',
        'All HK snapshots from one UID remain in exactly one partition.',
        f'Calibration UID count={report["calibration"]["uid_count"]}, rows={len(cal_idx)}, positives={int(yval[cal_idx].sum())}.',
        f'Policy UID count={len(set(policy_u))}, rows={len(policy_idx)}, positives={int(policy_y.sum())}.',
        'Calibration fit never sees policy labels. Model preprocessing and estimator fit only on Train.',
        'Important: full Validation was previously used for F model selection; this is not an unbiased holdout.', '',
        '## G3: Probability threshold (Validation POLICY only)', '',
        f'Threshold **{threshold["threshold"]:.2f}** maximizes F1 over fixed 0.05..0.80 grid step 0.01.',
        f'Policy F1={threshold["f1"]:.6f}, Precision={threshold["precision"]:.6f}, Recall={threshold["recall"]:.6f}.',
        f'Policy raw Brier={raw_measures["brier"]:.6f}; calibrated Brier={policy_measures["brier"]:.6f}.',
        f'Policy raw ECE10={raw_measures["ece_10bin"]:.6f}; calibrated ECE10={policy_measures["ece_10bin"]:.6f}.', '',
        '## G4: Alert budgets', '',
        'Top5% and Top10%: `ceil(N*budget)` per term, score descending, UID and term ascending for ties.',
        'Ranking budget is independent of absolute probability threshold.',
        f'Policy HK1 Recall@Top10={by_term["1"]["top10"]["recall_at_k"]:.6f}; '
        f'pooled Recall@Top10={policy_measures["top10"]["recall_at_k"]:.6f}.', '',
        '## G5: Frozen artifacts', '',
        f'- Model artifact: `{report["model_artifact"]}`',
        f'- SHA256: `{report["model_sha256"]}`',
        f'- Source Phase F report SHA256: `{report["source_f_report_sha256"]}`',
        f'- Save/load max probability delta: {delta:.3e}',
        '- Ordered 58 features and version/config/hashes in `model_selection_decision.json`.',
        '- Model frozen now. Phase H may read Test once; no retuning based on Test.',
        '- Synthetic simulation only; not validated for real CMC students.', '',
    ]
    freeze_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'PHASE_2_G_FULL_PASS; frozen={WINNER}; threshold={threshold["threshold"]:.2f}; '
          f'cal={len(cal_idx):,}; policy={len(policy_idx):,}; Test accessed=NO; report={freeze_json}')


if __name__ == '__main__':
    main()
