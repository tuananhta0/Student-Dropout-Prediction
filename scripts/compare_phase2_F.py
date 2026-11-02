"""Phase 2-F: fixed-budget ML comparison on CMC V3.2 Train/Validation only.

Never opens Test files. No threshold optimization, calibration, or final model
selection here: those belong to Phase G. All trials are pre-registered in code.

Requires existing src/ml_v32.py, src/features.py (Phase C/D).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.features import B5_FEATURES  # noqa: E402
from src.ml_v32 import (  # noqa: E402
    V32TrainingLoader, CATEGORICAL_B5_FEATURES,
    _categorical_to_object_nan, make_b5_pipeline, validate_feature_frame,
)

SEED = 221026
# Fixed budget: 1 LR reference + 2 configurations per tree family.
# Do NOT change configurations based on Test 2025.
TRIALS = (
    ('LR_B5', 'LogisticRegression', {}),
    ('RF_A', 'RandomForest', dict(n_estimators=80, max_depth=10, min_samples_leaf=10,
                                 max_features='sqrt', class_weight='balanced_subsample')),
    ('RF_B', 'RandomForest', dict(n_estimators=120, max_depth=12, min_samples_leaf=10,
                                 max_features='sqrt', class_weight=None)),
    ('XGB_A', 'XGBoost', dict(n_estimators=240, max_depth=4, learning_rate=0.06,
                             min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                             reg_lambda=5)),
    ('XGB_B', 'XGBoost', dict(n_estimators=350, max_depth=3, learning_rate=0.04,
                             min_child_weight=10, subsample=1.0, colsample_bytree=1.0,
                             reg_lambda=10)),
    ('LGBM_A', 'LightGBM', dict(n_estimators=250, num_leaves=15, learning_rate=0.05,
                               min_child_samples=40, subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.9, reg_lambda=3)),
    ('LGBM_B', 'LightGBM', dict(n_estimators=320, num_leaves=31, learning_rate=0.035,
                               min_child_samples=60, subsample=1.0, subsample_freq=0,
                               colsample_bytree=1.0, reg_lambda=8)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_preprocessing() -> ColumnTransformer:
    categorical = [c for c in B5_FEATURES if c in CATEGORICAL_B5_FEATURES]
    numeric = [c for c in B5_FEATURES if c not in CATEGORICAL_B5_FEATURES]
    if len(categorical) != 5 or len(numeric) != 53:
        raise ValueError('Unexpected B5 feature roles: must be 53 numeric + 5 categorical')
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median'))]), numeric),
        ('categorical', Pipeline([
            ('to_object', FunctionTransformer(_categorical_to_object_nan, validate=False)),
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])


def build_trial(trial: tuple, threads: int = 4) -> Pipeline:
    _, family, config = trial
    if family == 'LogisticRegression':
        raise ValueError('Call make_b5_pipeline for the reference LR')
    if family == 'RandomForest':
        model = RandomForestClassifier(**config, random_state=SEED, n_jobs=threads)
    elif family == 'XGBoost':
        model = XGBClassifier(**config, random_state=SEED, n_jobs=threads,
                              objective='binary:logistic', eval_metric='logloss',
                              tree_method='hist', verbosity=0)
    elif family == 'LightGBM':
        model = LGBMClassifier(**config, random_state=SEED, n_jobs=threads,
                               verbosity=-1, deterministic=True, force_col_wise=True)
    else:
        raise ValueError(f'Unknown model family: {family}')
    return Pipeline([('prep', tree_preprocessing()), ('clf', model)])


def alert_topk(y: np.ndarray, scores: np.ndarray, uid: np.ndarray,
               term: np.ndarray, fraction: float = .10) -> dict:
    if not (len(y) == len(scores) == len(uid) == len(term)) or not len(y):
        raise ValueError('Invalid top-k inputs')
    # Tie-breaking fixed by ascending UID and term; ranking is deterministic.
    order = np.lexsort((term.astype(int), uid.astype(str), -scores.astype(float)))
    k = max(1, math.ceil(len(y) * fraction))
    found = int(y[order[:k]].sum())
    positives = int(y.sum())
    return {'budget': fraction, 'k': k, 'positive_found': found,
            'precision_at_k': float(found / k),
            'recall_at_k': float(found / positives) if positives else None}


def metrics(y: np.ndarray, scores: np.ndarray,
            uid: np.ndarray, term: np.ndarray) -> dict:
    if len(y) == 0 or len(scores) != len(y) or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction scores')
    if np.min(scores) < 0 or np.max(scores) > 1:
        raise ValueError('Probabilities outside [0, 1]')
    positives = int(y.sum())
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, scores)) if positives else None
    out = {
        'n': int(len(y)), 'positive': positives, 'prevalence': prevalence,
        'pr_auc': ap,
        'pr_auc_lift': float(ap / prevalence) if prevalence and ap is not None else None,
        'roc_auc': float(roc_auc_score(y, scores)) if 0 < positives < len(y) else None,
        'brier': float(brier_score_loss(y, scores)),
        'f1_at_0_5': float(f1_score(y, scores >= .5, zero_division=0)),
        'precision_at_0_5': float(precision_score(y, scores >= .5, zero_division=0)),
        'recall_at_0_5': float(recall_score(y, scores >= .5, zero_division=0)),
        'top5': alert_topk(y, scores, uid, term, .05),
        'top10': alert_topk(y, scores, uid, term, .10),
    }
    return out


def select_shortlist(rows: list[dict]) -> dict:
    """Predefined heuristic, F only; NOT the frozen final G decision.

    Require >=0.01 absolute HK1 PR-AUC improvement over LR to displace it;
    otherwise LR reference is retained as a conservative default.
    Rank by HK1 PR-AUC, HK1 Recall@10, then pooled Brier.
    """
    lr = next(x for x in rows if x['trial'] == 'LR_B5')
    def sort_key(x):
        return (-x['hk1_pr_auc'], -x['hk1_recall_at_top10'], x['pooled_brier'], x['trial'])
    ranked = sorted(rows, key=sort_key)
    best = ranked[0]
    if best['trial'] != 'LR_B5' and best['hk1_pr_auc'] - lr['hk1_pr_auc'] >= .01:
        recommended = best
        reason = 'HK1 PR-AUC improves by >=0.010 absolute vs LR_B5 on Validation'
    else:
        recommended = lr
        reason = 'No candidate improves HK1 PR-AUC by >=0.010; prefer simpler LR_B5'
    return {'shortlist_for_G': [x['trial'] for x in ranked[:3]],
            'preliminary_recommendation': recommended['trial'],
            'reason': reason,
            'policy': 'Exploratory Validation-only Phase F; final selection/calibration in G'}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No rows to report')
    with path.open('w', encoding='utf-8', newline='') as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Phase 2-F: fixed-budget Train/Validation comparisons')
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--confirm-full', action='store_true', help='Required to run complete 7-trial comparison')
    p.add_argument('--data-root', type=Path, default=PROJECT / 'data/external/v3_2_main')
    p.add_argument('--report-dir', type=Path, default=PROJECT / 'reports/ml_v3_2')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke-train', type=int, default=1800)
    p.add_argument('--smoke-validation', type=int, default=900)
    args = p.parse_args()
    if args.mode == 'full' and not args.confirm_full:
        p.error('Full comparison requires --confirm-full')
    if args.threads < 1 or args.threads > 16:
        p.error('Require 1 <= threads <= 16')
    root = args.data_root.resolve()
    outdir = args.report_dir.resolve()
    if outdir.is_relative_to(root):
        p.error('Reports cannot be written inside dataset root')
    loader = V32TrainingLoader(root)
    train = loader.load('TRAIN_2022_23')
    val = loader.load('VALIDATION_2024')
    validate_feature_frame(val.features, reference=train.features)
    if set(train.keys.record_uid) & set(val.keys.record_uid):
        raise ValueError('Train/Validation UID overlap')

    if args.mode == 'smoke':
        rng = np.random.default_rng(SEED)
        train_idx = np.sort(rng.choice(len(train.labels), size=min(args.smoke_train, len(train.labels)), replace=False))
        val_idx = np.sort(rng.choice(len(val.labels), size=min(args.smoke_validation, len(val.labels)), replace=False))
        Xtrain = train.features.iloc[train_idx]
        ytrain = train.labels.iloc[train_idx].to_numpy(dtype=int)
        Xval = val.features.iloc[val_idx]
        yval = val.labels.iloc[val_idx].to_numpy(dtype=int)
        keys = val.keys.iloc[val_idx].reset_index(drop=True)
    else:
        Xtrain = train.features
        ytrain = train.labels.to_numpy(dtype=int)
        Xval = val.features
        yval = val.labels.to_numpy(dtype=int)
        keys = val.keys.reset_index(drop=True)
    if len(np.unique(ytrain)) != 2:
        raise ValueError('Training data needs both classes')
    terms = keys['snapshot_term_index'].to_numpy(dtype=int)
    uids = keys['record_uid'].to_numpy(dtype=str)
    if len(set(terms)) == 0 or not set(np.unique(terms)).issubset({1, 2, 3}):
        raise ValueError('Invalid snapshot terms')
    if (terms == 1).sum() == 0:
        raise ValueError('No HK1 validation rows')

    trials = []
    per_terms = []
    report = {
        'phase': '2-F', 'status': 'PENDING', 'mode': args.mode,
        'scenario': 'V3_2_MAIN', 'target': 'dropout_next_2_terms',
        'train_split': 'TRAIN_2022_23', 'validation_split': 'VALIDATION_2024',
        'test_accessed': False, 'seed': SEED, 'feature_count': len(B5_FEATURES),
        'features_ordered': B5_FEATURES,
        'trial_budget_predefined': len(TRIALS),
        'trial_configs': [{'trial': name, 'family': family, 'params': config}
                          for name, family, config in TRIALS],
        'preprocessing': 'Train-only; numeric median, categorical mode+OHE; scaler for LR only',
        'threshold_policy': 'F1 at fixed 0.5 exploratory; threshold optimization reserved for G',
        'group_rank_policy': 'Top5/Top10 sorted by score descending, UID/term ascending tie break',
        'environment': {'python': platform.python_version(), **{
            name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib')
        }},
        'train_rows': len(Xtrain), 'val_rows': len(Xval),
        'input_hashes': {f'{prefix}_{split}.csv.gz': sha256(root / 'splits/full_simulation' / f'{prefix}_{split}.csv.gz')
                         for prefix in ('X', 'y') for split in ('TRAIN_2022_23', 'VALIDATION_2024')},
        'results': [],
    }
    for trial in TRIALS:
        name, family, cfg = trial
        pipeline = make_b5_pipeline(Xtrain, seed=SEED) if name == 'LR_B5' else build_trial(trial, args.threads)
        started = perf_counter()
        pipeline.fit(Xtrain.loc[:, B5_FEATURES], ytrain)
        train_sec = perf_counter() - started
        started = perf_counter()
        scores = pipeline.predict_proba(Xval.loc[:, B5_FEATURES])[:, 1].astype(float)
        pred_sec = perf_counter() - started
        pooled = metrics(yval, scores, uids, terms)
        per_term = {}
        for term in (1, 2, 3):
            mask = terms == term
            if not mask.any():
                continue
            row = metrics(yval[mask], scores[mask], uids[mask], terms[mask])
            per_term[str(term)] = row
            per_terms.append({'trial': name, 'term': term, 'n': row['n'],
                              'positive': row['positive'], 'prevalence': row['prevalence'],
                              'pr_auc': row['pr_auc'], 'pr_auc_lift': row['pr_auc_lift'],
                              'roc_auc': row['roc_auc'], 'brier': row['brier'],
                              'recall_at_top5': row['top5']['recall_at_k'],
                              'recall_at_top10': row['top10']['recall_at_k'],
                              'f1_at_0_5': row['f1_at_0_5']})
        stream = io.BytesIO()
        joblib.dump(pipeline, stream)
        obj_bytes = stream.tell()
        summary = {
            'trial': name, 'family': family, 'features': len(B5_FEATURES),
            'hk1_pr_auc': per_term['1']['pr_auc'],
            'hk1_recall_at_top10': per_term['1']['top10']['recall_at_k'],
            'hk1_prevalence': per_term['1']['prevalence'],
            'pooled_pr_auc': pooled['pr_auc'], 'pooled_pr_auc_lift': pooled['pr_auc_lift'],
            'pooled_roc_auc': pooled['roc_auc'], 'pooled_brier': pooled['brier'],
            'pooled_f1_at_0_5': pooled['f1_at_0_5'],
            'pooled_recall_at_top10': pooled['top10']['recall_at_k'],
            'train_sec': round(train_sec, 4),
            'validation_predict_sec': round(pred_sec, 4),
            'model_bytes_estimate': obj_bytes,
        }
        report['results'].append({'trial': name, 'family': family, 'params': cfg,
                                  'metrics': {'pooled': pooled, 'by_term': per_term},
                                  'timing': {'train_sec': train_sec, 'val_predict_sec': pred_sec},
                                  'model_bytes_estimate': obj_bytes})
        trials.append(summary)
        print(f"{name}: HK1 PR-AUC={summary['hk1_pr_auc']:.6f}; "
              f"HK1 Recall@Top10={summary['hk1_recall_at_top10']:.4f}; "
              f"pooled PR-AUC={summary['pooled_pr_auc']:.6f}; Brier={summary['pooled_brier']:.6f}", flush=True)

    report['F_recommendation'] = select_shortlist(trials)
    report['status'] = 'PASS'
    report['run_finished_utc'] = datetime.now(timezone.utc).isoformat()
    # Never overwrite a pre-existing FULL report. Smoke has its own filenames.
    prefix = 'f_smoke_' if args.mode == 'smoke' else ''
    names = ('model_comparison.csv', 'model_comparison_by_term.csv', 'model_comparison.json', 'model_comparison.md')
    dest = [outdir / (prefix + name) for name in names]
    if any(file.exists() for file in dest):
        raise FileExistsError('Refusing to overwrite existing Phase F report: ' + ', '.join(str(f) for f in dest if f.exists()))
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(dest[0], trials)
    write_csv(dest[1], per_terms)
    dest[2].write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = [f'# Phase 2-F: model comparison ({args.mode})', '',
          'V3_2_MAIN synthetic, Train 2022–2023 / Validation 2024. Test NOT accessed.',
          '', '| Trial | HK1 PR-AUC | HK1 Recall@Top10% | Pooled PR-AUC | Brier | Fit sec |',
          '|---|---:|---:|---:|---:|---:|']
    for row in trials:
        md.append(f"| {row['trial']} | {row['hk1_pr_auc']:.6f} | {row['hk1_recall_at_top10']:.4f} | "
                  f"{row['pooled_pr_auc']:.6f} | {row['pooled_brier']:.6f} | {row['train_sec']:.2f} |")
    md += ['', f"Preliminary shortlist: {', '.join(report['F_recommendation']['shortlist_for_G'])}.",
           f"Recommendation for G analysis: {report['F_recommendation']['preliminary_recommendation']}.",
           f"Reason: {report['F_recommendation']['reason']}.",
           '', 'Threshold and calibration are NOT selected here; only Validation. All metrics are synthetic.',
           'Model size is serialized in-memory for estimation; no model artifacts saved in F.',
           'Final decision and any tuning/calibration belong to Phase G.', '']
    dest[3].write_text('\n'.join(md), encoding='utf-8')
    print(f"PHASE_2_F_{args.mode.upper()}_PASS; trials={len(TRIALS)}; Test accessed=NO; report={dest[2]}")


if __name__ == '__main__':
    main()
