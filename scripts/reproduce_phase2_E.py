"""Phase 2-E: reproduce synthetic V3.2 generator B0-B5 on Train/Validation ONLY.

No Test split is loaded or scored. Does not write or modify model artifacts.
Read benchmark *validation* metrics only from the original generator archive.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src import features as registry  # noqa: E402
from src.ml_v32 import (V32TrainingLoader, make_b5_pipeline,
                        CATEGORICAL_B5_FEATURES, _categorical_to_object_nan,
                        validate_feature_frame)  # noqa: E402

SEED = 221026
SOURCE_ROOT = 'CMC_Synthetic_Dataset_V3.2/'
SOURCE_CODE = SOURCE_ROOT + 'src/evaluate_baselines.py'
SOURCE_REPORT = SOURCE_ROOT + 'artifacts/runs/V3_2_MAIN/reports/ml_baseline_results.json'
EXPECTED_SOURCE_HASH = 'c501d1149d0d70f893689eeb5224b6d11385f330d5ad75aaee167b06e10f1ef4'
EXPECTED_BENCHMARK_HASH = '27489c8848b0a0d8259bf73f653662248fbc96f27f0058ce0e9c8405f8f18174'
GENERATOR_ZIP_NAME = 'CMC_Synthetic_Dataset_V3_2_RESEARCH_CANDIDATE.zip'
V31_EXCLUDED = (
    'lms_slope_last6w', 'attendance_slope_last4w', 'assignment_missed_streak',
    'longest_inactive_streak_6w', 'observed_concern_reports_term',
    'lms_first6_mean', 'lms_last6_mean',
)
GROUPS = ('ACADEMIC', 'BEHAVIOR', 'FINANCE', 'SUPPORT', 'CONTEXT')


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda: inp.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_reference(source_archive: Path) -> tuple[dict, dict]:
    if not source_archive.is_file():
        raise FileNotFoundError(f'Generator ZIP missing: {source_archive}\n'
                                'Pass --generator-zip with the correct local path.')
    with zipfile.ZipFile(source_archive) as archive:
        source_bytes = archive.read(SOURCE_CODE)
        report_bytes = archive.read(SOURCE_REPORT)
    if hash_bytes(source_bytes) != EXPECTED_SOURCE_HASH:
        raise ValueError('Generator evaluate_baselines.py checksum mismatch; stop and inspect source.')
    if hash_bytes(report_bytes) != EXPECTED_BENCHMARK_HASH:
        raise ValueError('Generator ml_baseline_results.json checksum mismatch; stop and inspect source.')
    tree = ast.parse(source_bytes.decode('utf-8'))
    source_lists = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in GROUPS:
                source_lists[name] = ast.literal_eval(node.value)
    for name in GROUPS:
        if name not in source_lists or list(getattr(registry, name)) != source_lists[name]:
            raise ValueError(f'Generator feature group differs from src/features.py: {name}')
    payload = json.loads(report_bytes.decode('utf-8'))
    names = ('B0_PRIOR', 'B1_GPA_ONLY', 'B2_ACADEMIC', 'B3_PLUS_BEHAVIOR',
             'B4_PLUS_FINANCE', 'B5_FULL_SAFE', 'B3_PLUS_BEHAVIOR_V31')
    # Restrict the benchmark reference to Validation fields only. TEST entries
    # present in the generator's original report are deliberately not used.
    reference = {name: payload['models'][name]['validation'] for name in names}
    provenance = {
        'generator_zip_path': str(source_archive),
        'generator_zip_sha256': sha256(source_archive),
        'generator_script_sha256': hash_bytes(source_bytes),
        'generator_benchmark_report_sha256': hash_bytes(report_bytes),
        'generator_version': 'V3_2_MAIN',
        'reference_metrics_scope': 'Validation only',
        'reference_feature_contract_identical': True,
    }
    return reference, provenance


def feature_sets() -> dict[str, list[str]]:
    a = registry.ACADEMIC
    b = registry.BEHAVIOR
    f = registry.FINANCE
    s = registry.SUPPORT
    c = registry.CONTEXT
    return {
        'B0_PRIOR': [],
        'B1_GPA_ONLY': ['gpa4_term'],
        'B2_ACADEMIC': list(a),
        'B3_PLUS_BEHAVIOR': list(a + b),
        'B4_PLUS_FINANCE': list(a + b + f),
        'B5_FULL_SAFE': list(a + b + f + s + c),
        'B3_PLUS_BEHAVIOR_V31': list(a) + [col for col in b if col not in V31_EXCLUDED],
    }


def subset_pipeline(cols: list[str]) -> Pipeline:
    cat = [name for name in cols if name in CATEGORICAL_B5_FEATURES]
    numeric = [name for name in cols if name not in CATEGORICAL_B5_FEATURES]
    transforms = []
    if numeric:
        transforms.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=False)),
            ('scale', StandardScaler()),
        ]), numeric))
    if cat:
        transforms.append(('categorical', Pipeline([
            ('to_object', FunctionTransformer(_categorical_to_object_nan, validate=False)),
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), cat))
    return Pipeline([
        ('prep', ColumnTransformer(transforms, verbose_feature_names_out=False)),
        ('clf', LogisticRegression(solver='liblinear', max_iter=350, random_state=SEED)),
    ])


def metric(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    positive = int(y.sum())
    return {
        'n': int(len(y)),
        'positive': positive,
        'prevalence': float(y.mean()),
        'pr_auc': float(average_precision_score(y, probs)) if positive else None,
        'roc_auc': float(roc_auc_score(y, probs)) if 0 < positive < len(y) else None,
        'brier': float(brier_score_loss(y, probs)),
        'threshold': float(threshold),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def top_k(y: np.ndarray, probs: np.ndarray, portion: float = .10) -> dict:
    k = max(1, math.ceil(len(y) * portion))
    idx = np.argsort(-probs, kind='stable')[:k]
    found = int(y[idx].sum())
    return {
        'k': k,
        'positive_found': found,
        'precision_at_k': float(found / k),
        'recall_at_k': float(found / max(1, int(y.sum()))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description='Phase 2-E Train/Validation-only B0–B5 reproduction')
    ap.add_argument('--data-root', type=Path, default=PROJECT / 'data/external/v3_2_main')
    ap.add_argument('--generator-zip', type=Path, required=True,
                    help='Original CMC V3.2 generator ZIP. Read-only; exact archive hash required.')
    ap.add_argument('--report-dir', type=Path, default=PROJECT / 'reports/ml_v3_2')
    ap.add_argument('--include-v31-ablation', action='store_true', help='Optional E6 additional ablation')
    args = ap.parse_args()
    # Reference verification is first, before training or writing any output.
    reference, provenance = verify_reference(args.generator_zip.resolve())
    loader = V32TrainingLoader(args.data_root)
    train = loader.load('TRAIN_2022_23')
    val = loader.load('VALIDATION_2024')
    validate_feature_frame(val.features, reference=train.features)
    if set(train.keys.record_uid) & set(val.keys.record_uid):
        raise ValueError('Cross-partition UID leakage')
    ytrain = train.labels.to_numpy(dtype=int)
    yval = val.labels.to_numpy(dtype=int)
    terms = val.keys['snapshot_term_index'].to_numpy()
    masks = {term: terms == term for term in (1, 2, 3)}
    models = feature_sets()
    if not args.include_v31_ablation:
        models.pop('B3_PLUS_BEHAVIOR_V31')
    results = {}
    comparison = []
    per_term = []
    max_b5_artifact_pred_diff = None
    for name, cols in models.items():
        if not set(cols).issubset(registry.B5_FEATURES):
            raise ValueError(f'Forbidden feature in {name}')
        if name == 'B0_PRIOR':
            prior = float(ytrain.mean())
            scores = np.full(len(yval), prior, dtype=float)
            threshold = .5
        else:
            pipeline = make_b5_pipeline(train.features) if name == 'B5_FULL_SAFE' else subset_pipeline(cols)
            pipeline.fit(train.features.loc[:, cols], ytrain)
            scores = pipeline.predict_proba(val.features.loc[:, cols])[:, 1]
            thresholds = np.linspace(.05, .8, 76)
            f1s = [f1_score(yval, scores >= th, zero_division=0) for th in thresholds]
            threshold = float(thresholds[int(np.argmax(f1s))])
            if name == 'B5_FULL_SAFE':
                artifact_path = PROJECT / 'models/v3_2/b5_reference_pipeline.joblib'
                if not artifact_path.is_file():
                    raise FileNotFoundError(f'B5 existing artifact missing: {artifact_path}')
                artifact = joblib.load(artifact_path)
                if artifact.get('feature_columns') != cols or artifact.get('target') != 'dropout_next_2_terms':
                    raise ValueError('Existing B5 artifact feature/target contract mismatch')
                recorded = artifact['pipeline'].predict_proba(val.features.loc[:, cols])[:, 1]
                max_b5_artifact_pred_diff = float(np.max(np.abs(scores - recorded)))
                if max_b5_artifact_pred_diff > 1e-7:
                    raise ValueError('B5 regenerated probabilities differ from existing provisional artifact')
        if not np.isfinite(scores).all() or np.min(scores) < 0 or np.max(scores) > 1:
            raise ValueError(f'Bad probability scores for {name}')
        main_metric = metric(yval, scores, threshold)
        expected = reference[name]
        deltas = {key: float(main_metric[key] - expected[key]) for key in ('pr_auc','roc_auc','brier')}
        match = all(abs(delta) < 0.0001 for delta in deltas.values())
        record = {
            'model': name, 'features': len(cols),
            'validation': main_metric,
            'validation_top10': top_k(yval, scores),
            'reference_validation': {k:expected[k] for k in ('pr_auc','roc_auc','brier','threshold')},
            'metric_deltas': deltas,
            'threshold_delta_vs_generator': float(threshold - expected['threshold']),
            'reference_match': match,
        }
        results[name] = record
        comparison.append({
            'model': name, 'features': len(cols), 'pr_auc': main_metric['pr_auc'],
            'roc_auc': main_metric['roc_auc'], 'brier': main_metric['brier'],
            'threshold_val_f1': threshold, 'precision': main_metric['precision'],
            'recall': main_metric['recall'], 'f1': main_metric['f1'],
            'top10_recall': record['validation_top10']['recall_at_k'],
            'reference_pr_auc': expected['pr_auc'], 'delta_pr_auc': deltas['pr_auc'],
            'reference_match': match,
        })
        for term, mask in masks.items():
            part = metric(yval[mask], scores[mask], threshold)
            per_term.append({
                'model':name, 'term': term, 'n': part['n'], 'positives': part['positive'],
                'prevalence': part['prevalence'], 'pr_auc': part['pr_auc'],
                'roc_auc': part['roc_auc'], 'brier': part['brier'],
                'precision': part['precision'], 'recall': part['recall'],
                'f1': part['f1'], 'top10_recall': top_k(yval[mask], scores[mask])['recall_at_k'],
            })
        print(f'{name}: PR-AUC={main_metric["pr_auc"]:.6f}; ROC-AUC={main_metric["roc_auc"]:.6f}; '
              f'Brier={main_metric["brier"]:.6f}; reference={"MATCH" if match else "MISMATCH"}', flush=True)
    for term in (1, 2, 3):
        if not masks[term].any():
            raise ValueError(f'No validation term {term} rows found')
    if any(not result['reference_match'] for result in results.values()):
        # Still save audit reports, but do NOT say PASS.
        status = 'INVESTIGATE'
    else:
        status = 'PASS'
    report = {
        'status': status, 'phase': '2', 'scope': 'E1-E5' + (' + optional E6' if args.include_v31_ablation else ''),
        'train_n':len(ytrain), 'train_positive':int(ytrain.sum()),
        'validation_n':len(yval), 'validation_positive':int(yval.sum()),
        'test_accessed': False, 'prediction_selection_data': 'Validation only',
        'seed':SEED, 'preprocessing_fit_partition': 'TRAIN_2022_23',
        'threshold_selection': 'Validation F1, first maximum on grid np.linspace(.05, .8, 76); B0=0.5',
        'reference_tolerance_abs_metrics': 0.0001,
        'b5_saved_artifact_max_probability_difference': max_b5_artifact_pred_diff,
        'versions': {name:version(name) for name in ('pandas','numpy','scikit-learn','joblib')},
        'reference': provenance,
        'models': results,
        'note': 'Research-only synthetic dataset; Validation is reused to select threshold so thresholded metrics are in-sample to selection.',
    }
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'baseline_reproduction.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',encoding='utf-8')
    write_csv(out / 'baseline_comparison.csv', comparison)
    write_csv(out / 'baseline_per_term_validation.csv', per_term)
    md = [
        '# PHASE 2 E: B0-B5 Validation baseline reproduction', '',
        f'**Status:** {status}. **Test accessed:** NO. **Seed:** {SEED}.',
        '', '| Model | Features | PR-AUC | ROC-AUC | Brier | Recall@Top10% | Match reference |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in comparison:
        md.append(f'| {row["model"]} | {row["features"]} | {row["pr_auc"]:.6f} | '
                  f'{row["roc_auc"]:.6f} | {row["brier"]:.6f} | {row["top10_recall"]:.4f} | '
                  f'{"PASS" if row["reference_match"] else "INVESTIGATE"} |')
    md.extend(['', 'All metrics are Validation only. Top10% is selected within each relevant Validation group.',
               'Threshold selected on Validation F1 is exploratory, not an unbiased threshold performance estimate.',
               'Generator source and report SHA-256 are recorded in baseline_reproduction.json.',
               'Model selection belongs to Phase F/G; no assumption that B5 is best.',
               'Synthetic only; cannot claim performance on real CMC students.', ''])
    (out / 'baseline_reproduction.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'PHASE_2_E_{status}; models={len(models)}; Test accessed=NO; report={out / "baseline_reproduction.json"}')
    if status != 'PASS':
        sys.exit(2)


if __name__ == '__main__':
    main()
