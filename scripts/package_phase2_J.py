"""Phase 2-J: build evidence-only metadata for already-frozen RF_B V3.2.

Never fits, re-calibrates, selects a threshold, reads Test labels, or rewrites
pre-Test model artifacts. Outputs are create-once. --preflight is read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.features import B5_FEATURES
from src.ml_v32 import CATEGORICAL_B5_FEATURES, validate_feature_frame
from src.model_bundle_v32 import (
    METADATA_NAME, MODEL_NAME, MODEL_ID, MODEL_VERSION, load_frozen_bundle,
    predict_features, require, sha256, validate_metadata,
)
from src.model_policy_v32 import apply_platt

ROOT = PROJECT / 'data/external/v3_2_main'
MODELS = PROJECT / 'models/v3_2'
REPORTS = PROJECT / 'reports/ml_v3_2'
SOURCE_ARCHIVE_SHA = '5DF81F995347B167C4996F3A83C63C11637F78674834F76DD3D28A0548E6ECD6'
METADATA_PATH = MODELS / METADATA_NAME
REPORT_PATH = REPORTS / 'artifact_reproducibility_J.json'
REPORT_MD = REPORTS / 'artifact_reproducibility_J.md'


def read_json(path: Path):
    require(path.is_file(), f'Missing required Phase 2 evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def assert_context(project_root=PROJECT, data_root=ROOT, model_dir=MODELS, report_dir=REPORTS):
    """Only checks existing evidence/model hashes. No joblib load, no Test open."""
    g = read_json(report_dir/'model_selection_decision.json')
    h = read_json(report_dir/'frozen_test_metrics.json')
    i = read_json(report_dir/'subgroup_audit.json')
    hr = read_json(report_dir/'reproducibility.json')
    b5 = read_json(report_dir/'b5_training_provisional.json')
    f = read_json(report_dir/'model_comparison.json')
    manifest_path = data_root/'manifests/generation_manifest.json'
    manifest = read_json(manifest_path)

    require(g.get('status') == 'PASS' and g.get('phase') == '2-G' and g.get('mode') == 'full', 'G freeze missing')
    require(g.get('model_id') == MODEL_ID and g.get('model_version') == MODEL_VERSION, 'Wrong frozen model identity')
    require(g.get('target') == 'dropout_next_2_terms' and g.get('features_ordered') == B5_FEATURES, 'G feature/target drift')
    require(g.get('threshold',{}).get('threshold') == .27 and g.get('seed') == 221026, 'G threshold or seed drift')
    require(g.get('test_accessed') is False, 'G no longer pre-Test evidence')
    require(h.get('status') == 'PASS' and h.get('phase') == '2-H' and h.get('test_evaluation_once') is True, 'H one-shot evidence missing')
    require(h.get('model_version') == MODEL_VERSION and h.get('threshold') == .27, 'H used different model/threshold')
    require(h.get('pooled',{}).get('n') == 9284 and h.get('pooled',{}).get('positive') == 911, 'H Test accounting changed')
    require(i.get('status') == 'PASS' and i.get('phase') == '2-I', 'I audit evidence missing')
    require(i.get('model_version') == MODEL_VERSION and i.get('test_rescored') is False, 'I rescored Test or wrong model')
    require(hr.get('model_artifact_sha256') == g.get('model_sha256'), 'H reproducibility model hash drift')
    require(f.get('status') == 'PASS' and f.get('mode') == 'full', 'F full report missing')
    require(sha256(report_dir/'model_comparison.json') == g.get('source_f_report_sha256'), 'F report changed since G')
    frozen = model_dir/MODEL_NAME
    require(frozen.is_file() and sha256(frozen) == g.get('model_sha256'), 'Frozen model SHA mismatch, STOP before joblib.load')
    b5path = model_dir/'b5_reference_pipeline.joblib'
    require(b5.get('status') == 'PASS' and b5path.is_file() and
            sha256(b5path) == b5.get('model_sha256'), 'B5 reference model changed')
    require(manifest.get('generator_version') == '3.2.0' and
            manifest.get('scenario_id') == 'V3_2_MAIN' and
            manifest.get('is_synthetic') is True and
            manifest.get('demo_as_of') == '2026-10-01', 'V3.2 manifest mismatch')
    for name, old_hash in g.get('source_input_sha256', {}).items():
        require(name in {f'{xy}_{split}.csv.gz' for xy in ('X', 'y')
                         for split in ('TRAIN_2022_23', 'VALIDATION_2024')}, 'Unexpected split input')
        require(sha256(data_root/'splits/full_simulation'/name) == old_hash, f'Train/Val source hash mismatch: {name}')
    require(len(g.get('source_input_sha256', {})) == 4, 'Incomplete Train/Val input hashes')
    # The expected sklearn version must match frozen training runtime.
    require(version('scikit-learn') == g.get('environment',{}).get('scikit-learn'),
            'scikit-learn version differs from frozen model training runtime')
    # Read git ignore rule, do not stage files. A local git repo is required on user's machine.
    if (project_root/'.git').exists():
        result = subprocess.run(['git','check-ignore','-q','--no-index', 'models/v3_2/frozen_rf_b_pipeline.joblib'],
                                cwd=project_root, check=False)
        require(result.returncode == 0, 'Frozen model binary is NOT ignored by Git')
    return g, h, i, hr, b5, manifest, manifest_path


def max_train_snapshot_cutoff(data_root: Path) -> str:
    # Purely temporal metadata from observable snapshots, NOT Test labels.
    splits = data_root/'splits/full_simulation'
    train = pd.read_csv(splits/'X_TRAIN_2022_23.csv.gz', usecols=['record_uid', 'snapshot_term_index'],
                        dtype={'record_uid':'string'})
    snapshots = pd.read_csv(data_root/'features/prediction_snapshots.csv.gz',
                            usecols=['record_uid', 'snapshot_term_index', 'snapshot_cutoff'],
                            dtype={'record_uid':'string'})
    require(not snapshots.duplicated(['record_uid','snapshot_term_index']).any(), 'Duplicate source snapshot keys')
    joined = train.merge(snapshots, on=['record_uid','snapshot_term_index'],
                         how='left', validate='one_to_one')
    require(len(joined) == 34304 and not joined.snapshot_cutoff.isna().any(), 'Missing Train snapshot cutoff')
    return pd.to_datetime(joined.snapshot_cutoff, format='%Y-%m-%d').max().date().isoformat()


def build_metadata(g, h, i, hr, b5, manifest, manifest_path, train_dtype, train_cutoff) -> dict:
    cats = [x for x in B5_FEATURES if x in CATEGORICAL_B5_FEATURES]
    nums = [x for x in B5_FEATURES if x not in CATEGORICAL_B5_FEATURES]
    return {
        'schema_version': '2-J-1', 'status': 'FROZEN',
        'model_file': MODEL_NAME, 'model_sha256': g['model_sha256'],
        'model_id': MODEL_ID, 'model_version': MODEL_VERSION,
        'target': 'dropout_next_2_terms', 'horizon': 'next 2 academic terms',
        'features_ordered': list(B5_FEATURES), 'feature_count': 58,
        'feature_roles': {'numeric': nums, 'categorical': cats},
        'observed_train_input_dtypes': train_dtype,
        'dtype_compatibility': 'Numeric pandas numeric dtypes, categorical pandas object/string; no implicit coerce or fill at API boundary',
        'missing_policy': {'numeric':'median fit on TRAIN_2022_23 only',
                           'categorical':'most_frequent fit on TRAIN_2022_23 only',
                           'source_missing_allowed':'Existing NaN only; never insert missing columns'},
        'category_policy': 'OneHotEncoder(handle_unknown=ignore), fit on Train only',
        'preprocessing': 'ColumnTransformer, numeric imputer+scaler; categorical object-NaN normalization+imputer+OneHotEncoder',
        'estimator': 'RandomForestClassifier', 'hyperparameters': g['hyperparameters'],
        'calibration': {'method':g['calibration']['method'], 'fit_source':g['calibration']['fit_source'],
                        'uid_count':g['calibration']['uid_count'], 'rows':g['calibration']['rows']},
        'threshold': g['threshold']['threshold'], 'alert_policy':g['alert_policy'],
        'seed':g['seed'],
        'train_split':'TRAIN_2022_23', 'training_cohorts':[2022,2023],
        'training_max_snapshot_cutoff': train_cutoff,
        'validation_split':'VALIDATION_2024', 'test_split':'TEST_2025',
        'serving_max_asof': manifest['demo_as_of'],
        'source': {
            'scenario':'V3_2_MAIN/FULL_SIMULATION synthetic',
            'source_archive_sha256_phase1_provenance': SOURCE_ARCHIVE_SHA,
            'source_archive_sha256_reverified_in_J':False,
            'generation_manifest_sha256':sha256(manifest_path),
            'generation_config_sha256':manifest['config_sha256'],
            'train_validation_input_sha256':g['source_input_sha256'],
            'test_input_sha256_h_evidence':hr['source_test_sha256'],
            'labels_permitted_in_serving':False,
        },
        'training_versions':g['environment'],
        'pretest_selection_f_report_sha256':g['source_f_report_sha256'],
        'evaluation': {
            'train_rows':g['train_rows'], 'validation_rows':g['validation_rows'],
            'validation_policy_metrics':g['policy_subset']['calibrated_metrics'],
            'test_pooled_metrics':h['pooled'], 'test_by_term':h['by_term'],
            'test_frozen_before_eval':True,
            'test_evaluated_at_utc':h['evaluated_at_utc'],
            'subgroup_support_counts':i['support_status_counts'],
        },
        'artifact_frozen_at_utc':g['created_utc'],
        'disclaimer':'Synthetic simulation only; not validated for real CMC student decisions',
        'security':'Trusted local joblib only. SHA pinning detects accidental drift; not a defense against a malicious file+metadata replacement',
    }


def main():
    parser=argparse.ArgumentParser(description='Phase 2-J metadata and reproducibility (no training, no Test scoring)')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--build', action='store_true')
    parser.add_argument('--confirm-bundle', action='store_true')
    args=parser.parse_args()
    if args.build and not args.confirm_bundle:
        parser.error('--build requires --confirm-bundle')
    if args.build:
        preexisting=[str(p) for p in (METADATA_PATH, REPORT_PATH, REPORT_MD) if p.exists()]
        require(not preexisting, 'Refusing to overwrite existing Phase J outputs: '+', '.join(preexisting))
    g,h,i,hr,b5,manifest,manifest_path=assert_context()
    if args.preflight:
        print('PHASE_2_J_PREFLIGHT_PASS; G/H/I SHA verified; model NOT loaded; Test NOT opened')
        return
    # Trusted, pinned model only. No fit and no access to Test files.
    frozen=MODELS/MODEL_NAME
    raw1=joblib.load(frozen)
    raw2=joblib.load(frozen)
    require(isinstance(raw1,dict) and raw1.get('feature_columns')==B5_FEATURES and
            raw1.get('model_version')==MODEL_VERSION, 'Frozen joblib contract differs from G report')
    xp=ROOT/'splits/full_simulation/X_VALIDATION_2024.csv.gz'
    # Validation input only; no y/labels. Explicitly use 58 features.
    val=pd.read_csv(xp, dtype={'record_uid':'string'}, low_memory=False)
    sample=val.loc[:63, B5_FEATURES].copy()
    validate_feature_frame(sample)
    score1=apply_platt(raw1['calibrator'], raw1['pipeline'].predict_proba(sample)[:,1])
    score2=apply_platt(raw2['calibrator'], raw2['pipeline'].predict_proba(sample)[:,1])
    max_delta=float(np.max(np.abs(score1-score2)))
    require(math.isfinite(max_delta) and max_delta <= 1e-10, 'Reloaded bundle probabilities not stable')
    trainframe=pd.read_csv(ROOT/'splits/full_simulation/X_TRAIN_2022_23.csv.gz',
                           dtype={'record_uid':'string'}, low_memory=False)
    validate_feature_frame(trainframe.loc[:, B5_FEATURES])
    dtype={k:str(trainframe[k].dtype) for k in B5_FEATURES}
    max_cutoff=max_train_snapshot_cutoff(ROOT)
    metadata=build_metadata(g,h,i,hr,b5,manifest,manifest_path,dtype,max_cutoff)
    validate_metadata(metadata)
    report={
        'status':'PASS', 'phase':'2-J', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_id':MODEL_ID, 'model_version':MODEL_VERSION, 'target':metadata['target'],
        'frozen_model_sha256':g['model_sha256'],
        'model_selection_decision_sha256':sha256(REPORTS/'model_selection_decision.json'),
        'frozen_test_metrics_sha256':sha256(REPORTS/'frozen_test_metrics.json'),
        'subgroup_audit_sha256':sha256(REPORTS/'subgroup_audit.json'),
        'source_archive_sha256_phase1_provenance':SOURCE_ARCHIVE_SHA,
        'manifest_sha256':sha256(manifest_path),
        'train_cutoff':max_cutoff,'seed':g['seed'],
        'saved_and_reloaded_independently':2, 'probabilities_checked':len(sample),
        'max_abs_proba_delta':max_delta, 'tolerance':1e-10,
        'test_rescored':False, 'model_refitted':False, 'model_rewritten':False,
        'metadata_path':'models/v3_2/model_metadata.json',
        'disclaimer':'Synthetic only; no claim of real CMC student performance',
    }
    MD=("# Phase 2-J: Frozen artifact & reproducibility\n\n"
        "**Status:** PASS (only if the companion JSON and model SHA validate).\n\n"
        f"- Model: `{MODEL_VERSION}` / SHA256 `{g['model_sha256']}`\n"
        f"- Metadata: `models/v3_2/model_metadata.json`\n"
        f"- Frozen threshold: `{metadata['threshold']}`; budgets: `5%`, `10%`\n"
        f"- Train cutoff (source snapshots): `{max_cutoff}`; as-of demo ceiling: `{metadata['serving_max_asof']}`\n"
        f"- Save/load maximum abs delta (64 Validation feature rows): `{max_delta:.3e}` (limit `1e-10`)\n"
        "- Model, calibration, feature list and threshold were **not** modified. Test was **not** rescored.\n"
        "- Original V2.2 artifacts remain separate; no auto-train or silent fallback.\n"
        "- Dataset archive hash is recorded from Phase 1 provenance, not re-hashed in J.\n"
        "- WARNING: `joblib.load` is pickle-based. Never load untrusted model files.\n"
        "- Synthetic only; outputs cannot be used as real CMC student decisions.\n")
    MODELS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    try:
        METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        report['metadata_sha256']=sha256(METADATA_PATH)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        REPORT_MD.write_text(MD, encoding='utf-8')
        # Full trusted-loader round-trip, then compare scores with original run.
        recovered=load_frozen_bundle(PROJECT)
        scores3=predict_features(sample, recovered)
        require(float(np.max(np.abs(scores3-score1)))<=1e-10, 'Verified loader prediction mismatch')
    except Exception:
        for path in (METADATA_PATH, REPORT_PATH, REPORT_MD):
            if path.exists(): path.unlink()
        raise
    print(f'PHASE_2_J_PASS; model SHA verified; metadata SHA={report["metadata_sha256"]}; '
          f'reload_delta={max_delta:.3e}; Test rescored=NO; artifact unchanged=YES')


if __name__=='__main__':
    main()
