"""Phase 2-K contract gate; read-only except a new evidence JSON on --run.

No training, no Test split files, no labels, no Test predictions.
Only the synthetic 2022 cohort is scored for end-to-end equality.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.data_access import V32DataAccess
from src.inference_v32 import V32InferenceService, ordered_frame_from_snapshot
from src.model_bundle_v32 import (METADATA_NAME, MODEL_NAME, load_frozen_bundle,
                                 predict_features, sha256, validate_metadata)

REPORT = PROJECT / 'reports/ml_v3_2/inference_contract_K.json'


def require(condition: bool, msg: str):
    if not condition:
        raise RuntimeError(msg)


def preflight() -> dict:
    meta_path = PROJECT / 'models/v3_2' / METADATA_NAME
    model_path = PROJECT / 'models/v3_2' / MODEL_NAME
    j_path = PROJECT / 'reports/ml_v3_2/artifact_reproducibility_J.json'
    cat_path = PROJECT / 'data/processed/v3_2/dataset_catalog.json'
    for path in (meta_path, model_path, j_path, cat_path):
        require(path.is_file(), f'Required source absent: {path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    validate_metadata(meta)
    jreport = json.loads(j_path.read_text(encoding='utf-8-sig'))
    catalog = json.loads(cat_path.read_text(encoding='utf-8-sig'))
    require(jreport['status'] == 'PASS' and jreport['metadata_sha256'] == sha256(meta_path),
            'J metadata checksum mismatch')
    require(meta['model_sha256'] == sha256(model_path) == jreport['frozen_model_sha256'],
            'Frozen model hash mismatch')
    require(catalog['dataset_version'] == '3.2.0'
            and catalog['scenario_id'] == 'V3_2_MAIN'
            and catalog['synthetic'] is True
            and catalog['demo_as_of'] == meta['serving_max_asof'] == '2026-10-01',
            'Phase 1 catalog or demo ceiling mismatch')
    return {'model_sha256': meta['model_sha256'], 'metadata_sha256': sha256(meta_path),
            'model_version': meta['model_version'], 'asof': meta['serving_max_asof']}


def run():
    require(not REPORT.exists(), f'Refuse to overwrite K evidence: {REPORT}')
    before = preflight()
    access = V32DataAccess()
    bundle = load_frozen_bundle(PROJECT)
    service = V32InferenceService(project_root=PROJECT, access=access, bundle=bundle)
    # Only cohort 2022 snapshots; not opening/sampling full-simulation Test split.
    students = access._students_by_uid
    uids = students.loc[students['cohort_year'].eq(2022)].index[:30].tolist()
    checked = 0
    max_delta = 0.0
    for uid in uids:
        if checked >= 3:
            break
        row = access.get_snapshot(str(uid), 1, asof=before['asof'])
        if row is None:
            continue
        actual = service.predict(str(uid), 1, asof=before['asof'])
        require(actual['status'] == 'ok', 'Visible snapshot was not predicted')
        frame = ordered_frame_from_snapshot(row, str(uid), 1, before['asof'])
        direct = float(predict_features(frame, bundle)[0])
        delta = abs(actual['probability']-direct)
        require(math.isfinite(delta) and delta <= 1e-10, 'API/direct probability mismatch')
        max_delta = max(max_delta, delta)
        require(actual['policy']['frozen_probability_threshold'] == .27,
                'Frozen threshold mismatch')
        require(actual['policy']['topk_requires_batch'] is True,
                'TopK must not become per-person decision')
        require(all(k not in actual for k in ('dropout_next_2_terms','label_observed','student_id')),
                'Leaked outcome or student profile field')
        # If as-of is earlier than cutoff, data access must not serve the row.
        day_before = (date.fromisoformat(row['snapshot_cutoff'])-timedelta(days=1)).isoformat()
        require(service.predict(str(uid),1,asof=day_before)['status']=='not_available',
                'Past-asof incorrectly exposed snapshot')
        checked += 1
    require(checked >= 2, 'Too few visible Train-cohort snapshots for K verification')
    require(service.predict('NO_SUCH_RECORD_UID',1,asof=before['asof'])['status']=='not_available',
            'Unknown UID not rejected')
    future_rejected = False
    try:
        service.predict(str(uids[0]),1,'2026-10-02')
    except ValueError:
        future_rejected = True
    require(future_rejected, 'Future asof was not rejected')
    after = preflight()
    require(before == after, 'Frozen artifact or metadata changed during inference')
    result = {
        'status': 'PASS', 'phase':'2-K', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'model_version': before['model_version'], 'model_sha256': before['model_sha256'],
        'metadata_sha256': before['metadata_sha256'],
        'source':'V3_2_MAIN synthetic ASOF_2026_DEMO',
        'serving_max_asof': before['asof'], 'feature_count':58,
        'visible_train_cohort_snapshots_checked': checked,
        'api_direct_max_abs_probability_delta': max_delta, 'tolerance':1e-10,
        'unknown_uid_not_available':True, 'past_asof_not_available':True,
        'future_asof_rejected':True, 'no_fit': True, 'no_test_split_opened':True,
        'no_labels_accessed':True, 'no_test_predictions_accessed':True,
        'model_and_metadata_unchanged':True,
        'note':'No real-student validity claim. TopK requires batch ranking. Phase 1 E1-E5 must remain PASS.',
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'PHASE_2_K_PASS; checked={checked}; direct_delta={max_delta:.3e}; '
          'model=FROZEN; Test rescored=NO; no labels loaded')
    print(f'report={REPORT}')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--run', action='store_true')
    parser.add_argument('--confirm-api', action='store_true')
    args = parser.parse_args()
    if args.run and not args.confirm_api:
        parser.error('--run requires --confirm-api')
    if args.preflight:
        info=preflight()
        print('PHASE_2_K_PREFLIGHT_PASS; frozen hashes verified; model NOT loaded; Test NOT opened')
    else:
        run()


if __name__ == '__main__':
    main()
