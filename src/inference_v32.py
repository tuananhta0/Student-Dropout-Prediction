"""Phase 2-K: read-only, as-of-safe single-snapshot V3.2 inference.

No training, labels, evaluation artifacts, future access, UI or V2.2 fallback.
The verified model and demo data access are existing Phase 1/J contracts.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_access import V32DataAccess
from src.features import B5_FEATURES
from src.ml_v32 import CATEGORICAL_B5_FEATURES, validate_feature_frame
from src.model_bundle_v32 import load_frozen_bundle, predict_features

BOOLEAN_B5_FEATURES = frozenset({"warning_current", "academic_recovery_flag"})

FORBIDDEN_SERVING_FIELDS = frozenset({
    'dropout_next_2_terms', 'label_observed', 'censor_reason',
    'first_dropout_event_id', 'exit_reason_for_evaluation', 'split',
    'hidden_latent', 'dropout_probability_true', 'label', 'y_true',
})
DISCLAIMER = ('Synthetic V3.2 research simulation only. Risk scores support '
              'prioritizing academic outreach; not a CMC dismissal decision '
              'or validated prediction for real students.')


class InferenceContractError(ValueError):
    """Refuse malformed, ambiguous, unsafe or temporally invalid requests."""


def _iso(value: object, field: str) -> date:
    if not isinstance(value, str) or len(value) != 10:
        raise InferenceContractError(f'{field} must be YYYY-MM-DD')
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise InferenceContractError(f'Invalid {field} date') from exc
    if result.isoformat() != value:
        raise InferenceContractError(f'{field} must be YYYY-MM-DD')
    return result


def ordered_frame_from_snapshot(snapshot: dict, uid: str, term: int, asof: str) -> pd.DataFrame:
    """Validate source identity/time and convert ONLY frozen features to one row.

    Permitted boundary conversion: Python int/float -> float64, two documented
    boolean flags -> 0.0/1.0; numeric None ->
    NaN and categorical None -> NaN. String-to-number or new missing *columns*
    are prohibited. Fit-time imputation belongs inside the frozen pipeline.
    """
    if not isinstance(snapshot, dict):
        raise InferenceContractError('Serving snapshot must be a dict')
    forbidden = set(snapshot) & FORBIDDEN_SERVING_FIELDS
    if forbidden:
        raise InferenceContractError(f'Outcome/latent fields in serving snapshot: {sorted(forbidden)}')
    if snapshot.get('record_uid') != uid or type(snapshot.get('snapshot_term_index')) is not int or snapshot['snapshot_term_index'] != term:
        raise InferenceContractError('Serving snapshot composite key mismatch')
    cutoff = _iso(snapshot.get('snapshot_cutoff'), 'snapshot_cutoff')
    latest = _iso(snapshot.get('feature_source_latest_at'), 'feature_source_latest_at')
    effective = _iso(asof, 'asof')
    if latest > cutoff or cutoff > effective:
        raise InferenceContractError('Snapshot violates as-of temporal boundary')
    missing = sorted(set(B5_FEATURES) - snapshot.keys())
    if missing:
        raise InferenceContractError(f'Missing required B5 features: {missing}')
    cols = {}
    for name in B5_FEATURES:
        value = snapshot[name]
        if name in CATEGORICAL_B5_FEATURES:
            if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                normalized = np.nan
            elif isinstance(value, str) and value.strip():
                normalized = value
            else:
                raise InferenceContractError(f'Invalid categorical feature: {name}')
            cols[name] = pd.Series([normalized], dtype='object')
        else:
            if value is None or value is pd.NA:
                normalized = np.nan
            elif name in BOOLEAN_B5_FEATURES and isinstance(value, (bool, np.bool_)):
                normalized = float(value)
            elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value):
                normalized = float(value)
            else:
                raise InferenceContractError(f'Invalid numeric feature (no coercion): {name}')
            cols[name] = pd.Series([normalized], dtype='float64')
    frame = pd.DataFrame(cols).loc[:, B5_FEATURES]
    validate_feature_frame(frame)
    return frame


class V32InferenceService:
    """Explicit service dependency boundary, useful for unit testing and UI."""
    def __init__(self, *, project_root: str | Path | None = None,
                 access: V32DataAccess | None = None, bundle=None):
        root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
        # Production path ALWAYS verifies model/metadata hashes and trusted sklearn runtime.
        self.bundle = bundle if bundle is not None else load_frozen_bundle(root)
        self.access = access if access is not None else V32DataAccess()
        meta = self.bundle.metadata
        if (meta.get('model_version') != 'v3.2-rf-b-platt-g1'
                or meta.get('serving_max_asof') != '2026-10-01'
                or meta.get('threshold') != .27
                or meta.get('features_ordered') != B5_FEATURES):
            raise InferenceContractError('Unsupported frozen V3.2 inference metadata')
        self.max_asof = meta['serving_max_asof']

    def predict(self, uid: str, term: int, asof: str | None = None) -> dict:
        if not isinstance(uid, str) or not uid.strip():
            raise InferenceContractError('uid must be a non-empty record_uid')
        uid = uid.strip()
        if type(term) is not int or term not in (1, 2, 3):
            raise InferenceContractError('term must be integer 1, 2 or 3')
        effective = asof if asof is not None else self.max_asof
        if _iso(effective, 'asof') > _iso(self.max_asof, 'serving_max_asof'):
            raise InferenceContractError('asof beyond ASOF_2026_DEMO cutoff')
        # Single approved data-access API call: composite key + as-of.
        snapshot = self.access.get_snapshot(uid, term, asof=effective)
        base = {'model_version': self.bundle.metadata['model_version'],
                'snapshot_term_index': term, 'asof': effective,
                'disclaimer': DISCLAIMER}
        if snapshot is None:
            return {'status': 'not_available', 'reason': 'snapshot_not_visible_or_unknown', **base}
        frame = ordered_frame_from_snapshot(snapshot, uid, term, effective)
        proba = predict_features(frame, self.bundle)
        p = float(proba[0])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise InferenceContractError('Non-finite calibrated probability')
        threshold = float(self.bundle.metadata['threshold'])
        budgets = self.bundle.metadata['alert_policy']['top_budgets']
        return {
            'status': 'ok', 'probability': p, **base,
            'snapshot_cutoff': snapshot['snapshot_cutoff'],
            'policy': {
                'frozen_probability_threshold': threshold,
                'alert_at_threshold': bool(p >= threshold),
                'top_budgets': list(budgets),
                'topk_requires_batch': True,
                'topk_note': 'Top5%/Top10% require ranking all eligible snapshots in a term; not a single-UID decision',
            },
        }


@lru_cache(maxsize=1)
def _default_service() -> V32InferenceService:
    return V32InferenceService()


def predict(uid: str, term: int, asof: str | None = None) -> dict:
    """Public Phase 3 integration contract: predict(record_uid, term, asof)."""
    return _default_service().predict(uid, term, asof)
