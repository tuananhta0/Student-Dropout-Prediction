"""Phase 2-G immutable probability calibration and alert-policy primitives.

Calibrator is fitted on disjoint Validation UID groups, never on the threshold
policy subset. These functions do not access data files or labels at inference.
"""
from __future__ import annotations

import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

SEED = 221026
EPS = 1e-12


def check_probs(probs):
    a = np.asarray(probs, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Probabilities must be a nonempty finite 1-D [0,1] array')
    return a


def logits(probs):
    p = np.clip(check_probs(probs), EPS, 1.0-EPS)
    return np.log(p/(1-p)).reshape(-1, 1)


def split_uids(uids, fraction=.30, seed=SEED):
    """Group-preserving, deterministic calibration/policy split by record_uid."""
    a = np.asarray(uids, dtype=str)
    if a.ndim != 1 or not len(a) or not all(a) or not 0 < fraction < 1:
        raise ValueError('Invalid UID split arguments')
    unique = np.unique(a)
    if len(unique) < 4:
        raise ValueError('Not enough distinct UIDs')
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_cal = max(1, min(len(unique)-1, round(fraction*len(unique))))
    cal_mask = np.isin(a, shuffled[:n_cal])
    policy_mask = ~cal_mask
    if not cal_mask.any() or not policy_mask.any():
        raise ValueError('Empty calibration/policy partition')
    if set(a[cal_mask]) & set(a[policy_mask]):
        raise ValueError('Calibration UID leakage')
    return cal_mask, policy_mask


def fit_platt(cal_probs, cal_y, seed=SEED):
    """Platt sigmoid on logit(base_probability) from Calibration UID subset ONLY."""
    p = check_probs(cal_probs)
    y = np.asarray(cal_y, dtype=int)
    if len(p) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Calibration labels must contain both classes')
    obj = LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)
    obj.fit(logits(p), y)
    if not np.isfinite(obj.coef_).all() or float(obj.coef_[0, 0]) <= 0:
        raise ValueError('Calibration slope must be positive; ranking cannot invert')
    return obj


def apply_platt(calibrator, raw_probs):
    p = check_probs(raw_probs)
    out = calibrator.predict_proba(logits(p))[:, 1].astype(float)
    if len(out) != len(p):
        raise ValueError('Invalid calibrated output length')
    return check_probs(out)


def expected_calibration_error(y, probs, bins=10):
    p = check_probs(probs)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('Bad labels for ECE')
    if not isinstance(bins, int) or bins < 2:
        raise ValueError('Invalid bins')
    edges = np.linspace(0, 1, bins+1)
    ix = np.minimum(np.searchsorted(edges, p, side='right')-1, bins-1)
    return float(sum(np.count_nonzero(ix == b)/len(y) * abs(float(y[ix == b].mean())-float(p[ix == b].mean()))
                     for b in range(bins) if (ix == b).any()))


def best_f1_threshold(y, probs):
    """Predefined grid, F1 objective; tie-breaker chooses higher threshold."""
    p = check_probs(probs)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Policy labels must contain both classes')
    candidates = np.round(np.arange(.05, .801, .01), 2)
    rows = []
    for t in candidates:
        preds = (p >= t).astype(int)
        rows.append({'threshold': float(t), 'f1': float(f1_score(y, preds, zero_division=0)),
                     'precision': float(precision_score(y, preds, zero_division=0)),
                     'recall': float(recall_score(y, preds, zero_division=0)),
                     'alerts': int(preds.sum())})
    return max(rows, key=lambda r: (r['f1'], r['threshold']))


def ranked_budget(y, probs, uids, terms, fraction):
    """Deterministic top-k, ceil(n*budget), score desc then UID and term asc."""
    p = check_probs(probs)
    u = np.asarray(uids, dtype=str)
    t = np.asarray(terms, dtype=int)
    labels = np.asarray(y, dtype=int)
    if not (len(p) == len(u) == len(t) == len(labels)) or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('Bad ranking inputs')
    if len(set(zip(u.tolist(), t.tolist()))) != len(p):
        raise ValueError('Duplicate UID/term keys')
    if not 0 < fraction < 1:
        raise ValueError('Budget must be in (0,1)')
    order = np.lexsort((t, u, -p))
    k = math.ceil(len(p)*fraction)
    found = int(labels[order[:k]].sum())
    positives = int(labels.sum())
    return {'budget': fraction, 'k': k, 'positive_found': found,
            'precision_at_k': float(found/k),
            'recall_at_k': float(found/positives) if positives else None}
