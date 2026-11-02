"""Phase 2-J: trusted, hash-checked immutable V3.2 model bundle loader.

Never trains, never falls back to legacy V2.2, never accesses labels or Test.
Only load joblib artifacts created and trusted by this project. Do not load
joblib files from unknown or untrusted sources; pickle may execute code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import B5_FEATURES
from src.ml_v32 import CATEGORICAL_B5_FEATURES, validate_feature_frame
from src.model_policy_v32 import apply_platt, check_probs

MODEL_NAME = "frozen_rf_b_pipeline.joblib"
METADATA_NAME = "model_metadata.json"
MODEL_ID = "RF_B"
MODEL_VERSION = "v3.2-rf-b-platt-g1"


class ModelBundleError(RuntimeError):
    """Explicit refusal to serve when a verified frozen V3.2 bundle is absent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: Any, why: str) -> None:
    if not bool(ok):
        raise ModelBundleError(why)


def _json(path: Path) -> dict:
    require(path.is_file(), f"Missing V3.2 bundle metadata: {path}. Run Phase 2-J packaging; never auto-train.")
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ModelBundleError(f"Invalid metadata JSON: {path}") from exc
    require(isinstance(doc, dict), "Invalid metadata object")
    return doc


def validate_metadata(meta: dict) -> None:
    """Fail-fast static validation before opening any pickle/joblib payload."""
    require(meta.get("status") == "FROZEN" and meta.get("schema_version") == "2-J-1",
            "Bundle metadata is not a supported frozen contract")
    require(meta.get("model_id") == MODEL_ID and meta.get("model_version") == MODEL_VERSION,
            "Wrong model ID/version; V2.2 fallback is forbidden")
    require(meta.get("target") == "dropout_next_2_terms", "Wrong target")
    require(meta.get("features_ordered") == B5_FEATURES and meta.get("feature_count") == 58,
            "Mismatch in frozen ordered 58 features")
    roles = meta.get("feature_roles", {})
    expected_cats = [f for f in B5_FEATURES if f in CATEGORICAL_B5_FEATURES]
    require(roles.get("categorical") == expected_cats,
            "Categorical feature policy has changed")
    require(roles.get("numeric") == [f for f in B5_FEATURES if f not in CATEGORICAL_B5_FEATURES],
            "Numeric feature policy has changed")
    require(meta.get("missing_policy", {}).get("numeric") == "median fit on TRAIN_2022_23 only" and
            meta.get("missing_policy", {}).get("categorical") == "most_frequent fit on TRAIN_2022_23 only",
            "Unapproved missingness policy")
    require(meta.get("category_policy") == "OneHotEncoder(handle_unknown=ignore), fit on Train only",
            "Unapproved category policy")
    require(meta.get("model_file") == MODEL_NAME, "Unexpected model filename")
    h = meta.get("model_sha256", "")
    require(isinstance(h, str) and len(h) == 64 and all(ch in "0123456789abcdef" for ch in h),
            "Invalid frozen model hash")
    require(meta.get("threshold") == 0.27, "Frozen policy threshold changed")
    policy = meta.get("alert_policy", {})
    require(policy.get("top_budgets") == [.05, .10] and
            policy.get("topk_independent_of_threshold") is True,
            "Frozen TopK policy changed")
    require(meta.get("source", {}).get("scenario") == "V3_2_MAIN/FULL_SIMULATION synthetic",
            "Incorrect synthetic source/scenario")
    require(meta.get("seed") == 221026, "Seed changed")
    require(meta.get("serving_max_asof") == "2026-10-01", "Unapproved serving cutoff")
    require(meta.get("training_versions", {}).get("scikit-learn"), "Missing sklearn training version")


@dataclass(frozen=True)
class VerifiedBundle:
    pipeline: Any
    calibrator: Any
    metadata: dict


def load_frozen_bundle(project_root: str | Path) -> VerifiedBundle:
    """Load only verified trusted artifact from fixed project-local path.

    Cryptographic hash pinning detects accidental changes, not malicious edits
    by someone who can replace BOTH the model and trusted metadata/report.
    """
    root = Path(project_root).resolve()
    model_dir = root / "models" / "v3_2"
    meta_path = model_dir / METADATA_NAME
    meta = _json(meta_path)
    validate_metadata(meta)
    # The Phase J report is a separate integrity anchor for the metadata file.
    jreport = _json(root / "reports" / "ml_v3_2" / "artifact_reproducibility_J.json")
    require(jreport.get("status") == "PASS" and
            jreport.get("metadata_sha256") == sha256(meta_path),
            "Bundle metadata changed since Phase J packaging")
    frozen = model_dir / MODEL_NAME
    require(frozen.is_file(), f"Frozen V3.2 model missing: {frozen}. Restore trusted model by Phase 2-J runbook; no auto-train or V2.2 fallback.")
    require(sha256(frozen) == meta["model_sha256"],
            "Frozen V3.2 model SHA256 mismatch; refuse to unpickle")
    # Exact sklearn version check avoids unsafe binary cross-version reuse.
    require(version("scikit-learn") == meta["training_versions"]["scikit-learn"],
            "scikit-learn version mismatch; restore documented training environment")
    # SECURITY: joblib.load is pickle based. This is an explicitly trusted, hash-verified local file.
    try:
        raw = joblib.load(frozen)
    except Exception as exc:
        raise ModelBundleError("Could not deserialize trusted frozen V3.2 model") from exc
    require(isinstance(raw, dict), "Invalid joblib bundle object")
    for key in ("pipeline", "calibrator", "feature_columns", "model_id", "model_version", "threshold", "target", "alert_policy"):
        require(key in raw, f"Missing joblib field: {key}")
    require(raw["feature_columns"] == B5_FEATURES, "Joblib feature schema changed")
    require(raw["model_id"] == MODEL_ID and raw["model_version"] == MODEL_VERSION,
            "Joblib model version changed")
    require(raw["target"] == meta["target"] and raw["threshold"] == meta["threshold"],
            "Joblib target/threshold changed")
    require(raw["alert_policy"] == meta["alert_policy"], "Joblib alert policy changed")
    require(hasattr(raw["pipeline"], "predict_proba") and
            hasattr(raw["calibrator"], "predict_proba"), "Frozen model is not fitted")
    return VerifiedBundle(raw["pipeline"], raw["calibrator"], meta)


def predict_features(frame: pd.DataFrame, bundle: VerifiedBundle) -> np.ndarray:
    """Predict calibrated probabilities from *exactly* 58 ordered features.

    Caller must obtain as-of snapshot through Phase 1 serving data access.
    This function never opens datasets, labels, reports or legacy models.
    """
    require(isinstance(frame, pd.DataFrame) and len(frame) > 0,
            "Input must be a nonempty Pandas DataFrame")
    try:
        validate_feature_frame(frame)
    except ValueError as exc:
        raise ModelBundleError(f"Invalid inference feature frame: {exc}") from exc
    raw = check_probs(bundle.pipeline.predict_proba(frame)[:, 1])
    calibrated = apply_platt(bundle.calibrator, raw)
    require(len(calibrated) == len(frame), "Model returned wrong probability count")
    return calibrated
