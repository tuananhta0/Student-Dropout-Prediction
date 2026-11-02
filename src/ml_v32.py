"""Phase 2 V3.2 train/validation-only loading, leakage guards and B5 pipeline.

This module deliberately has no TEST_2025 loader and no import from the legacy
V2.2 trainer. Only the explicitly approved 58 features enter the estimator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import B5_FEATURES

KEY = ["record_uid", "snapshot_term_index"]
TARGET = "dropout_next_2_terms"
META = ["record_uid", "cohort_year", "administrative_action_eligible_at_cutoff", "weeks_observed"]
ALLOWED_SPLITS = {
    "TRAIN_2022_23": (34304, 3136, {2022, 2023}),
    "VALIDATION_2024": (13981, 1240, {2024}),
}
SOURCE_TIMESTAMP_COLS = KEY + ["snapshot_cutoff", "feature_source_latest_at"]
# Pandas 3.x infers textual data as dtype="str" rather than object.
# Freeze semantic feature roles instead of inferring roles from Pandas dtypes.
CATEGORICAL_B5_FEATURES = (
    "major_code", "degree_type", "education_mode",
    "admission_method", "living_status_at_cutoff",
)


@dataclass
class Partition:
    name: str
    features: pd.DataFrame
    labels: pd.Series
    keys: pd.DataFrame
    metadata: pd.DataFrame


def _fail(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _check_keys(frame: pd.DataFrame, label: str) -> None:
    _fail(set(KEY).issubset(frame.columns), f"{label}: missing composite key")
    _fail(not frame[KEY].isna().any().any(), f"{label}: null composite key")
    _fail(not frame.duplicated(KEY).any(), f"{label}: duplicate UID/term composite key")
    _fail(frame["snapshot_term_index"].isin((1, 2, 3)).all(), f"{label}: invalid term")


def validate_raw_x(frame: pd.DataFrame) -> None:
    required = set(B5_FEATURES) | set(META)
    _fail(len(frame.columns) == 62 and set(frame.columns) == required,
          f"X schema mismatch: missing={sorted(required-set(frame))}, "
          f"unexpected/forbidden={sorted(set(frame)-required)}")
    _check_keys(frame, "X")


def validate_raw_y(frame: pd.DataFrame) -> None:
    _fail(list(frame.columns) == KEY + [TARGET], "y schema mismatch: expected key + label")
    _check_keys(frame, "y")
    _fail(not frame[TARGET].isna().any(), "Missing outcome label")
    _fail(frame[TARGET].isin((0, 1)).all(), "Non-binary outcome label")


def validate_feature_frame(frame: pd.DataFrame, *, reference: pd.DataFrame | None = None) -> None:
    """Strict contract for already-ordered model input, especially serving."""
    _fail(list(frame.columns) == B5_FEATURES, "Ordered 58 B5 feature schema mismatch")
    cats = CATEGORICAL_B5_FEATURES
    for name in B5_FEATURES:
        if name in cats:
            _fail(pd.api.types.is_object_dtype(frame[name]) or pd.api.types.is_string_dtype(frame[name]),
                  f"Categorical dtype mismatch: {name}")
        else:
            _fail(pd.api.types.is_numeric_dtype(frame[name]), f"Numeric dtype mismatch: {name}")
    if reference is not None:
        _fail(list(reference.columns) == B5_FEATURES, "Training feature contract mismatch")
        for name in B5_FEATURES:
            _fail(str(frame[name].dtype) == str(reference[name].dtype),
                  f"Dtype drift in {name}")


def join_x_y(raw_x: pd.DataFrame, raw_y: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Keyed 1:1 join: shuffling y must not change x->label mapping."""
    validate_raw_x(raw_x)
    validate_raw_y(raw_y)
    joined = raw_x.merge(raw_y, on=KEY, how="left", validate="one_to_one", indicator=True, sort=False)
    _fail(len(joined) == len(raw_x) == len(raw_y), "X/y row count mismatch")
    _fail(joined["_merge"].eq("both").all(), "Unmatched X/y composite keys")
    features = joined.loc[:, B5_FEATURES].copy()
    validate_feature_frame(features)
    return features, joined[TARGET].astype("int64"), joined[KEY].copy()


def validate_timestamps(source: pd.DataFrame, keys: pd.DataFrame) -> None:
    _fail(set(SOURCE_TIMESTAMP_COLS).issubset(source.columns),
          "Source timestamps missing required column")
    _check_keys(source, "source timestamps")
    _check_keys(keys, "training keys")
    matched = keys.merge(source.loc[:, SOURCE_TIMESTAMP_COLS], on=KEY,
                         how="left", validate="one_to_one", indicator=True)
    _fail(matched["_merge"].eq("both").all(), "Training snapshot absent from source")
    _fail(not matched[["snapshot_cutoff", "feature_source_latest_at"]].isna().any().any(),
          "Missing temporal source data")
    cutoff = pd.to_datetime(matched["snapshot_cutoff"], errors="raise", format="%Y-%m-%d")
    latest = pd.to_datetime(matched["feature_source_latest_at"], errors="raise", format="%Y-%m-%d")
    _fail((latest <= cutoff).all(), "Future feature_source_latest_at beyond snapshot_cutoff")


class V32TrainingLoader:
    """Training-only loader. No TEST partition permitted, by design."""
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        _fail(self.root.is_dir(), f"V3.2 source root missing: {self.root}")
        manifest_path = self.root / "manifests/generation_manifest.json"
        _fail(manifest_path.is_file(), f"Missing V3.2 manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        _fail(manifest.get("generator_version") == "3.2.0" and
              manifest.get("scenario_id") == "V3_2_MAIN" and
              manifest.get("is_synthetic") is True,
              "Manifest mismatch: require synthetic V3_2_MAIN v3.2.0")

    @property
    def _source_timestamps(self) -> pd.DataFrame:
        path = self.root / "features" / "prediction_snapshots.csv.gz"
        _fail(path.is_file(), f"Snapshot timestamps unavailable: {path}")
        return pd.read_csv(path, usecols=SOURCE_TIMESTAMP_COLS,
                           dtype={"record_uid": "string"}, low_memory=False)

    def load(self, split: str) -> Partition:
        _fail(split in ALLOWED_SPLITS, f"Forbidden split: {split}; only Train and Validation")
        root = self.root / "splits" / "full_simulation"
        xp, yp = (root / f"X_{split}.csv.gz", root / f"y_{split}.csv.gz")
        _fail(xp.is_file() and yp.is_file(), f"Missing full-simulation split: {split}")
        raw_x = pd.read_csv(xp, dtype={"record_uid": "string"}, low_memory=False)
        raw_y = pd.read_csv(yp, dtype={"record_uid": "string"}, low_memory=False)
        features, labels, keys = join_x_y(raw_x, raw_y)
        expected_rows, expected_positives, expected_years = ALLOWED_SPLITS[split]
        _fail(len(labels) == expected_rows and int(labels.sum()) == expected_positives,
              f"Split row/label total mismatch: {split}")
        _fail(set(raw_x["cohort_year"].unique()) == expected_years,
              f"Cohort partition mismatch: {split}")
        validate_timestamps(self._source_timestamps, keys)
        metadata = raw_x.loc[:, META].copy()
        return Partition(split, features, labels, keys, metadata)


def _categorical_to_object_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Pandas 2/3 textual dtypes and missing tokens for sklearn.

    Pandas `str` uses np.nan; nullable `string` uses pd.NA. sklearn's
    SimpleImputer reliably handles NumPy object columns with np.nan.
    """
    result = frame.astype("object")
    return result.where(pd.notna(result), np.nan)


def make_b5_pipeline(train_features: pd.DataFrame, seed: int = 221026) -> Pipeline:
    """Reference B5 preprocessing mirrors the generator Train-only pipeline."""
    validate_feature_frame(train_features)
    categorical = [c for c in B5_FEATURES if c in CATEGORICAL_B5_FEATURES]
    numeric = [c for c in B5_FEATURES if c not in CATEGORICAL_B5_FEATURES]
    _fail(len(categorical) == 5 and len(numeric) == 53,
          "B5 numeric/categorical signature changed")
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("to_object", FunctionTransformer(_categorical_to_object_nan, validate=False)),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=350, solver="liblinear", random_state=seed)),
    ])
