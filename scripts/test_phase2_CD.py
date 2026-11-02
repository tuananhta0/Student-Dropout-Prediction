"""Phase 2 C4/C5/D1-D5 guard tests. Run via: python -B -m unittest scripts.test_phase2_CD -v.

Uses a tiny subset of actual Train data, never opens TEST and never writes models.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import B5_FEATURES
from src.ml_v32 import (
    V32TrainingLoader, join_x_y, make_b5_pipeline, validate_feature_frame,
    validate_raw_x, validate_timestamps,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/v3_2_main"


class TestPhase2CD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = V32TrainingLoader(DATA)
        cls.train = cls.loader.load("TRAIN_2022_23")
        cls.val = cls.loader.load("VALIDATION_2024")
        cls.split_path = DATA / "splits/full_simulation"
        cls.raw_x = pd.read_csv(cls.split_path / "X_TRAIN_2022_23.csv.gz",
                                dtype={"record_uid": "string"}, low_memory=False)
        cls.raw_y = pd.read_csv(cls.split_path / "y_TRAIN_2022_23.csv.gz",
                                dtype={"record_uid": "string"}, low_memory=False)

    def test_01_loader_rejects_test_access(self):
        with self.assertRaisesRegex(ValueError, "Forbidden split"):
            self.loader.load("TEST_2025")

    def test_02_train_val_disjoint(self):
        self.assertFalse(set(self.train.keys.record_uid) & set(self.val.keys.record_uid))

    def test_03_exact_feature_contract(self):
        self.assertEqual(list(self.train.features), B5_FEATURES)
        self.assertEqual(list(self.val.features), B5_FEATURES)
        validate_feature_frame(self.val.features, reference=self.train.features)

    def test_04_injected_target_rejected(self):
        bad = self.raw_x.head(3).copy()
        bad["dropout_next_2_terms"] = 0
        with self.assertRaisesRegex(ValueError, "unexpected/forbidden"):
            validate_raw_x(bad)

    def test_05_missing_feature_rejected(self):
        bad = self.train.features.head(5).drop(columns=["gpa4_term"])
        with self.assertRaisesRegex(ValueError, "Ordered 58"):
            validate_feature_frame(bad)

    def test_06_reordered_feature_rejected(self):
        bad = self.train.features.head(5).loc[:, list(reversed(B5_FEATURES))]
        with self.assertRaisesRegex(ValueError, "Ordered 58"):
            validate_feature_frame(bad)

    def test_07_duplicate_keys_rejected(self):
        duplicated = pd.concat([self.raw_x.head(1), self.raw_x.head(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate UID/term"):
            validate_raw_x(duplicated)

    def test_08_keyed_join_stable_if_y_shuffled(self):
        x, y = self.raw_x.head(200).copy(), self.raw_y.head(200).copy()
        fx, ly, k = join_x_y(x, y)
        fx2, ly2, k2 = join_x_y(x, y.sample(frac=1, random_state=17).reset_index(drop=True))
        pd.testing.assert_frame_equal(fx, fx2)
        pd.testing.assert_series_equal(ly, ly2)
        pd.testing.assert_frame_equal(k, k2)

    def test_09_missing_label_rejected(self):
        with self.assertRaisesRegex(ValueError, "row count mismatch|Unmatched"):
            join_x_y(self.raw_x.head(20), self.raw_y.head(19))

    def test_10_future_timestamp_rejected(self):
        row = self.loader._source_timestamps.head(1).copy()
        cutoff = pd.to_datetime(row["snapshot_cutoff"].iloc[0])
        row.loc[:, "feature_source_latest_at"] = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with self.assertRaisesRegex(ValueError, "Future feature_source"):
            validate_timestamps(row, row[["record_uid", "snapshot_term_index"]].copy())

    def test_11_numeric_dtype_drift_rejected(self):
        bad = self.train.features.head(5).copy()
        bad["gpa4_term"] = bad["gpa4_term"].astype("str")
        with self.assertRaisesRegex(ValueError, "Numeric dtype mismatch"):
            validate_feature_frame(bad)

    def test_12_imputer_uses_train_only_and_unseen_cat_safe(self):
        # Tiny real Train subset: verify fit statistics are train-only.
        raw = self.train.features
        y = self.train.labels
        rng = np.random.default_rng(221026)
        subset = np.sort(rng.choice(len(y), 700, replace=False))
        X_train, y_train = raw.iloc[subset], y.iloc[subset]
        self.assertEqual(y_train.nunique(), 2)
        pipeline = make_b5_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        num_cols = pipeline.named_steps["prep"].transformers_[0][2]
        numeric_imputer = pipeline.named_steps["prep"].named_transformers_["numeric"].named_steps["impute"]
        gpa_idx = num_cols.index("gpa4_term")
        self.assertAlmostEqual(numeric_imputer.statistics_[gpa_idx], float(X_train["gpa4_term"].median()))
        X_val = self.val.features.head(8).copy()
        X_val.loc[:, "gpa4_term"] = 9999.0
        X_val.loc[:, "major_code"] = "NEW_UNSEEN_MAJOR"
        scores = pipeline.predict_proba(X_val)[:, 1]  # only transform; NO fit on Val
        self.assertEqual(len(scores), len(X_val))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(numeric_imputer.statistics_[gpa_idx], float(X_train["gpa4_term"].median()))


    def test_13_pandas_string_dtype_categorical_stays_categorical(self):
        # Pandas 3 read_csv infers dtype="str"; nullable dtype="string" uses pd.NA.
        # Both must be treated as categories, not sent to numeric median imputer.
        train = self.train.features.head(800).copy()
        labels = self.train.labels.head(800).copy()
        train.loc[:, "major_code"] = train["major_code"].astype("string")
        train.loc[:, "living_status_at_cutoff"] = train["living_status_at_cutoff"].astype("string")
        train.loc[train.index[0], "major_code"] = pd.NA
        pipe = make_b5_pipeline(train)
        pipe.fit(train, labels)
        numeric = pipe.named_steps["prep"].transformers_[0][2]
        categorical = pipe.named_steps["prep"].transformers_[1][2]
        self.assertEqual((len(numeric), len(categorical)), (53, 5))
        self.assertIn("major_code", categorical)
        self.assertNotIn("major_code", numeric)
        p = pipe.predict_proba(self.val.features.head(8).copy())[:, 1]
        self.assertTrue(np.isfinite(p).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
