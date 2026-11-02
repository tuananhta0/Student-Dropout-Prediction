"""Phase 2-J: synthetic fixture tests; do not re-score Test or retrain frozen model."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import B5_FEATURES
from src.ml_v32 import CATEGORICAL_B5_FEATURES
from src.model_bundle_v32 import (
    MODEL_NAME, MODEL_VERSION, ModelBundleError, VerifiedBundle,
    load_frozen_bundle, predict_features, sha256, validate_metadata,
)


def fixture_meta():
    cats = [x for x in B5_FEATURES if x in CATEGORICAL_B5_FEATURES]
    return {'status':'FROZEN', 'schema_version':'2-J-1', 'model_id':'RF_B',
            'model_version':MODEL_VERSION, 'target':'dropout_next_2_terms',
            'feature_count':58, 'features_ordered':list(B5_FEATURES),
            'feature_roles':{'categorical':cats,'numeric':[x for x in B5_FEATURES if x not in cats]},
            'missing_policy':{'numeric':'median fit on TRAIN_2022_23 only',
                              'categorical':'most_frequent fit on TRAIN_2022_23 only'},
            'category_policy':'OneHotEncoder(handle_unknown=ignore), fit on Train only',
            'model_file':MODEL_NAME, 'model_sha256':'a'*64,
            'threshold':.27, 'seed':221026, 'serving_max_asof':'2026-10-01',
            'alert_policy':{'top_budgets':[.05,.10],'topk_independent_of_threshold':True},
            'source':{'scenario':'V3_2_MAIN/FULL_SIMULATION synthetic'},
            'training_versions':{'scikit-learn':'1.9.1'}}


class FakePipeline:
    def predict_proba(self, X):
        p=np.repeat(.4,len(X))
        return np.column_stack([1-p,p])


class FakeCalibrator:
    def predict_proba(self, X):
        p=np.repeat(.37,len(X))
        return np.column_stack([1-p,p])


class TestPhase2J(unittest.TestCase):
    def test_01_valid_metadata(self):
        validate_metadata(fixture_meta())

    def test_02_wrong_target_fails(self):
        m=fixture_meta();m['target']='academic_status'
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_03_reordered_features_fail(self):
        m=fixture_meta();m['features_ordered']=list(reversed(m['features_ordered']))
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_04_reclassified_categorical_fails(self):
        m=fixture_meta();m['feature_roles']['categorical']=[]
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_05_threshold_mutation_fails(self):
        m=fixture_meta();m['threshold']=.21
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_06_policy_budget_mutation_fails(self):
        m=fixture_meta();m['alert_policy']['top_budgets']=[.10]
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_07_missing_artifact_does_not_train_or_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ModelBundleError,'Missing V3.2 bundle metadata'):
                load_frozen_bundle(d)

    def test_08_metadata_json_must_be_frozen(self):
        m=fixture_meta();m['status']='DRAFT'
        with self.assertRaises(ModelBundleError): validate_metadata(m)

    def test_09_predict_rejects_missing_feature(self):
        fake=VerifiedBundle(FakePipeline(),FakeCalibrator(),fixture_meta())
        frame=pd.DataFrame({k:[1.] for k in B5_FEATURES})
        frame=frame.drop(columns=['gpa4_term'])
        with self.assertRaises(ModelBundleError):predict_features(frame,fake)

    def test_10_predict_rejects_extra_label(self):
        fake=VerifiedBundle(FakePipeline(),FakeCalibrator(),fixture_meta())
        frame=pd.DataFrame({k:[1.] for k in B5_FEATURES})
        frame['dropout_next_2_terms']=1
        with self.assertRaises(ModelBundleError):predict_features(frame,fake)

    def test_11_valid_prediction_no_train_or_labels(self):
        fake=VerifiedBundle(FakePipeline(),FakeCalibrator(),fixture_meta())
        frame=pd.DataFrame({k:[1.] for k in B5_FEATURES})
        for cat in CATEGORICAL_B5_FEATURES:
            frame[cat]='A'
        out=predict_features(frame,fake)
        self.assertEqual(len(out),1)
        self.assertAlmostEqual(out[0],.37)

    def test_12_sha256_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'candidate.joblib'
            p.write_bytes(b'alpha')
            first=sha256(p)
            p.write_bytes(b'beta')
            self.assertNotEqual(first,sha256(p))


if __name__=='__main__':
    unittest.main()
