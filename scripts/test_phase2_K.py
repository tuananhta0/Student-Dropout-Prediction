"""Phase 2-K unit/mutation tests. No external dataset or real joblib needed."""
import math
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.features import B5_FEATURES
from src.ml_v32 import CATEGORICAL_B5_FEATURES
from src.inference_v32 import (
    FORBIDDEN_SERVING_FIELDS, InferenceContractError, V32InferenceService,
    ordered_frame_from_snapshot,
)
from src.model_bundle_v32 import ModelBundleError, predict_features


class FakePipeline:
    def predict_proba(self, frame):
        assert list(frame.columns) == B5_FEATURES
        assert len(frame.columns) == 58
        # Deterministic, finite raw probability; no fit() exists.
        p = np.full(len(frame), 0.35, dtype=float)
        return np.column_stack((1-p, p))


class FakePlatt:
    def predict_proba(self, logits):
        z = np.asarray(logits).reshape(-1)
        p = 1 / (1 + np.exp(-z))
        return np.column_stack((1-p, p))


def snapshot(uid='SIMV31_000001', term=1, cutoff='2024-04-20'):
    data = {k: ('AI' if k in CATEGORICAL_B5_FEATURES else 1.0) for k in B5_FEATURES}
    data.update({'record_uid': uid, 'snapshot_term_index': term,
                 'snapshot_cutoff': cutoff, 'feature_source_latest_at': cutoff,
                 'cohort_year': 2022, 'student_id': 'NON_AUTHORITATIVE_STUDENT_ID'})
    return data


def bundle():
    return SimpleNamespace(pipeline=FakePipeline(), calibrator=FakePlatt(), metadata={
        'model_version': 'v3.2-rf-b-platt-g1',
        'serving_max_asof': '2026-10-01',
        'threshold': .27, 'features_ordered': list(B5_FEATURES),
        'alert_policy': {'top_budgets': [.05, .10], 'topk_independent_of_threshold': True},
    })


class Access:
    def __init__(self, row=None):
        self.row = row if row is not None else snapshot()
        self.calls = []
    def get_snapshot(self, uid, term, asof=None):
        self.calls.append((uid,term,asof))
        if uid != self.row['record_uid'] or term != self.row['snapshot_term_index'] or asof < self.row['snapshot_cutoff']:
            return None
        return deepcopy(self.row)


class TestPhase2K(unittest.TestCase):
    def setUp(self):
        self.access=Access()
        self.service=V32InferenceService(access=self.access, bundle=bundle())

    def test_01_exact_composite_lookup(self):
        result=self.service.predict('SIMV31_000001', 1, '2026-10-01')
        self.assertEqual(self.access.calls, [('SIMV31_000001', 1, '2026-10-01')])
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['snapshot_term_index'],1)

    def test_02_direct_bundle_probability_equivalence(self):
        result=self.service.predict('SIMV31_000001',1,'2026-10-01')
        frame=ordered_frame_from_snapshot(self.access.row,'SIMV31_000001',1,'2026-10-01')
        expected=float(predict_features(frame,self.service.bundle)[0])
        self.assertLessEqual(abs(result['probability']-expected),1e-10)
        self.assertEqual(list(frame.columns),B5_FEATURES)
        self.assertTrue(result['policy']['alert_at_threshold'])
        self.assertTrue(result['policy']['topk_requires_batch'])
        self.assertEqual(result['policy']['top_budgets'],[.05,.10])

    def test_03_unknown_uid_not_available(self):
        out=self.service.predict('MISSING_UID',1,'2026-10-01')
        self.assertEqual(out['status'],'not_available')
        self.assertNotIn('probability',out)

    def test_04_before_snapshot_not_available(self):
        out=self.service.predict('SIMV31_000001',1,'2024-04-19')
        self.assertEqual(out['status'],'not_available')

    def test_05_bad_uid_and_student_id_dont_substitute(self):
        for uid in (None,'','   ',42):
            with self.assertRaises(InferenceContractError): self.service.predict(uid,1)
        self.assertEqual(self.service.predict('NON_AUTHORITATIVE_STUDENT_ID',1)['status'],'not_available')

    def test_06_invalid_term_rejected(self):
        for term in (0,4,1.0,'1',True,None):
            with self.assertRaises(InferenceContractError): self.service.predict('SIMV31_000001',term)

    def test_07_invalid_asof_rejected(self):
        for asof in ('2026-10-02','2026-02-30','2026-10-1',123,'2026/10/01'):
            with self.assertRaises(InferenceContractError): self.service.predict('SIMV31_000001',1,asof)

    def test_08_missing_feature_fail_fast(self):
        self.access.row.pop('gpa4_term')
        with self.assertRaisesRegex(InferenceContractError,'Missing required'):
            self.service.predict('SIMV31_000001',1)

    def test_09_target_label_injected_rejected(self):
        for field in ('dropout_next_2_terms','label_observed','censor_reason','y_true'):
            self.access.row=snapshot();self.access.row[field]=1
            with self.assertRaisesRegex(InferenceContractError,'Outcome/latent'):
                self.service.predict('SIMV31_000001',1)

    def test_10_numeric_string_and_boolean_rejected(self):
        for bad in ('3.2',True,float('inf'),float('-inf')):
            self.access.row=snapshot();self.access.row['gpa4_term']=bad
            with self.assertRaisesRegex(InferenceContractError,'Invalid numeric'):
                self.service.predict('SIMV31_000001',1)

    def test_11_categorical_wrong_type_rejected(self):
        self.access.row['major_code']=1.0
        with self.assertRaisesRegex(InferenceContractError,'Invalid categorical'):
            self.service.predict('SIMV31_000001',1)

    def test_12_existing_missing_values_allowed(self):
        self.access.row['gpa4_previous']=None
        self.access.row['major_code']=None
        out=self.service.predict('SIMV31_000001',1)
        self.assertEqual(out['status'],'ok')
        frame=ordered_frame_from_snapshot(self.access.row,'SIMV31_000001',1,'2026-10-01')
        self.assertTrue(math.isnan(frame['gpa4_previous'].iat[0]))
        self.assertTrue(math.isnan(frame['major_code'].iat[0]))

    def test_13_temporal_mutation_fails(self):
        self.access.row['feature_source_latest_at']='2024-04-21'
        with self.assertRaisesRegex(InferenceContractError,'temporal'):
            self.service.predict('SIMV31_000001',1)

    def test_14_key_mismatch_fails(self):
        self.access.row['snapshot_term_index']=2
        # fake access returns None for requested term; force a corrupt adapter for key validation
        self.access.get_snapshot=lambda *args,**kwargs: self.access.row
        with self.assertRaisesRegex(InferenceContractError,'composite key'):
            self.service.predict('SIMV31_000001',1)

    def test_15_frozen_bundle_missing_refuses_no_fallback(self):
        with patch('src.inference_v32.load_frozen_bundle', side_effect=ModelBundleError('missing frozen model')):
            with self.assertRaisesRegex(ModelBundleError,'missing frozen model'):
                V32InferenceService(access=self.access)

    def test_16_hash_mismatch_refuses_no_fallback(self):
        with patch('src.inference_v32.load_frozen_bundle', side_effect=ModelBundleError('SHA256 mismatch')):
            with self.assertRaisesRegex(ModelBundleError,'SHA256 mismatch'):
                V32InferenceService(access=self.access)

    def test_17_frozen_metadata_mutation_refused(self):
        for key,value in [('threshold',.22),('model_version','legacy'),('serving_max_asof','2028-12-31')]:
            b=bundle(); b.metadata[key]=value
            with self.assertRaises(InferenceContractError):
                V32InferenceService(access=self.access,bundle=b)

    def test_21_duplicate_key_error_propagates(self):
        # Phase 1 V32DataAccess rejects duplicate composite keys at index creation.
        self.access.get_snapshot=Mock(side_effect=ValueError('Duplicate snapshot composite keys'))
        with self.assertRaisesRegex(ValueError,'Duplicate snapshot composite keys'):
            self.service.predict('SIMV31_000001',1)

    def test_19_two_frozen_boolean_features_convert_to_numeric(self):
        self.access.row['warning_current']=True
        self.access.row['academic_recovery_flag']=False
        frame=ordered_frame_from_snapshot(self.access.row,'SIMV31_000001',1,'2026-10-01')
        self.assertEqual(frame['warning_current'].iat[0],1.0)
        self.assertEqual(frame['academic_recovery_flag'].iat[0],0.0)
        self.assertEqual(self.service.predict('SIMV31_000001',1)['status'],'ok')

    def test_20_boolean_wrong_numeric_field_rejected(self):
        self.access.row['gpa4_term']=True
        with self.assertRaisesRegex(InferenceContractError,'Invalid numeric'):
            self.service.predict('SIMV31_000001',1)

    def test_18_response_never_contains_labels_or_profile(self):
        out=self.service.predict('SIMV31_000001',1)
        self.assertTrue(set(out).isdisjoint(FORBIDDEN_SERVING_FIELDS))
        self.assertNotIn('student_id',str(out))
        self.assertNotIn('features',str(out))
        self.assertIn('Synthetic',out['disclaimer'])


if __name__=='__main__': unittest.main()
