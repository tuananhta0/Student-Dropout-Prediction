"""Unit tests for synthetic Phase 2-I audit; do not fit model or score Test."""
import unittest
import numpy as np
import pandas as pd
from scripts.audit_phase2_I import (
    THRESHOLD, wilson, topk_indices, support, metrics_of_group,
    merge_audit, risk_budgets, assert_unique_keys, ece
)

class TestPhase2I(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({'record_uid':['a','b','c','d'],
                             'snapshot_term_index':[1,1,1,1],
                             'y_true':[1,0,1,0],
                             'calibrated_probability':[.9,.7,.1,.2],
                             'model_id':['RF_B']*4,
                             'model_version':['v3.2-rf-b-platt-g1']*4})

    def test_01_wilson_bounds(self):
        lo,hi=wilson(2,10)
        self.assertTrue(0 <= lo <= .2 <= hi <= 1)
        self.assertIsNone(wilson(0,0))

    def test_02_topk_ceil_budget(self):
        p=np.linspace(.01,.99,23)
        self.assertEqual(len(topk_indices(p,[str(i) for i in range(23)],[1]*23,.05)),2)
        self.assertEqual(len(topk_indices(p,[str(i) for i in range(23)],[1]*23,.10)),3)

    def test_03_topk_tie_uid_deterministic(self):
        p=[.9,.9,.9,.1]; u=['b','a','c','d'];t=[1]*4
        self.assertEqual(topk_indices(p,u,t,.10).tolist(),[1])

    def test_04_support_rule(self):
        self.assertEqual(support(100,20,80,50),'SUFFICIENT')
        self.assertEqual(support(1000,19,981,500),'INSUFFICIENT_SUPPORT')

    def test_05_group_low_support_suppresses_performance(self):
        small=self.fixture()
        r=metrics_of_group(small,'gender','x')
        self.assertEqual(r['status'],'INSUFFICIENT_SUPPORT')
        self.assertIsNone(r['pr_auc'])
        self.assertIsNone(r['false_negative_rate_threshold'])
        self.assertEqual(r['false_negatives_threshold'],1)

    def test_06_group_large_wilson(self):
        g=pd.concat([self.fixture()]*40,ignore_index=True)
        g['record_uid']=[f'uid_{i}' for i in range(len(g))]
        r=metrics_of_group(g,'major','IT')
        self.assertEqual(r['status'],'SUFFICIENT')
        self.assertIsNotNone(r['recall_wilson95'])
        self.assertTrue(0<=r['pr_auc']<=1)

    def test_07_duplicate_key_fails(self):
        with self.assertRaisesRegex(ValueError,'duplicate composite keys'):
            assert_unique_keys(pd.concat([self.fixture(),self.fixture()]),'pred')

    def test_08_missing_gender_fails(self):
        with self.assertRaisesRegex(ValueError,'gender'):
            merge_audit(pd.DataFrame(),pd.DataFrame(),pd.DataFrame())

    def test_09_bad_probability_fails(self):
        with self.assertRaisesRegex(ValueError,'Invalid scores'):
            topk_indices([.5,1.3],['a','b'],[1,1],.10)

    def test_10_wrong_budget_fails(self):
        with self.assertRaisesRegex(ValueError,'locked'):
            topk_indices([.5,.7],['a','b'],[1,1],.15)

    def test_11_ece_zero_for_perfect_predictions(self):
        self.assertEqual(ece([0,1],[0.,1.]),0.)

    def test_12_frozen_threshold_metrics_fixed(self):
        r=risk_budgets(self.fixture())
        self.assertEqual(r['threshold']['threshold'],THRESHOLD)
        self.assertEqual(r['threshold']['false_negatives'],1)
        self.assertEqual(r['top10']['alerts'],1)
        self.assertEqual(r['top10']['true_positives'],1)

    def test_13_missing_uid_key_fails(self):
        with self.assertRaisesRegex(ValueError,'missing keys'):
            assert_unique_keys(pd.DataFrame({'record_uid':['a']}),'x')

    def test_14_student_metadata_not_used_as_model_feature(self):
        # The audit joins gender as a retrospective subgroup dimension only.
        from scripts.audit_phase2_I import MODEL_VERSION
        self.assertEqual(MODEL_VERSION, 'v3.2-rf-b-platt-g1')
        self.assertNotIn('gender', __import__('src.features',fromlist=['B5_FEATURES']).B5_FEATURES)

if __name__ == '__main__':
    unittest.main()
