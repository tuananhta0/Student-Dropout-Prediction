"""Test Phase H metric invariants WITHOUT opening frozen Test dataset."""
from __future__ import annotations

import unittest
import numpy as np

from scripts.evaluate_phase2_H import (
    EXPECTED, REFERENCE_B5, binary_metrics, bootstrap_uid, require,
    stratify_by_term, verify_reference_metrics,
)

class TestPhase2H(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,1,0,1,0,1,0,1,0,1,0,0],dtype=int)
        self.p=np.array([.05,.7,.2,.8,.3,.6,.1,.9,.3,.45,.2,.11])
        self.u=np.array([f'S{i:02d}' for i in range(len(self.y))])
        self.t=np.tile([1,2,3],4)

    def test_01_confusion_at_locked_threshold(self):
        d=binary_metrics(self.y,self.p,self.u,self.t,.27)
        self.assertEqual(d['confusion_matrix'],[[5,2],[0,5]])
        self.assertEqual(d['positive'],5)

    def test_02_policy_topk_independent_of_threshold(self):
        a=binary_metrics(self.y,self.p,self.u,self.t,.27)
        b=binary_metrics(self.y,self.p,self.u,self.t,.80)
        self.assertEqual(a['top10'],b['top10'])
        self.assertNotEqual(a['alerts_at_threshold'],b['alerts_at_threshold'])

    def test_03_per_term_counts(self):
        d=stratify_by_term(self.y,self.p,self.u,self.t,.27)
        self.assertEqual(set(d),{'1','2','3'})
        self.assertEqual([d[str(i)]['n'] for i in (1,2,3)],[4,4,4])

    def test_04_nonbinary_labels_fail(self):
        bad=self.y.copy();bad[0]=2
        with self.assertRaises(ValueError):binary_metrics(bad,self.p,self.u,self.t,.27)

    def test_05_duplicate_key_fail(self):
        with self.assertRaises(ValueError):binary_metrics(self.y,self.p,np.repeat('S1',12),np.ones(12,int),.27)

    def test_06_bad_probability_fail(self):
        bad=self.p.copy();bad[0]=np.nan
        with self.assertRaises(ValueError):binary_metrics(self.y,bad,self.u,self.t,.27)

    def test_07_reference_check_exact_match(self):
        a={'pooled':{'pr_auc':REFERENCE_B5['pooled_pr_auc']},
           'by_term':{'1':{'pr_auc':REFERENCE_B5['hk1_pr_auc'],
                           'top10':{'recall_at_k':REFERENCE_B5['hk1_recall_at_top10']}}}}
        self.assertEqual(verify_reference_metrics(a)['status'],'MATCH')

    def test_08_reference_check_bad_fails(self):
        a={'pooled':{'pr_auc':.2},'by_term':{'1':{'pr_auc':.2,'top10':{'recall_at_k':.2}}}}
        self.assertEqual(verify_reference_metrics(a)['status'],'INVESTIGATE')

    def test_09_expected_test_sizes_fixed(self):
        self.assertEqual(EXPECTED['n'],9284)
        self.assertEqual(EXPECTED['by_term'][1],(3234,248))

    def test_10_bootstrap_determinism_with_duplicate_uid_draws(self):
        uid=np.repeat([f'U{i:03d}' for i in range(80)],3)
        term=np.tile([1,2,3],80)
        y=np.asarray([(j//3)%3==0 or ((j%3)==1 and (j//3)%2==0) for j in range(240)], dtype=int)
        p=np.where(y==1,.65,.24)
        a=bootstrap_uid(y,p,uid,term,.27,reps=100,seed=18)
        b=bootstrap_uid(y,p,uid,term,.27,reps=100,seed=18)
        self.assertEqual(a,b)
        self.assertEqual(a['valid_reps'],100)

    def test_11_require_rejects_false(self):
        with self.assertRaisesRegex(ValueError,'expected error'):
            require(False,'expected error')

    def test_12_bootstrap_ci_ordered(self):
        uid=np.repeat([f'U{i:03d}' for i in range(80)],3)
        term=np.tile([1,2,3],80)
        y=np.asarray([(j//3)%3==0 or ((j%3)==1 and (j//3)%2==0) for j in range(240)], dtype=int)
        p=np.where(y==1,.7,.2)
        d=bootstrap_uid(y,p,uid,term,.27,reps=100)
        self.assertLessEqual(d['intervals']['pooled_pr_auc']['ci_2_5'],
                             d['intervals']['pooled_pr_auc']['ci_97_5'])

if __name__=='__main__':unittest.main()
