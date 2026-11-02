"""Phase 2-G unit tests. Synthetic arrays only, no Test or source data opened."""
import unittest
import numpy as np
from src.model_policy_v32 import (
    apply_platt, best_f1_threshold, check_probs,
    expected_calibration_error, fit_platt, ranked_budget, split_uids,
)
from scripts.freeze_phase2_G import (CALIBRATION_FRACTION, MODEL_VERSION, WINNER)


class TestPhase2G(unittest.TestCase):
    def test_01_frozen_candidate_is_rf_b(self):
        self.assertEqual(WINNER, 'RF_B')
        self.assertEqual(MODEL_VERSION, 'v3.2-rf-b-platt-g1')

    def test_02_calibration_policy_uid_groups_disjoint(self):
        uid = np.repeat([f'UID{i:04d}' for i in range(101)], 3)
        cal, pol = split_uids(uid)
        self.assertTrue(np.all(cal ^ pol))
        self.assertFalse(set(uid[cal]) & set(uid[pol]))
        self.assertEqual(cal.sum()+pol.sum(), len(uid))
        self.assertAlmostEqual(CALIBRATION_FRACTION, .30)

    def test_03_uid_partition_reproducible_when_rows_shuffled(self):
        uid = np.repeat([f'UID{i:04d}' for i in range(101)], 3)
        c1, _ = split_uids(uid)
        rng = np.random.default_rng(3)
        order = rng.permutation(len(uid))
        c2, _ = split_uids(uid[order])
        self.assertEqual(set(uid[c1]), set(uid[order][c2]))

    def test_04_invalid_probs_rejected(self):
        for x in ([-.1, .3], [1.3], [np.nan], [], [[.1,.2]]):
            with self.assertRaises(ValueError):
                check_probs(x)

    def test_05_platt_outputs_probabilities_and_monotonic(self):
        p = np.array([.01,.03,.06,.12,.3,.48,.55,.8,.95,.98] * 5)
        y = np.array([0,0,0,0,0,0,1,1,1,1] * 5)
        cal = fit_platt(p, y)
        result = apply_platt(cal, np.array([.01,.1,.5,.9]))
        self.assertTrue(np.all(np.diff(result)>0))
        self.assertTrue(np.all((result>=0)&(result<=1)))

    def test_06_platt_requires_two_classes(self):
        with self.assertRaises(ValueError):
            fit_platt([.1,.3],[0,0])

    def test_07_ece_extremes(self):
        self.assertAlmostEqual(expected_calibration_error([0,1], [.0,1.0]), 0)
        self.assertAlmostEqual(expected_calibration_error([0,1], [.5,.5]), 0)
        self.assertGreater(expected_calibration_error([0,0], [.7,.8]), .7)

    def test_08_threshold_grid_reproducible(self):
        y = np.array([0,0,1,1])
        p = np.array([.01,.2,.65,.8])
        r = best_f1_threshold(y,p)
        self.assertTrue(.05 <= r['threshold'] <= .80)
        self.assertEqual(r['f1'], 1.)
        self.assertEqual(r, best_f1_threshold(y,p))

    def test_09_threshold_not_automatically_reference_point_21(self):
        y = np.array([0,0,0,1,1])
        p = np.array([.01,.03,.15,.75,.9])
        r = best_f1_threshold(y,p)
        self.assertNotEqual(r['threshold'], .21)

    def test_10_topk_ceil_and_deterministic_tie(self):
        y = [1,0,0,1,1,0,0,0,1,0,0]
        p = [.5]*11
        ids = ['U09','U02','U08','U01','U03','U11','U04','U07','U05','U10','U06']
        terms = [1]*11
        out = ranked_budget(y,p,ids,terms,.10)
        self.assertEqual(out['k'],2)
        self.assertEqual(out['positive_found'],1)
        order = np.array([8,4,1,10,7,2,0,9,3,5,6])
        b = ranked_budget(np.array(y)[order],np.array(p)[order],np.array(ids)[order],np.array(terms)[order],.10)
        self.assertEqual(out,b)

    def test_11_reject_duplicate_keys(self):
        with self.assertRaises(ValueError):
            ranked_budget([1,0],[.5,.5],['U1','U1'],[1,1],.10)

    def test_12_both_groups_have_valid_budget(self):
        y = [0,1,0,0,1,0,0,0]
        p = [.1,.4,.7,.2,.9,.3,.8,.6]
        u = [f'U{i}' for i in range(8)]
        one = [1]*8
        self.assertEqual(ranked_budget(y,p,u,one,.05)['k'],1)
        self.assertEqual(ranked_budget(y,p,u,one,.10)['k'],1)


if __name__ == '__main__':
    unittest.main()
