"""Unit tests for Phase F fixed comparison; no model fit, no Test files."""
import unittest

import numpy as np
import pandas as pd

from scripts.compare_phase2_F import (TRIALS, alert_topk, build_trial,
                                       metrics, select_shortlist, tree_preprocessing)
from src.features import B5_FEATURES


class TestPhase2F(unittest.TestCase):
    def test_01_fixed_budget_unique_ids(self):
        self.assertEqual(len(TRIALS), 7)
        self.assertEqual(len({t[0] for t in TRIALS}), 7)

    def test_02_expected_families(self):
        families = [t[1] for t in TRIALS]
        self.assertEqual(families.count('LogisticRegression'), 1)
        for family in ('RandomForest', 'XGBoost', 'LightGBM'):
            self.assertEqual(families.count(family), 2)

    def test_03_feature_count(self):
        self.assertEqual(len(B5_FEATURES), 58)
        self.assertEqual(len(set(B5_FEATURES)), 58)

    def test_04_transform_has_fixed_role_counts(self):
        transformer = tree_preprocessing()
        self.assertEqual(len(transformer.transformers_[0][2]), 53) if hasattr(transformer, 'transformers_') else self.assertEqual(len(transformer.transformers[0][2]), 53)
        self.assertEqual(len(transformer.transformers[1][2]), 5)

    def test_05_tree_builds_without_fitting(self):
        for trial in TRIALS[1:]:
            pipe = build_trial(trial)
            self.assertIn('prep', pipe.named_steps)
            self.assertIn('clf', pipe.named_steps)

    def test_06_top10_tie_break_deterministic(self):
        y = np.array([1, 0, 0, 1, 0, 1])
        scores = np.full(len(y), .3)
        uids = np.array(['z', 'a', 'b', 'c', 'd', 'e'])
        terms = np.ones(len(y), dtype=int)
        first = alert_topk(y, scores, uids, terms, .5)
        perm = np.array([5, 4, 3, 2, 1, 0])
        second = alert_topk(y[perm], scores[perm], uids[perm], terms[perm], .5)
        self.assertEqual(first, second)
        self.assertEqual(first['k'], 3)

    def test_07_pr_auc_lift(self):
        y = np.array([1, 0, 0, 1, 0, 0])
        s = np.array([.9, .1, .2, .8, .3, .4])
        m = metrics(y, s, np.array(['a', 'b', 'c', 'd', 'e', 'f']), np.ones(6))
        self.assertGreater(m['pr_auc_lift'], 1.0)
        self.assertEqual(m['n'], 6)

    def test_08_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.array([0, 1]), np.array([0.1, np.nan]),
                    np.array(['a', 'b']), np.array([1, 1]))

    def test_09_shortlist_conservative(self):
        lr = dict(trial='LR_B5', hk1_pr_auc=.25, hk1_recall_at_top10=.4, pooled_brier=.09)
        alt = dict(trial='RF_A', hk1_pr_auc=.255, hk1_recall_at_top10=.45, pooled_brier=.08)
        self.assertEqual(select_shortlist([lr, alt])['preliminary_recommendation'], 'LR_B5')
        alt['hk1_pr_auc'] = .27
        self.assertEqual(select_shortlist([lr, alt])['preliminary_recommendation'], 'RF_A')


if __name__ == '__main__':
    unittest.main()
