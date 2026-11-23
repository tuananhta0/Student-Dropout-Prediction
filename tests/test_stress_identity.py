"""Kiem tra scope dinh danh MAIN/STRESS V3.2.1 (khong can dataset that)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from scripts.review_ml_extended import verify_stress


class StressIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.main_run = root / 'V3_2_MAIN'
        stress = root / 'V3_2_STRESS' / 'manifests'
        stress.mkdir(parents=True)
        (stress / 'generation_manifest.json').write_text('{}', encoding='utf-8')
        self.main = pd.DataFrame({
            'scenario_id':['V3_2_MAIN'] * 2,
            'record_uid':['SIMV31_000001', 'SIMV31_000002'],
            'student_id':['BIT220001', 'BIT220002'],
        })
        self.stress = pd.DataFrame({
            'scenario_id':['V3_2_STRESS'] * 2,
            'record_uid':['SIMV31_000001', 'SIMV31_000002'],
            'student_id':['BIT229991', 'BIT229992'],
        })

    def check(self, main=None, stress=None):
        main = self.main if main is None else main
        stress = self.stress if stress is None else stress
        source = SimpleNamespace(run=self.main_run, table=lambda _: main)
        with patch('scripts.review_ml_extended.StressDataset') as mock_dataset:
            mock_dataset.return_value.table.side_effect = (
                lambda name: stress if name == 'students' else pd.DataFrame())
            return verify_stress(source, require_stress=True)

    def test_reused_local_record_uids_are_valid_across_scenarios(self):
        result = self.check()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['identity_scope'], 'scenario_id + record_uid')
        self.assertEqual(result['record_uid_overlap_local'], 2)
        self.assertEqual(result['student_id_overlap'], 0)

    def test_same_student_id_across_scenarios_is_rejected(self):
        stress = self.stress.copy()
        stress.loc[0, 'student_id'] = self.main.loc[0, 'student_id']
        with self.assertRaisesRegex(ValueError, 'student_id'):
            self.check(stress=stress)

    def test_duplicate_local_record_uid_is_rejected(self):
        stress = self.stress.copy()
        stress.loc[1, 'record_uid'] = stress.loc[0, 'record_uid']
        with self.assertRaisesRegex(ValueError, 'record_uid'):
            self.check(stress=stress)

    def test_wrong_scenario_is_rejected(self):
        stress = self.stress.copy()
        stress.loc[1, 'scenario_id'] = 'V3_2_MAIN'
        with self.assertRaisesRegex(ValueError, 'scenario_id'):
            self.check(stress=stress)


if __name__ == '__main__':
    unittest.main()
