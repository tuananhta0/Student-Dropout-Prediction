"""Focused Phase 2-L tests. No model loads or dataset reads."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import close_phase2_L as l


class TestPhase2L(unittest.TestCase):
    def test_01_required_report_inventory(self):
        for name in ('preflight.md','data_contract_verification.json',
                     'baseline_comparison.csv','model_comparison.csv',
                     'model_selection_decision.md','frozen_test_metrics.json',
                     'per_term_metrics.csv','reproducibility.json'):
            self.assertIn(name,l.REQUIRED_REPORTS)

    def test_02_per_row_test_predictions_must_not_be_stageable(self):
        self.assertIn('test_predictions_frozen.csv.gz',l.REQUIRED_REPORTS)
        self.assertNotIn('reports/ml_v3_2/test_predictions_frozen.csv.gz',l.SAFE_PATHS)

    def test_03_joblib_and_external_data_not_stageable(self):
        self.assertIn('models/v3_2/model_metadata.json',l.SAFE_PATHS)
        for forbidden in ('models/v3_2/frozen_rf_b_pipeline.joblib',
                          'models/dropout_model.joblib','data/external/v3_2_main/labels/snapshot_labels.csv.gz',
                          'data/processed/v3_2/dataset_catalog.json'):
            self.assertNotIn(forbidden,l.SAFE_PATHS)

    def test_04_no_binary_or_prediction_files_in_allowlist(self):
        for name in l.SAFE_PATHS:
            self.assertFalse(name.lower().endswith(('.joblib','.csv.gz','.zip','.pem','.key')),name)

    def test_05_stage_checker_rejects_predictions(self):
        with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
             patch.object(l,'staged_paths',return_value={'reports/ml_v3_2/test_predictions_frozen.csv.gz'}):
            with self.assertRaises(l.GateError): l.stage_check()

    def test_06_stage_checker_rejects_unknown_script(self):
        with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
             patch.object(l,'staged_paths',return_value={'scripts/unsafe.py'}):
            with self.assertRaises(l.GateError): l.stage_check()

    def test_07_stage_checker_accepts_safe_file(self):
        with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
             patch.object(l,'staged_paths',return_value={'src/inference_v32.py'}):
            l.stage_check()

    def test_08_unstamped_handover_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'handover.md';path.write_text(l.PLACEHOLDER,encoding='utf-8')
            with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
                 patch.object(l,'staged_paths',return_value={'PHASE_2_ML_Training_Handover.md'}), \
                 patch.object(l,'HANDOVER',path):
                with self.assertRaises(l.GateError): l.stage_check()

    def test_09_stamped_handover_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'handover.md';path.write_text('SHA: '+'a'*40,encoding='utf-8')
            with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
                 patch.object(l,'staged_paths',return_value={'PHASE_2_ML_Training_Handover.md'}), \
                 patch.object(l,'HANDOVER',path):
                l.stage_check()

    def test_10_sha256_detects_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data'
            path.write_bytes(b'a'); old=l.sha(path)
            path.write_bytes(b'b'); self.assertNotEqual(old,l.sha(path))

    def test_11_required_file_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(l.GateError): l.sha(Path(tmp)/'absent')

    def test_12_no_extra_gate_added(self):
        self.assertEqual(len('ABCDEFGHIJK'),11)
        self.assertEqual(l.EXPECTED_TESTS,93)
        self.assertEqual(len(l.TEST_MODULES),7)

    def test_13_stage_checker_rejects_empty_index(self):
        with patch.object(l,'ensure_branch'), patch.object(l,'git',return_value=''), \
             patch.object(l,'staged_paths',return_value=set()):
            with self.assertRaises(l.GateError): l.stage_check()


if __name__ == '__main__':
    unittest.main()
