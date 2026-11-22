import unittest
import pandas as pd
from data_pipeline import certificates, asof_state, visible_attempts, course_features, unique, merge_exact, Dataset
from pathlib import Path
import tempfile
import json


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.snap = pd.DataFrame({'record_uid':['A','A'],'snapshot_term_index':[1,2],
                                  'snapshot_cutoff':['2026-02-03','2026-07-15']})

    def test_certificate_not_future(self):
        events = pd.DataFrame({'record_uid':['A'],'requirement_type':['PE'], 'completion_status':['VERIFIED_SYNTHETICALLY'], 'available_at':['2026-07-15']})
        self.assertEqual(certificates(self.snap,events),['Chưa hoàn thành','PE'])

    def test_certificate_unverified(self):
        events = pd.DataFrame({'record_uid':['A','A'],'requirement_type':['PE','PE'], 'completion_status':['VERIFIED_SYNTHETICALLY','REVOKED'], 'available_at':['2026-02-03','2026-07-15']})
        self.assertEqual(certificates(self.snap,events),['PE','Chưa hoàn thành'])

    def test_state_asof(self):
        states = pd.DataFrame({'record_uid':['A','A'],'to_status':['ENROLLED','DROPOUT'],'effective_at':['2026-02-03','2026-07-15']})
        self.assertEqual(asof_state(self.snap,states,'snapshot_cutoff').tolist(),['ENROLLED','DROPOUT'])

    def test_attempt_cutoff_includes_boundary_excludes_future(self):
        attempts = pd.DataFrame({'record_uid':['A']*3,'term_index':[1,1,2], 'available_at':['2026-02-03','2026-02-04','2026-02-03'], 'score_published_at':['2026-02-03']*3})
        self.assertEqual(len(visible_attempts(attempts,self.snap.iloc[:1])),1)

    def test_unpublished_grade(self):
        attempts = pd.DataFrame({'record_uid':['A'],'term_index':[1],'available_at':['2026-02-03'],'score_published_at':['2026-02-04']})
        self.assertTrue(visible_attempts(attempts,self.snap.iloc[:1]).empty)

    def test_course_feature_asof(self):
        a = pd.DataFrame({'record_uid':['A','A'],'term_index':[1,1],'available_at':['2026-02-03','2026-02-04'],'score_published_at':['2026-02-03','2026-02-04'],'course_final_score10':[9,0],'exam_final_score10':[8,0],'exam_eligible':[True,False],'unexcused_absent_hours':[0,20]})
        r = course_features(self.snap,a).iloc[0]
        self.assertEqual(r.ca_score10_min,9)
        self.assertEqual(r.ca_ineligible_exams,0)

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            unique(pd.concat([self.snap,self.snap]),['record_uid','snapshot_term_index'],'test')

    def test_missing_merge_key(self):
        with self.assertRaises(ValueError):
            merge_exact(self.snap,self.snap.iloc[:1],['record_uid','snapshot_term_index'],'test')

    def test_reject_stress(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'manifests';p.mkdir(); (p/'generation_manifest.json').write_text(json.dumps({'scenario_id':'V3_2_STRESS'}))
            with self.assertRaises(ValueError):
                Dataset(root)
