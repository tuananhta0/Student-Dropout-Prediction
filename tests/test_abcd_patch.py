"""Regression A–D bổ sung. Không dùng khóa thật hoặc dữ liệu sinh viên thật."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from core.retrieval import RegulationSearch, is_tuition_deadline_query
from scripts.benchmark_rag import endpoint
from scripts.review_rag_semantics import summarize, FIELDS
from scripts.review_ml_extended import group_stats, calibration_bins, verify_stress
from scripts.smoke_live_http import probe
from scripts.setup_shared_gemini import save_env

ROOT=Path(__file__).resolve().parents[1]


class RAGPatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieval=RegulationSearch()

    def test_three_fixed_cases_http(self):
        cases=[
            ('Em đang bệnh nên muốn nghỉ học một kỳ, thủ tục thế nào?',21),
            ('Mình đang điều trị bệnh xin nghỉ học một kỳ được không?',21),
            ('Có thể chuyển đến trường khác để học không?',22),
            ('Tôi muốn đổi trường đại học khác thì làm gì?',22),
            ('Khi nào đóng học phí kỳ tới?',None),
            ('Hạn chót nộp học phí của kỳ sau?',None),
        ]
        with endpoint() as ask:
            for q,expected in cases:
                with self.subTest(q=q):
                    r=ask(q)
                    arts={c['article'] for c in r['regulations']}
                    self.assertIn(expected,arts) if expected is not None else self.assertFalse(arts)
                    if expected is None:
                        self.assertIn('không có lịch',r['reply'])

    def test_compound_policy_tuition_deadline(self):
        with endpoint() as ask:
            r=ask('Em có thể bảo lưu và ngày đóng học phí là khi nào?')
            self.assertEqual({x['article'] for x in r['regulations']},{21})
            self.assertIn('ngày hoặc hạn đóng học phí',r['reply'])
            r=ask('Có thể bảo lưu và chuyển đến trường khác?')
            self.assertTrue({21,22}.issubset({x['article'] for x in r['regulations']}))

    def test_tuition_deadline_distinct_from_debt(self):
        self.assertTrue(is_tuition_deadline_query('Khi nào đóng học phí kỳ tới?'))
        self.assertFalse(is_tuition_deadline_query('Em còn nợ học phí bao nhiêu?'))
        self.assertFalse(is_tuition_deadline_query('Hoàn tiền học phí khi rút môn?'))

    def test_40_proposals_retrieval_and_http(self):
        cases=json.loads((ROOT/'tests/cases/bo_cau_ra_soat_40_de_xuat.json').read_text(encoding='utf-8'))
        self.assertEqual(len(cases),40)
        with endpoint() as ask:
            for question,expected in cases:
                with self.subTest(question=question):
                    r=ask(question)
                    returned={c['article'] for c in r['regulations']}
                    if expected is None:self.assertFalse(returned)
                    else:self.assertIn(expected,returned)


class AcceptanceUtilities(unittest.TestCase):
    def test_manual_review_must_be_graded(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path=Path(tmp)/'review.csv'
            with csv_path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['cau_hoi','phuong_phap','ghi_chu_nguoi_cham',*FIELDS]);writer.writeheader()
                writer.writerow({'cau_hoi':'Test','phuong_phap':'bm25'})
            with self.assertRaises(ValueError):summarize(csv_path,'Reviewer')
            with csv_path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['cau_hoi','phuong_phap','ghi_chu_nguoi_cham',*FIELDS]);writer.writeheader()
                writer.writerow({'cau_hoi':'Test','phuong_phap':'bm25',**{k:'1' for k in FIELDS}})
            self.assertEqual(summarize(csv_path,'Reviewer')['dat_tat_ca_tieu_chi'],1)

    def test_one_label_does_not_report_auc(self):
        df=pd.DataFrame({'dropout_next_2_terms':[1]*30,'risk_probability':[.3]*30})
        r=group_stats(df,.12);self.assertFalse(r['coverage_both_labels'])
        self.assertNotIn('roc_auc',r);self.assertNotIn('ap',r)

    def test_calibration_uses_actual_bins(self):
        df=pd.DataFrame({'dropout_next_2_terms':[0,1],'risk_probability':[.05,.95]})
        report=calibration_bins(df)
        self.assertEqual(sum(b['n'] for b in report['bins']),2)
        self.assertGreaterEqual(report['ece'],0)

    def test_stress_not_installed_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=SimpleNamespace(run=Path(tmp)/'V3_2_MAIN')
            self.assertEqual(verify_stress(source)['status'],'NOT_INSTALLED')
            with self.assertRaises(FileNotFoundError):verify_stress(source,True)

    def test_live_http_rejects_non_localhost(self):
        with self.assertRaises(ValueError):probe('https://example.com')
        with self.assertRaises(ValueError):probe('http://192.168.1.5:8000')

    def test_shared_key_setup_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'.env';save_env(target,'A'*30)
            self.assertIn('CMC_RAG_CLOUD=1',target.read_text())
            with self.assertRaises(FileExistsError):save_env(target,'B'*30)
            with self.assertRaises(ValueError):save_env(Path(tmp)/'bad.env','bad\nkey')


if __name__=='__main__':unittest.main()
