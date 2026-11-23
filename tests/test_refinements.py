"""Hồi quy cho các lỗi sau nghiệm thu RAG, phạm vi demo và số liệu dashboard."""
import os
from collections import Counter
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from fastapi import HTTPException
from core.retrieval import RegulationSearch, public_focus, retry_transient
from scripts.benchmark_rag import EXTENDED
from core.security import Principal, RateLimiter, authenticate, require_class
from core.chatbot_engine import CMCChatbotEngine
from core.intent_router import route
from core.academic_rules import projection
from tests.test_security import FakeEngine

class RefinementTests(unittest.TestCase):
    def test_paraphrases(self):
        s=RegulationSearch()
        for q,a in EXTENDED:
            with self.subTest(q=q):
                rows=s.retrieve(q)
                self.assertEqual(rows[0]['article'] if rows else None,a)
    def test_public_projection_privacy(self):
        focus=public_focus('Làm sao xin chấm lại bài thi BIT250093, secret-123?')
        self.assertIn('phuc tra',focus);self.assertNotIn('BIT250093',focus);self.assertNotIn('secret',focus)
    def test_semantic_recovers_missed_candidate(self):
        s=RegulationSearch();i=next(i for i,c in enumerate(s.chunks) if c['article']==16)
        score=np.zeros(len(s.chunks));score[i]=.95
        # lexical misses all candidates; semantic query still runs on full corpus.
        with patch.object(s,'bags',[Counter() for _ in s.chunks]),patch.object(s,'_cloud',return_value=object()),patch.object(s,'_semantic',return_value=score):
            self.assertEqual(s.retrieve('phúc tra kết quả thi',cloud=True)[0]['article'],16)
    def test_retry(self):
        exc=RuntimeError();exc.code=503;call=Mock(side_effect=[exc,'OK'])
        with patch('core.retrieval.time.sleep'):self.assertEqual(retry_transient(call),'OK')
        self.assertEqual(call.call_count,2)
        exc.code=429;call=Mock(side_effect=exc)
        with self.assertRaises(RuntimeError):retry_transient(call)
        self.assertEqual(call.call_count,1)
    def test_context_and_intent(self):
        r=RegulationSearch();p=Principal('BIT250093','student');e=FakeEngine();ctx={}
        with patch.dict(os.environ,{'CMC_RAG_CLOUD':'0'}):
            self.assertEqual(route(e,r,p,'Xin chào',ctx)['intent'],'greeting')
            route(e,r,p,'Điều 18',ctx);route(e,r,p,'Điều 99',ctx)
            self.assertFalse(route(e,r,p,'Khoản 2 thì sao?',ctx)['regulations'])
            for q,a in [('Em có thể bảo lưu khi bị ốm không?',21),('Điều kiện học phần tiên quyết của em là gì?',3)]:
                self.assertEqual(route(e,r,p,q,ctx)['regulations'][0]['article'],a)
            self.assertFalse(route(e,r,p,'Em có thể đóng tiền bằng thẻ ngân hàng không?',ctx)['regulations'])
    def test_demo_password_matches_account(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'accounts.json';f.write_text('{"__demo_shared__": {"hash": "old"}}')
            with patch.dict(os.environ,{'CMC_ACCOUNTS_FILE':str(f)}):
                e=FakeEngine()
                for uid,role in [('ADMIN','admin'),('GV001','advisor'),('GV002','advisor'),('GV003','advisor'),('BIT250093','student'),('BBA250001','student')]:
                    self.assertEqual(authenticate(e,uid,uid).role,role)
                self.assertEqual(authenticate(e,' bit250093 ','BIT250093').user_id,'BIT250093')
                for uid,pw in [('ADMIN','wrong'),('ADMIN','BIT250093'),('BIT250093','ADMIN'),('ADMIN','admin'),('ADMIN','mật khẩu'),('INVALID','INVALID'),('A','A')]:
                    with self.assertRaises(HTTPException) as exc:authenticate(e,uid,pw)
                    self.assertEqual(exc.exception.status_code,401)
    def test_rate_and_class_scope(self):
        limit=RateLimiter(2,60);limit.check('a');limit.check('a');limit.check('b')
        with self.assertRaises(HTTPException) as cm:limit.check('a')
        self.assertEqual(cm.exception.status_code,429)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'scope.json';f.write_text('{"GV001":["IT25-09"]}')
            with patch.dict(os.environ,{'CMC_ADVISOR_SCOPE_FILE':str(f)}):
                p=Principal('GV001','advisor');require_class(p,'IT25-09')
                with self.assertRaises(HTTPException):require_class(p,'IT25-10')
    def test_historical_gpa(self):
        e=CMCChatbotEngine.__new__(CMCChatbotEngine)
        rec=pd.DataFrame([dict(student_id='S',snapshot_term_index=1,gpa4_term=2.20),dict(student_id='S',snapshot_term_index=3,gpa4_term=3.30)])
        out=e.answer_personal_metric_query('GPA học kỳ 1 của em',rec)
        self.assertIn('2.20',out);self.assertNotIn('3.30',out)
        self.assertIn('Chưa có',e.answer_personal_metric_query('GPA học kỳ 2 của em',rec))
    def test_coverage_and_null(self):
        e=CMCChatbotEngine.__new__(CMCChatbotEngine);e.meta={'threshold':.12}
        e.active_latest_df=pd.DataFrame([dict(record_uid='a',class_code='IT25-01',major_code='IT',major_name='IT',gpa4_cumulative=np.nan,credits_owed_cumulative=0,failed_credit_ratio=0,warning_bool=False,intervention_risk=np.nan,tuition_debt_end_vnd=0)])
        f=e.get_faculties_overview()[0];self.assertIsNone(f['high_risk_pct']);self.assertEqual(f['valid_predictions'],0);self.assertIsNone(f['gpa_mean'])
        self.assertEqual(e.get_all_classes_summary()[0]['coverage_pct'],0)
    def test_near_graduation_budget(self):
        row=dict(gpa_quality_points=240,gpa_credit_count=120,gpa4_cumulative=2,snapshot_term_index=8,credits_required=125,credits_earned_cumulative=120,credits_owed_cumulative=2,new_course_credits_available=20,registration_max_credits=22)
        self.assertEqual(projection(row)['next_tc'],3)

    def test_embedding_cache(self):
        with tempfile.TemporaryDirectory() as t:
            a=RegulationSearch();a.root=Path(t)
            matrix=np.zeros((len(a.chunks),768));matrix[:,0]=1
            query=matrix[:1]
            with patch.object(a,'_embed',side_effect=[matrix,query]) as call:
                self.assertTrue(np.allclose(a._semantic('gpa'),1));self.assertEqual(call.call_count,2)
            b=RegulationSearch();b.root=Path(t)
            with patch.object(b,'_embed',return_value=query) as call:
                self.assertTrue(np.allclose(b._semantic('gpa'),1));self.assertEqual(call.call_count,1)
    def test_review_existing_predictions(self):
        from scripts.review_experiment import review
        from scripts.train_model import evaluate
        rows=[]
        for cohort,cutoff,end in [(2022,'2024-02-03','2025-02-03'),(2024,'2025-02-03','2026-02-03'),(2025,'2027-02-03','2028-02-03')]:
            for i,y in enumerate((0,1)):
                for term in (1,2):rows.append(dict(record_uid=f'{cohort}-{i}',snapshot_term_index=term,cohort_year=cohort,snapshot_cutoff=cutoff,label_observation_end=end,label_observed=True,dropout_next_2_terms=y,risk_probability=.8 if y else .1,major_code='IT',degree_type='Bachelor'))
        data=pd.DataFrame(rows);students=data[['record_uid']].drop_duplicates().assign(gender='F')
        r=review(data,students,dict(model_available_as_of='2027-02-03',supported_terms=[1,2],threshold=.12,test_pr_auc=1.),{'so_luong_tap':{'test':4}})
        self.assertEqual(r['test_snapshot']['n'],4);self.assertEqual(r['test_sinh_vien_snapshot_dau_tien']['n'],2);self.assertAlmostEqual(r['ece'],.15)
    def test_package_research_duplicate_basename(self):
        from scripts.package_research import pack_research
        from core.data_pipeline import digest
        from scripts.release_model import MODEL_FILES
        from types import SimpleNamespace
        import zipfile,json
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source.zip';source.write_bytes(b'source')
            main=root/'V3_2_MAIN';stress=root/'V3_2_STRESS';model=root/'model';model.mkdir()
            for name in MODEL_FILES:(model/name).write_text(name)
            dsets=[]
            for run in (main,stress):
                entries={}
                for i in ('full','demo'):
                    f=run/i/'X.csv.gz';f.parent.mkdir(parents=True);f.write_text(i)
                    entries[i]={'file':'X.csv.gz','sha256':digest(f)}
                dsets.append(SimpleNamespace(run=run,manifest={'parent_archive_sha256':digest(source),'table_artifacts':entries}))
            out=root/'package.zip'
            with patch('scripts.package_research.verify_pair',return_value={'version':'test'}),patch('scripts.package_research.Dataset',return_value=dsets[0]),patch('scripts.package_research.StressDataset',return_value=dsets[1]):
                pack_research(source,root,model,out)
            with zipfile.ZipFile(out) as z:
                self.assertEqual(z.read('du_lieu/V3_2_MAIN/full/X.csv.gz'),b'full');self.assertEqual(z.read('du_lieu/V3_2_STRESS/demo/X.csv.gz'),b'demo');self.assertIsNone(z.testzip())
