"""Hồi quy các lỗi tìm thấy ở commit 4097616; không cần API key hoặc model thật."""
import csv
import json
import os
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from core.intent_router import route
from core.retrieval import RegulationSearch, public_focus, cloud_call, CloudBudgetError
from core.security import Principal
from core.academic_rules import projection
from scripts.benchmark_rag import quality_pass, endpoint, CASES, EXTENDED, REGRESSION
from tests.test_security import FakeEngine
from collections import deque

class FinalValidationTests(unittest.TestCase):
    def setUp(self):
        self.search=RegulationSearch()
        self.e=FakeEngine()
        self.p=Principal('BIT250093','student')
        self.context={}
        env=patch.dict(os.environ,{'CMC_RAG_CLOUD':'0'})
        env.start();self.addCleanup(env.stop)

    def ask(self,q): return route(self.e,self.search,self.p,q,self.context)

    def test_regression_endpoint(self):
        with endpoint() as ask:
            for q,article in dict(CASES+EXTENDED+REGRESSION).items():
                with self.subTest(q=q):
                    out=ask(q);self.assertEqual(out['regulations'][0]['article'] if out['regulations'] else None,article)
                    if article is not None: self.assertIn('#page=',out['reply'])

    def test_phuc_tra_scope(self):
        out=self.search.answer('Làm sao xin chấm lại bài thi?')
        self.assertEqual({c['article'] for c in out['regulations']},{16})

    def test_multiple_articles(self):
        out=self.search.answer('Điều 3, Điều 12, Điều 13, Điều 18, Điều 20, Điều 25')
        self.assertEqual({c['article'] for c in out['regulations']},{3,12,13,18,20,25})
        self.assertIn('99',self.search.answer('Điều 18, Điều 99')['reply'])

    def test_calculation_source(self):
        out=self.ask('Em có nợ 25 tín chỉ theo Điều 16 thì sao?')
        self.assertEqual({c['article'] for c in out['regulations']},{18})
        self.assertIn('Bị cảnh báo',out['reply'])
        self.assertIn('không phải căn cứ',out['reply'])

    def test_supplement(self):
        out=self.search.answer('Các hình thức thôi học')
        self.assertEqual({c['article'] for c in out['regulations']},{12,18,21,25})
        self.assertIn('Quy-dinh-thoi-hoc-DH-CMC.pdf#page=',out['reply'])
        self.assertIn('Bản tổng hợp tham khảo',out['reply'])

    def test_historical_risk(self):
        earlier=self.e.df.iloc[[0]].copy()
        earlier['snapshot_term_index']=1;earlier['risk_probability']=.2;earlier['prediction_status']='EXPIRED'
        self.e.df=pd.concat([earlier,self.e.df],ignore_index=True)
        out=self.ask('Nguy cơ thôi học học kỳ 1 của em')
        self.assertIn('20.0%',out['reply']);self.assertIn('lịch sử',out['reply']);self.assertNotIn('4.0%',out['reply'])
        self.assertIn('Chưa có',self.ask('Nguy cơ thôi học học kỳ 2 của em')['reply'])

    def test_target_gpa(self):
        self.e.df=self.e.df.assign(gpa4_cumulative=2.75,gpa_quality_points=115.5,gpa_credit_count=42,credits_required=128,credits_earned_cumulative=42,credits_owed_cumulative=0,new_course_credits_available=86,registration_max_credits=22)
        self.assertIn('Không khả thi',self.ask('Em muốn đạt GPA tích lũy 3.5 thì kỳ sau cần bao nhiêu?')['reply'])
        self.assertIn('3.70/4',self.ask('Em muốn đạt GPA tích lũy 3.0 thì kỳ sau cần bao nhiêu?')['reply'])
        self.e.df=self.e.df.drop(columns='registration_max_credits')
        self.assertIn('Chưa đủ dữ liệu',self.ask('Em muốn đạt GPA tích lũy 3.0 thì kỳ sau cần bao nhiêu?')['reply'])

    def test_graduation(self):
        self.e.df=self.e.df.assign(credits_required=128,credits_earned_cumulative=42,gpa4_cumulative=2.75)
        out=self.ask('Em đã đủ điều kiện tốt nghiệp chưa?')
        self.assertEqual(out['intent'],'graduation');self.assertIn('Chưa đủ điều kiện',out['reply']);self.assertIn('Chưa xác minh',out['reply'])

    def test_public_focus(self):
        q=public_focus('Làm sao xin chấm lại bài thi BIT250093 secret-678?')
        self.assertIn('thủ tục',q);self.assertNotIn('secret',q);self.assertNotIn('250093',q)

    def test_budget(self):
        with patch('core.retrieval._cloud_calls',deque()),patch('core.retrieval.CLOUD_LIMIT_MINUTE',1):
            self.assertEqual(cloud_call(lambda:'ok'),'ok')
            with self.assertRaises(CloudBudgetError): cloud_call(lambda:self.fail('Không gọi khi quá ngân sách'))

    def test_deadline_fallback(self):
        event=threading.Event()
        original=self.search._answer
        def slow(q,cloud):
            if cloud: event.wait(1)
            return original(q,False)
        try:
            with patch('core.retrieval.CLOUD_DEADLINE',.02),patch.object(self.search,'_answer',side_effect=slow):
                out=self.search.answer('Điều 18',cloud=True)
                self.assertEqual(out['generation_status'],'DEADLINE');self.assertTrue(out['regulations'])
        finally: event.set()

    def test_quality_gate(self):
        x=dict(so_cau=1,hit_rate_at_5=1,mrr_at_5=1,top1=1,tu_choi_ngoai_pham_vi=1,trich_dan_hop_le=True)
        self.assertTrue(quality_pass(x));x['tu_choi_ngoai_pham_vi']=0;self.assertFalse(quality_pass(x))

    def test_manual_review_requires_scores(self):
        from scripts.review_rag import summarize
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'review.csv'
            p.write_text('cau_hoi,phuong_phap,dung_ngu_nghia,day_du\nQ,gemini,,\n',encoding='utf-8')
            with self.assertRaises(ValueError): summarize(p,'reviewer')
            p.write_text('cau_hoi,phuong_phap,dung_ngu_nghia,day_du\nQ,gemini,1,0\nF,bm25,,\n',encoding='utf-8')
            out=summarize(p,'reviewer');self.assertEqual(out['so_cau_gemini'],1);self.assertEqual(out['dat_ca_hai'],0)

    def test_bootstrap_cluster(self):
        from scripts.review_experiment import bootstrap_students
        df=pd.DataFrame({'record_uid':['A','A','B','B','C','C'],'dropout_next_2_terms':[0,0,1,1,1,0],'risk_probability':[.1,.1,.8,.7,.6,.2]})
        out=bootstrap_students(df,.5,30)
        self.assertEqual(out['n_sinh_vien'],3)
        self.assertGreater(out['n_luot_hop_le'],0)
        self.assertEqual(out,bootstrap_students(df,.5,30))

    def test_sdk_timeout_config(self):
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock
        from core.retrieval import CLOUD_DEADLINE, CLOUD_SDK_TIMEOUT_MS
        sdk=ModuleType('google.genai')
        sdk.types=SimpleNamespace(HttpOptions=lambda **kw:kw, HttpRetryOptions=lambda **kw:kw)
        sdk.Client=Mock()
        google=ModuleType('google');google.genai=sdk
        with patch.dict('sys.modules',{'google':google,'google.genai':sdk}),patch.dict(os.environ,{'CMC_RAG_CLOUD':'1','GEMINI_API_KEY':'test-only'}):
            RegulationSearch()._cloud()
        options=sdk.Client.call_args.kwargs['http_options']
        self.assertEqual(options['timeout'],CLOUD_SDK_TIMEOUT_MS)
        self.assertGreaterEqual(options['timeout'],10000)
        self.assertLess(options['timeout']/1000,CLOUD_DEADLINE)
        self.assertEqual(options['retry_options']['attempts'],1)

    def test_http400_deadline_diagnostics(self):
        from core.retrieval import safe_api_error, retry_transient
        from unittest.mock import Mock
        exc=RuntimeError('Manually set deadline 8s is too short. Minimum allowed deadline is 10s. secret-test');exc.code=400
        self.assertEqual(safe_api_error(exc),'HTTP_400_DEADLINE')
        call=Mock(side_effect=exc)
        with self.assertRaises(RuntimeError): retry_transient(call)
        self.assertEqual(call.call_count,1)
        other=RuntimeError('Invalid request');other.code=400
        self.assertEqual(safe_api_error(other),'HTTP_400')

    def test_cloud_quality_exit(self):
        from scripts.benchmark_rag import cloud_quality, main
        r=self.search.answer('Khoản 2 Điều 18')
        r['generation_status']='HTTP_400'
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'report.json';cases=Path(t)/'cases.json'
            cases.write_text(json.dumps([['Khoản 2 Điều 18',18]]),encoding='utf-8')
            with patch('sys.argv',['benchmark_rag','--cloud','--cases',str(cases),'--output',str(out)]),patch('scripts.benchmark_rag.RegulationSearch.answer',return_value=r),patch('builtins.print'):
                self.assertEqual(main(),1)
                report=json.loads(out.read_text(encoding='utf-8'))
                self.assertTrue(report['quality_gate']);self.assertFalse(report['cloud_quality_gate'])
                r['method']='gemini';r['generation_status']='OK'
                self.assertEqual(main(),0)
        rows=[{'dieu_mong_doi':18,'phuong_phap':'gemini','trang_thai_llm':'OK'}]*9+[{'dieu_mong_doi':18,'phuong_phap':'bm25','trang_thai_llm':'HTTP_400'}]
        self.assertTrue(cloud_quality(rows)['cloud_quality_gate'])
        self.assertFalse(cloud_quality([])['cloud_quality_gate'])
