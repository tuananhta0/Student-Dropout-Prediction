"""Kiểm thử HTTP với hồ sơ giả nhỏ; không tải model hoặc dữ liệu thật khi chạy unit test."""
import importlib
import sys
import unittest
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from core.chatbot_engine import CMCChatbotEngine
from core.security import Principal, SessionStore

class FakeEngine(CMCChatbotEngine):
    def __init__(self):
        self.meta={'threshold':.12,'test_recall_top10':.47,'version':'test'}
        self.latest_df=pd.DataFrame([dict(student_id='BIT250093',record_uid='A',class_code='IT25-09',major_code='IT',major_name='CNTT',state_as_of='ENROLLED',admission_score=25,admission_method='THPT',gender='F',risk_probability=.04,prediction_status='VALID',snapshot_term_index=3),dict(student_id='BBA250001',record_uid='B',class_code='BA25-01',major_code='BA',major_name='QTKD',state_as_of='ENROLLED',admission_score=24,admission_method='THPT',gender='M',risk_probability=.2,prediction_status='VALID',snapshot_term_index=3)])
        self.df=self.latest_df; self.active_latest_df=self.df;self.ml_df=self.df;self.df_attempts=pd.DataFrame();self.data_mode='DEMO';self.data_as_of='2027-02-03'
    def get_student_records(self,id): return self.df[self.df.student_id.eq(id)|self.df.record_uid.eq(id)]
    def get_student_courses(self,*args):return []
    def evaluate_student_warnings(self,row):return []
    def calc_next_term_gpa_projection(self,row):return None
    def calculate_retake(self,*args):return {'gpa_after':3.0}
    def calculate_payment_scenario(self,*args):return {'difference_pp':0}
    def get_all_classes_summary(self,major=''):
        return [{'major_code':r.major_code,'class_code':r.class_code,'n_students':1,'gpa_mean':2.,'high_risk_count':0} for r in self.df.itertuples() if not major or r.major_code==major]
    def get_class_detail(self,code):
        r=self.df[self.df.class_code.eq(code)]
        return {} if r.empty else dict(major_code=r.iloc[0].major_code,summary_text='Tóm tắt lớp',students=[])
    def get_faculties_overview(self):return []
    def get_faculty_detail(self,major):return {'major_code':major}

class SecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=sys.modules.pop('app',None)
        with patch('core.runtime_config.configure_runtime',return_value='dummy'),patch('core.chatbot_engine.CMCChatbotEngine',return_value=FakeEngine()):
            cls.module=importlib.import_module('app')
        cls.password_patch=patch('core.security.check_password',side_effect=lambda u,p:u==p)
        cls.password_patch.start()
        cls.client=TestClient(cls.module.app,raise_server_exceptions=False)
    @classmethod
    def tearDownClass(cls):
        cls.password_patch.stop();cls.client.close();sys.modules.pop('app',None)
        if cls.original is not None:sys.modules['app']=cls.original
    def login(self,u):
        r=self.client.post('/api/login',json={'username':u,'password':u});self.assertEqual(r.status_code,200,r.text)
        return {'Authorization':'Bearer '+r.json()['access_token']}
    def chat(self,h,q,**extra):return self.client.post('/api/chat',headers=h,json={'message':q,**extra})
    def test_unauthenticated(self):
        for url in ['/api/student/me?user_id=BIT250093','/api/advisor/classes','/api/admin/faculties','/api/data-view','/api/advisor/class/IT25-09','/api/admin/faculty/IT']:
            with self.subTest(url=url): self.assertEqual(self.client.get(url).status_code,401)
        for url in ['/api/student/gpa-scenario','/api/student/payment-scenario','/api/chat']:
            self.assertEqual(self.client.post(url,json={'user_id':'BIT250093','message':'Điều 18','payment':1}).status_code,401)
    def test_login_error_and_revocation(self):
        self.assertEqual(self.client.post('/api/login',json={'username':'ADMIN','password':'wrong'}).status_code,401)
        h=self.login('ADMIN');self.assertEqual(self.client.post('/api/logout',headers=h).status_code,200)
        self.assertEqual(self.client.get('/api/admin/faculties',headers=h).status_code,401)
        self.assertEqual(self.client.get('/api/admin/faculties',headers={'Authorization':'Bearer bogus'}).status_code,401)
    def test_student_scope_and_whitelist(self):
        h=self.login('BIT250093')
        r=self.client.get('/api/student/me?user_id=BIT250093',headers=h);self.assertEqual(r.status_code,200,r.text)
        self.assertNotIn('gender',r.json()['latest']);self.assertNotIn('record_uid',r.json()['latest']);self.assertNotIn('risk_probability',r.json()['latest'])
        for path in ['/api/student/me?user_id=BBA250001&role=admin','/api/admin/faculties?role=admin','/api/advisor/classes?role=admin']:
            self.assertEqual(self.client.get(path,headers=h).status_code,403)
        self.assertNotIn('active_students',self.client.get('/api/data-view',headers=h).json())
    def test_scenarios(self):
        h=self.login('BIT250093')
        for path in ['/api/student/gpa-scenario','/api/student/payment-scenario']:
            self.assertEqual(self.client.post(path,headers=h,json={'user_id':'BBA250001','payment':1,'role':'admin'}).status_code,403)
            self.assertEqual(self.client.post(path,headers=h,json={'user_id':'BIT250093','payment':1}).status_code,200)
    def test_advisor_scope(self):
        h=self.login('GV001')
        self.assertEqual([r['major_code'] for r in self.client.get('/api/advisor/classes',headers=h).json()['classes']],['IT'])
        self.assertEqual(self.client.get('/api/advisor/class/IT25-09',headers=h).status_code,200)
        self.assertEqual(self.client.get('/api/advisor/class/BA25-01',headers=h).status_code,403)
        self.assertEqual(self.client.get('/api/student/me?user_id=BBA250001',headers=h).status_code,403)
        self.assertEqual(self.client.get('/api/admin/faculty/IT',headers=h).status_code,403)
    def test_chat_intents(self):
        h=self.login('BIT250093')
        r=self.chat(h,'Điểm đầu vào của em');self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['intent'],'admission');self.assertIn('25',r.json()['reply'])
        self.assertEqual(self.chat(h,'Nguy cơ thôi học của em').json()['intent'],'risk')
        for q in ['Lớp mạnh nhất','Học bổng có giữ chân','Tổng số sinh viên','Xem hồ sơ BBA250001','Xem hồ sơ BIT250093 và BBA250001']:
            self.assertEqual(self.chat(h,q,role='admin',logged_in_user='ADMIN').status_code,403,q)
        self.assertEqual(self.chat(h,'Điều 18').json()['intent'],'regulation')
    def test_context_isolated(self):
        h=self.login('BIT250093');other=self.login('BIT250093')
        self.chat(h,'Điều 18')
        r=self.chat(h,'Khoản 2 thì sao?');self.assertEqual([c['id'] for c in r.json()['regulations']],['D18.K2'])
        self.assertFalse(self.chat(other,'Khoản 2 thì sao?').json()['regulations'])
    def test_xss_sanitizer(self):
        html=self.module.sanitize_reply('<img src=x onerror=alert(1)><script>alert(1)</script><b onclick="evil()">ok</b><a href="javascript:alert(1)">x</a>')
        for bad in ('<img','<script','onclick','javascript:'):self.assertNotIn(bad,html)
        self.assertIn('<b>ok</b>',html)
    def test_expired(self):
        store=SessionStore(ttl=-1);t=store.issue(Principal('ADMIN','admin'))
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):store.resolve(t)
    def test_admin(self):
        h=self.login('ADMIN');self.assertEqual(self.client.get('/api/admin/faculties',headers=h).status_code,200)
        self.assertEqual(self.client.get('/api/student/me?user_id=BBA250001',headers=h).status_code,200)
    def test_rule_calculation_with_article(self):
        h=self.login('BIT250093')
        response=self.chat(h,'Theo Điều 18, em có GPA 0.79 HK1 bị cảnh báo không?')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['intent'],'regulation')
        self.assertIn('Bị cảnh báo vì GPA',response.json()['reply'])
        self.assertTrue(response.json()['regulations'])

    def test_three_digit_class_scope(self):
        h=self.login('BIT250093')
        self.assertEqual(self.chat(h,'Xem BA25-123').status_code,403)
