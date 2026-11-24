"""Natural conversation acceptance tests with local synthetic records."""
import importlib
import os
import sys
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from core.chatbot_engine import CMCChatbotEngine
from core.intent_router import route
from core.security import Principal
import tests.test_chat_query_v2 as fixture


class ConversationBasicsTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.QueryV2Tests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.e, self.r = self.f.e, self.f.r
        self.p = Principal('BIT250093', 'student')

    def ask(self, message, context=None, principal=None):
        return route(self.e, self.r, principal or self.p, message, {} if context is None else context)

    def test_identity_variants_local_only(self):
        with patch.object(self.r, 'answer', side_effect=AssertionError('identity sent to RAG')):
            for q in ('tôi là ai', 'Em là ai ạ?', 'toi la ai', 'Tôi đang học ngành gì?', 'Mình học lớp nào?'):
                with self.subTest(q=q):
                    out = self.ask(q)
                    self.assertEqual(out['intent'], 'personal')
                    self.assertIn('BIT250093', out['reply'])
                    self.assertIn('IT25-09', out['reply'])
                    self.assertNotIn('BBA250001', out['reply'])
                    self.assertFalse(out['regulations'])

    def test_assistant_identity_and_thanks(self):
        with patch.object(self.r, 'answer', side_effect=AssertionError('small talk sent to RAG')):
            self.assertIn('CMC Academic AI', self.ask('Bạn là ai?')['reply'])
            self.assertIn('Không có gì', self.ask('Cảm ơn bạn nhé!')['reply'])
            self.assertEqual(self.ask('xin chào ?')['intent'], 'greeting')

    def test_leave_clarification_and_followups(self):
        for first in ('Tôi có thể nghỉ học hay không', 'Em có được nghỉ học không ạ?', 'toi co the nghi hoc ko', 'Tôi muốn nghỉ học'):
            for follow, source, phrase in [('bảo lưu', 'D21.K1', 'ít nhất một học kỳ'), ('hẳn luôn', 'D21.K2–3', 'dự tuyển lại'), ('một buổi', 'D25.K1', '20%')]:
                with self.subTest(first=first,follow=follow):
                    ctx = {}
                    out = self.ask(first,ctx)
                    self.assertEqual(out['intent'],'clarify')
                    self.assertIn('thôi học hẳn',out['reply'])
                    out = self.ask(follow,ctx)
                    self.assertEqual(out['intent'],'regulation')
                    self.assertIn(source,{c['id'] for c in out['regulations']})
                    self.assertIn(phrase,out['reply'].split('<details>')[0])
                    self.assertFalse(ctx.get('basic_leave'))

    def test_leave_context_not_reused_after_new_topic(self):
        ctx={}
        self.ask('Tôi có thể nghỉ học không?',ctx)
        self.ask('Tôi là ai?',ctx)
        self.assertNotIn('basic_leave',ctx)
        self.assertNotIn('D25.K1',{c['id'] for c in self.ask('một buổi',ctx)['regulations']})

    def test_personal_concerns_actual_rule_engine(self):
        values = {'snapshot_term_index':3,'gpa4_term':.9,'gpa4_cumulative':1.8,'credits_registered_term':18,'credits_failed_term':10,'credits_owed_cumulative':25,'warning_streak':1,'warnings_total_to_date':1,'counted_duration_semesters':3,'degree_type':'Bachelor'}
        for k,v in values.items(): self.e.df.loc[0,k]=v
        with patch.object(self.e,'evaluate_student_warnings',side_effect=lambda row: CMCChatbotEngine.evaluate_student_warnings(self.e,row)), patch.object(self.r,'answer',side_effect=AssertionError('private record sent to RAG')):
            for q in ('Tôi đang mắc lỗi gì','Em đang gặp vấn đề gì về học tập?','Tình trạng học tập của tôi thế nào?'):
                out=self.ask(q)
                self.assertEqual(out['intent'],'personal')
                for fragment in ('GPA học kỳ dưới ngưỡng','Trượt quá 50% tín chỉ','Nợ quá 24 tín chỉ','không phải kết luận kỷ luật'):
                    self.assertIn(fragment,out['reply'])
                self.assertNotIn('BBA250001',out['reply'])

    def test_privacy_and_scope(self):
        for role,uid in [('advisor','GV001'),('admin','ADMIN')]:
            p=Principal(uid,role)
            out=self.ask('Tôi là ai?',principal=p)
            self.assertIn(uid,out['reply']);self.assertNotIn('BIT250093',out['reply'])
            self.assertEqual(self.ask('Tôi đang mắc lỗi gì?',principal=p)['intent'],'clarify')
        for q in ('Tôi là ai? Xem hồ sơ BBA250001','Tôi đang mắc lỗi gì? Xem hồ sơ BBA250001'):
            with self.assertRaises(HTTPException) as err:self.ask(q)
            self.assertEqual(err.exception.status_code,403)
        out=self.ask('Tôi đang mắc lỗi gì?')
        self.assertIn('Chưa có kết quả đánh giá đủ căn cứ',out['reply'])

    def test_detailed_policy_preserved(self):
        for q,article in [('Em bị bệnh nên muốn nghỉ học một kỳ',21),('Điều 18 Khoản 2',18),('Tôi muốn nghỉ học hẳn',21)]:
            out=self.ask(q);self.assertIn(article,{c['article'] for c in out['regulations']})

    def test_http_sequence(self):
        original=sys.modules.pop('app',None)
        try:
            with patch('core.runtime_config.configure_runtime',return_value='dummy'),patch('core.chatbot_engine.CMCChatbotEngine',return_value=self.e):
                module=importlib.import_module('app')
            with TestClient(module.app) as client:
                login=client.post('/api/login',json={'username':self.p.user_id,'password':self.p.user_id})
                self.assertEqual(login.status_code,200)
                h={'Authorization':'Bearer '+login.json()['access_token']}
                for q,intent in [('Tôi có thể nghỉ học hay không','clarify'),('một buổi','regulation'),('Tôi đang mắc lỗi gì','personal'),('xin chào ?','greeting'),('tôi là ai','personal')]:
                    response=client.post('/api/chat',headers=h,json={'message':q})
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.json()['intent'],intent,response.text)
                self.assertIn(self.p.user_id,response.json()['reply'])
        finally:
            sys.modules.pop('app',None)
            if original is not None:sys.modules['app']=original
