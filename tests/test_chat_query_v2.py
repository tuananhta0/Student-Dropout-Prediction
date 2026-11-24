"""V2.2 developer regression, synthetic fixture only; no cloud calls."""
import os
import unittest
from unittest.mock import patch
import pandas as pd
from fastapi import HTTPException
from core.intent_router import route
from core.retrieval import RegulationSearch
from core.security import Principal
from tests.test_security import FakeEngine


class QueryV2Tests(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ,{'CMC_RAG_CLOUD':'0'}); env.start(); self.addCleanup(env.stop)
        self.e=FakeEngine()
        self.e.df=self.e.df.assign(cohort_year=[2025,2024],gpa4_term=[2.8,3.1],gpa4_cumulative=[2.7,3.0],warning_current=[False,True],warnings_total_to_date=[0,1],courses_failed_cumulative=[2,4],assignment_on_time_rate_term=[.75,.80])
        self.e.latest_df=self.e.df.copy()
        self.e.active_latest_df=self.e.latest_df.copy()
        self.r=RegulationSearch()

    def ask(self,q,role='student',uid=None):
        return route(self.e,self.r,Principal(uid or {'student':'BIT250093','advisor':'GV001','admin':'ADMIN'}[role],role),q,{})

    def test_credit_warning(self):
        for fail,verdict in [(10,'Bị'),(9,'Không bị')]:
            out=self.ask(f'Em đăng ký 18 tín chỉ và trượt {fail} tín chỉ thì có bị cảnh báo không?')
            self.assertTrue(out['reply'].startswith('<p>'+verdict))
            self.assertEqual([c['id'] for c in out['regulations']],['D18.K1'])
            self.assertIn('<details>',out['reply'])

    def test_hypothetical_gpa(self):
        for q,verdict in [('GPA học kỳ đầu của em là 0,9 thì có bị cảnh báo không?','Không bị'),('GPA học kỳ 3 của em là 0,95 thì có bị cảnh báo không?','Bị')]:
            self.assertTrue(self.ask(q)['reply'].startswith('<p>'+verdict))

    def test_policy_lookups(self):
        for q,article,fragment in [('GPA học kỳ bao nhiêu thì bị cảnh báo học tập?',18,'0,8'),('Quay lại học sau khi nghỉ có giữ nguyên mã sinh viên không?',9,'giữ mã'),('Bao lâu thì có kết quả phúc tra?',16,'một tuần'),('Thời gian hoàn thành khóa luận là bao lâu?',12,'3 đến 4 tháng'),('Điểm B+ tương ứng bao nhiêu trên thang 4?',17,'3.5/4'),('Sinh viên có 85 tín chỉ tích lũy thuộc trình độ năm mấy?',17,'năm thứ ba')]:
            with self.subTest(q=q):
                out=self.ask(q);self.assertEqual({c['article'] for c in out['regulations']},{article});self.assertIn(fragment,out['reply'].split('<details>')[0])

    def test_personal_metrics(self):
        for q,fragment in [('Số môn đã trượt tích lũy của em hiện tại là bao nhiêu?','2 môn'),('Tỷ lệ nộp bài đúng hạn của em hiện tại là bao nhiêu?','75.0%')]:
            out=self.ask(q);self.assertEqual(out['intent'],'personal');self.assertIn(fragment,out['reply'])

    def test_risk_scope_and_limits(self):
        out=self.ask('Em có nguy cơ bị thôi học không?');self.assertEqual(out['intent'],'risk');self.assertIn('4.0%',out['reply'])
        out=self.ask('Nếu em tăng chuyên cần lên 90% thì nguy cơ giảm bao nhiêu?');self.assertIn('Không thể suy ra',out['reply']);self.assertNotIn('4.0%',out['reply'])
        self.e.df.loc[0,'prediction_status']='RETROSPECTIVE'
        out=self.ask('Em có nguy cơ bị thôi học không?');self.assertIn('hồi cứu',out['reply']);self.assertNotIn('4.0%',out['reply'])
        self.assertIn('không dự báo',self.ask('Mô hình AI dự báo nguy cơ rớt môn của em?')['reply'])

    def test_analytics_permission_and_denominator(self):
        q='Ngành Công nghệ thông tin có bao nhiêu sinh viên thuộc nhóm nguy cơ thôi học cao ở học kỳ gần nhất?'
        self.assertIn('0/1 SV có nguy cơ',self.ask(q,'advisor')['reply'])
        for role,query in [('student',q),('advisor',q.replace('Công nghệ thông tin','Quản trị kinh doanh'))]:
            with self.assertRaises(HTTPException) as err:self.ask(query,role)
            self.assertEqual(err.exception.status_code,403)
        self.e.latest_df.loc[0,'prediction_status']='OUT_OF_SCOPE'
        self.assertIn('0/1 SV đang học',self.ask(q,'advisor')['reply'])
        out=self.ask('Tỷ lệ sinh viên tốt nghiệp theo khóa là bao nhiêu?','admin');self.assertEqual(out['intent'],'analytics');self.assertIn('2024',out['reply'])

    def test_no_feedback_data(self):
        out=self.ask('Góp ý nào nhiều nhất tháng này?','advisor');self.assertEqual(out['intent'],'none');self.assertIn('chưa có kho dữ liệu',out['reply'])

    def test_advisor_class_scope(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'scope.json';f.write_text(json.dumps({'GV001':['IT25-99']}))
            with patch.dict(os.environ,{'CMC_ADVISOR_SCOPE_FILE':str(f)}):
                out=self.ask('Ngành Công nghệ thông tin có bao nhiêu sinh viên thuộc nhóm nguy cơ thôi học cao?','advisor')
                self.assertIn('Không có hồ sơ',out['reply'])

    def test_probability_course_is_not_dropout(self):
        q='Môn Xác suất thống kê em đi thi cuối kỳ được 1.5 điểm nhưng điểm quá trình của em được 6.5 điểm thì tính ra có bị điểm liệt F không hả bot?'
        out=self.ask(q);self.assertEqual(out['intent'],'regulation');self.assertIn(13,[c['article'] for c in out['regulations']])
        out=self.ask('Chuyên cần của em có nguy cơ vượt ngưỡng cấm thi không?');self.assertNotEqual(out['intent'],'risk')

    def test_no_unrelated_aggregate(self):
        from core.chat_query_v2 import scoped_analytics
        from core.chatbot_engine import strip_accents
        p=Principal('ADMIN','admin')
        for q in ['Bao nhiêu sinh viên nghỉ tạm thời và đã quay lại?', 'Báo cáo tổng hợp số lượng sinh viên học cùng lúc hai chương trình trong toàn trường', 'bao nhiêu sv đang bị cảnh báo 2 lần liên tiếp']:
            self.assertIsNone(scoped_analytics(self.e,p,strip_accents(q),self.e.parse_flexible_query(q),set()))
