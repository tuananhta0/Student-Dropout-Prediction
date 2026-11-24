"""Integrated V2.5 developer checks; not an independent semantic review."""
import unittest
from unittest.mock import patch
from fastapi import HTTPException
import pandas as pd
from core.intent_router import route
from core.security import Principal
from core.policy_focus_v3 import FACETS, normalize_chat
from core.retrieval import public_focus
import tests.test_chat_query_v2 as fixture

class ChatV25(unittest.TestCase):
    def setUp(self):
        self.f=fixture.QueryV2Tests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.e=self.f.e;self.r=self.f.r
    def ask(self,q,role='student',ctx=None):
        p=Principal({'student':'BIT250093','advisor':'GV001','admin':'ADMIN'}[role],role)
        return route(self.e,self.r,p,q,ctx if ctx is not None else {})
    def test_facet_catalog(self):
        self.assertTrue({k for k,*_ in FACETS}<={c['id'] for c in self.r.chunks})
        for q,key in [('Nghỉ vì ốm hoặc thai sản cần giấy tờ gì?','D21.K1'),('Điều kiện để chuyển sang ngành khác là gì?','D22.K1–3'),('Mang điện thoại vào phòng thi không tắt nguồn bị xử lý ra sao?','D25.K4–6'),('bảo vệ online được ko ạ','D13.K2')]:
            self.assertIn(key,[c['id'] for c in self.r.retrieve(q)],q)
    def test_public_projection_private(self):
        q='Mang điện thoại vào phòng thi không tắt nguồn bị xử lý ra sao? BIT250093 secret-zxq'
        self.assertNotIn('secret-zxq',public_focus(q));self.assertNotIn('BIT250093',public_focus(q))
        self.assertIn('BIT250093',normalize_chat('SV BIT250093'))
        self.assertFalse(self.r.retrieve('Thư viện mở cửa bao lâu?'))
    def test_numeric(self):
        out=self.ask('Điểm 6,95 quy đổi về thang 4 thế nào?');self.assertIn('7.0/10',out['reply']);self.assertIn('3/4',out['reply']);self.assertEqual({c['article'] for c in out['regulations']},{13,17})
        out=self.ask('GPA 3,3 xếp loại gì?');self.assertIn('Giỏi',out['reply']);self.assertIn('hạng tốt nghiệp',out['reply'])
        out=self.ask('Đăng ký 12 tín trượt 6 tín thì có vượt quá 50% khối lượng học tập không?');self.assertIn('không vượt',out['reply']);self.assertEqual({c['article'] for c in out['regulations']},{18})
    def test_personal_followup(self):
        c={};o=self.ask('cho mình xem điểm tích lũy',ctx=c);self.assertIn('2.70',o['reply']);self.assertEqual(o['intent'],'personal')
        # No cross-user or cross-session lookup.
        o=self.ask('GPA tích lũy của em',ctx=c);self.assertEqual(o['intent'],'personal');o=self.ask('còn HK2?',ctx=c);self.assertIn('Chưa có',o['reply'])
        self.assertNotEqual(self.ask('còn HK2?')['intent'],'personal')
    def test_public_followup(self):
        c={};self.ask('Điều 16',ctx=c);o=self.ask('Còn thời hạn?',ctx=c);self.assertEqual({x['article'] for x in o['regulations']},{16});self.assertIn('một tuần',o['reply'])
        self.assertFalse(self.ask('Còn thời hạn?')['regulations'])
    def test_multi_public(self):
        o=self.ask('Mang điện thoại vào phòng thi không tắt nguồn bị xử lý ra sao? Điều kiện để chuyển sang ngành khác là gì?');self.assertEqual({c['article'] for c in o['regulations']},{22,25});self.assertIn('Ý 2',o['reply'])
        with self.assertRaises(HTTPException):self.ask('Mang điện thoại vào phòng thi xử lý sao? Xem hồ sơ BBA250001')
    def test_semester_warning(self):
        out=self.ask('Số lượng cảnh báo học tập mỗi học kỳ có xu hướng ra sao?',role='admin');self.assertEqual(out['intent'],'analytics');self.assertIn('1/2',out['reply']);self.assertIn('không phải kỳ lịch',out['reply'])
    def test_courses_scope_and_asof(self):
        self.e.df_attempts=pd.DataFrame([dict(record_uid='A',course_id='C1',term_index=3,course_final_score10=8,grade_letter='B+',is_passed=True,available_at='2027-01-01',score_published_at='2027-01-02'),dict(record_uid='B',course_id='C1',term_index=3,course_final_score10=2,grade_letter='F',is_passed=False,available_at='2027-01-01',score_published_at='2027-01-02'),dict(record_uid='A',course_id='C1',term_index=3,course_final_score10=1,grade_letter='F',is_passed=False,available_at='2027-02-01',score_published_at='2027-02-02')]);self.e.course_names={'C1':'Toán'}
        self.e.df=self.e.df.assign(snapshot_cutoff='2027-01-15');self.e.latest_df=self.e.df.copy()
        q='Tỷ lệ trượt từng môn học kỳ 3 của ngành Công nghệ thông tin?'
        o=self.ask(q,'advisor');self.assertEqual(o['intent'],'analytics');self.assertIn('0/1',o['reply']);self.assertNotIn('1/2',o['reply'])
        with self.assertRaises(HTTPException) as e:self.ask(q.replace('Công nghệ thông tin','Quản trị kinh doanh'),'advisor')
        self.assertEqual(e.exception.status_code,403)
        o=self.ask('Phổ điểm từng môn học kỳ 3 toàn trường?', 'admin');self.assertIn('B+: 1',o['reply']);self.assertIn('F: 1',o['reply'])
    def test_missing_course_data(self):
        o=self.ask('Phổ điểm từng môn toàn trường?', 'admin');self.assertIn('Chưa có dữ liệu',o['reply'])
    def test_explicit_all_school_forbidden(self):
        with self.assertRaises(HTTPException) as e:self.ask('GPA trung bình theo từng học kỳ toàn trường?', 'advisor')
        self.assertEqual(e.exception.status_code,403)
    def test_comparison_scope(self):
        o=self.ask('Ngành nào có GPA trung bình cao nhất?', 'admin');self.assertEqual(o['intent'],'analytics');self.assertIn('BA:',o['reply']);self.assertIn('IT:',o['reply']);self.assertLess(o['reply'].index('BA:'),o['reply'].index('IT:'))

    def test_model_explanation(self):
        self.e.meta['test_recall']=.5774
        o=self.ask('Dự báo này chính xác đến mức nào?');self.assertEqual(o['intent'],'model');self.assertIn('57.7%',o['reply']);self.assertIn('mô phỏng',o['reply'])
    def test_causal_unavailable_and_scope(self):
        q='Học bổng ảnh hưởng thế nào đến việc giữ chân sinh viên ngành Công nghệ thông tin?'
        o=self.ask(q,'advisor');self.assertIn('nhân quả',o['reply'])
        with self.assertRaises(HTTPException):self.ask(q.replace('Công nghệ thông tin','Quản trị kinh doanh'),'advisor')
    def test_course_title_is_not_population_statistics(self):
        q='Em vừa xem điểm thi kết thúc học phần môn Xác suất thống kê được có 4.0 điểm, thấp hơn kỳ vọng nhiều thì thủ tục nộp đơn phúc tra ở Phòng Đảm bảo chất lượng thế nào ạ?'
        out=self.ask(q)
        self.assertEqual(out['intent'],'regulation')
        self.assertIn(16,{c['article'] for c in out['regulations']})
        # Independent population-statistic cue must still enforce authorization.
        with self.assertRaises(HTTPException) as err:
            self.ask('Thống kê tỷ lệ trượt môn Xác suất thống kê toàn trường?')
        self.assertEqual(err.exception.status_code,403)

    def test_review_correction_and_sampling(self):
        from scripts.prepare_v25_review import review,choose,CORRECTIONS
        k='SV-0092';q=CORRECTIONS[k][0];b={'id':k,'question':q,'role':'Sinh viên','type':'Quy chế','source':'Điều 12; Điều 13'}
        r={'id':k,'question':q,'role':'Sinh viên','type':'Quy chế','source_reference':b['source'],'displayed_articles':'[26]','article_overlap':'False','expected_intent':'khoa_luan'}
        rows=review([b],[r]);self.assertEqual(rows[0]['source_reference'],b['source']);self.assertEqual(rows[0]['expected_articles_reviewed'],[26]);self.assertTrue(rows[0]['article_overlap_reviewed']);self.assertEqual(rows[0]['semantic_review_status'],'NOT_RUN');self.assertEqual(choose(rows,1),choose(rows,1))
        with self.assertRaises(ValueError):review([b],[dict(r,question='wrong')])
