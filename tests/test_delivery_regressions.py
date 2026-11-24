"""Hồi quy các lỗi phát hiện sau nghiệm thu 60524ac; không dùng mạng."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.academic_rules import projection
from core.intent_router import route
from core.retrieval import RegulationSearch, numbers, public_focus
from core.security import Principal
from scripts.benchmark_rag import endpoint, quality_pass
from scripts.report_provenance import provenance, sidecar, read_report_origin
from tests.test_security import FakeEngine


class DeliveryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.search=RegulationSearch();self.engine=FakeEngine();self.ctx={}
        self.engine.df=self.engine.df.assign(gpa4_cumulative=2.75,gpa_quality_points=115.5,gpa_credit_count=42,credits_required=128,credits_earned_cumulative=42,credits_owed_cumulative=0,new_course_credits_available=86,registration_min_credits=10,registration_max_credits=22)
        env=patch.dict('os.environ',{'CMC_RAG_CLOUD':'0'});env.start();self.addCleanup(env.stop)

    def ask(self,q):
        return route(self.engine,self.search,Principal('BIT250093','student'),q,self.ctx)

    def test_explicit_gpa_credits(self):
        out=self.ask('Em muốn đạt GPA tích lũy 3,1 thì kỳ sau cần bao nhiêu nếu học 22 tín chỉ?')
        self.assertIn('22 tín chỉ mới',out['reply']);self.assertIn('3.77/4',out['reply'])
        for n in ('0','-1','2.5','23','100'):
            with self.subTest(n=n):
                result=self.ask(f'Em muốn đạt GPA tích lũy 3.1 nếu học {n} tín chỉ?')
                self.assertNotIn('GPA kỳ tới tối thiểu',result['reply']);self.assertEqual(result['regulations'][0]['article'],17)
        row=self.engine.df.iloc[0].to_dict();self.assertIsNone(projection(row,3.1,new_credits=23))
        self.assertIsNone(projection(dict(row,new_course_credits_available=10),3.1,new_credits=22))
        self.assertIsNone(projection(dict(row,credits_required=50),3.1,new_credits=22))
        self.assertIn('một số tín chỉ',self.ask('Em muốn đạt GPA 3.1 nếu học 12 tín chỉ hoặc 22 tín chỉ?')['reply'])

    def test_registration_and_term(self):
        for q in ('Em đăng ký 24 tín chỉ có được không?','Em đăng ký 24 TC?','Đăng ký 24 tín chỉ'):
            with self.subTest(q=q):
                self.assertIsNone(self.engine.parse_flexible_query(q)['term_filter'])
                out=self.ask(q);self.assertEqual(out['regulations'][0]['id'],'D12.K2');self.assertIn('vượt giới hạn tối đa 22',out['reply'])
        self.assertEqual(self.engine.parse_flexible_query('GPA học kỳ 3')['term_filter'],3)
        self.assertEqual(self.engine.parse_flexible_query('GPA kỳ 3')['term_filter'],3)
        self.assertEqual(self.engine.parse_flexible_query('HK3')['term_filter'],3)
        self.assertIn('nằm trong khoảng',self.ask('Em đăng ký 15 tín chỉ được không?')['reply'])
        self.assertIn('ngoại lệ',self.ask('Em đăng ký 5 tín chỉ được không?')['reply'])
        self.engine.df=self.engine.df.drop(columns='registration_max_credits')
        self.assertIn('Chưa đủ dữ liệu',self.ask('Em đăng ký 24 tín chỉ được không?')['reply'])

    def test_withdrawal_boundaries(self):
        for q,label in [('Em có 3 lần cảnh báo liên tiếp có bị thôi học không?','Đủ tiêu chí'),('Em có 2 lần cảnh báo liên tiếp có bị thôi học không?','Chưa đủ tiêu chí'),('Em bị cảnh báo 4 lần không liên tiếp','Đủ tiêu chí'),('Em bị cảnh báo 3 lần không liên tiếp','Chưa đủ tiêu chí')]:
            with self.subTest(q=q):
                out=self.ask(q);self.assertIn(label,out['reply']);self.assertEqual([c['id'] for c in out['regulations']],['D18.K2'])

        for n in ('-3','2.5'):
            out=self.ask(f'Em có {n} lần cảnh báo liên tiếp thì sao?');self.assertIn('không nhất quán',out['reply']);self.assertNotIn('Đủ tiêu chí',out['reply'])

    def test_certificate_debt(self):
        for q in ('Em nợ 25 tín chỉ chứng chỉ thì sao?','Em nợ 25 tín chỉ gồm cả môn chứng chỉ'):
            out=self.ask(q);self.assertNotIn('Bị cảnh báo do nợ',out['reply']);self.assertEqual(out['regulations'][0]['id'],'D18.K1')
        self.assertIn('Nợ tính cảnh báo 22 tín chỉ',self.ask('Em nợ 25 tín chỉ trong đó 3 tín chỉ chứng chỉ thì sao?')['reply'])
        self.assertIn('Không bị cảnh báo',self.ask('Em nợ 24 tín chỉ không tính chứng chỉ')['reply'])
        self.assertIn('Bị cảnh báo',self.ask('Em nợ 25 tín chỉ không tính chứng chỉ')['reply'])
        self.assertIn('không hợp lệ',self.ask('Em nợ 2 tín chỉ trong đó 3 tín chỉ chứng chỉ')['reply'])

    def test_missing_grade_citation(self):
        out=self.ask('Em thi cuối kỳ 4 điểm có qua môn không?');self.assertIn('còn phụ thuộc tổng điểm',out['reply']);self.assertIn(13,{c['article'] for c in out['regulations']})
        self.assertIn(13,{c['article'] for c in self.ask('Em thi cuối kỳ 11 điểm')['regulations']})

    def test_numbers_and_context(self):
        self.assertEqual(numbers('Điều 18 và 21, 25'),([18,21,25],[]))
        self.assertEqual(numbers('Điều 18 Khoản 1 và 99'),([18],[1,99]))
        self.assertEqual(numbers('Điều 18 và 25 tín chỉ'),([18],[]))
        out=self.ask('Điều 18 và 21');self.assertEqual({c['article'] for c in out['regulations']},{18,21})
        out=self.ask('Điều 18 Khoản 1 và Khoản 99');self.assertEqual([c['id'] for c in out['regulations']],['D18.K1']);self.assertIn('Khoản 99',out['reply'])
        self.assertFalse(self.ask('Khoản 2')['regulations'])
        self.assertIn('Khoản 99',self.ask('Điều 18 Khoản 99')['reply'])

    def test_multiple_topics_and_refusal(self):
        for q in ('Lịch nghỉ lễ của sinh viên?', 'Thời hạn nộp học phí kỳ tới?', 'Thư viện trường mở cửa mấy giờ?', 'Món ăn sinh viên thích nhất?'):
            with self.subTest(q=q): self.assertFalse(self.ask(q)['regulations'])
        out=self.ask('Bảo lưu và phúc tra');self.assertEqual({c['article'] for c in out['regulations']},{16,21})
        out=self.ask('Bảo lưu và lịch nghỉ lễ của sinh viên?');self.assertEqual({c['article'] for c in out['regulations']},{21});self.assertIn('Một phần câu hỏi',out['reply'])
        focus=public_focus('Bảo lưu và phúc tra BIT250093 bí mật-123')
        self.assertIn('bao luu',focus);self.assertIn('phuc tra',focus);self.assertNotIn('250093',focus)

    def test_http_new_cases(self):
        with endpoint() as ask:
            for q,ids in [('Điều 18 và 21',{18,21}),('Em có 3 lần cảnh báo liên tiếp có bị thôi học không?',{18}),('Em thi cuối kỳ 4 điểm có qua môn không?',{13}),('Lịch nghỉ lễ của sinh viên?',set())]:
                out=ask(q);self.assertEqual({c['article'] for c in out['regulations']},ids)
            self.assertIn('Chưa đủ dữ liệu',ask('Em đăng ký 24 tín chỉ được không?')['reply'])
        with endpoint(self.engine) as ask:
            self.assertIn('3.77/4',ask('Em muốn đạt GPA tích lũy 3.1 nếu học 22 tín chỉ?')['reply'])
            self.assertIn('vượt giới hạn tối đa 22',ask('Em đăng ký 24 tín chỉ được không?')['reply'])

    def test_displayed_source_gate(self):
        r=dict(so_cau=1,so_cau_co_can_cu=1,trich_dan_hop_le=True,hit_rate_at_5=1,mrr_at_5=1,top1=1,tu_choi_ngoai_pham_vi=None,ty_le_dieu_mong_doi_duoc_hien_thi=1)
        self.assertTrue(quality_pass(r))
        for v in (0,.99,None):
            self.assertFalse(quality_pass(dict(r,ty_le_dieu_mong_doi_duoc_hien_thi=v)))
        del r['ty_le_dieu_mong_doi_duoc_hien_thi'];self.assertFalse(quality_pass(r))

    def test_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.csv';p.write_text('a,b\n1,2\n')
            self.assertEqual(read_report_origin(p)['status'],'unknown')
            info={'source_commit':'abc','source_dirty':False}
            sidecar(p,info);self.assertEqual(read_report_origin(p)['status'],'recorded')
            p.write_text('a,b\n1,3\n');self.assertEqual(read_report_origin(p)['status'],'changed_after_recording')
            old=Path(tmp)/'old.json';old.write_text(json.dumps({'metric':1}));self.assertEqual(read_report_origin(old)['status'],'unknown')
            self.assertIn('created_utc',provenance(root=tmp))
    def test_r01_tuition_exemption_excludes_article_25(self):
        for q in (
            'Sinh viên có được miễn giảm học phí không?',
            'Chính sách miễn, giảm học phí của trường như thế nào?',
            'Điều kiện giảm học phí cho sinh viên khó khăn?',
            'Em có được miễn đóng học phí không?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertNotIn(25, {c['article'] for c in out['regulations']})
                self.assertFalse(out['regulations'])

    def test_r02_tuition_fee_amount_refusal_and_guidance(self):
        for q in (
            'Học phí một tín chỉ bao nhiêu tiền?',
            'Mức học phí ngành IT một năm là bao nhiêu?',
            'Đơn giá tín chỉ hiện tại là bao nhiêu?',
            '1 tín chỉ hết bao nhiêu tiền?',
            'Học phí mỗi kỳ là bao nhiêu?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertFalse(out['regulations'])
                self.assertIn('thông báo học phí', out['reply'].lower())
        self.engine.df = self.engine.df.assign(tuition_debt_end_vnd=0)
        self.assertEqual(self.ask('Học phí còn nợ của em là bao nhiêu?')['intent'], 'personal')
        self.assertIn(12, {c['article'] for c in self.ask('Rút học phần có được hoàn học phí không?')['regulations']})

    def test_q01_target_gpa_phrasings(self):
        for q in (
            'Em muốn kéo GPA lên 3.1 nếu học 22 tín chỉ',
            'Nâng điểm trung bình tích lũy lên 3,1 với 22 tín chỉ thì kỳ sau cần bao nhiêu?',
            'Cải thiện GPA lên 3.1 nếu học 22 TC',
            'GPA mục tiêu của em là 3.1, học 22 tín chỉ',
            'Em muốn đạt 3.1 GPA nếu học 22 tín chỉ',
            'Em muốn nâng GPA lên tối thiểu 3.1 với 22 tín chỉ',
            'Học kỳ 2 em đạt GPA 2.5, kỳ sau muốn lên 3.1 với 22 tín chỉ thì cần bao nhiêu?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'gpa')
                self.assertIn('3.77/4', out['reply'])
        rank_out = self.ask('GPA 3.2 xếp loại gì?')
        self.assertEqual(rank_out['intent'], 'regulation')
        self.assertIn('Giỏi', rank_out['reply'])

    def test_q02_improve_policy_vs_gpa_scenario(self):
        for q in (
            'Quy định học cải thiện như thế nào?',
            'Em có được đăng ký học cải thiện không?',
            'Điểm D học cải thiện thì lấy điểm cao nhất hay điểm lần sau?',
            'Học cải thiện điểm có giới hạn số lần không?',
            'Học cải thiện có tính vào GPA không?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'regulation')
                self.assertIn(15, {c['article'] for c in out['regulations']})
        for q in (
            'Kịch bản cải thiện GPA kỳ sau của em',
            'Cải thiện GPA kỳ tới thế nào?',
            'Em muốn cải thiện GPA lên 3.1 nếu học 22 tín chỉ',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'gpa')

    def test_policy_improvement_class_phrase(self):
        for q in (
            'Điều kiện học lớp cải thiện khi đạt C?',
            'Em có được học lớp cải thiện điểm C không?',
            'Học lớp cải thiện với điểm D thì lấy điểm nào?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'regulation')
                self.assertIn(15, {c['article'] for c in out['regulations']})
                self.assertIn('Điều 15', out['reply'])
        with endpoint() as ask:
            out = ask('Điều kiện học lớp cải thiện khi đạt C?')
            self.assertIn(15, {c['article'] for c in out['regulations']})

    def test_partial_tuition_and_study_leave(self):
        for q in (
            'Học phí cao quá em muốn nghỉ học một năm được không?',
            'Học phí cao quá nên em muốn nghỉ học một năm được không?',
            'Mức học phí cao nên em cần xin nghỉ học tạm thời một năm?',
            'Học phí một tín chỉ bao nhiêu, em có được nghỉ học một năm không?',
            'Học phí một tín chỉ bao nhiêu và bảo lưu thế nào?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'regulation')
                self.assertIn(21, {c['article'] for c in out['regulations']})
                self.assertIn('Điều 21', out['reply'])
                self.assertNotIn('mức thu:', out['reply'].lower())
        # Hỏi đơn giá riêng phải từ chối, không kéo Điều 21 vào sai ngữ cảnh.
        out = self.ask('Học phí mỗi tín chỉ là bao nhiêu?')
        self.assertFalse(out['regulations'])
        with endpoint() as ask:
            out = ask('Học phí cao quá em muốn nghỉ học một năm được không?')
            self.assertIn(21, {c['article'] for c in out['regulations']})

    def test_r03_military_service_leave(self):
        for q in (
            'Nghỉ học để thực hiện nghĩa vụ quân sự có được bảo lưu không?',
            'Em phải đi nghĩa vụ quân sự thì nghỉ học thế nào?',
            'Đi nhập ngũ có được tạm dừng học không?',
            'Đi bộ đội có được bảo lưu kết quả học tập không?',
        ):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'regulation')
                self.assertIn(21, {c['article'] for c in out['regulations']})
        focus = public_focus('Em phải đi nghĩa vụ quân sự')
        self.assertIn('nghia vu quan su', focus)
        self.assertIn('nghi hoc tam thoi', focus)

    def test_study_leave_and_retake_followups(self):
        """Câu hỏi mới ngoài bộ hồi quy 70 câu: giữ đúng căn cứ và ý nghĩa."""
        questions = (
            ('Mức học phí một tín chỉ bao nhiêu? Có được nghỉ học do khó khăn tài chính?', 21),
            ('Em có thể nghỉ 3 kỳ vì lý do cá nhân không?', 21),
            ('Em có thể nghỉ 3 học kỳ vì lý do cá nhân được không?', 21),
            ('Em xin nghỉ 1 học kỳ vì ốm có phải nộp đơn không?', 21),
            ('Em đạt điểm C nhưng muốn thi lại để cải thiện thì sao?', 15),
            ('Em đạt điểm B nhưng muốn thi lại để cải thiện thì sao?', 15),
            ('Em đạt điểm B+ nhưng muốn thi lại để cải thiện thì sao?', 15),
            ('Em xin nghỉ 4 kỳ vì lý do cá nhân được không?', 21),
        )
        for question, expected in questions:
            with self.subTest(q=question):
                out = self.ask(question)
                self.assertEqual(out['intent'], 'regulation')
                self.assertEqual({c['article'] for c in out['regulations']}, {expected})
                self.assertIn(f'Điều {expected}', out['reply'])
        for q in (
            'Mức học phí một tín chỉ bao nhiêu? Có được nghỉ học do khó khăn tài chính?',
            'Học phí cao quá em muốn nghỉ học một năm được không?',
        ):
            self.assertIn('thông báo', self.ask(q)['reply'].lower())
        allowed = self.ask('Em có thể nghỉ 3 kỳ vì lý do cá nhân không?')['reply']
        self.assertIn('giới hạn tối đa 3 học kỳ', allowed)
        self.assertIn('nhà trường quyết định', allowed)
        exceeded = self.ask('Em xin nghỉ 4 kỳ vì lý do cá nhân được không?')['reply']
        self.assertIn('vượt giới hạn', exceeded)
        self.assertNotIn('Điều 25', exceeded)
        sick = self.ask('Em xin nghỉ 1 học kỳ vì ốm có phải nộp đơn không?')['reply']
        self.assertIn('chứng nhận', sick)

    def test_hypothetical_term_gpa_warning(self):
        """GPA giả định dùng số nêu trong câu, không đọc GPA hồ sơ cá nhân."""
        checks = (
            ('GPA học kỳ thứ hai 0.99 có bị cảnh báo?', 'Bị cảnh báo'),
            ('GPA học kỳ 2 là 0.99 có bị cảnh báo không?', 'Bị cảnh báo'),
            ('GPA học kỳ đầu 0.79 có bị cảnh báo không?', 'Bị cảnh báo'),
            ('GPA học kỳ đầu 0.80 có bị cảnh báo không?', 'Không bị cảnh báo'),
            ('GPA học kỳ 3 1.00 có bị cảnh báo?', 'Không bị cảnh báo'),
        )
        for question, phrase in checks:
            with self.subTest(q=question):
                out = self.ask(question)
                self.assertEqual(out['intent'], 'regulation')
                self.assertEqual({c['article'] for c in out['regulations']}, {18})
                self.assertIn(phrase, out['reply'])
                self.assertIn('còn tiêu chí tín chỉ', out['reply'])
        self.assertNotIn('Bị cảnh báo vì GPA', self.ask('GPA học kỳ 2 là bao nhiêu?')['reply'])

    def test_no_semantic_regressions_from_new_leave_phrases(self):
        """Nghỉ không phép, thi lại F và học phí riêng vẫn giữ ý định cũ."""
        with self.subTest('nghi-khong-phep'):
            out = self.ask('Nghỉ không phép quá 20% số giờ môn thì sao?')
            self.assertIn(25, {c['article'] for c in out['regulations']})
            self.assertNotIn(21, {c['article'] for c in out['regulations']})
        out = self.ask('Thi lại khi bị điểm F được không?')
        self.assertIn(14, {c['article'] for c in out['regulations']})
        self.assertNotIn(15, {c['article'] for c in out['regulations']})
        out = self.ask('Mức học phí một tín chỉ bao nhiêu?')
        self.assertFalse(out['regulations'])
        self.assertIn('thông báo học phí', out['reply'].lower())
        out = self.ask('Em có thể nghỉ 1 kỳ vì ốm không?')
        self.assertIn(21, {c['article'] for c in out['regulations']})
        with endpoint() as invoke:
            for question, expected in (
                ('GPA học kỳ thứ hai 0.99 có bị cảnh báo?', 18),
                ('Mức học phí một tín chỉ bao nhiêu? Có được nghỉ học do khó khăn tài chính?', 21),
                ('Em đạt điểm B nhưng muốn thi lại để cải thiện thì sao?', 15),
            ):
                with self.subTest(q=question):
                    out = invoke(question)
                    self.assertEqual({c['article'] for c in out['regulations']}, {expected})
    def test_short_queries_and_professional_tables(self):
        for q in ('bảng điểm', 'xem bảng điểm chi tiết các học phần', 'môn trượt'):
            with self.subTest(q=q):
                self.assertEqual(self.ask(q)['intent'], 'courses')
        self.engine.calc_next_term_gpa_projection = lambda row: projection(row) if row.get('state_as_of') in (None, 'ENROLLED') else None
        for q in ('dự đoán gpa', 'dự đoán gpa kì sau', 'kịch bản gpa'):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'gpa')
                self.assertIn('<table', out['reply'])
        for q in ('nguy cơ thôi học', 'nguy cơ nghỉ học'):
            with self.subTest(q=q):
                out = self.ask(q)
                self.assertEqual(out['intent'], 'risk')
                self.assertNotIn('Chân trời:', out['reply'])
                self.assertNotIn('snapshot', out['reply'].lower())
        for q in ('thanh toán học phí', 'What-if thanh toán học phí'):
            with self.subTest(q=q):
                self.assertEqual(self.ask(q)['intent'], 'payment')
    def test_partial_student_id_blocked_and_short_credit_unit(self):
        from fastapi import HTTPException
        for q in ('xem hồ sơ 250007', 'xem hồ sơ 999999', 'xem hồ sơ sinh viên khác'):
            with self.subTest(q=q):
                with self.assertRaises(HTTPException) as ctx:
                    self.ask(q)
                self.assertEqual(ctx.exception.status_code, 403)
        own = self.ask('xem hồ sơ 250093')
        self.assertEqual(own['intent'], 'student')
        out_target = self.ask('nâng gpa lên 3.05 nếu học 18 tín')
        self.assertEqual(out_target['intent'], 'gpa')
        self.assertIn('18 tín chỉ mới', out_target['reply'])
        self.assertIn('3.75/4', out_target['reply'])
        out_proj = self.ask('dự đoán gpa kì sau nếu học 20 tín')
        self.assertEqual(out_proj['intent'], 'gpa')
        self.assertIn('Giả định 20 TC mới', out_proj['reply'])
    def test_block_faculty_stats_for_student_and_plain_tuition_payment(self):
        from fastapi import HTTPException
        for q in ('xem thống kê của khoa', 'xem thống kê của khoa it', 'thống kê của lớp', 'xem thống kê của trường'):
            with self.subTest(q=q):
                with self.assertRaises(HTTPException) as ctx:
                    self.ask(q)
                self.assertEqual(ctx.exception.status_code, 403)
        self.engine.df = self.engine.df.assign(tuition_debt_end_vnd=176336, tuition_overdue_days_end=0)
        owed = self.ask('thanh toán học phí')
        self.assertEqual(owed['intent'], 'payment')
        self.assertIn('176,336 VNĐ', owed['reply'])
        self.assertNotIn('độ nhạy', owed['reply'].lower())
        self.assertNotIn('what-if', owed['reply'].lower())
        self.engine.df = self.engine.df.assign(tuition_debt_end_vnd=0, tuition_overdue_days_end=0)
        zero = self.ask('thanh toán học phí')
        self.assertEqual(zero['intent'], 'payment')
        self.assertIn('0 VNĐ', zero['reply'])
        self.assertNotIn('độ nhạy', zero['reply'].lower())
    def test_advisor_and_admin_dropout_rate_explanation(self):
        from fastapi import HTTPException
        self.engine.df = self.engine.df.assign(
            cohort_year=2025,
            state_as_of='ENROLLED',
            prediction_status='VALID',
            risk_probability=0.18,
            attendance_rate_term=88.0,
            warning_bool=False,
        )
        self.engine.latest_df = self.engine.df.copy()
        self.engine.active_latest_df = self.engine.df.copy()
        with self.assertRaises(HTTPException) as stu_err:
            self.ask('tỉ lệ thôi học')
        self.assertEqual(stu_err.exception.status_code, 403)
        gv_out = route(self.engine, self.search, Principal('GV001', 'advisor'), 'dự đoán tỉ lệ thôi học', {})
        self.assertEqual(gv_out['intent'], 'analytics')
        self.assertIn('Random Forest', gv_out['reply'])
        self.assertIn('Dự báo kỳ vọng (AI)', gv_out['reply'])
        with self.assertRaises(HTTPException) as scope_err:
            route(self.engine, self.search, Principal('GV001', 'advisor'), 'tỉ lệ thôi học toàn trường', {})
        self.assertEqual(scope_err.exception.status_code, 403)
