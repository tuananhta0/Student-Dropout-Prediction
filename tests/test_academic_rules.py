import unittest
from core.academic_rules import (registration_bounds, warning_reasons, withdrawal_reasons,
                            exam_barred, course_grade, official_gpa, projected_gpa,
                            projection, graduation_conditions)
from core.chatbot_engine import CMCChatbotEngine


class AcademicRulesTests(unittest.TestCase):
    def test_registration(self):
        self.assertEqual(registration_bounds(15),(10,22))
        self.assertEqual(registration_bounds(18),(12,27))
        with self.assertRaises(ValueError):
            registration_bounds(0)

    def test_warning_boundaries(self):
        self.assertEqual(warning_reasons(1,.8,12,6,24),dict(gpa=False,failed=False,owed=False))
        self.assertTrue(all(warning_reasons(2,.99,12,7,25).values()))
        self.assertTrue(warning_reasons(1,0,12,0,0)['gpa'])
        self.assertIsNone(warning_reasons(1,None,12,None,0)['gpa'])
        with self.assertRaises(ValueError):
            warning_reasons(1,2,12,13,0)

    def test_withdrawal(self):
        self.assertFalse(any(withdrawal_reasons(2,3,18,'Bachelor').values()))
        self.assertTrue(all(withdrawal_reasons(3,4,19,'Bachelor').values()))
        self.assertFalse(withdrawal_reasons(0,0,22,'Engineer')['duration'])
        self.assertTrue(withdrawal_reasons(0,0,23,'Engineer')['duration'])
        self.assertIsNone(withdrawal_reasons(None,None,None,'Unknown')['duration'])

    def test_exam(self):
        self.assertFalse(exam_barred(100,20,10))
        self.assertTrue(exam_barred(100,20.01,0))
        self.assertTrue(exam_barred(100,15,15.01))
        self.assertFalse(exam_barred(100,0,25))
        self.assertIsNone(exam_barred(None,0,0))
        with self.assertRaises(ValueError):
            exam_barred(100,90,20)

    def test_grade(self):
        self.assertEqual(course_grade(9,2),'F')
        self.assertEqual(course_grade(9,2.01),'A+')
        self.assertEqual(course_grade(3.95,3),'D')
        self.assertEqual(course_grade(3.94,3),'F')
        self.assertEqual(course_grade(7.95,7),'B+')
        self.assertEqual(course_grade(10,10,retest=True),'C')
        self.assertEqual(course_grade(10,10,barred=True),'F')
        self.assertIsNone(course_grade(None,8))
        self.assertEqual(course_grade(5,5,gpa_eligible=False),'P')

    def test_best_grade_ledger(self):
        def attempt(cid, grade, eligible=True):
            return dict(course_id=cid,registered_credits=3,grade_letter=grade,is_gpa_eligible=eligible)
        rows = [attempt('A','F'),attempt('A','B'),attempt('A','D'),attempt('B','F'),
                attempt('PE','P',False),attempt('X','X'),attempt('R','R')]
        self.assertEqual(official_gpa(rows),dict(credits=6,quality_points=9,gpa=1.5))
        self.assertEqual(official_gpa(rows+[attempt('B','A')])['gpa'],3.5)

    def test_projection(self):
        self.assertEqual(projected_gpa(28.5,15,3,15),2.45)
        self.assertEqual(projected_gpa(0,0,0,15),0)
        self.assertIsNone(projected_gpa(None,0,3,15))
        p=projection(dict(gpa_quality_points=28.5,gpa_credit_count=15,gpa4_cumulative=1.9,
                          snapshot_term_index=1,credits_required=120,credits_earned_cumulative=12,credits_owed_cumulative=3,new_course_credits_available=105,registration_max_credits=22))
        self.assertEqual(p['next_tc'],15)
        self.assertEqual(p['pred_cum_imp'],2.15)
        self.assertEqual(projection({}),None)

    def test_graduation_unknown(self):
        c=graduation_conditions(dict(credits_earned_cumulative=120,credits_required=120,gpa4_cumulative=2,
                                     tuition_debt_end_vnd=0,verified_certificates='ENGLISH, PE, DEFENSE'))
        self.assertTrue(c['credits'] and c['gpa'])
        self.assertIsNone(c['discipline_and_criminal_clearance'])
        self.assertIsNone(c['financial'])
        self.assertFalse(graduation_conditions(dict(tuition_debt_end_vnd=1))['financial'])

    def test_trap(self):
        engine=CMCChatbotEngine.__new__(CMCChatbotEngine)
        self.assertIn('Cấm thi',engine.evaluate_regulation_trap('Nghỉ 25% không có phép'))
        self.assertIn('Không vượt',engine.evaluate_regulation_trap('Nghỉ 25% có phép'))
        self.assertIn('Không bị',engine.evaluate_regulation_trap('GPA 4.0 HK1 cảnh báo'))
        self.assertIn('GPA phải',engine.evaluate_regulation_trap('GPA 5 HK1 cảnh báo'))

    def test_expired_prediction(self):
        engine=CMCChatbotEngine.__new__(CMCChatbotEngine)
        engine.meta={'threshold':.21}
        warnings=engine.evaluate_student_warnings(dict(state_as_of='ENROLLED',snapshot_term_index=1,gpa4_term=3,
            credits_registered_term=15,credits_failed_term=0,credits_owed_cumulative=0,warning_streak=0,
            warnings_total_to_date=0,counted_duration_semesters=1,degree_type='Bachelor',risk_probability=.8,prediction_status='STALE'))
        self.assertTrue(any('hết hạn' in w['title'] for w in warnings))
        self.assertFalse(any('Nguy cơ thôi học dự báo' in w['title'] for w in warnings))

    def test_strict_bool_and_invalid_counts(self):
        from core.academic_rules import strict_bool, number
        import numpy as np
        self.assertIsNone(strict_bool(np.nan))
        self.assertIsNone(number(True))
        with self.assertRaises(ValueError):
            strict_bool('False')
        with self.assertRaises(ValueError):
            withdrawal_reasons(3,1,3,'Bachelor')
        self.assertIsNone(warning_reasons(1.5,0,12,0,0)['gpa'])
        with self.assertRaises(ValueError):
            official_gpa([dict(course_id='X',registered_credits=3,grade_letter='A',is_gpa_eligible=np.nan)])

    def test_retake_projection(self):
        from core.academic_rules import retake_projection
        rows=[dict(course_id='X',registered_credits=3,grade_letter='F',is_gpa_eligible=True),dict(course_id='Y',registered_credits=3,grade_letter='B',is_gpa_eligible=True)]
        self.assertEqual(retake_projection(rows,{'X':'B'})['gpa_after'],3)
        self.assertEqual(retake_projection(rows,{'X':'B'})['gpa_credits'],6)
        self.assertEqual(retake_projection(rows,{'Y':'D'})['gpa_after'],1.5)
        self.assertEqual(retake_projection(rows,{'X':'B'},4,3)['gpa_after'],3.33)
        with self.assertRaises(ValueError):
            retake_projection(rows,{'Z':'B'})

    def test_clearance_flags(self):
        c=graduation_conditions(dict(clear_finance=True,clear_discipline=False,clear_english=False,verified_certificates='ENGLISH',clear_no_outstanding_academic_debt=True,credits_owed_cumulative=3))
        self.assertTrue(c['financial']); self.assertFalse(c['english']); self.assertFalse(c['academic_debt']); self.assertFalse(c['discipline_and_criminal_clearance'])
