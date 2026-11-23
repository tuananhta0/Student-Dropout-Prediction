"""Kiểm thử giá trị biên, tính toàn vẹn feature và kịch bản ML."""
import unittest
import numpy as np
import pandas as pd

from core.academic_rules import retake_projection, exam_barred, course_grade, withdrawal_reasons
from core.prediction import model_inputs, payment_scenario, apply_prediction_scope, risk_level
from scripts.train_model import evaluate, choose_threshold, split_at_freeze


class PredictionTests(unittest.TestCase):
    def test_missing_categories_not_defaults(self):
        original = pd.DataFrame({'major_code':['IT',None],'gpa':[0.,np.nan]})
        meta = {'feature_cols':['major_code','gpa'],'cat_cols':['major_code'],'num_cols':['gpa'],'categorical_missing_token':'__MISSING__'}
        before = original.copy(deep=True)
        actual = model_inputs(original,meta)
        self.assertEqual(actual.major_code.tolist(),['IT','__MISSING__'])
        self.assertTrue(pd.isna(actual.gpa.iloc[1]))
        self.assertEqual(actual.gpa.iloc[0],0.)
        self.assertTrue(pd.isna(original.major_code.iloc[1]))
        pd.testing.assert_frame_equal(original, before)
        with self.assertRaises(ValueError):model_inputs(original.drop(columns='gpa'),meta)
        with self.assertRaises(ValueError):model_inputs(original.assign(gpa=np.inf),meta)

    def test_payment_consistency(self):
        initial={'tuition_debt_end_vnd':2e6,'tuition_debt_to_due_ratio':.2,'tuition_overdue_days_end':20,'tuition_overdue_terms_to_date':3,'gpa4_term':2.5}
        after=payment_scenario(initial,2e6)
        self.assertEqual(after['tuition_debt_end_vnd'],0)
        self.assertEqual(after['tuition_debt_to_due_ratio'],0)
        self.assertEqual(after['tuition_overdue_days_end'],0)
        self.assertEqual(after['tuition_overdue_terms_to_date'],2)
        self.assertEqual(initial['tuition_debt_end_vnd'],2e6)
        self.assertEqual(after['gpa4_term'],2.5)
        self.assertAlmostEqual(payment_scenario(initial,1e6)['tuition_debt_to_due_ratio'],.1)
        for x in [0,-1,2e6+1,True,float('nan')]:
            with self.subTest(x=x),self.assertRaises(ValueError):payment_scenario(initial,x)

    def test_prediction_scope(self):
        frame=pd.DataFrame({'prediction_status':['VALID','VALID','VALID','STALE'],'prediction_term':[3,7,3,3],'prediction_cutoff':['2027-02-03','2027-02-03','2026-07-15','2027-02-03']})
        meta={'model_available_as_of':'2027-02-03','supported_terms':[1,2,3]}
        out=apply_prediction_scope(frame,meta)
        self.assertEqual(out.prediction_status.tolist(),['VALID','OUT_OF_SCOPE','RETROSPECTIVE','STALE'])
        self.assertEqual(frame.prediction_status.iloc[1],'VALID')

    def test_threshold_boundaries(self):
        self.assertEqual(risk_level(.21,.21),'Cao')
        self.assertEqual(risk_level(.12,.21),'Trung bình')
        self.assertEqual(risk_level(.45,.21),'Rất cao')
        self.assertNotEqual(risk_level(.45,.6),'Rất cao')
        self.assertEqual(risk_level(.07,.07),'Cao')

    def test_metrics(self):
        m=evaluate([0,0,1,1],[.1,.8,.9,.2],.5)
        self.assertEqual(m['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(m['baseline'],.5)
        self.assertEqual(m['recall_top10'],.5)
        self.assertEqual(m['precision'],.5)
        t=choose_threshold([0]*8+[1]*2,[.01]*8+[.8,.9])
        self.assertLessEqual(np.mean(np.array([.01]*8+[.8,.9])>=t),.3)

    def test_temporal_split(self):
        rows=[]
        for cohort,cutoff,end in [(2022,'2024-02-03','2025-02-03'),(2024,'2025-02-03','2026-02-03'),(2025,'2027-02-03','2028-02-03')]:
            for y in (0,1):rows.append(dict(record_uid=f'{cohort}-{y}',cohort_year=cohort,snapshot_cutoff=cutoff,label_observation_end=end,label_observed=True,dropout_next_2_terms=y))
        rows.append(dict(record_uid='censored',cohort_year=2022,snapshot_cutoff='2026-07-15',label_observation_end='2027-07-15',label_observed=False,dropout_next_2_terms=np.nan))
        rows.append(dict(record_uid='immature-positive',cohort_year=2023,snapshot_cutoff='2026-07-15',label_observation_end='2027-07-15',label_observed=True,dropout_next_2_terms=1))
        parts=split_at_freeze(pd.DataFrame(rows))
        self.assertEqual([len(x) for x in parts],[2,2,2])

    def test_multi_course_projection(self):
        attempts=[dict(course_id='A',registered_credits=3,is_gpa_eligible=True,grade_letter='F'),dict(course_id='B',registered_credits=3,is_gpa_eligible=True,grade_letter='C')]
        result=retake_projection(attempts,{'A':'B','B':'A'},3,3)
        self.assertEqual(result['gpa_credits'],9)
        self.assertEqual(result['quality_points'],30)
        self.assertEqual(result['gpa_after'],3.33)
        self.assertEqual(retake_projection(attempts,{'B':'F'})['gpa_after'],1.)

    def test_policy_boundaries(self):
        for value,expected in [(3.94,'F'),(3.95,'D'),(4.,'D'),(7.95,'B+'),(8.95,'A+')]:
            self.assertEqual(course_grade(value,5),expected)
        self.assertFalse(exam_barred(45,9,4.5))
        self.assertTrue(exam_barred(45,9.0001,0))
        self.assertTrue(exam_barred(45,9,4.5001))
        self.assertFalse(withdrawal_reasons(2,3,18,'Bachelor')['duration'])
        self.assertTrue(withdrawal_reasons(2,3,19,'Bachelor')['duration'])

if __name__=='__main__':unittest.main()
