"""Wave 2 RAG regressions: offline HTTP with synthetic fixture, no cloud or PII.

These examples were written by the developer of this patch. They are NOT a
blind/independent human evaluation of semantic quality or Gemini.
"""
import unittest

from core.retrieval import (
    RegulationSearch, is_tuition_amount_query, is_tuition_exemption_query,
    topic_articles,
)
from scripts.benchmark_rag import endpoint
from scripts.smoke_live_http import scenario_valid


class RAGWave2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = RegulationSearch()

    def test_six_confirmed_fixes_and_variants_over_http(self):
        cases = [
            ('Một tín chỉ có bao nhiêu giờ học?', {3}),
            ('1 tín chỉ học bao nhiêu tiếng?', {3}),
            ('Mỗi tín chỉ có bao nhiêu giờ tín chỉ lý thuyết?', {3}),
            ('Học phí có giảm cho sinh viên giỏi không?', set()),
            ('Học phí có được miễn cho sinh viên nghèo không?', set()),
            ('Em muốn chuyển ngành IT sang BA có được không?', {22}),
            ('Em muốn đổi IT sang BA được không?', {22}),
            ('Đổi từ IT sang BA được không?', {22}),
            ('Em có thể tạm dừng việc học vì tai nạn phải điều trị 6 tháng không?', {21}),
            ('Tôi muốn tạm ngưng học vì phải phẫu thuật dài ngày', {21}),
            ('Cách quy đổi điểm 8,5 theo thang 4?', {17}),
            ('Điểm 7,5 quy đổi về thang điểm 4 thế nào?', {17}),
            ('Cho em biết số lần cảnh báo học tập bao nhiêu thì bị đuổi học', {18}),
            ('Cho em hỏi quy định về số lần cảnh báo học tập?', {18}),
            ('Em hỏi số lần bị cảnh báo học tập tối đa theo quy chế?', {18}),
        ]
        with endpoint() as ask:
            for question, expected in cases:
                with self.subTest(question=question):
                    result = ask(question)
                    found = {c['article'] for c in result['regulations']}
                    self.assertEqual(found, expected)
                    self.assertEqual(result['intent'], 'regulation')
                    if expected:
                        self.assertTrue(all('pages' in c for c in result['regulations']))

    def test_multi_topic_retrieval_and_unsupported_separation(self):
        cases = [
            ('Em muốn nghỉ học hai kỳ do tai nạn và chuyển ngành có được không?', {21, 22}),
            ('Em đang học IT muốn chuyển sang BA và nghỉ học do tai nạn?', {21, 22}),
            ('Có được bảo lưu và chuyển sang trường khác không?', {21, 22}),
            ('Học phí có giảm cho sinh viên giỏi và có được bảo lưu không?', {21}),
        ]
        with endpoint() as ask:
            for question, expected in cases:
                with self.subTest(question=question):
                    result = ask(question)
                    self.assertEqual({c['article'] for c in result['regulations']}, expected)

    def test_negative_personal_queries_keep_correct_routing(self):
        with endpoint() as ask:
            result = ask('Em đã bị cảnh báo học tập mấy lần?')
            self.assertIn(result['intent'], ('student','personal'))
            self.assertFalse(result['regulations'])
            fee = ask('Một tín chỉ có giá bao nhiêu tiền?')
            self.assertFalse(fee['regulations'])
            self.assertIn('học phí', fee['reply'].lower())

    def test_http_scenarios_require_business_success(self):
        self.assertFalse(scenario_valid('gpa', 422, {'detail': 'invalid'}))
        self.assertFalse(scenario_valid('gpa', 200, {'gpa_after': 3.0}))
        self.assertTrue(scenario_valid('gpa', 200, {'gpa_before': 2.0, 'gpa_after': 2.4}))
        self.assertFalse(scenario_valid('payment', 422, {'detail': 'invalid'}))
        self.assertFalse(scenario_valid('payment', 200, {'difference_pp': -2.0}))
        self.assertTrue(scenario_valid('payment', 200, {
            'before': {'risk_probability': 0.4},
            'after': {'risk_probability': 0.3}, 'difference_pp': -10.0,
        }))

    def test_refusal_and_preprocessing_boundaries(self):
        self.assertFalse(is_tuition_amount_query('Một tín chỉ có bao nhiêu giờ học?'))
        self.assertTrue(is_tuition_amount_query('Giá một tín chỉ là bao nhiêu?'))
        self.assertTrue(is_tuition_exemption_query('Học phí có giảm cho sinh viên giỏi không?'))
        self.assertEqual(topic_articles('Cách quy đổi điểm 8,5 theo thang 4?'), {17})
        self.assertEqual(topic_articles('Em muốn đổi IT sang BA được không?'), {22})


if __name__ == '__main__':
    unittest.main()
