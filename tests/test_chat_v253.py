"""Regression for precise public evidence and cloud citation selection."""
import json
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch, Mock
from core.retrieval import RegulationSearch, public_focus


class ChatV253(unittest.TestCase):
    def setUp(self):
        self.r = RegulationSearch()

    def test_precise_certificate_debt(self):
        for q in ('Môn chứng chỉ có tính vào 24 tín chỉ nợ đọng không?',
                  'Tin chi mon chung chi co tinh vao tong no dong khong?',
                  'Tín chỉ nợ có kể môn chứng chỉ không?'):
            with self.subTest(q=q), patch.object(self.r, '_semantic', side_effect=AssertionError('unnecessary embedding')):
                self.assertEqual([c['id'] for c in self.r.retrieve(q, cloud=True)], ['D18.K1'])
                self.assertIn('môn chứng chỉ', public_focus(q))
                self.assertIn('không kể môn chứng chỉ', self.r.answer(q)['reply'])

    def test_online_assessment(self):
        q = 'Đánh giá trực tuyến được chiếm tối đa bao nhiêu phần trăm điểm học phần?'
        self.assertEqual([c['id'] for c in self.r.retrieve(q)], ['D13.K2'])
        self.assertIn('50%', self.r.answer(q)['reply'])
        self.assertIn('trọng', public_focus(q))

    def test_specific_certificate_subject_is_public(self):
        q = 'hết thời gian học mà thiếu gdtc thì sao secret123'
        focus = public_focus(q)
        self.assertIn('Nội dung được hỏi: chứng chỉ Giáo dục thể chất', focus)
        self.assertNotIn('secret123', focus)

    def test_wrong_cloud_source_falls_back(self):
        sdk = ModuleType('google.genai')
        sdk.types = SimpleNamespace(GenerateContentConfig=lambda **kw:kw)
        client = Mock()
        q = 'Môn chứng chỉ có tính vào 24 tín chỉ nợ đọng không?'
        with patch.dict('sys.modules', {'google.genai':sdk}), patch.object(self.r, '_cloud', return_value=client):
            client.models.generate_content.return_value = SimpleNamespace(candidates=[], text=json.dumps({'answer':'Một tín chỉ có 15 giờ','citations':['D3.K1']}))
            out = self.r.answer(q, cloud=True)
            self.assertEqual(out['method'], 'bm25')
            self.assertEqual(out['generation_status'], 'INVALID_CITATIONS')
            self.assertEqual([c['id'] for c in out['regulations']], ['D18.K1'])
            self.assertNotIn('15 giờ', out['reply'])
            client.models.generate_content.return_value.text=json.dumps({'answer':'Không tính môn chứng chỉ.','citations':['D18.K1']})
            out=self.r.answer(q,cloud=True)
            self.assertEqual(out['method'],'gemini')
            self.assertEqual([c['id'] for c in out['regulations']],['D18.K1'])
            self.assertIn('D18.K1', client.models.generate_content.call_args.kwargs['config']['response_schema']['properties']['citations']['items']['enum'])

    def test_multi_topic_not_forced_into_certificate_debt(self):
        from core.policy_focus_v3 import specific_facets
        self.assertFalse(specific_facets('Môn chứng chỉ có tính vào tín chỉ nợ đọng không? GPA tính thế nào?'))
        self.assertFalse(specific_facets('Hồ sơ của em có tính môn chứng chỉ vào tín chỉ nợ đọng không?'))

    def test_exact_clause_coverage_not_only_article(self):
        # Two distinct clauses in one article must not be silently reduced
        # to one of them by generation.
        sdk=ModuleType('google.genai'); sdk.types=SimpleNamespace(GenerateContentConfig=lambda **kw:kw)
        client=Mock();client.models.generate_content.return_value=SimpleNamespace(candidates=[],text=json.dumps({'answer':'chỉ một khoản','citations':['D18.K1']}))
        with patch.dict('sys.modules',{'google.genai':sdk}), patch.object(self.r,'_cloud',return_value=client), patch('core.retrieval.specific_facets',return_value=['D18.K1','D18.K2']):
            out=self.r.answer('cảnh báo học tập',cloud=True)
        self.assertEqual(out['method'],'bm25')
        self.assertEqual({c['id'] for c in out['regulations']},{'D18.K1','D18.K2'})
