"""Kiểm thử truy xuất quy chế, trích dẫn và từ chối ngoài phạm vi."""
import unittest
from core.retrieval import RegulationSearch, numbers

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=RegulationSearch()
    def test_all_articles(self):
        for i in range(1,28):
            with self.subTest(i=i): self.assertTrue(all(x['article']==i for x in self.r.retrieve(f'Điều {i}')) and self.r.retrieve(f'Điều {i}'))
    def test_clause(self):
        self.assertEqual([x['id'] for x in self.r.retrieve('Khoản 2 Điều 18')],['D18.K2'])
        self.assertFalse(self.r.retrieve('Khoản 9 Điều 18'))
        self.assertEqual(numbers('Khoản 2 Điều 18'),([18],[2]))
    def test_sources(self):
        out=self.r.answer('Điều 25')
        self.assertIn('#page=25',out['reply'])
        self.assertIn('20%',out['reply'])
        self.assertIn('30%',out['reply'])
        self.assertTrue(all(c['document'].endswith('.pdf') for c in out['regulations']))
    def test_unknown(self):
        for q in ['thiên văn','trái đất có bao nhiêu mặt trăng','Điều 99','xyzxyzxyz']:
            with self.subTest(q=q): self.assertFalse(self.r.retrieve(q))
    def test_topics(self):
        for q,a in [('rút học phần',12),('bảo lưu',21),('điểm liệt',13),('cấm thi',25),('thi lại',14),('cảnh báo học tập',18),('chuyển ngành',22),('tiếng Anh B2',19),('phúc khảo',16),('học song ngành',23)]:
            with self.subTest(q=q): self.assertIn(a,[c['article'] for c in self.r.retrieve(q)])

    def test_cloud_citation_validation(self):
        # Giả lập đúng hợp đồng SDK; không gọi mạng hoặc dùng API key thật.
        import json
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch, Mock
        sdk=ModuleType('google.genai')
        sdk.types=SimpleNamespace(GenerateContentConfig=lambda **kw:kw)
        search=RegulationSearch()
        client=Mock()
        client.models.generate_content.return_value.candidates=[]
        client.models.generate_content.return_value.text=json.dumps({'answer':'<script>không tin cậy</script>','citations':['D18.K2']})
        with patch.dict('sys.modules',{'google.genai':sdk}),patch.object(search,'_cloud',return_value=client):
            result=search.answer('Khoản 2 Điều 18 bí mật cá nhân',cloud=True)
            self.assertEqual(result['method'],'gemini')
            self.assertEqual(result['generation_status'],'OK')
            self.assertEqual(client.models.generate_content.call_args.kwargs['config']['max_output_tokens'],4096)
            self.assertNotIn('<script>',result['reply'])
            self.assertNotIn('bí mật cá nhân',client.models.generate_content.call_args.kwargs['contents'])
            client.models.generate_content.return_value.text=json.dumps({'answer':'Không có chứng cứ','citations':['D99.K1']})
            result=search.answer('Khoản 2 Điều 18',cloud=True)
            self.assertEqual(result['method'],'bm25')
            self.assertEqual(result['generation_status'],'INVALID_CITATIONS')
            self.assertNotIn('Không có chứng cứ',result['reply'])

    def test_semantic_rerank(self):
        import numpy as np
        from unittest.mock import patch
        search=RegulationSearch();trace={}
        with patch.object(search,'_cloud',return_value=object()),patch.object(search,'_semantic',return_value=np.zeros(len(search.chunks))) as semantic:
            rows=search.retrieve('cảnh báo học tập bí mật',cloud=True,trace=trace)
            self.assertTrue(rows)
            self.assertTrue(trace['embedding_used'])
            self.assertNotIn('bí mật',semantic.call_args.args[0])

    def test_generation_diagnostics(self):
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch, Mock
        sdk=ModuleType('google.genai')
        sdk.types=SimpleNamespace(GenerateContentConfig=lambda **kw:kw)
        client=Mock();search=RegulationSearch()
        with patch.dict('sys.modules',{'google.genai':sdk}),patch.object(search,'_cloud',return_value=client):
            for raw,finish,expected in [(None,'MAX_TOKENS','MAX_TOKENS'),(None,'STOP','EMPTY_RESPONSE'),('{','STOP','INVALID_JSON')]:
                client.models.generate_content.return_value=SimpleNamespace(text=raw,candidates=[SimpleNamespace(finish_reason=finish)])
                with self.assertLogs('core.retrieval',level='WARNING') as logs:
                    result=search.answer('Điều 18',cloud=True)
                self.assertEqual(result['generation_status'],expected)
                self.assertEqual(result['method'],'bm25')
                self.assertIn(expected,logs.output[0])
            exc=RuntimeError('bí mật không được ghi log');exc.code=429
            client.models.generate_content.side_effect=exc
            with self.assertLogs('core.retrieval',level='WARNING') as logs:
                result=search.answer('Điều 18',cloud=True)
            self.assertEqual(result['generation_status'],'HTTP_429')
            self.assertNotIn('bí mật',str(logs.output))
