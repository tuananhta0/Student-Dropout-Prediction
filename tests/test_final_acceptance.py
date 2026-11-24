"""Acceptance gates must not silently promote missing/failing evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.smoke_browser import localhost_url
from scripts.summarize_remaining import evaluate, summarize, DEFAULT_FILES, markdown


class FinalAcceptanceTests(unittest.TestCase):
    def test_browser_only_accepts_localhost(self):
        self.assertEqual(localhost_url('http://127.0.0.1:8000'), 'http://127.0.0.1:8000')
        for x in ('https://127.0.0.1', 'http://example.com', 'http://localhost:8000/evil',
                  'http://admin:secret@localhost:8000', 'http://127.0.0.1:8000?x=y'):
            with self.subTest(value=x), self.assertRaises(ValueError):localhost_url(x)

    def test_unavailable_evidence_never_passes(self):
        for name in DEFAULT_FILES:
            self.assertEqual(evaluate(name, None)[0], 'NOT_RUN')

    def test_cloud_does_not_count_bm25_fallback(self):
        case = dict(cloud_requested=True, so_cau=40, so_cau_gemini=0, cloud_quality_gate=True)
        self.assertNotEqual(evaluate('gemini_cloud', case)[0], 'PASS')
        case['so_cau_gemini'] = 38
        self.assertEqual(evaluate('gemini_cloud', case)[0], 'PASS')

    def test_http_422_never_accepted(self):
        obj = {'failed':0,'passed':8,'checks':[{'name':name,'status':'PASS'} for name in
               ['Login student','Login advisor','Login admin','Student cannot access other profile','Scenario business gpa']]}
        self.assertNotEqual(evaluate('http_live', obj)[0], 'PASS')
        obj['checks'].append({'name':'Scenario business payment','status':'PASS'})
        self.assertEqual(evaluate('http_live', obj)[0], 'PASS')

    def test_audit_review_counts_are_not_blocking(self):
        obj = {'blocking_errors':{'foreign_key':0,'label_cutoff':0},'review_required':{'registration_exceptions':4331}}
        self.assertEqual(evaluate('main_audit', obj)[0], 'PASS')
        obj['blocking_errors']['foreign_key'] = 1
        self.assertEqual(evaluate('main_audit', obj)[0], 'FAIL')

    def test_browser_requires_all_roles_and_sizes(self):
        obj = {'failed':0, 'checks':[{'role':r,'viewport':w,'failed':0,'passed':6}
               for r in ('student','advisor','admin') for w in ('desktop','mobile')]}
        self.assertEqual(evaluate('browser_live', obj)[0], 'PASS')
        obj['checks'].pop()
        self.assertNotEqual(evaluate('browser_live', obj)[0], 'PASS')

    def test_provenance_mismatch_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'cloud.json'
            p.write_text(json.dumps({'cloud_requested':True,'so_cau':40,'so_cau_gemini':40,
                         'cloud_quality_gate':True,'provenance':{'source_commit':'OLD','source_dirty':False}}))
            result = summarize({'gemini_cloud':p}, 'NEW')
            self.assertEqual(result['items']['gemini_cloud']['status'], 'STALE')
            self.assertIn('MANUAL_PENDING', markdown(result))

    def test_independent_semantics_never_auto_certified(self):
        obj = {'so_cau':40,'nguoi_cham':'MEMBER_2'}
        self.assertEqual(evaluate('semantic_review', obj)[0], 'HUMAN_REVIEWED')


if __name__ == '__main__':unittest.main()
