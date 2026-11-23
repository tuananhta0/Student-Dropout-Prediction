"""Hồi quy đọc ZIP runtime public; không cần asset lớn hay chạy model."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from scripts.review_public_runtime import inspect, scan_stream


class PublicRuntimeReviewTests(unittest.TestCase):
    def test_safe_mock_archive_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mock.zip'
            content=b'public academic simulation, no secret'
            name='model/analytics_summary.json'
            manifest={'version':'cmc-research-v1','files':{name:{'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}}}
            with zipfile.ZipFile(path,'w') as z:
                z.writestr(name,content)
                z.writestr('release_manifest.json',json.dumps(manifest))
            result=inspect(path,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result['files_checked'],1)
            self.assertTrue(result['verified_manifest'])
            self.assertEqual(result['secret_alerts'],0)
            self.assertTrue(result['manual_confirmation_required'])
            with self.assertRaisesRegex(ValueError,'SHA256'):
                inspect(path,'0'*64)

    def test_secret_detection_never_displays_payload(self):
        issues=[]
        token=b'AIza' + b'Q'*35
        scan_stream(io.BytesIO(b'model_token='+token),'model/example.json',issues)
        self.assertEqual(issues,[{'file':'model/example.json','kind':'google_api_key'}])
        self.assertNotIn(token.decode(),str(issues))

    def test_sensitive_columns_flag_for_manual_review(self):
        issues=[]
        scan_stream(io.BytesIO(b'student_id,email,gpa\n1234,example@example.com,3.0'),
                    'dataset/table.csv',issues,sample_header=True)
        self.assertIn('sensitive_columns_need_manual_review',{v['kind'] for v in issues})

    def test_reject_unsafe_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.zip'
            content=b'not a model'
            name='../secret.txt'
            manifest={'version':'cmc-research-v1','files':{name:{'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}}}
            with zipfile.ZipFile(path,'w') as z:
                z.writestr(name,content)
                z.writestr('release_manifest.json',json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'không an toàn'):
                inspect(path,hashlib.sha256(path.read_bytes()).hexdigest())
