"""Kiểm thử đóng gói, kiểm tra hash và từ chối cấu hình cũ."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import zipfile
from core.data_pipeline import digest, MAIN
from scripts.release_model import DATA_FILES,MODEL_FILES,pack,install
from scripts.create_accounts import create
from core.security import check_password
from core.runtime_config import configure_runtime
from scripts.audit_dataset import resolve_scored_path

class ReleaseTests(unittest.TestCase):
    def test_pack_install_and_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ds=root/'dataset';model=root/'model';model.mkdir()
            for rel in DATA_FILES:
                f=ds/MAIN/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(rel)
            for name in MODEL_FILES:(model/name).write_text(name)
            archive=root/'release.zip';dest=root/'installed'
            # Kiểm tra quy trình file trên mẫu; verify_pair thật bắt buộc khi chạy CLI.
            with patch('scripts.release_model.verify_pair',return_value={'dataset_manifest_sha256':'example'}):
                pack(ds,model,archive)
                with self.assertRaises(ValueError):install(archive,dest,'0'*64)
                install(archive,dest,digest(archive))
                self.assertEqual((dest/'model'/MODEL_FILES[0]).read_bytes(),(model/MODEL_FILES[0]).read_bytes())
                with self.assertRaises(ValueError):install(archive,dest,digest(archive))
                bad=root/'bad.zip'
                with zipfile.ZipFile(archive) as src,zipfile.ZipFile(bad,'w') as out:
                    for name in src.namelist():out.writestr(name,b'x' if name=='model/'+MODEL_FILES[0] else src.read(name))
                with self.assertRaises(ValueError):install(bad,root/'bad',digest(bad))
                self.assertFalse((root/'bad').exists())
    def test_no_fallback(self):
        with tempfile.TemporaryDirectory() as t,patch.dict('os.environ',{'CMC_MODEL_DIR':t,'CMC_DATASET_ROOT':t,'CMC_ALLOW_BASELINE':'0'},clear=True):
            with self.assertRaises(RuntimeError):configure_runtime()
    def test_audit_uses_runtime_model(self):
        with patch('core.runtime_config.configure_runtime',return_value='data/models/cmc-research-v1') as configure:
            self.assertEqual(resolve_scored_path(False),Path('data/models/cmc-research-v1/student_snapshots_scored.csv.gz'))
            configure.assert_called_once_with()
    def test_audit_explicit_and_source_only(self):
        with patch('core.runtime_config.configure_runtime') as configure:
            self.assertIsNone(resolve_scored_path(True))
            self.assertEqual(resolve_scored_path(False,Path('custom/scored.csv.gz')),Path('custom/scored.csv.gz'))
            configure.assert_not_called()
    def test_password_file(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'accounts.json';values=create(f,['ADMIN','BIT250093']);self.assertNotEqual(values['ADMIN'],values['BIT250093']);self.assertEqual(create(f,['ADMIN']),{})
            self.assertNotIn(values['ADMIN'],f.read_text())
            with patch.dict('os.environ',{'CMC_ACCOUNTS_FILE':str(f)}):
                self.assertTrue(check_password('ADMIN',values['ADMIN']));self.assertFalse(check_password('ADMIN','ADMIN'))
