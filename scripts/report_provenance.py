"""Gắn nguồn phiên bản cho báo cáo tại lúc chạy; không nâng phiên bản báo cáo cũ."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from core.data_pipeline import ROOT


def git(*args, root=ROOT):
    try:
        return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError):
        return None


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def provenance(root=ROOT, **inputs):
    head=git('rev-parse','HEAD',root=root)
    tree=git('rev-parse','HEAD^{tree}',root=root)
    status=git('status','--porcelain',root=root)
    diff=git('diff','--binary','HEAD',root=root)
    # Dấu vân tay thay đổi tracked, không lưu diff, đường dẫn hoặc biến môi trường.
    return {'source_commit':head.decode().strip() if head else None,
            'source_tree':tree.decode().strip() if tree else None,
            'source_dirty':bool(status) if status is not None else None,
            'tracked_diff_sha256':hashlib.sha256(diff).hexdigest() if diff is not None else None,
            'created_utc':datetime.now(timezone.utc).isoformat(), 'inputs':inputs}


def sidecar(path, info):
    path=Path(path)
    dest=path.with_name(path.name+'.provenance.json')
    dest.write_text(json.dumps({'provenance':info,'report_sha256':sha256(path)},ensure_ascii=False,indent=2),encoding='utf-8')
    return dest


def read_report_origin(path):
    """Báo cáo cũ không có thông tin phải được ghi là chưa xác minh."""
    path=Path(path)
    try:
        if path.suffix=='.json':
            obj=json.loads(path.read_text(encoding='utf-8-sig'))
        else:
            obj=json.loads(path.with_name(path.name+'.provenance.json').read_text(encoding='utf-8-sig'))
            if obj.get('report_sha256')!=sha256(path): return {'status':'changed_after_recording','provenance':obj.get('provenance')}
        info=obj.get('provenance')
        return {'status':'recorded' if isinstance(info,dict) and info.get('source_commit') else 'unknown','provenance':info}
    except (OSError,ValueError,AttributeError):
        return {'status':'unknown','provenance':None}
