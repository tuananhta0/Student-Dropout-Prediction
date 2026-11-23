"""Kiểm tra đọc-chỉ ZIP runtime public: toàn vẹn, đường dẫn, chỉ dấu bí mật.

Không giải nén ra đĩa, không nạp joblib, không ghi lại giá trị bí mật.
Phát hiện theo mẫu không chứng minh dữ liệu là giả lập hay hoàn toàn vô hại.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

EXPECTED_SHA = '72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848'
MAX_UNCOMPRESSED = 1024 * 1024 * 1024
PATTERNS = {
    'google_api_key': re.compile(rb'AIza[0-9A-Za-z_-]{35}'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    'private_key': re.compile(rb'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'),
    'api_key_assignment': re.compile(rb'(?:GEMINI_API_KEY|GOOGLE_API_KEY|GITHUB_TOKEN|OPENAI_API_KEY)\s*["\']?\s*[:=]\s*["\']?[A-Za-z0-9_-]{16,}', re.I),
}
SENSITIVE_COLUMNS = re.compile(r'(?i)(?:^|[,;\t])\s*(?:full_name|student_name|ho_ten|email|e_mail|phone|so_dien_thoai|cccd|cmnd|national_id)\s*(?:[,;\t]|$)')


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        for part in iter(lambda: src.read(1024 * 1024),b''):
            h.update(part)
    return h.hexdigest()


def safe_member(info):
    path=PurePosixPath(info.filename)
    mode=(info.external_attr >> 16) & 0o170000
    return (not path.is_absolute() and len(path.parts)>1 and '..' not in path.parts
            and ':' not in info.filename and '\\' not in info.filename
            and not info.is_dir() and mode != stat.S_IFLNK)


def scan_stream(stream, member, issues, *, sample_header=False):
    """Đọc hết nội dung, không giữ payload hay secret trong kết quả báo cáo."""
    seen=set()
    overlap=b''
    consumed=0
    for part in iter(lambda:stream.read(65536),b''):
        consumed+=len(part)
        if consumed>MAX_UNCOMPRESSED:
            raise ValueError('Nội dung giải nén quá giới hạn an toàn')
        window=overlap+part
        for kind,pattern in PATTERNS.items():
            if pattern.search(window):seen.add(kind)
        if sample_header and consumed==len(part):
            # Chỉ xem tên cột, không đọc/ghi dữ liệu sinh viên.
            header=window.split(b'\n',1)[0][:8192].decode('utf-8','replace')
            if SENSITIVE_COLUMNS.search(header): seen.add('sensitive_columns_need_manual_review')
        overlap=window[-256:]
    for kind in sorted(seen):
        issues.append({'file':member,'kind':kind})


def inspect(archive, expected_sha=EXPECTED_SHA):
    archive=Path(archive)
    sha=checksum(archive)
    if sha!=expected_sha.lower():raise ValueError('SHA256 toàn bộ ZIP không khớp nguồn tin cậy')
    issues=[]
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>1000 or len(set(x.filename for x in infos))!=len(infos):
            raise ValueError('Tên tệp trùng hoặc số lượng tệp bất thường')
        if sum(i.file_size for i in infos)>MAX_UNCOMPRESSED:
            raise ValueError('Tổng dung lượng giải nén vượt giới hạn')
        if not all(i.filename=='release_manifest.json' or safe_member(i) for i in infos):
            raise ValueError('Đường dẫn hoặc loại tệp không an toàn trong ZIP')
        manifest=json.loads(z.read('release_manifest.json'))
        entries=manifest.get('files')
        if not isinstance(entries,dict) or manifest.get('version')!='cmc-research-v1':
            raise ValueError('Release manifest không hợp lệ')
        if set(entries)|{'release_manifest.json'} != {i.filename for i in infos}:
            raise ValueError('Danh sách file không trùng release manifest')
        # Manifest cũng là nội dung công bố nên phải quét, không chỉ các file con.
        with z.open('release_manifest.json') as manifest_stream:
            scan_stream(manifest_stream,'release_manifest.json',issues)
        for info in infos:
            if info.filename=='release_manifest.json':
                continue
            record=entries[info.filename]
            if record.get('bytes')!=info.file_size:
                raise ValueError('Kích thước sai: '+info.filename)
            h=hashlib.sha256()
            with z.open(info) as stream:
                for chunk in iter(lambda: stream.read(65536),b''):
                    h.update(chunk)
            if h.hexdigest()!=record.get('sha256'):
                raise ValueError('SHA256 thành phần sai: '+info.filename)
            # Quét lại từng tệp, gồm cả binary, không thực thi nội dung.
            with z.open(info) as stream:
                scan_stream(stream,info.filename,issues,
                            sample_header=info.filename.lower().endswith('.csv'))
            if info.filename.lower().endswith('.csv.gz'):
                with z.open(info) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
                    scan_stream(stream,info.filename,issues,sample_header=True)
    return {
        'archive_sha256':sha,
        'version':manifest['version'],
        'verified_manifest':True,
        'files_checked':len(entries),
        'issues':issues,
        'secret_alerts':sum(x['kind']!='sensitive_columns_need_manual_review' for x in issues),
        'manual_confirmation_required':True,
        'note':'Mẫu phát hiện bí mật không thay thế xác nhận dữ liệu chỉ là giả lập. Không nạp joblib từ nguồn không tin cậy.'
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path,help='ZIP tải từ GitHub Release')
    parser.add_argument('--sha256',default=EXPECTED_SHA,help='Checksum nhận từ nguồn tin cậy, không tính từ file rồi tự chấp nhận')
    parser.add_argument('--output',type=Path,default=Path('data/kiem_tra_ban_giao/bao_cao_runtime_public.json'))
    args=parser.parse_args()
    try:report=inspect(args.archive,args.sha256)
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile,gzip.BadGzipFile) as exc:
        print('FAIL:',type(exc).__name__,str(exc))
        return 1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='issues'},ensure_ascii=False,indent=2))
    print('Phát hiện chỉ dấu (không in giá trị):',len(report['issues']),'; báo cáo:',args.output)
    return 0 if not report['secret_alerts'] else 1

if __name__=='__main__':raise SystemExit(main())
