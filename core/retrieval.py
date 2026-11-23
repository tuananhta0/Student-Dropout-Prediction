"""Tra cứu quy chế theo đoạn đã đối chiếu ảnh PDF; BM25 và embedding tùy chọn."""
from collections import Counter
import csv
import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import ContextVar
import unicodedata

import numpy as np
from core.chatbot_engine import strip_accents

ROOT = Path(__file__).resolve().parents[1]
PDF = 'Quy-che-dao-tao-trinh-do-dai-hoc.pdf'
SUPPLEMENT = 'Quy-dinh-thoi-hoc-DH-CMC.pdf'
SUPPLEMENT_SHA = '6c5b0c2e085bd7f7820f4cbf827acf8e555295b95ad6f1147842c3a90e6ec042'
# Bản tổng hợp, không phải quy phạm độc lập; chỉ gắn vào các đoạn chính tương ứng.
SUPPLEMENT_PAGES = {'D3.K5–6':'1–2','D9.K1–2':'2','D12.K2':'2','D12.K4':'2','D13.K1':'3','D14.K3':'3','D15.K1–3':'2','D17.K1':'3','D17.K3–4':'3','D18.K1':'1','D18.K2':'1','D18.K3–4':'1','D19.K1–5':'4','D20.K5–6':'3','D21.K1':'2','D21.K2–3':'2','D25.K1':'3','D25.K2–3':'3','D25.K4–6':'3'}
PDF_SHA = '10692552bf4245f4ee5a5487cbad69d123ea7bdb2fde6c13d1658d97ed857ea7'
STOP = set('la cua va trong cho ve co duoc hay nhu the nao nhieu gi toi em mot nhung cac khong sinh vien truong cmc theo quy dinh quy che'.split())


# Cụm diễn đạt công khai, độc lập với kết quả BM25. Không gửi từ do người dùng nhập.
PARAPHRASES = {
    'cham lai bai thi': 'phuc tra ket qua thi', 'cham lai diem': 'phuc tra ket qua thi',
    'nhac nho lien tiep': 'canh bao hoc tap lien tiep', 'duoi hoc': 'buoc thoi hoc',
    'hoc het bao nhieu nam thi phai ra truong': 'thoi gian hoc toi da', 'hoc het bao lau': 'thoi gian hoc toi da',
    'chuyen sang truong': 'chuyen truong', 'chuyen sang dai hoc': 'chuyen truong',
    'chuyen den truong': 'chuyen truong', 'chuyen den dai hoc': 'chuyen truong',
    'chuyen qua truong': 'chuyen truong', 'doi truong': 'chuyen truong',
    'doi nganh': 'chuyen nganh', 'doi chuyen nganh': 'chuyen nganh',
    'doi sang truong': 'chuyen truong',
    'dang ky hoc phan': 'dang ky tin chi', 'tiet hoc': 'moi tiet',
    'bi om': 'nghi hoc tam thoi bao luu', 'nghi vi benh': 'nghi hoc tam thoi bao luu',
    'phuc khao': 'phuc tra', 'song nganh': 'hai chuong trinh',
    'mon bat buoc':'hoc phan bat buoc', 'nganh thu hai':'hai chuong trinh', 'nghi khong phep':'vang khong phep', 'luu bao lau':'luu tru', 'thoi gian toi da': 'thoi gian hoc toi da', 'rot mon': 'hoc lai', 'truot mot mon muon nang diem': 'hoc lai cai thien diem', 'luan van': 'khoa luan', 'nghi hoc do benh':'nghi hoc tam thoi bao luu', 'nghi hoc vi benh':'nghi hoc tam thoi bao luu', 'nghi hoc mot nam':'nghi hoc tam thoi bao luu', 'nghi mot nam':'nghi hoc tam thoi', 'cham nguoi than':'ly do ca nhan nghi hoc tam thoi', 'hoc lop cai thien':'hoc cai thien', 'lop cai thien':'hoc cai thien', 'vang mat':'vang', 'tu xin thoi hoc':'tu nguyen thoi hoc',
}
TOPICS = ('phuc tra ket qua thi|thoi gian hoc toi da|canh bao hoc tap|buoc thoi hoc|chuyen truong|chuyen nganh|bao luu|nghi hoc tam thoi|hoc phan tien quyet|hoc phan bat buoc|hoc phan tu chon|dang ky tin chi|rut hoc phan|diem liet|thi lai|hoc lai|cai thien diem|gpa|tot nghiep|tieng anh|hoc truc tuyen|ke hoach dao tao|quy mo lop|co van hoc tap|trach nhiem sinh vien|muc tieu dao tao|lop nhap hoc|trao doi sinh vien|luu tru khoa luan|hieu luc thi hanh|cam thi|ha hang tot nghiep|hoc phi rut mon|hoc phi|khoa luan|moi tiet|quy che ap dung|hai chuong trinh|tin chi|diem thi|tu nguyen thoi hoc|ly do ca nhan|chung nhan|thoi hoc|vang khong phep|dieu kien du thi|thi cuoi ky|cong nhan tin chi|quyen loi sau thoi hoc|hoc cai thien|nghia vu quan su').split('|')


def normalize_query(question):
    query=strip_accents(question)
    for old,new in PARAPHRASES.items():
        query=query.replace(old,new)
    # Các cụm thông dụng không nhất thiết dùng từ pháp lý 'nghỉ học tạm thời'.
    # Chỉ nối ý định nghỉ học khi có yêu cầu nghỉ học theo kỳ/lý do, không bắt
    # nhầm 'nghỉ thi', 'nghỉ không phép' hay hỏi giờ vắng mặt trong học phần.
    if re.search(r'\b(?:chuyen|doi)\s+(?:den|qua|sang)?\s*(?:mot\s+)?(?:truong|dai hoc)\s+(?:dai hoc\s+)?khac\b', query):
        query += ' chuyen truong'
    # Hỏi giờ tín chỉ không phải hỏi đơn giá học phí. Điều 3.1 định nghĩa
    # khối lượng học tập theo giờ tín chỉ; Điều 5 quy định độ dài mỗi tiết.
    if (re.search(r'\b(?:mot|1|moi)\s+tin\s+chi\b',query)
            and re.search(r'\b(?:hoc\s+)?(?:bao nhieu|may|so|tinh)\s+(?:gio|tiet|phut|tieng)\b',query)
            and 'hoc phi' not in query):
        query += ' gio tin chi ly thuyet'
    if (re.search(r'\b(?:mot|1|moi)\s+tiet(?:\s+hoc)?\b',query)
            and any(x in query for x in ('bao lau','bao nhieu phut','keo dai','thoi luong'))):
        query += ' moi tiet'
    if 'hoc phan' in query and 'diem' in query and any(x in query for x in ('lam tron','ra chu','diem chu')):
        query += ' diem chu'
    # Câu chuyển ngành chỉ dùng mã ngành, thường không nhắc từ 'ngành'.
    # Chỉ nhận khi có động từ chuyển/đổi cùng hướng chuyển giữa hai mã viết hoa.
    original_codes = re.findall(r'\b[A-Z]{2,3}\b', question)
    if (len(set(original_codes)) >= 2 and re.search(r'\b(?:doi|chuyen)\b',query)
            and re.search(r'\b(?:sang|qua|den)\b',query)):
        query += ' chuyen nganh'
    if (re.search(r'\b(?:tam dung|dung|tam ngung|ngung)\s+(?:viec\s+)?hoc\b', query)
            and any(x in query for x in ('tai nan','dieu tri','benh','om','phau thuat','nam vien'))):
        query += ' nghi hoc tam thoi bao luu'
    if re.search(r'\bquy doi\s+(?:ve\s+|sang\s+)?(?:thang\s+)?(?:diem\s+)?4\b', query):
        query += ' quy doi thang 4'
    if 'gpa tich luy' in query and any(x in query for x in ('duoc tinh', 'cong thuc', 'tinh tu', 'tinh nhu the nao')):
        query += ' cong thuc gpa'
    if 'bao nhieu nam' in query and any(x in query for x in ('hoc','dao tao','ra truong')): query+=' thoi gian hoc toi da'
    if re.search(r'(?:vang|nghi)\s*\d+(?:[.,]\d+)?\s*%',query): query+=' cam thi'
    # Các câu xin nghỉ có thời hạn/lý do thường không nói nguyên cụm "bảo lưu".
    # Không áp dụng với vắng giờ học, nghỉ không phép hoặc nghỉ thi.
    study_leave = (
        re.search(r'\b(?:xin|muon|co the|duoc)\s+nghi\s+(?:hoc\s+)?\d+\s*(?:hoc\s*)?ky\b',query)
        or re.search(r'\bnghi\s+(?:hoc\s+)?\d+\s*(?:hoc\s*)?ky\s+(?:vi|do)\b',query)
        or re.search(r'\b(?:xin|muon|co the|duoc)\s+nghi\s+\d+\s*(?:hoc\s*)?ky\b',query)
        or ('nghi hoc' in query and any(x in query for x in ('kho khan tai chinh','bi om','vi om','vi benh','do benh','dang benh','bi benh','benh nang','dieu tri','tai nan','phau thuat','nam vien','ly do ca nhan','mot hoc ky','moi hoc ky')))
        or (re.search(r'\b(?:xin|muon|dinh|can)\s+nghi\s+(?:hoc\s+)?(?:mot|hai|ba|bon|1|2|3|4)\s*(?:hoc\s*)?ky\b',query)
            and not any(x in query for x in ('nghi khong phep','nghi thi')))
    )
    if study_leave and not any(x in query for x in ('nghi khong phep','nghi thi','nghi le','nghi tet')):
        query += ' nghi hoc tam thoi bao luu'
    if (
        any(x in query for x in ('nghia vu quan su', 'di nghia vu', 'nhap ngu', 'di bo doi', 'luc luong vu trang'))
        or re.search(r'\bnvqs\b', query)
    ):
        query += ' nghia vu quan su nghi hoc tam thoi bao luu luc luong vu trang'
    return query


def public_focus(question):
    """Chỉ xuất cụm có sẵn trong danh sách công khai, không sao chép đầu vào."""
    q=normalize_query(question)
    if any(x in q for x in ('the ngan hang','chuyen khoan','vi dien tu','so tai khoan','thanh toan bang','mat khau','api key')):
        return ''
    matched=[t for t in TOPICS if re.search(r'\b'+re.escape(t)+r'\b',q)]
    # Bổ sung khía cạnh hỏi bằng cụm công khai, không chuyển bất kỳ token gốc nào.
    aspects={'bao nhieu':'ngưỡng hoặc số lượng','toi da':'giới hạn tối đa','toi thieu':'giới hạn tối thiểu','lam sao':'thủ tục','giay xac nhan':'minh chứng','can gi':'điều kiện','khi nao':'thời hạn','co duoc':'điều kiện cho phép','ca nhan':'lý do cá nhân','co phep':'vắng có phép'}
    matched=list(dict.fromkeys(matched+[t for terms in POLICY_TOPICS.values() for t in terms if re.search(r'\b'+re.escape(t)+r'\b',q)]))
    if not matched: return ''
    return '; '.join(matched+[v for k,v in aspects.items() if k in q])


# Chủ đề hẹp được ánh xạ tới Điều có nội dung tương ứng. Chủ đề rộng như GPA,
# tín chỉ, học phí không tự động khóa vào một Điều duy nhất.
POLICY_TOPICS = {
    1: ('quy che ap dung','pham vi doi tuong'), 2: ('muc tieu dao tao',),
    3: ('thoi gian hoc toi da','hoc phan tien quyet','gio tin chi ly thuyet'), 4: ('hoc phan bat buoc',),
    5: ('moi tiet','gio hoc ly thuyet'), 6: ('trach nhiem sinh vien',),
    7: ('co van hoc tap','giao vien chu nhiem'), 8: ('ke hoach dao tao',),
    9: ('lop nhap hoc',), 10: ('quy mo lop',), 11: ('hoc truc tuyen',),
    12: ('rut hoc phan','rut mon','dang ky tin chi','hoc phi rut mon','lam do an','lam khoa luan'),
    13: ('diem liet','thi cuoi ky','diem thanh phan','diem chu'),
    14: ('thi lai','thi bo sung','dieu kien du thi'), 15: ('cai thien diem','hoc lai','hoc cai thien'),
    16: ('phuc tra','phuc khao'), 17: ('cong thuc gpa','quy doi thang 4','quy doi diem','diem theo thang 4','xep loai hoc luc'),
    18: ('canh bao hoc tap','buoc thoi hoc','canh bao lien tiep','so lan canh bao hoc tap'),
    19: ('tieng anh','cong nhan tin chi'), 20: ('dieu kien tot nghiep','ha hang tot nghiep'),
    21: ('bao luu','nghi hoc tam thoi','tu nguyen thoi hoc','nghia vu quan su'),
    22: ('chuyen truong','chuyen nganh'), 23: ('hai chuong trinh',),
    24: ('trao doi sinh vien',), 25: ('cam thi','vang khong phep','thi ho'),
    26: ('luu tru khoa luan','luu tru','khoa luan phai duoc luu'), 27: ('hieu luc thi hanh',),
}


def topic_articles(question):
    q=normalize_query(question)
    out={a for a,terms in POLICY_TOPICS.items() if any(re.search(r'\b'+re.escape(t)+r'\b',q) for t in terms)}
    if 4 in out and not any(t in q for t in ('khong dat','truot','rot','phai lam gi')):
        out.discard(4);out.add(3)
    # Đã đạt điểm chữ thì hỏi "thi lại để cải thiện" là quy định học lại
    # cải thiện (Điều 15), không phải thi lại điểm F (Điều 14).
    if ('thi lai' in q and 'cai thien' in q
            and re.search(r'\bdiem\s*[abcd]\+?(?=\s|[.,!?;]|$)',q)):
        out.discard(14)
        out.add(15)
    return out


def is_tuition_exemption_query(question):
    q = strip_accents(question)
    if any(x in q for x in ('rut hoc phan', 'rut mon', 'hoan hoc phi', 'hoan tra')):
        return False
    if re.search(r'\b(?:mien\s*[,/&\-]?\s*giam|mien|giam|ho tro)\s+(?:dong\s+|tien\s+)?hoc\s+phi\b', q):
        return True
    return 'hoc phi' in q and (
        any(x in q for x in ('mien giam', 'duoc mien', 'duoc giam', 'uu dai hoc phi'))
        or re.search(r'\bhoc\s*phi\b.{0,45}\b(?:mien|giam|ho tro)\b',q) is not None
    )


def is_tuition_amount_query(question):
    q = strip_accents(question)
    if is_tuition_exemption_query(question):
        return False
    if re.search(r'\b(?:muc|so)\s+tien\s+(?:mot|1|moi)\s+tin\s+chi\b', q):
        return True
    if any(x in q for x in ('con no', 'no hoc phi', 'cong no', 'tong no', 'qua han', 'hoan hoc phi', 'hoan tra', 'rut hoc phan', 'rut mon', 'thanh toan hoc phi')):
        return False
    if any(x in q for x in (
        'hoc phi bao nhieu', 'muc hoc phi', 'muc thu hoc phi', 'hoc phi mot tin', 'hoc phi 1 tin',
        'hoc phi bao nhieu mot thang', 'hoc phi moi tin', 'hoc phi tung tin', 'hoc phi moi hoc phan',
        'don gia tin chi', 'don gia hoc phi', 'gia mot tin chi', 'gia 1 tin chi', 'gia hoc phi', 'bieu phi'
    )):
        return True
    if 'hoc phi' in q and re.search(
        r'\b(?:bao nhieu(?:\s+tien)?|may\s+trieu|muc\s+thu|don\s+gia|bao\s+tien|het\s+bao\s+nhieu|dong\s+bao\s+nhieu|so\s+tien|muc\s+tien|1\s*tin|mot\s+tin|moi\s+tin|moi\s+ky|mot\s+ky|1\s*ky|moi\s+nam|mot\s+nam|1\s*nam|mot\s+thang|moi\s+thang)\b',
        q,
    ):
        return True
    # Không suy đoán học phí chỉ từ 'một tín chỉ' + 'bao nhiêu': câu đó
    # có thể đang hỏi giờ/tiết học. Cần tín hiệu tiền tệ rõ ràng.
    if re.search(r'\b(?:1|mot|moi)\s*(?:tin chi|tc)\b', q) and re.search(
        r'\b(?:bao nhieu tien|bao tien|may trieu|gia|dong bao nhieu|het bao nhieu|so tien|hoc phi)\b',
        q,
    ):
        return True
    return False


def is_tuition_deadline_query(question):
    """Lịch thu tiền cụ thể không có trong quy chế học vụ, kể cả 'kỳ tới'.

    Tách riêng khỏi số tiền còn nợ hay thủ tục bảo lưu có lý do tài chính.
    """
    q = strip_accents(question)
    if not re.search(r'\b(?:hoc\s*phi|dong\s*tien\s*hoc)\b', q):
        return False
    if any(x in q for x in ('no hoc phi', 'cong no', 'so ngay qua han', 'dong duoc bao nhieu')):
        return False
    return bool(
        re.search(r'\b(?:khi nao|bao gio|ngay nao|ngay may|ngay bao nhieu|han chot|han cuoi|han dong|lich dong|lich nop|thoi han|deadline)\b', q)
        or re.search(r'\b(?:dong|nop|thanh toan)\s+(?:tien\s+)?hoc\s*phi\b.*\b(?:ky toi|ki toi|ky sau|hoc ky toi|thang \d{1,2}|nam \d{4})\b', q)
    )


def unsupported(question):
    q=strip_accents(question)
    if is_tuition_exemption_query(question) or is_tuition_amount_query(question) or is_tuition_deadline_query(question):
        return True
    # Lịch/ngày giờ/giá cụ thể thuộc thông báo vận hành, không phải quy chế.
    if re.search(r'\b(?:lich|ngay|thoi han|han chot)\s+(?:nghi le|nghi tet|nop hoc phi|dong hoc phi|thi cu the|thi lai cu the)',q): return True
    if re.search(r'\b(?:thu vien|can tin|nha xe)\b',q) or 'nghi le' in q or 'nghi tet' in q: return True
    return any(x in q for x in ('the ngan hang','chuyen khoan','vi dien tu','so tai khoan','thanh toan bang','mat khau','api key','thoi tiet','mat trang','nau an','gia vang','vay tien','gui xe','cau lac bo','bong da','ky tuc xa','dia chi truong','so dien thoai','diem chuan nam','lich thi cu the'))


# Ngân sách theo tiến trình; reset khi khởi động lại. Đếm cả embedding và retry.
CLOUD_DEADLINE = 25.0
# SDK gửi X-Server-Timeout theo giây; Gemini có thể từ chối deadline dưới 10 giây.
CLOUD_SDK_TIMEOUT_MS = 20000
CLOUD_LIMIT_MINUTE = 30
CLOUD_LIMIT_DAY = 500
_cloud_calls = deque()
_cloud_budget_lock = threading.Lock()
_cloud_slots = threading.BoundedSemaphore(2)
_cloud_pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix='quy-che-cloud')
_deadline = ContextVar('rag_deadline',default=None)

class CloudBudgetError(RuntimeError):
    pass

def cloud_call(call):
    now=time.monotonic()
    deadline=_deadline.get()
    if deadline is not None and deadline-now < 1: raise TimeoutError('Hết thời gian cloud')
    with _cloud_budget_lock:
        while _cloud_calls and now-_cloud_calls[0]>=86400: _cloud_calls.popleft()
        if len(_cloud_calls)>=CLOUD_LIMIT_DAY or sum(now-t<60 for t in _cloud_calls)>=CLOUD_LIMIT_MINUTE:
            raise CloudBudgetError('Đã đạt giới hạn gọi cloud')
        _cloud_calls.append(now)
    return call()

def retry_transient(call):
    # Một lần thử lại cho lỗi dịch vụ; không thử lại 429 hoặc lỗi dữ liệu.
    try: return call()
    except Exception as exc:
        if safe_api_error(exc) not in ('HTTP_503','HTTP_504','TIMEOUT'): raise
        time.sleep(1)
        return call()


def tokens(text):
    return [w for w in re.findall(r'[a-z0-9]+', strip_accents(text)) if w not in STOP]


def numbers(text):
    """Danh sách số ngay sau nhãn; không nuốt số tín chỉ/học kỳ ở câu kế tiếp."""
    q = strip_accents(text)
    def collect(label):
        pattern = r'\b(?:'+label+r')\s*(?:so\s*)?(\d{1,3}\b(?:\s*(?:,|;|va|hoac|&|/)\s*\d{1,3}\b(?!\s*(?:tin chi|tc|hoc ky|diem|%)))*)'
        return list(dict.fromkeys(int(x) for m in re.finditer(pattern,q) for x in re.findall(r'\d+',m.group(1))))
    return collect(r'dieu|quy\s*che|quy\s*dinh'), collect('khoan')


def span(s):
    ends = [int(x) for x in s.split('–')]
    return range(ends[0], ends[-1]+1)


def safe_api_error(exc):
    """Chỉ ghi mã lỗi, không ghi thông báo từ SDK có thể chứa bí mật."""
    code = getattr(exc, 'code', None)
    if code is None:
        code = getattr(exc, 'status_code', None)
    if code == 400:
        # Chỉ phân loại thông báo tại chỗ, tuyệt đối không xuất chuỗi lỗi gốc.
        message = str(exc).lower()
        if 'deadline' in message and ('too short' in message or 'minimum allowed deadline' in message):
            return 'HTTP_400_DEADLINE'
    if isinstance(code, int) and 400 <= code <= 599:
        return f'HTTP_{code}'
    if isinstance(exc, CloudBudgetError): return 'CLOUD_BUDGET'
    if isinstance(exc, TimeoutError) or 'timeout' in type(exc).__name__.lower():
        return 'TIMEOUT'
    return 'SDK_ERROR'


def finish_status(response):
    candidates = getattr(response, 'candidates', None) or []
    if not candidates:
        return 'NO_CANDIDATES'
    reason = getattr(candidates[0], 'finish_reason', None)
    reason = getattr(reason, 'value', reason)
    # Không ghi dữ liệu tùy ý từ phản hồi vào log.
    return reason if isinstance(reason, str) and reason in {
        'STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION', 'OTHER', 'BLOCKLIST',
        'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL'
    } else 'UNKNOWN'


class RegulationSearch:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        pdf = self.root/'regulations'/PDF
        with pdf.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != PDF_SHA:
                raise ValueError('PDF quy chế đã thay đổi; phải đối chiếu lại kho bằng chứng.')
        if hashlib.sha256((self.root/'regulations'/SUPPLEMENT).read_bytes()).hexdigest()!=SUPPLEMENT_SHA:
            raise ValueError('Bản tổng hợp thôi học đã thay đổi; cần đối chiếu lại.')
        path = self.root/'regulations/quy_che_chuan_hoa.tsv'
        self.corpus_sha = hashlib.sha256(path.read_bytes()+SUPPLEMENT_SHA.encode()).hexdigest()
        with path.open(encoding='utf-8',newline='') as f:
            self.chunks = list(csv.DictReader(f,delimiter='\t'))
        for c in self.chunks:
            c.update(article=int(c['article']), id=f"D{c['article']}.K{c['clause']}")
        if {page for c in self.chunks for page in span(c['pages'])} != set(range(5,28)):
            raise ValueError('Kho bằng chứng chưa phủ đủ trang nội dung PDF 5–27')
        if set(c['article'] for c in self.chunks) != set(range(1,28)):
            raise ValueError('Kho quy chế chưa đủ 27 điều')
        if len(set(c['id'] for c in self.chunks)) != len(self.chunks):
            raise ValueError('Trùng đoạn quy chế')
        self.bags = [Counter(tokens(c['title']+' '+c['title']+' '+c['text'])) for c in self.chunks]
        self.lengths = np.array([sum(b.values()) for b in self.bags])
        self.avg = float(self.lengths.mean())
        self.df = Counter(w for b in self.bags for w in b)
        self.client = None
        self.matrix = None
        self.lock = threading.Lock()
        self.matrix_lock = threading.Lock()
        self.matrix_model = None

    def _cloud(self):
        # Một file .env KHÔNG commit có thể được nhóm chia sẻ nội bộ. Biến môi
        # trường thật luôn ưu tiên và chế độ cloud vẫn phải được bật rõ ràng.
        from dotenv import load_dotenv
        load_dotenv(self.root / '.env', override=False)
        # Không gửi hồ sơ, lịch sử chat, token hay câu hỏi gốc cho LLM.
        if os.getenv('CMC_RAG_CLOUD') != '1' or not os.getenv('GEMINI_API_KEY'):
            return None
        with self.lock:
            if self.client is None:
                from google import genai
                from google.genai import types
                options={'timeout':CLOUD_SDK_TIMEOUT_MS}
                if hasattr(types,'HttpRetryOptions'):
                    options['retry_options']=types.HttpRetryOptions(attempts=1)
                self.client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(**options))
        return self.client

    def _embed(self, texts, task):
        from google.genai import types
        response = retry_transient(lambda: cloud_call(lambda: self._cloud().models.embed_content(model=os.getenv('CMC_EMBED_MODEL','gemini-embedding-001'), contents=texts,
                config=types.EmbedContentConfig(task_type=task,output_dimensionality=768))))
        m = np.asarray([x.values for x in response.embeddings],dtype=float)
        if m.shape != (len(texts),768) or not np.isfinite(m).all():
            raise ValueError('Embedding không hợp lệ')
        norms = np.linalg.norm(m,axis=1,keepdims=True)
        if (norms==0).any(): raise ValueError('Embedding rỗng')
        return m/norms

    def _semantic(self, focus):
        model=os.getenv('CMC_EMBED_MODEL','gemini-embedding-001')
        key=hashlib.sha256(f'{self.corpus_sha}|{model}|768|RETRIEVAL_DOCUMENT|v1'.encode()).hexdigest()
        path=self.root/'data/kiem_tra_rag/cache'/f'{key}.npy'
        with self.matrix_lock:
            if self.matrix is None or self.matrix_model!=model:
                matrix=None
                try:
                    candidate=np.load(path,allow_pickle=False)
                    if candidate.shape==(len(self.chunks),768) and np.isfinite(candidate).all() and np.allclose(np.linalg.norm(candidate,axis=1),1,atol=1e-5): matrix=candidate
                except (OSError,ValueError): pass
                if matrix is None:
                    matrix=self._embed([c['title']+' '+c['text'] for c in self.chunks], 'RETRIEVAL_DOCUMENT')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npy',delete=False) as tmp:
                        np.save(tmp,matrix,allow_pickle=False); name=tmp.name
                    os.replace(name,path)
                self.matrix,self.matrix_model=matrix,model
            matrix=self.matrix
        return matrix @ self._embed([focus], 'RETRIEVAL_QUERY')[0]

    def retrieve(self, question, limit=5, cloud=False, trace=None):
        arts, clauses = numbers(question)
        if clauses and not arts: return []
        if len(arts)>1 and clauses: return []
        if arts:
            return [c for c in self.chunks if c['article'] in arts and (not clauses or any(k in span(c['clause']) for k in clauses))]
        if unsupported(question):
            return []
        query0=normalize_query(question)
        if any(x in query0 for x in ('cac hinh thuc thoi hoc','cac con duong dan den thoi hoc','phan biet cac loai thoi hoc')):
            wanted=('D18.K2','D25.K4–6','D21.K2–3','D12.K4','D21.K1')
            return [next(c for c in self.chunks if c['id']==key) for key in wanted]
        if 'quyen loi sau thoi hoc' in query0:
            return [next(c for c in self.chunks if c['id']==key) for key in ('D20.K5–6','D19.K1–5','D18.K3–4')]
        query = normalize_query(question)
        focus = public_focus(question)
        strong = topic_articles(question)
        # Độ dài tiết học là Điều 5, không cần hai từ hiếm trùng với
        # đoạn tóm lược (câu 'kéo dài bao lâu' thường không có trong PDF).
        if 5 in strong and re.search(r'\bmoi\s+tiet\b',query):
            return [c for c in self.chunks if c['article']==5][:limit]
        q = set(tokens(query))
        original = set(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', question.lower())))
        accented = {w for w in original if w != strip_accents(w) and strip_accents(w) in q}
        phrase = ' '.join(tokens(query))
        if not q: return []
        if not (strong or focus):
            # Với câu chưa xác định được chủ đề, chỉ nhận cụm từ có nghĩa xuất hiện
            # nguyên cụm trong nhan đề; không dựa vào vài từ chung như sinh viên/học.
            if not any(re.search(r'\b'+re.escape(strip_accents(c['title']))+r'\b',query) for c in self.chunks):
                return []
        scores = []
        for i,b in enumerate(self.bags):
            score = sum(math.log(1+(len(self.bags)-self.df[w]+.5)/(self.df[w]+.5))*b[w]*2.5/(b[w]+1.5*(.25+.75*self.lengths[i]/self.avg)) for w in q if b[w])
            title = ' '.join(tokens(self.chunks[i]['title']))
            if phrase and len(phrase.split()) >= 2 and phrase in title: score += 12
            if accented:
                original_doc = set(re.findall(r'[^\W_]+',self.chunks[i]['text'].lower()+' '+self.chunks[i]['title'].lower()))
                if not accented & original_doc and query == strip_accents(question): score=0
            scores.append(score)
        # Lọc từ hiếm/trùng chủ đề trước, không lấy kết quả ngẫu nhiên khi không có căn cứ.
        ranking = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        candidates = [i for i in ranking if scores[i]>=max(2.,scores[ranking[0]]*.50) and (len(q & self.bags[i].keys())>=2 or (len(q)==1 and any(len(w)>=4 and self.df[w]<=2 for w in q & self.bags[i].keys())))]
        if strong:
            candidates=[i for i in candidates if self.chunks[i]['article'] in strong]
        if cloud and focus:
            try:
                if self._cloud():
                    similarities = self._semantic(focus)
                    if trace is not None: trace['embedding_used']=True
                    # Tìm ngữ nghĩa trên toàn corpus, không giới hạn bởi BM25.
                    semantic=[i for i in np.argsort(-similarities) if similarities[i]>=.55 and (not strong or self.chunks[i]['article'] in strong)][:limit]
                    candidates=list(dict.fromkeys(candidates+semantic))
                    candidates.sort(key=lambda i:scores[i]/max(1,max(scores))+.65*float(similarities[i]),reverse=True)
            except Exception as exc:
                code = safe_api_error(exc)
                if trace is not None: trace['embedding_error'] = code
                logging.getLogger(__name__).warning('Embedding không khả dụng (%s); dùng BM25.', code)
        if len(strong)>1:
            # Bảo toàn ít nhất một bằng chứng cho mỗi chủ đề nhận diện được,
            # không để một chủ đề có nhiều đoạn lấn hết chủ đề khác.
            first=[]
            for a in sorted(strong):
                ranked=[i for i in candidates if self.chunks[i]['article']==a]
                if not ranked:
                    ranked=[i for i in ranking if self.chunks[i]['article']==a and scores[i]>0]
                if ranked: first.append(ranked[0])
            first.sort(key=lambda i: scores[i],reverse=True)
            candidates=list(dict.fromkeys(first+candidates))
        return [self.chunks[i] for i in candidates[:max(limit,len(strong))]]

    def answer(self, question, cloud=False):
        # Deadline tổng thể cho khách; tối đa hai worker, không xếp hàng vô hạn.
        if not cloud:
            return self._answer(question,False)
        if not _cloud_slots.acquire(blocking=False):
            result=self._answer(question,False);result['generation_status']='CLOUD_BUSY';return result
        def work():
            token=_deadline.set(time.monotonic()+CLOUD_DEADLINE)
            try: return self._answer(question,True)
            finally:
                _deadline.reset(token)
                _cloud_slots.release()
        try:
            pending=_cloud_pool.submit(work)
        except BaseException:
            _cloud_slots.release()
            raise
        try: return pending.result(timeout=CLOUD_DEADLINE)
        except FutureTimeout:
            # SDK đồng bộ đang chạy không thể cưỡng bức hủy; slot được trả khi worker kết thúc.
            result=self._answer(question,False);result['generation_status']='DEADLINE';return result

    def _answer(self, question, cloud=False):
        trace={'embedding_used':False,'generation_status':'NOT_REQUESTED'}
        # Học phí là hoàn cảnh, không mặc nhiên là câu hỏi về đơn giá: ưu tiên
        # phần hỏi nghỉ học/bảo lưu có căn cứ ở Điều 21. Không suy diễn học phí.
        if (not numbers(question)[0] and 'hoc phi' in strip_accents(question)
                and 21 in topic_articles(question)
                and unsupported(question)):
            policy = self._answer('Điều kiện nghỉ học tạm thời bảo lưu', False)
            if policy['regulations']:
                boundary = ('ngày hoặc hạn đóng học phí' if is_tuition_deadline_query(question)
                            else 'mức tiền hoặc chính sách học phí cụ thể')
                policy['reply'] = ('<p>Quy chế này không cung cấp '+boundary+'; '
                                   'hãy đối chiếu thông báo học phí của trường. Về nghỉ học tạm thời/bảo lưu, '
                                   'cần đáp ứng các điều kiện tại Điều 21:</p>' + policy['reply'])
                return policy
        # Với câu ghép có một chủ đề ngoài phạm vi: chỉ trả nguồn của phần có
        # căn cứ, không gửi câu thô hay tự suy đoán phần thiếu.
        if not numbers(question)[0] and unsupported(question):
            parts=re.split(r'(?i)\s+(?:và|va|đồng thời)\s+|[;?]+',question)
            supported=[self._answer(part,False) for part in parts if part.strip() and not unsupported(part) and topic_articles(part)]
            supported=[out for out in supported if out['regulations']]
            if supported:
                evidence=list({c['id']:c for out in supported for c in out['regulations']}.values())
                return {'reply':'<p>Một phần câu hỏi nằm ngoài phạm vi quy chế; chỉ hiển thị các nội dung có căn cứ dưới đây.</p>'+''.join(out['reply'] for out in supported),'regulations':evidence,'retrieved':evidence,'method':'bm25',**trace}
        chunks = self.retrieve(question,cloud=cloud,trace=trace)
        requested,clauses=numbers(question)
        missing=sorted(set(requested)-{c['article'] for c in self.chunks})
        missing_clauses=sorted(k for k in set(clauses) if not any(k in span(c['clause']) for c in chunks)) if len(requested)==1 else []
        diagnostics=''
        if missing: diagnostics+='<p>Không có căn cứ cho Điều '+', '.join(map(str,missing))+' trong kho đang sử dụng.</p>'
        if missing_clauses: diagnostics+='<p>Không tìm thấy Khoản '+', '.join(map(str,missing_clauses))+' của Điều '+str(requested[0])+' trong kho đang sử dụng.</p>'
        if len(requested)>1 and clauses: diagnostics+='<p>Hãy nêu từng Điều và Khoản riêng để tránh ghép nhầm.</p>'
        if not chunks:
            if not requested and is_tuition_deadline_query(question):
                return {'reply':diagnostics+'<p>Quy chế đào tạo đang sử dụng không có lịch hoặc ngày đến hạn đóng học phí của kỳ cụ thể. Hãy xem thông báo học phí chính thức hoặc liên hệ bộ phận tài chính của trường; không suy đoán ngày từ Điều/Khoản không liên quan.</p>', 'regulations':[], 'method':'none',**trace}
            if not requested and is_tuition_amount_query(question):
                return {'reply':diagnostics+'<p>Quy chế đào tạo (QĐ 22/2023) không quy định mức tiền học phí cụ thể. Vui lòng tra cứu Thông báo học phí của nhà trường tương ứng khóa học, ngành và chương trình đào tạo.</p>','regulations':[], 'method':'none',**trace}
            if not requested and is_tuition_exemption_query(question):
                return {'reply':diagnostics+'<p>Quy chế đào tạo (QĐ 22/2023) không quy định chính sách miễn, giảm học phí cụ thể. Vui lòng tra cứu thông báo học phí và quy định hỗ trợ sinh viên của nhà trường.</p>','regulations':[], 'method':'none',**trace}
            return {'reply':diagnostics+'<p>Chưa tìm thấy căn cứ trong quy chế. Hãy nêu Điều, Khoản hoặc chủ đề cụ thể.</p>','regulations':[], 'method':'none',**trace}
        evidence = [{'id':c['id'],'article':c['article'],'clause':c['clause'],'pages':c['pages'],'document':PDF,'title':c['title'],'text':c['text']} for c in chunks]
        for c in evidence:
            if c['id'] in SUPPLEMENT_PAGES:
                c['supplement']={'document':SUPPLEMENT,'pages':SUPPLEMENT_PAGES[c['id']],'kind':'Bản tổng hợp tham khảo'}
        method = 'bm25'
        intro = ''
        retrieved=list(evidence)
        if cloud and len(chunks)<=12 and not missing and not missing_clauses:
            code = 'SDK_ERROR'
            try:
                client = self._cloud()
                trace['generation_status'] = 'DISABLED'
                if client:
                    trace['generation_status'] = 'REQUESTED'
                    from google.genai import types
                    # Không gửi câu hỏi gốc hoặc dữ liệu cá nhân ra ngoài. LLM chỉ diễn giải chủ đề công khai được truy xuất.
                    prompt = ('Tóm lược bằng tiếng Việt các bằng chứng quy chế dưới đây. Không thêm quy định, số liệu hoặc kết luận cá nhân. '
                              'Nếu thiếu căn cứ, nói rõ. Trả JSON gồm answer (chuỗi) và citations (mảng ID chỉ từ bằng chứng).\n'
                              'Ý định công khai: '+ (public_focus(question) or '; '.join(dict.fromkeys(c['title'] for c in evidence)))+'\n'+json.dumps(evidence,ensure_ascii=False))
                    allowed = {c['id'] for c in evidence}
                    schema = {'type':'OBJECT','properties':{
                        'answer':{'type':'STRING'},
                        'citations':{'type':'ARRAY','items':{'type':'STRING','enum':sorted(allowed)}}
                    },'required':['answer','citations']}
                    r = retry_transient(lambda: cloud_call(lambda: client.models.generate_content(model=os.getenv('CMC_LLM_MODEL','gemini-3.1-flash-lite'),contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type='application/json',response_schema=schema,temperature=0,max_output_tokens=4096))))
                    # Một số model dùng chung ngân sách token cho suy luận và đầu ra.
                    status = finish_status(r)
                    if status not in ('STOP','UNKNOWN','NO_CANDIDATES'):
                        code = status
                        raise ValueError('Phản hồi không hoàn tất')
                    code = 'EMPTY_RESPONSE' if status in ('UNKNOWN','STOP','NO_CANDIDATES') else status
                    raw = r.text
                    if not isinstance(raw,str) or not raw.strip():
                        raise ValueError('Phản hồi rỗng')
                    code = 'INVALID_JSON'
                    obj = json.loads(raw)
                    code = 'INVALID_CITATIONS'
                    ids = obj.get('citations') if isinstance(obj,dict) else None
                    if not isinstance(ids,list) or not ids or not all(isinstance(x,str) and x in allowed for x in ids) or not isinstance(obj.get('answer'),str) or not obj['answer'].strip():
                        raise ValueError('Trích dẫn không hợp lệ')
                    intro = '<p><b>Diễn giải AI (đối chiếu bằng chứng bên dưới):</b> '+html.escape(obj['answer'][:6000])+' ['+html.escape(', '.join(ids))+']</p>'
                    # Giữ toàn bộ đoạn khi người dùng tra cứu Điều/Khoản trực tiếp.
                    if not requested:
                        cited=[c for c in evidence if c['id'] in ids]
                        # Không để LLM bỏ mất một chủ đề đã truy xuất; giữ bằng chứng
                        # trực tiếp và báo fallback nếu thiếu nguồn của một chủ đề.
                        if len(topic_articles(question))>1 and {c['article'] for c in cited}!={c['article'] for c in evidence}:
                            intro=''
                            raise ValueError('Thiếu nguồn cho một chủ đề')
                        evidence=cited
                    method='gemini'
                    trace['generation_status']='OK'
            except Exception as exc:
                # Không ghi exception gốc, nội dung phản hồi, prompt hoặc khóa API.
                if code == 'SDK_ERROR': code = safe_api_error(exc)
                trace['generation_status']=code
                logging.getLogger(__name__).warning('LLM không khả dụng (%s); trả bằng chứng trực tiếp.',code)
        intro=diagnostics+intro
        if len(chunks)>12:
            intro='<p>Truy vấn nhiều Điều: hiển thị đầy đủ các đoạn tóm lược, không yêu cầu AI rút gọn.</p>'+intro
        blocks = []
        for c in evidence:
            page = c['pages'].split('–')[0]
            extra=c.get('supplement')
            supplement_link=f"<p><a href='/regulations/{SUPPLEMENT}#page={extra['pages'].split('–')[0]}' target='_blank' rel='noopener noreferrer'>Bản tổng hợp tham khảo · trang {extra['pages']}</a></p>" if extra else ''
            blocks.append(f"<div class='border border-slate-600 rounded-lg p-3'><b>Điều {c['article']}, Khoản {c['clause']}: {html.escape(c['title'])}</b><p>{html.escape(c['text'])}</p><p><a href='/regulations/{PDF}#page={page}' target='_blank' rel='noopener noreferrer'>QĐ 22/2023 · trang PDF {c['pages']}</a></p>{supplement_link}</div>")
        return {'reply':intro+''.join(blocks)+'<p>Bản tóm lược đã đối chiếu ảnh trang; văn bản PDF gốc là căn cứ. Áp dụng từ năm học 2023–2024.</p>', 'regulations':evidence, 'retrieved':retrieved, 'method':method,'retrieval_method':'hybrid' if trace['embedding_used'] else 'bm25',**trace}
