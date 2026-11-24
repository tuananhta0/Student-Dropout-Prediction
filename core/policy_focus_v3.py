"""Public clause facets. No question-bank IDs, private data or raw cloud prompts."""
import re
from core.chatbot_engine import strip_accents

# Concept vocabulary, not full questions/answers. Each group is an AND of ORs.
FACETS = [
 ('D18.K1', ('canh bao',), ('gpa','tin chi','no dong','nhieu ly do','nhieu tieu chi','muc nao')),
 ('D18.K2', ('canh bao','duoi hoc','buoc thoi hoc'), ('lien tiep','khong lien tiep','may lan','bao nhieu lan','so lan')),
 ('D3.K5–6', ('cu nhan','ky su','chuong trinh'), ('thoi gian thiet ke','thoi gian hoc toi da','hoc toi da','bao nhieu hoc ky','bao nhieu nam')),
 ('D21.K1', ('nghi','bao luu','tro lai','quay lai','hoc lai sau khi nghi'), ('thai san','om','dieu tri','tai nan','ly do ca nhan','nop don','giay to','6 hoc ky','sau khi nghi')),
 ('D22.K1–3', ('chuyen sang nganh','chuyen nganh','doi nganh','doi sang nganh','chuyen truong','chuyen sang truong'),),
 ('D23.K1–2', ('song bang','hai chuong trinh','2 bang','nganh thu hai','hoc 2 nganh','hoc hai nganh'),),
 ('D12.K1', ('dang ky','dang ki'), ('truoc khi hoc ky','truoc ky','thoi han','han dang ky','bao lau truoc')),
 ('D12.K2', ('dang ky','dang ki'), ('tin chi','khoi luong','lop thay the','huy hoc phan','con cho','tai khoan')),
 ('D12.K3', ('khoa luan','do an tot nghiep'), ('dang ky','gpa','bao ve','gia han','huong dan','hoan thanh','hoi dong','khong dat','lam lai')),
 ('D12.K4', ('rut mon','rut hoc phan','dung lop hoc phan','tu y bo hoc','thoi han rut','ngung hoc phan'),),
 ('D13.K2', ('bao ve','khoa luan'), ('truc tuyen','online','ghi hinh')),
 ('D13.K3', ('diem p','diem i','diem x','diem r'), ('tinh gpa','tinh vao','thang diem','nghia la')),
 ('D3.K2', ('hoc phan tuong duong','mon tuong duong'), ('noi dung','tin chi','phan tram')),
 ('D6.K1–6', ('tai khoan','mat khau'), ('trach nhiem','bao mat')),
 ('D13.K1', ('diem liet','diem thi cuoi ky'), ('bao nhieu','nguong','2 diem','duoi','bang','tro xuong')),
 ('D13.K3', ('diem chu','thang 10','he 10','trong so','ty trong','lam tron diem hoc phan','diem hoc phan','diem tong ket'), ('diem','thang','lam tron','50','70','tuong ung','khoang')),
 ('D14.K2', ('vang thi','vang buoi thi','vang mat buoi thi','thi bu','hai giang vien cham','2 giang vien cham'),),
 ('D14.K3', ('thi lai','thi bo sung'), ('diem toi da','diem cao nhat','toi da','gioi han')),
 ('D16.K1–3', ('phuc tra','phuc khao','cham lai diem','khieu nai diem','khieu nai','chua dung'), ('diem','bai thi','ket qua','don','bao lau','giang vien','nop')),
 ('D17.K1', ('thang 4','he 4','thang diem 4'), ('quy doi','tuong ung','diem chu','diem a','diem b','diem c','diem d','diem f')),
 ('D17.K2', ('gpa','diem trung binh'), ('cong thuc','lam tron','tinh nhu the nao','duoc tinh','tinh tu')),
 ('D17.K3–4', ('xep loai','hoc luc','trinh do nam','nam thu may'), ('gpa','tin chi','tich luy','diem trung binh')),
 ('D19.K6', ('chuan ngoai ngu','chuan tieng anh','b1','b2','cefr','bac 3/6','bac 4/6'), ('sm','gm','he','tuong duong','cefr','chuan')),
 ('D19.K1–5', ('cong nhan','chuyen doi','mien hoc'), ('tin chi','hoc phan','ket qua hoc tap')),
 ('D20.K4', ('hang tot nghiep','ha hang','xep hang tot nghiep'), ('hoc lai','ky luat','canh cao','5%','giam','tut')),
 ('D20.K5–6', ('het thoi gian','thoi hoc'), ('thieu chung chi','thieu ngoai ngu','chung nhan','quyen loi')),
 ('D25.K1', ('nghi co phep','nghi khong phep','vang co phep','vang khong phep','tong nghi','cam thi'), ('phan tram','bao nhieu','qua','hoc lai','%')),
 ('D25.K2–3', ('phong thi','bai thi'), ('dung tai lieu','su dung tai lieu','mang tai lieu','den muon','vao muon','15 phut','trao doi')),
 ('D25.K4–6', ('phong thi','thi cu','thi ho','nho thi ho','ho so gia'), ('dien thoai','thiet bi','tat nguon','thi ho','ho so gia','bang gia','chung chi gia')),
 ('D26.K1–2', ('luu tru','luu bao lau','luu giu','luu vinh vien'), ('khoa luan','bang diem','ho so','quyet dinh')),
 ('D27.K1–3', ('quy che','quy dinh'), ('ap dung tu nam','hieu luc','nam hoc nao','khoa 2022')),
 ('D1.K1–3', ('dao tao tu xa','lien ket dao tao','dao tao lien ket'), ('ap dung','pham vi','doi tuong')),
 ('D7.K2', ('co van hoc tap',), ('trach nhiem','nhiem vu','vai tro','tu van','ho tro gi')),
 ('D8.K1–2', ('ke hoach dao tao','thoi khoa bieu'), ('cong bo','thong bao','truoc','ban hanh')),
]

def has(q, words):
    return any(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)', q) for w in words)

def normalize_chat(text):
    # Preserve accent-bearing text, numeric data, IDs and course codes.
    mapping={'ko':'không','k':'không','khum':'không','hok':'không','đk':'đăng ký','dk':'đăng ký','tc':'tín chỉ','cbht':'cảnh báo học tập','ctsv':'công tác sinh viên','hk':'học kỳ','sv':'sinh viên','đuổi học':'buộc thôi học'}
    text=re.sub(r"(?<=\d)\s+tín\b(?!\s+chỉ\b)", " tín chỉ", text, flags=re.I)
    # Single k is too ambiguous (grade/symbol); leave untouched.
    mapping.pop('k')
    return re.sub(r'(?<!\w)(?:'+ '|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))+r')(?!\w)',lambda m:mapping[m[0].lower()],text,flags=re.I)

def facet_ids(question):
    q=strip_accents(normalize_chat(question))
    # Public policy only: no reading a personal metric, no statistics or hypotheticals.
    if has(q, ('cua em','cua toi','cua minh','ho so','du bao','nguy co','ty le sinh vien','bao nhieu sinh vien','thong ke','so luong sinh vien','toan truong','bao nhieu em','da nghi','da bi','can dat','muc tieu','kich ban','what-if','gia lap','diem cua','hoc lai de cai thien')):
        return []
    if not has(q, ('bao nhieu','may','the nao','nhu the nao','ra sao','gi','khong','duoc','can','phai','khi nao','bao lau','toi da','toi thieu','cong thuc','lam tron','ap dung','tuong ung','quy doi','xu ly','chuyen nganh','phuc tra','phuc khao','dieu kien','quy dinh','quy che')):
        return []
    ids=[key for key,*groups in FACETS if all(has(q,g) for g in groups)]
    if has(q,('buoi','tiet','gio','vang thi','vang khong phep')) and not has(q,('hoc ky','bao luu')): ids=[k for k in ids if k!='D21.K1']
    if has(q,('giu nguyen ma sinh vien',)): ids=['D9.K1–2']
    if has(q,('khoi luong','phan tram')) and has(q,('cong nhan',)): ids=[k for k in ids if not k.startswith('D12.')]
    if has(q,('bao ve',)) and has(q,('online','truc tuyen','ghi hinh')): ids=[k for k in ids if not k.startswith('D12.')]
    if has(q,('trach nhiem',)) and has(q,('tai khoan',)): ids=[k for k in ids if not k.startswith('D12.')]
    # Only narrow selections; conflicting facets are left to the existing multi-topic engine.
    return list(dict.fromkeys(ids)) if len(set(ids))<=2 else []


def public_facets(question):
    """Return only fixed vocabulary for a recognized facet; never copy input."""
    q=strip_accents(normalize_chat(question)); ids=set(facet_ids(question))
    return list(dict.fromkeys(term for key,*groups in FACETS if key in ids for group in groups for term in group if has(q,(term,))))


# Narrow public question facets. These express concepts, not question-bank IDs.
# Prefer a precise clause to generic topic words such as GPA, thi or tin chi.
# A match is a retrieval hint, not a claim that the student's facts are verified.
def specific_facets(question):
    q=strip_accents(normalize_chat(question))
    if (q.count('?')>1 or ';' in q or has(q,('dong thoi','ngoai ra','them nua','con neu','va neu','toan truong','ty le sinh vien','bao nhieu sinh vien','thong ke','du bao','nguy co','ho so','kich ban','gia lap','what-if','muc tieu','cua toi','cua minh'))):
        return []
    if re.search(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6}|[A-Z]{2}\d{2}-\d{2,3})\b',question.upper()):
        return []
    found=[]
    def add(ids,groups,exclude=()):
        if all(has(q,g) for g in groups) and not has(q,exclude):
            found.append(ids)
    # A certificate-debt question concerns the warning criterion, not the
    # generic definition of a credit. Keep compound and personal requests separate.
    add(('D18.K1',), [('chung chi',),('no dong','tin chi no','no tin chi'),('tinh vao','co tinh','tinh ca','khong ke','loai tru','co ke')], ('cua em','cua toi','cua minh','va','hoac','nhung','gpa','tot nghiep','hoc phi','buoc thoi hoc'))
    add(('D13.K2',), [('danh gia truc tuyen','danh gia online','kiem tra truc tuyen'),('trong so','phan tram','%','ty trong'),('toi da','bao nhieu','gioi han','vuot')], ('va','hoac','nhung','thi lai','diem liet'))
    add(('D21.K1',), [('chua hoc ky nao','chua hoc hoc ky nao','chua hoc mot ky','chua hoc het ky dau','chua hoc du mot ky'),('nghi','bao luu')], ('vang thi','cam thi','nghi le','nghi tet'))
    add(('D17.K2',), [('gpa tich luy','diem trung binh tich luy'),('gpa ky','gpa hoc ky','diem trung binh hoc ky'),('khac','phan biet','giong','khac nhau','so sanh')], ('cua em','ky toi','ky sau','tang','giam'))
    year_query=re.sub(r'\btich luy \d+ tin chi\b','tich luy tin chi',q)
    if year_query!=q and has(year_query,('trinh do nam','nam thu may','nam may','thuoc dien nao','thuoc dien gi')) and not has(q,('thieu','con lai','du dieu kien tot nghiep','can bao nhieu')):
        found.append(('D17.K3–4',))
    else: add(('D17.K3–4',), [('tin chi tich luy','tich luy tin chi'),('thuoc dien nao','thuoc dien gi','trinh do nam','nam thu may','nam may')], ('thieu','con lai','du dieu kien tot nghiep','can bao nhieu'))
    add(('D20.K5–6',), [('het thoi gian hoc','het thoi gian dao tao','qua thoi gian hoc toi da'),('thieu','chua co','chua dat'),('gdtc','gdqp','gdqp-an','giao duc the chat','quoc phong','ngoai ngu','tieng anh','chung chi')])
    add(('D20.K1','D19.K6'), [('tieng anh','ngoai ngu'),('tot nghiep',),('thieu','chua dat','khong dat','chua co')], ('het thoi gian','hoc toi da','gpa','tin chi','nghia vu tai chinh','no hoc phi','em da'))
    add(('D14.K2',), [('vang thi','vang buoi thi','vang mat buoi thi','vang mat trong buoi thi','vang mat trong ky thi','bo thi'),('ly do','diem may','diem gi','thi bu')], ('cam thi','nghi hoc','phuc tra'))
    add(('D14.K1',), [('giang vien','danh sach'),('cam thi','khong du dieu kien','du dieu kien du thi'),('gui danh sach','bao lau','may ngay','han gui','thoi han gui')], ('ty le','bao nhieu em'))
    add(('D23.K1–2',), [('song bang','hai chuong trinh','2 bang','nganh thu hai','hoc hai nganh','hoc 2 nganh'),('gpa','hoc luc','dieu kien','co duoc','duoc khong')], ('chuyen nganh','chuyen truong','tot nghiep','ky sau','muc tieu','tang','nang','cai thien'))
    add(('D3.K5–6',), [('cu nhan','ky su'),('tin chi',),('khoang nao','dao dong','bao nhieu','khoi luong')], ('dang ky','no dong','truot','tich luy','hoc phi','hoc phan bat buoc','mon bat buoc'))
    add(('D11.K1–5',), [('hoc online','hoc truc tuyen'),('toan bo','100%','het chuong trinh','ca chuong trinh','toi da','bao nhieu phan tram')], ('bao ve','khoa luan','danh gia','thi online','diem'))
    add(('D12.K4',), [('khong di hoc','khong hoc mon','bo hoc phan','ngung hoc','dung hoc'),('mon do nua','mon nay nua','hoc phan do nua','diem sao','nhan diem gi')], ('nghi mot buoi','vang buoi','nghi khong phep','cam thi','vang thi','ty le','phan tram'))
    # Conflicts retain the existing multi-topic retrieval logic.
    return list(found[0]) if len(found)==1 else []


# Public projection labels are constants; never include the question or values.
SPECIFIC_ASPECTS={
    'D18.K1':'môn chứng chỉ có được tính vào tổng tín chỉ nợ đọng khi xét cảnh báo học tập hay không; phân biệt ngưỡng vượt 24 với đúng 24',
    'D13.K2':'giới hạn tỷ trọng đánh giá trực tuyến trong điểm học phần và ngoại lệ bảo vệ khóa luận',
    'D21.K1':'điều kiện nghỉ trước khi học đủ một kỳ, phân biệt lý do cá nhân và các trường hợp đặc biệt',
    'D17.K2':'phân biệt phạm vi tính GPA học kỳ và GPA tích lũy',
    'D17.K3–4':'xếp trình độ năm học theo tín chỉ tích lũy, không kết luận đủ điều kiện tốt nghiệp',
    'D20.K5–6':'hoàn thiện chứng chỉ hoặc chuẩn ngoại ngữ sau hết thời gian tối đa',
    'D20.K1':'điều kiện tốt nghiệp khi chưa đạt chuẩn ngoại ngữ',
    'D14.K2':'vắng thi có hoặc không có lý do chính đáng',
    'D14.K1':'hạn gửi danh sách đủ và không đủ điều kiện dự thi tính từ khi kết thúc giảng dạy',
    'D23.K1–2':'điều kiện đăng ký ngành thứ hai theo học lực và tín chỉ',
    'D3.K5–6':'khung tổng tín chỉ chương trình cử nhân và kỹ sư, các nội dung không tính vào khung',
    'D11.K1–5':'giới hạn tỷ lệ thời lượng học trực tuyến của chương trình',
    'D12.K4':'điểm và học phí khi ngừng học phần, phân biệt rút đúng hạn với tự ý bỏ học',
}


def ambiguous_penalty(question):
    q=strip_accents(normalize_chat(question)).strip()
    return bool(re.fullmatch(r'(?:the |vay |con )?(?:lan|lan thu) (?:2|hai) (?:thi )?(?:bi )?tru bao nhieu diem(?: a| nhi| vay)?[?.! ]*',q))
