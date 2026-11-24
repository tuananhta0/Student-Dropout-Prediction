"""Focused local queries; no raw question or private records sent to Gemini."""
import html
import re
import pandas as pd
from core.chatbot_engine import MAJOR_ALIASES, row_to_clean_dict, risk_display
from core.security import require_role, require_major, scoped_frame, permitted_student
from core.retrieval import unsupported
from core.academic_rules import number, warning_reasons, GRADE_POINTS


def response(text, intent):
    return {'reply': '<p>'+html.escape(text)+'</p>', 'regulations': [], 'intent': intent, 'method': 'python'}


def has(q, *words):
    return any(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)', q) for w in words)


def scoped_analytics(engine, principal, q, parsed, classes):
    """Only explicit, supported aggregate queries. Refuse ambiguous dimensions.

    Latest states are not incidence rates; historical semester rows are GPA only.
    No causal estimates, no inferred exit reasons, no unrestricted aggregates.
    """
    if unsupported(q): return None
    # Require a population statistic, never a public rule or individual request.
    if not (has(q, 'ty le', 'phan tram', 'trung binh', 'tong so', 'thong ke', 'phan bo', 'so luong', 'tong no', 'cao nhat', 'thap nhat') or re.search(r'\b(?:bao nhieu|may)\s+(?:sinh vien|sv|em|ho so)\b',q)):
        return None
    metric = None
    if has(q, 'nguy co', 'rui ro', 'dropout') and has(q, 'cao', 'rat cao'):
        metric = 'risk'
    elif has(q, 'canh bao'):
        metric = 'warning'
    elif has(q, 'tot nghiep'):
        metric = 'graduated'
    elif has(q, 'thoi hoc', 'bo hoc'):
        metric = 'dropout'
    elif has(q, 'gpa', 'diem trung binh'):
        metric = 'gpa'
    elif has(q, 'hoc phi', 'cong no'):
        metric = 'debt'
    elif has(q, 'sinh vien', 'sv'):
        metric = 'count'
    if not metric:
        return None
    if metric=='count' and not (has(q,'tong so sinh vien','tong so sv','so luong sinh vien','phan bo sinh vien') or re.search(r'\bbao nhieu\s+(?:sinh vien|sv)\s*(?:[?!.]|$)',q)):
        return None
    if sum((has(q,'canh bao'),has(q,'tot nghiep'),has(q,'thoi hoc','bo hoc') and metric!='risk',has(q,'gpa','diem trung binh'),has(q,'hoc phi','cong no'))) > 1:
        return None
    ask = has(q, 'bao nhieu', 'may em', 'may sv', 'ty le', 'phan tram', 'trung binh', 'tong so', 'thong ke', 'phan bo', 'so luong', 'tong no', 'cao nhat', 'thap nhat')
    population = bool(parsed['major_code'] or classes or has(q, 'sinh vien', 'sv', 'nganh', 'khoa', 'lop', 'toan truong', 'toan khoa', 'tung khoa', 'tung lop', 'tung nganh', 'he sm', 'he gm', 'bao nhieu em'))
    if not (ask and population):
        return None
    # Specific requests not implemented must not turn into unrelated totals.
    if has(q, 'vi sao', 'nguyen nhan', 'tac dong', 'anh huong', 'lien quan', 'hieu qua', 'lam giam', 'co giam', 'sau do', 'sau khi', 'truoc do', 'can thiep', 'hoc bong', 'do ky luat', 'buoc thoi hoc', 'tu xin', 'dung han', 'qua han', 'chua gap', 'chua tung', 'khong gap', 'moi hoc ky', 'tung hoc ky', 'theo hoc ky', 'nam qua', 'thang nay', 'hom nay', 'duoi', 'tren', 'so voi nam', 'chua co', 'hoan thanh', 'ngoai le', 'toi thieu', 'toi da', 'mon hoc', 'tung mon', 'hoc phan', 'diem liet', 'phuc hoi', 'nghi hoc', 'nha o', 'noi o', 'gioi tinh', 'phu dao', 'co van', 'lms', 'nop bai', 'lam them', 'tu hoc', 'chuyen can', 'trong nam', 'tuan', 'loai', 'top', 'sv nao', 'sinh vien nao', 'thi lai', 'hoc lai', 'ky luat', 'dang hoc', 'dang cho', 'tam nghi', 'can ho tro', 'chua', 'khong', 'moi nam', 'theo nam', 'phan bo diem', 'diem chu', 'diem f', 'diem a', 'chung chi', 'dang ky', 'tin chi', 'ket qua hoc tap', 'dau vao', 'du bao sai', 'bo sot', 'bao nham', 'nguy co thap', 'nguy co trung binh', 'nghi tam thoi', 'quay lai', 'xac nhan hanh chinh', 'dang bi', 'lien tiep', 'no dong', 'no trung binh', 'no bao nhieu tin', 'sau canh bao', 'o tro', 'o cung gia dinh', 'ktx', 'ho tro som', 'giu lai', 'co nen', 'cong bo', 'qua cac ky', 'song bang', 'hai chuong trinh', 'cung luc', 'hoc ky vua qua'):
        return None
    # Policy questions must not be interpreted as aggregates.
    if has(q, 'quy che', 'quy dinh', 'dieu kien', 'nguong', 'co duoc', 'phai lam gi', 'may lan', 'bao nhieu lan', 'muc hoc phi'):
        return None
    require_role(principal, 'advisor', 'admin')
    cohort_scope = (
        bool(re.search(r'\b20\d{2}\b', q))
        and has(q, 'qua cac khoa', 'theo khoa', 'tung khoa', 'khoa tuyen sinh')
    ) or has(q, 'qua cac khoa', 'cac khoa tuyen sinh', 'theo khoa tuyen sinh')

    if principal.role == 'advisor' and (
        has(q, 'toan truong', 'so voi truong', 'ca truong', 'cac nganh')
        or (has(q, 'cac khoa') and not cohort_scope)
    ):
        require_role(principal, 'admin')
    majors = {k for k, aliases in MAJOR_ALIASES.items() if any(has(q, a) for a in aliases)}
    majors.update(re.findall(r'\b(?:khoa|nganh)\s+('+'|'.join(k.lower() for k in MAJOR_ALIASES)+r')\b',q))
    majors = {m.upper() for m in majors}
    if parsed['major_code']:
        majors.add(parsed['major_code'])
    for major in majors:
        require_major(principal, major)
    if len(majors)>1 or (majors and has(q, 'so voi toan truong', 'so voi truong')):
        return response('Hãy nêu một phạm vi thống kê mỗi lần; so sánh giữa các phạm vi chưa được hỗ trợ ở truy vấn này.', 'clarify')
    term = parsed['term_filter']
    if term is not None and metric != 'gpa':
        return response('Chưa hỗ trợ chỉ số này theo học kỳ lịch sử. Có thể hỏi tại mốc dữ liệu gần nhất.', 'analytics')
    # Avoid mixing term-state semantics with current state. GPA historical only.
    frame = engine.latest_df if term is None else engine.df[engine.df.snapshot_term_index.eq(term)]
    frame = scoped_frame(principal, frame)
    if majors: frame = frame[frame.major_code.isin(majors)]
    if classes: frame = frame[frame.class_code.isin(classes)]
    years = set(map(int, re.findall(r'\b20\d{2}\b', q)))
    if len(years)>2: return response('Hãy nêu một khóa hoặc khoảng khóa cần thống kê.', 'clarify')
    if years:
        if 'cohort_year' not in frame: return response('Chưa có dữ liệu khóa tuyển sinh.', 'analytics')
        frame = frame[frame.cohort_year.between(min(years),max(years))]
    group = None
    if has(q, 'theo khoa', 'tung khoa', 'cac khoa', 'qua cac khoa'):
        group = 'cohort_year' if years or has(q,'khoa tuyen sinh','tung khoa','qua cac khoa','theo khoa') else 'major_code'
    if has(q, 'theo nganh', 'tung nganh', 'nganh nao'): group = 'major_code'
    if has(q, 'theo lop', 'tung lop', 'lop nao'): group = 'class_code'
    if has(q, 'he sm', 'he gm', 'theo he', 'sm va gm', 'gm va sm', 'sm voi gm', 'gm voi sm'): group = 'education_mode'
    if group=='education_mode' and not has(q,'theo he','sm va gm','gm va sm','sm voi gm','gm voi sm'):
        modes={m for m in ('SM','GM') if has(q,'he '+m.lower())}
        if len(modes)==1:
            if 'education_mode' not in frame: return response('Chưa có dữ liệu hệ đào tạo.', 'analytics')
            frame=frame[frame.education_mode.isin(modes)]
    if has(q, 'cu nhan', 'ky su'):
        group = 'degree_type'
        degrees={v for k,v in [('cu nhan','Bachelor'),('ky su','Engineer')] if has(q,k)}
        if len(degrees)==1:
            if 'degree_type' not in frame: return response('Chưa có dữ liệu bậc đào tạo.', 'analytics')
            frame=frame[frame.degree_type.isin(degrees)]
    if has(q, 'phuong thuc tuyen sinh', 'theo phuong thuc', 'hoc ba', 'thpt') or re.search(r'\bcmc\d{3}',q):
        if not has(q, 'theo phuong thuc','phan bo','tung phuong thuc'): return None
        group = 'admission_method'
    if has(q,'cao nhat','thap nhat','so voi','so sanh','khac nhau'):
        return response('Chưa hỗ trợ phép so sánh này; hãy hỏi từng nhóm và chỉ số cụ thể.', 'clarify')
    if group and group not in frame: return response('Chưa có dữ liệu phân nhóm được yêu cầu.', 'analytics')
    if frame.empty: return response('Không có hồ sơ trong phạm vi và kỳ/khóa được chọn.', 'analytics')
    def calc(g):
        n = len(g)
        if metric == 'count': return f'{n} hồ sơ có kết quả học kỳ'
        if metric in ('dropout','graduated'):
            k = int(g.state_as_of.eq('DROPOUT' if metric=='dropout' else 'GRADUATED').sum())
            return f'{k}/{n} hồ sơ ({k/n:.1%}) ở trạng thái '+('thôi học' if metric=='dropout' else 'tốt nghiệp')
        if metric == 'risk':
            active = g[g.state_as_of.eq('ENROLLED')]
            v = active[active.prediction_status.eq('VALID')]
            vals = pd.to_numeric(v.risk_probability, errors='coerce'); vals = vals[vals.between(0,1)]
            k = int((vals >= engine.meta['threshold']).sum())
            if has(q, 'rat cao'): k = int((vals >= max(.45,engine.meta['threshold'])).sum())
            label = 'rất cao' if has(q,'rat cao') else 'cao hoặc rất cao'
            if not len(vals): return f'Chưa có dự báo hợp lệ (0/{len(active)} SV đang học)'
            return f'{k}/{len(vals)} SV có nguy cơ {label} ({k/len(vals):.1%}); độ bao phủ {len(vals)}/{len(active)} SV đang học'
        if metric == 'warning':
            col = 'warnings_total_to_date' if has(q,'tu dau khoa','da bi','tung bi','qua cac khoa') else 'warning_current'
            if col not in g: return 'Chưa có dữ liệu cảnh báo'
            vals = g[col].dropna(); flags = pd.to_numeric(vals,errors='coerce').dropna().gt(0) if col=='warnings_total_to_date' else vals.astype(str).str.lower().map({'true':True,'false':False})
            flags = flags.dropna(); k=int(flags.sum())
            return f'{k}/{len(flags)} ({k/len(flags):.1%}) có cảnh báo '+('từ đầu khóa' if col=='warnings_total_to_date' else 'ở học kỳ gần nhất') if len(flags) else 'Chưa có dữ liệu cảnh báo'
        col = ('gpa4_cumulative' if has(q,'tich luy','toan khoa') else 'gpa4_term') if metric=='gpa' else 'tuition_debt_end_vnd'
        if metric=='debt': g = g[g.state_as_of.eq('ENROLLED')]
        if col not in g: return 'Chưa có chỉ số yêu cầu'
        vals = pd.to_numeric(g[col],errors='coerce').dropna()
        if not len(vals): return 'Chưa có giá trị hợp lệ'
        return f'GPA trung bình {vals.mean():.2f}/4 ({len(vals)} hồ sơ)' if metric=='gpa' else f'Tổng công nợ nhóm đang học {vals.sum():,.0f} đồng ({len(vals)} hồ sơ có số liệu)'
    parts=[]
    if group:
        groups=list(frame.groupby(group,dropna=False))
        if len(groups)>30: return response('Có quá nhiều nhóm. Hãy thu hẹp theo ngành hoặc khóa.', 'clarify')
        for key,g in groups: parts.append(f'{key}: {calc(g)}')
    else: parts.append(calc(frame))
    scope = 'toàn trường' if principal.role=='admin' else 'ngành/lớp được phân công'
    if majors: scope += ', '+', '.join(sorted(majors))
    if classes: scope += ', '+', '.join(sorted(classes))
    if years: scope += ', khóa '+str(min(years))+('–'+str(max(years)) if len(years)>1 else '')
    note = f'Dữ liệu mô phỏng, {scope}; '+(f'HK{term}' if term else 'trạng thái tại '+str(engine.data_as_of))+'.'
    if metric in ('dropout','graduated'): note += ' Không phải tỷ suất theo kỳ hay tỷ lệ của toàn bộ người nhập học.'
    return {'reply':'<p>'+html.escape(note)+'</p><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in parts)+'</ul>','regulations':[],'method':'python','intent':'analytics'}


def focused_policy(retrieval, q, arts, clauses):
    if arts or clauses or has(q,'va neu','dong thoi','hay la','hay khong','so voi','nhung neu'):
        return None
    # Only patterns with an unambiguous numerical question; keep all rule evidence.
    def answer(text, article, clause):
        out=retrieval.answer(f'Điều {article} Khoản {clause}',cloud=False)
        if not out['regulations']: return None
        citation = '; '.join(dict.fromkeys(
            f"Điều {c['article']}, khoản {c['clause']}"
            for c in out['regulations']
        ))
        out['reply'] = (
            '<p>' + html.escape(text) + '</p>'
            + '<p>Căn cứ: ' + html.escape(citation) + '.</p>'
            + '<details><summary>Căn cứ quy chế chi tiết</summary>'
            + out['reply'] + '</details>'
        )
        out['intent']='regulation'
        return out

    if q.count('?') <= 1 and ';' not in q and not has(
        q, 'dong thoi', 'ngoai ra', 'them nua', 'con neu'
    ):
        if has(q, 'chung chi') and has(q, 'no dong', 'tin chi no') and has(q, 'tinh vao', 'co tinh'):
            return answer(
                'Không. Tín chỉ môn chứng chỉ không tính vào tiêu chí nợ trên 24 tín chỉ khi xét cảnh báo học tập.',
                18, 1
            )

        if has(q, 'gpa') and has(q, 'canh bao') and has(q, 'duoi bao nhieu', 'duoi muc nao') and not has(q, 'tin chi', 'buoc thoi hoc'):
            if has(q, 'ky dau', 'ky dau tien', 'ky thu nhat'):
                return answer('GPA học kỳ đầu dưới 0,8 là một tiêu chí cảnh báo; còn các tiêu chí tín chỉ.', 18, 1)
            if has(q, 'ky thu hai', 'ky tiep theo', 'tu ky 2', 'hoc ky 2'):
                return answer('Từ học kỳ thứ hai, GPA học kỳ dưới 1,0 là một tiêu chí cảnh báo; còn các tiêu chí tín chỉ.', 18, 1)

        if has(q, 'canh bao') and has(q, 'tin chi', 'tc') and has(q, 'bao nhieu phan tram') and has(q, 'truot', 'khong dat'):
            return answer('Tín chỉ không đạt vượt quá 50% số tín chỉ đăng ký trong kỳ là một tiêu chí cảnh báo. Còn tiêu chí GPA và tín chỉ nợ.', 18, 1)

        if has(q, 'gpa') and has(q, 'cong thuc') and not has(q, 'muc tieu', 'ky sau', 'cai thien', 'hoc lai'):
            return answer('GPA = tổng điểm hệ 4 từng học phần nhân số tín chỉ, chia tổng số tín chỉ tương ứng. Làm tròn hai chữ số thập phân, không tính môn chứng chỉ.', 17, 2)

        if has(q, 'gpa') and has(q, 'lam tron') and has(q, 'chu so', 'thap phan'):
            return answer('GPA được làm tròn đến hai chữ số thập phân.', 17, 2)

        if has(q, 'cu nhan', 'ky su') and has(q, 'hoc toi da', 'thoi gian hoc toi da') and has(q, 'bao nhieu hoc ky', 'may hoc ky') and not has(q, 'va', 'hoac'):
            n = 18 if has(q, 'cu nhan') else 22
            return answer(f'Thời gian học tối đa là {n} học kỳ theo khung chương trình tương ứng.', 3, 5)

        if has(q, 'khoa luan') and has(q, 'gpa tich luy') and has(q, 'dang ky', 'co duoc'):
            m = re.search(r'\bgpa\s+tich luy\s*(?:la\s*)?(\d+(?:[.,]\d+)?)', q)
            if m:
                value = number(m[1].replace(',', '.'), 0, 4)
                if value is not None:
                    verdict = 'Đạt' if value >= 2.5 else 'Chưa đạt'
                    return answer(f'{verdict} ngưỡng GPA 2,5 để đăng ký khóa luận. Cần đáp ứng thêm các điều kiện của khoa/ngành nếu có.', 12, 3)

        if has(q, 'hoc song bang', 'hoc hai chuong trinh', 'hoc 2 bang', 'hoc nganh thu hai') and has(q, 'gpa') and has(q, 'co duoc', 'duoc khong'):
            return answer('Cần tích lũy ít nhất 40 tín chỉ. Sinh viên học lực khá phải đáp ứng ngưỡng bảo đảm chất lượng; học lực trung bình phải đáp ứng điều kiện trúng tuyển chương trình thứ hai. Còn các điều kiện riêng và phê duyệt của trường.', 23, 1)

        if has(q, 'vang thi', 'vang buoi thi', 'vang mat buoi thi') and has(q, 'ly do chinh dang'):
            if has(q, 'khong co ly do chinh dang'):
                return answer('Vắng thi không có lý do chính đáng nhận điểm 0.', 14, 2)
            if has(q, 'thi bu'):
                return answer('Vắng thi có lý do chính đáng được thi bù vào đợt kế tiếp.', 14, 2)

        if has(q, 'tai lieu') and has(q, 'phong thi') and has(q, 'dung tai lieu', 'su dung tai lieu') and not has(q, 'duoc phep', 'cho phep', 'de mo', 'chua su dung'):
            return answer('Sử dụng tài liệu vi phạm trong phòng thi: lần đầu hủy kết quả học phần; lần hai hủy kết quả học kỳ; lần ba buộc thôi học.', 25, 3)

        if has(q, 'diem qua trinh', 'diem thanh phan') and has(q, 'khieu nai') and not has(q, 'cuoi ky', 'phuc tra'):
            return answer('Khiếu nại trực tiếp giảng viên khi công bố điểm, trước khi nộp điểm. Sau khi đã nộp điểm, thực hiện qua Phòng Đào tạo.', 16, 1)

        if has(q, 'chuan ngoai ngu', 'chuan tieng anh') and has(q, 'gm') and has(q, 'sm'):
            return answer('Hệ SM yêu cầu bậc 3/6 hoặc B1 CEFR tương đương; hệ GM yêu cầu bậc 4/6 hoặc B2 CEFR tương đương.', 19, 6)

        if has(q, 'hoc lai') and has(q, 'hang tot nghiep', 'ha hang') and re.search(r'5\s*%', q):
            return answer('Hạng tốt nghiệp xuất sắc hoặc giỏi bị giảm một mức nếu khối lượng học phần phải học lại vượt 5% tổng tín chỉ chương trình, hoặc bị kỷ luật từ cảnh cáo trở lên.', 20, 4)

        if has(q, 'het thoi gian hoc toi da') and has(q, 'thieu') and has(q, 'chung chi quoc phong', 'giao duc the chat', 'ngoai ngu'):
            return answer('Trong ba năm kể từ khi thôi học, sinh viên có thể hoàn thiện các điều kiện GDQP-AN, GDTC hoặc ngoại ngữ còn thiếu và đề nghị xét tốt nghiệp.', 20, 5)

        if has(q, 'tu y bo hoc') and has(q, 'phan tram hoc phi'):
            return answer('Nếu tự ý bỏ học phần sau thời hạn rút học phần, sinh viên nhận điểm F và phải đóng 100% học phí học phần. Trường hợp đặc biệt do hiệu trưởng quyết định.', 12, 4)

    m=re.search(r'\bdang ky\s+(\d+)\s*(?:tin chi|tc)\s+va\s+(?:truot|rot|khong dat)\s+(\d+)\s*(?:tin chi|tc)',q)
    if m and has(q,'canh bao') and not has(q,'gpa','no dong','buoc thoi hoc'):
        reg,fail=map(int,m.groups())
        if reg<=0 or fail>reg: return answer('Số tín chỉ đăng ký/trượt không hợp lệ.',18,1)
        bad=warning_reasons(None,None,reg,fail,None)['failed']
        return answer(f'{"Bị" if bad else "Không bị"} cảnh báo theo tiêu chí tín chỉ trượt: {fail}/{reg} ({fail/reg:.2%}), ngưỡng vượt 50%. Còn các tiêu chí GPA và nợ tín chỉ.',18,1)
    m=re.search(r'\bgpa\s+(?:hoc\s*)?ky\s*(?:thu\s*)?(dau|nhat|[0-9]+)(?:\s+cua\s+(?:em|toi|minh))?\s*(?:la|dat|=)?\s*(-?\d+(?:[.,]\d+)?)\b',q)
    if m and has(q,'canh bao') and not has(q,'tin chi','buoc thoi hoc'):
        t=1 if m[1] in ('dau','nhat') else int(m[1]); v=number(m[2].replace(',','.'),0,4)
        if t<1 or v is None: return answer('GPA hoặc học kỳ không hợp lệ.',18,1)
        threshold=.8 if t==1 else 1.
        bad=warning_reasons(t,v,None,None,None)["gpa"]
        return answer(f'{"Bị" if bad else "Không bị"} cảnh báo theo tiêu chí GPA kỳ {t}: {v:.2f}/4, ngưỡng dưới {threshold:.1f}; còn tiêu chí tín chỉ trượt và tín chỉ nợ.',18,1)
    # Targeted lookups based on the reviewed corpus, not a generated FAQ bank.
    if re.search(r'\bgpa\s+(?:hoc ky|ky)\s+bao nhieu\b',q) and has(q,'canh bao') and not has(q,'buoc thoi hoc','tin chi'):
        return answer('GPA học kỳ đầu dưới 0,8 hoặc từ kỳ thứ hai dưới 1,0 là một tiêu chí cảnh báo. Còn tiêu chí tín chỉ không đạt và tín chỉ nợ.',18,1)
    if has(q,'giu nguyen ma sinh vien') and has(q,'nghi','quay lai','tro lai'):
        return answer('Có. Khi trở lại sau nghỉ tạm thời, sinh viên được xếp lớp phù hợp và giữ mã sinh viên.',9,1)
    if has(q,'phuc tra','phuc khao') and has(q,'bao lau','khi nao') and has(q,'ket qua') and not has(q,'nop don','han nop','le phi'):
        return answer('Muộn nhất một tuần sau khi kết thúc thời gian nhận đơn phúc tra.',16,1)
    if has(q,'khoa luan') and has(q,'thoi gian hoan thanh','lam trong bao lau') and not has(q,'gia han','khong hoan thanh','bao ve','luu tru'):
        return answer('Thời gian thực hiện khóa luận từ 3 đến 4 tháng.',12,3)
    if has(q,'thang 4','thang diem 4','he 4') and has(q,'quy doi','tuong ung'):
        m=re.search(r'\bdiem\s+([abcd]\+?|f)(?![\w+])',q)
        if m and not has(q,'va','hoac','diem liet','thi lai') and not re.search(r'\bdiem\s+\d',q):
            grade=m[1].upper()
            return answer(f'Điểm {grade} tương ứng {GRADE_POINTS[grade]:g}/4.',17,1)
    if has(q,'tich luy','tin chi tich luy') and has(q,'nam may','nam thu may','trinh do nam') and not has(q,'toi da','tot nghiep','thieu','con lai'):
        m=re.search(r'\b(\d+)\s*(?:tin chi|tc)\b',q)
        if m:
            n=int(m[1]); rank='năm thứ nhất' if n<40 else 'năm thứ hai' if n<80 else 'năm thứ ba' if n<120 else 'diện tốt nghiệp (không đồng nghĩa đủ điều kiện tốt nghiệp)'
            return answer(f'{n} tín chỉ tích lũy: trình độ {rank}.',17,3)
    return None


def student_risk(engine, principal, q, parsed, ids):
    if principal.role!='student' or ids or not (has(q,'nguy co','rui ro','du bao') or has(q,'xac suat thoi hoc','xac suat bo hoc')):
        return None
    if has(q,'rot mon','truot mon','nguy co bi canh bao','du bao diem','diem giua ky','qua mon') and not has(q,'thoi hoc','bo hoc','dropout'):
        return response('Mô hình hiện tại dự báo thôi học trong hai học kỳ, không dự báo xác suất trượt một môn hoặc cảnh báo giữa kỳ. Có thể kiểm tra điều kiện học vụ khi đủ dữ kiện.', 'risk')
    if has(q,'ai duoc xem','ai xem duoc','ai xem','ai biet','cho ai biet'):
        return response('Trong bản demo, sinh viên chỉ xem hồ sơ của mình; cố vấn chỉ xem sinh viên thuộc ngành/lớp được phân công; quản trị viên có quyền quản lý toàn trường. Dự báo không phải quyết định học vụ.', 'privacy')
    if has(q,'gui') and has(q,'gia dinh','bo me','phu huynh') and not has(q,'quyet dinh buoc thoi hoc'):
        return response('Bản demo không có chức năng tự gửi kết quả dự báo cho gia đình. Thông báo quyết định buộc thôi học theo quy chế là việc khác với dự báo rủi ro.', 'privacy')
    if has(q,'cam thi','dieu kien du thi','diem liet','phuc tra','phuc khao','xoa','lich su chat','cap nhat','giang vien','phan hoi','co van co lien he','loi','nhap so lieu gia','bao mat'):
        return None
    if has(q,'chac chan') and has(q,'thoi hoc','bo hoc'):
        return response('Không. Xác suất dự báo không phải kết luận chắc chắn hoặc quyết định buộc thôi học. Quyết định học vụ phải căn cứ quy chế và hồ sơ được xác minh.', 'risk')
    if has(q,'nen lam gi','uu tien','lam sao de giam','can gap ai'):
        return response('Hãy gặp cố vấn học tập để kiểm tra nguyên nhân cụ thể, ưu tiên học phần chưa đạt, chuyên cần và kế hoạch tín chỉ; nếu có công nợ, trao đổi bộ phận tài chính. Chưa thể khẳng định mức giảm xác suất do từng biện pháp.', 'risk')
    if has(q,'vi sao','yeu to','anh huong','tac dong','keo nguy co','o tro','gia dinh') and not has(q,'quy che','quy dinh'):
        return response('Mô hình dùng các đặc trưng học vụ và hành vi tại mốc dự báo. Chưa có giải thích đóng góp từng đặc trưng cho hồ sơ này, nên không thể khẳng định nguyên nhân cá nhân hay tác động nhân quả.', 'risk')
    # Do not answer policy, privacy, data-export or hypothetical model changes as a risk score.
    if has(q,'ai duoc','ai xem','gui','gia dinh','bo me','quyen','bao mat','hoc bong','ky luat','quy che','quy dinh','xoa','lich su chat','chinh xac','tin tuong','cong bang','mo hinh','vi sao','yeu to','so voi','thay doi','cap nhat','khi nao','hoc lai','bo nop bai','neu','tang','giam','cai thien','ky sau','ky toi','lam them','ky tuc xa','thue tro','nghi hoc','bao luu','canh bao','duoi hoc','buoc thoi hoc'):
        if has(q,'neu','tang','giam','cai thien','lam sao') and not has(q,'quy che','quy dinh','buoc thoi hoc','duoi hoc','bao luu','nghi hoc','gui','gia dinh','bo me'):
            return response('Không thể suy ra mức giảm xác suất từ thay đổi giả định này. Mô hình dự báo thôi học trong hai học kỳ, không chứng minh tác động nhân quả. Hãy trao đổi với cố vấn về kế hoạch học tập; chỉ dùng kịch bản đã được hệ thống hỗ trợ.', 'risk')
        return None
    if not has(q,'em','toi','minh') or not (has(q,'nguy co thoi hoc','nguy co bi thoi hoc','nguy co bo hoc','xac suat thoi hoc','xac suat bo hoc','muc nguy co','thuoc nhom nguy co','nguy co cua em','nguy co cua toi','nguy co cua minh') or re.search(r'\b(?:em|minh|toi)\s+co\s+nguy co\s+(?:cao|thap)\b',q)):
        return None
    if re.search(r'\d+(?:[.,]\d+)?\s*%',q): return None
    rec=permitted_student(engine,principal,principal.user_id)
    term=parsed['term_filter']
    if term is not None:
        # Preserve existing historical-risk path.
        return None
    row=row_to_clean_dict(rec.iloc[-1]); label=risk_display(row,engine.meta['threshold'])
    return response(label+'. Dự báo mô phỏng, không phải quyết định buộc thôi học.', 'risk')
