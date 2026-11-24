"""Explicit aggregate plans over authorized, as-of data. No SQL or LLM execution."""
import html
import re
import pandas as pd
from core.chatbot_engine import MAJOR_ALIASES, strip_accents
from core.security import require_role, require_major, require_class, scoped_frame, permitted_student
from core.academic_rules import number, rounded, GRADE_POINTS, course_grade
from core.policy_focus_v3 import has


def response(text, intent='analytics', rows=()):
    return {'reply':'<p>'+html.escape(text)+'</p>'+('<ul>'+''.join('<li>'+html.escape(str(r))+'</li>' for r in rows)+'</ul>' if rows else ''), 'intent':intent,'method':'python','regulations':[]}


def simple_major_count(engine,p,q):
    """Execute unambiguous population counts without a cloud plan.

    Full-match grammar: never drop a requested filter, comparison, time range,
    or second intent. Implicit advisor scope stays local; explicit schoolwide
    scope requires admin. Counts are latest academic records, not admissions.
    """
    q=re.sub(r'\s+',' ',q).strip().rstrip('?! .')
    q=re.sub(r'^(?:(?:xin|vui long|hay|cho (?:toi|em|minh) (?:biet|xem)|thong ke|liet ke|bao cao)\s+){0,4}','',q)
    group=r'(?:theo (?:tung |cac |tat ca cac )?nganh|(?:cua )?(?:tung|moi|cac|tat ca cac) nganh)'
    scope=r'(?: (?:toan truong|ca truong))?'
    tail=r'(?: hien tai)?(?: (?:la )?bao nhieu)?'
    patterns=(
        rf'(?:tong )?(?:so(?: luong)?|phan bo) sinh vien(?: dang hoc)?{scope} {group}{scope}{tail}',
        rf'(?:tung|moi) nganh{scope} co (?:tat ca )?bao nhieu sinh vien(?: dang hoc)?(?: hien tai)?',
        rf'bao nhieu sinh vien(?: dang hoc)?{scope} {group}{scope}(?: hien tai)?',
    )
    grouped=any(re.fullmatch(pattern,q) for pattern in patterns)
    # Match one *complete* known major name or code. Do not consume a partial
    # alias, a second question, a cohort, a course or a geographic qualifier.
    aliases={}
    for code,names in MAJOR_ALIASES.items():
        for name in (code.lower(),*names):
            name=re.sub(r'^(?:nganh|khoa) ', '', name)
            aliases.setdefault(name,set()).add(code)
    major_names='|'.join(re.escape(x) for x in sorted(aliases,key=lambda x:(-len(x),x)))
    named=rf'(?:nganh|khoa) (?P<major>{major_names})'
    count=r'(?:tong )?so(?: luong)? sinh vien(?: dang hoc)?'
    single_patterns=(
        rf'{count} (?:cua )?{named}{tail}',
        rf'(?:co )?(?:tat ca )?bao nhieu sinh vien(?: dang hoc)? (?:cua |thuoc )?{named}(?: hien tai)?',
        rf'{named}(?: hien tai)? co (?:tat ca )?bao nhieu sinh vien(?: dang hoc)?(?: hien tai)?',
        rf'{named}(?: hien tai)? co {count}(?: la)? bao nhieu',
    )
    match=next((m for pat in single_patterns if (m:=re.fullmatch(pat,q))),None)
    school_patterns=(
        rf'{count} (?:cua )?(?:toan truong|ca truong){tail}',
        r'(?:toan truong|ca truong)(?: hien tai)? co (?:tat ca )?bao nhieu sinh vien(?: dang hoc)?(?: hien tai)?',
        r'(?:co )?(?:tat ca )?bao nhieu sinh vien(?: dang hoc)? (?:trong |cua )?(?:toan truong|ca truong)(?: hien tai)?',
    )
    school=any(re.fullmatch(pat,q) for pat in school_patterns)
    if not (grouped or match or school):return None
    # Authorization must not depend on model output, even if the model is
    # unavailable or proposes a malformed or unrelated plan.
    require_role(p,'advisor','admin')
    if school or has(q,('toan truong','ca truong','cac nganh')):require_role(p,'admin')
    selected=None
    if match:
        candidates=aliases[match['major']]
        if len(candidates)!=1:return response('Tên ngành chưa rõ; hãy nhập mã ngành chính xác.','clarify')
        selected=next(iter(candidates))
        require_major(p,selected)
    frame=engine.latest_df
    required={'record_uid','major_code','state_as_of'}
    if not required.issubset(frame.columns):return response('Chưa có đủ dữ liệu để đếm sinh viên theo ngành.','clarify')
    if p.role=='advisor' and 'class_code' not in frame.columns:
        return response('Chưa có đủ thông tin lớp để kiểm tra phạm vi cố vấn.','clarify')
    frame=scoped_frame(p,frame)
    if selected is not None:frame=frame[frame.major_code.eq(selected)]
    if frame.empty:return response('Không có hồ sơ có kết quả học kỳ trong phạm vi được phép.')
    if frame.record_uid.isna().any() or frame.record_uid.duplicated().any() or frame.major_code.isna().any() or frame.state_as_of.isna().any():
        return response('Dữ liệu định danh hoặc trạng thái chưa đủ tin cậy để đếm sinh viên.','clarify')
    active_only=has(q,('dang hoc',))
    rows=[]
    for code,g in frame.groupby('major_code',sort=True):
        active=int(g.state_as_of.eq('ENROLLED').sum())
        name=''
        if 'major_name' in g:
            names=g.major_name.dropna().astype(str).unique()
            if len(names)==1:name=' ('+names[0]+')'
        rows.append(f'{code}: {active} đang học{name}' if active_only else f'{code}: {len(g)} hồ sơ; {active} đang học{name}')
    active=int(frame.state_as_of.eq('ENROLLED').sum())
    scope='toàn trường' if p.role=='admin' else 'ngành/lớp được phân công'
    if selected is not None:scope+=', ngành '+selected
    total=f'Tổng {active} sinh viên đang học' if active_only else f'Tổng {len(frame)} hồ sơ, trong đó {active} sinh viên đang học'
    return response(f'{total}. Dữ liệu mô phỏng, {scope}, tại mốc {engine.data_as_of}; chỉ tính sinh viên có kết quả học kỳ, không phải toàn bộ danh sách tuyển sinh.',rows=rows)


def scope_frame(engine,p,q,parsed,classes,frame):
    require_role(p,'advisor','admin')
    if p.role=='advisor' and has(q,('toan truong','ca truong','so voi truong','cac nganh','tat ca cac khoa')):
        require_role(p,'admin')
    majors={k for k,aliases in MAJOR_ALIASES.items() if has(q,aliases)}
    majors.update(m.upper() for m in re.findall(r'\b(?:khoa|nganh)\s+('+'|'.join(k.lower() for k in MAJOR_ALIASES)+r')\b',q))
    if parsed.get('major_code'): majors.add(parsed['major_code'])
    if not majors and re.search(r'\bnganh\s+(?!nao\b|hoc\b|va\b|theo\b|lop\b|co\b|voi\b|cua\b|trong\b|duoc\b|khac\b|tuong ung\b|do\b|nay\b|dang\b|thi\b|sinh vien\b|tren\b|tai\b|tuyen sinh\b|phan cong\b|bao nhieu\b|gpa\b|ty le\b|truot\b)[a-z]+',q):
        return None,'Chưa nhận diện được ngành; hãy nhập mã ngành chính xác.'
    for m in majors: require_major(p,m)
    for c in classes: require_class(p,c)
    frame=scoped_frame(p,frame)
    if majors: frame=frame[frame.major_code.isin(majors)]
    if classes: frame=frame[frame.class_code.isin(classes)]
    # Explicit intake-year range only; do not confuse school-year/calendar queries.
    years=sorted(set(map(int,re.findall(r'\b20\d{2}\b',q))))
    if years:
        if not has(q,('khoa','tuyen sinh')): return None,'Chưa hỗ trợ lọc theo năm lịch; hãy nêu khóa tuyển sinh hoặc học kỳ.'
        if len(years)>2 or 'cohort_year' not in frame: return None,'Chưa xác định được khóa tuyển sinh.'
        if len(years)==2 and not re.search(r'20\d{2}\s*(?:[-–—]|den|toi)\s*20\d{2}',q): frame=frame[frame.cohort_year.isin(years)]
        else: frame=frame[frame.cohort_year.between(years[0],years[-1])]
    for col,labels in [('education_mode',{'he sm':'SM','he gm':'GM'}),('degree_type',{'cu nhan':'Bachelor','ky su':'Engineer'})]:
        selected={v for k,v in labels.items() if has(q,(k,))}
        if selected:
            if col not in frame: return None,'Chưa có dữ liệu phân loại yêu cầu.'
            frame=frame[frame[col].isin(selected)]
    scope='toàn trường' if p.role=='admin' else 'ngành/lớp được phân công'
    if majors: scope+=', '+', '.join(sorted(majors))
    if classes: scope+=', '+', '.join(sorted(classes))
    if years: scope+=', khóa '+ '–'.join(map(str,years))
    return frame,scope


def aggregate_v3(engine,p,q,parsed,classes):
    # Additional structured queries only, existing simple queries remain in V2.
    # A course title is not an instruction to perform a population statistic.
    # Keep independent aggregate cues (e.g. "thong ke ty le truot mon ...").
    q = re.sub(r'\bxac suat thong ke\b', 'xac suat', q)
    if not has(q,('ty le','so luong','bao nhieu','may','trung binh','phan bo','pho diem','thong ke','cao nhat','thap nhat','nhieu nhat','it nhat','so sanh','so voi','xu huong')): return None
    if not has(q,('sinh vien','sv','nganh','khoa','lop','toan truong','he sm','he gm','mon hoc','hoc phan','tung mon','pho diem','gpa','canh bao hoc tap','chuyen can')): return None
    if has(q,('quy che','quy dinh','dieu kien','nguong','co duoc','phai','duoc phep','cua em','cua toi','cua minh','em co','em da','em con','can dat','toi thieu','toi da')): return None
    # Prevent causal or unimplemented slices from yielding unrelated statistics.
    if has(q,('tac dong','anh huong','hieu qua','nguyen nhan','vi sao','lam giam','giam duoc','du doan','sau canh bao','truoc khi','sau khi','can thiep','giu chan','bao luu','tam nghi','quay lai','hoc bong','gia dinh','noi o','gioi tinh','lam them','tu hoc','buoc thoi hoc','tu xin','can ho tro','chua','lien tiep','do ky luat','canh bao 2','canh bao 3','top','chung chi','tin chi','dau vao','phuong thuc','lms','nop bai','phu dao','co van','trong nam','thang','tuan','ky luat')): return None
    course=has(q,('mon hoc','hoc phan','tung mon','cac mon','mon nao','pho diem','diem chu','diem f')) and has(q,('diem','truot','rot','khong dat','qua mon','pho diem','diem f'))
    historical=has(q,('moi hoc ky','tung hoc ky','theo hoc ky','qua cac ky','qua cac hoc ky','xu huong'))
    comparison=has(q,('cao nhat','thap nhat','nhieu nhat','it nhat','so sanh','so voi','khac nhau'))
    if not (course or historical or comparison or has(q,('chuyen can','qua han','phan bo diem','nguy co thap','nguy co trung binh'))): return None
    if has(q,('khong',)) and not (course and has(q,('khong dat',))): return None
    if not course and not has(q,('canh bao','gpa','diem trung binh','chuyen can','qua han','nguy co thap','nguy co trung binh')): return None
    if not course and sum((has(q,('canh bao',)),has(q,('gpa','diem trung binh')),has(q,('chuyen can',)),has(q,('qua han',)),has(q,('nguy co thap','nguy co trung binh'))))>1: return None
    # Students can still ask public rules; only explicitly population-oriented questions enter here.
    if p.role=='student' and not has(q,('toan truong','tung nganh','cac nganh','tung lop','cac lop','tung khoa','cac khoa','bao nhieu sinh vien','ty le sinh vien','thong ke')): return None
    term=parsed.get('term_filter')
    if historical and term is not None: return response('Hãy chọn một học kỳ hoặc yêu cầu thống kê theo nhiều học kỳ.', 'clarify')
    if has(q,('hoc ky gan nhat','ky gan nhat')) and historical: return response('Hãy chọn mốc gần nhất hoặc chuỗi học kỳ.', 'clarify')
    if historical or term is not None:
        base=engine.df
        if term is not None: base=base[base.snapshot_term_index.eq(term)]
    else: base=engine.latest_df
    frame,scope=scope_frame(engine,p,q,parsed,classes,base)
    if frame is None: return response(scope,'clarify')
    if frame.empty: return response('Không có hồ sơ phù hợp trong phạm vi được phép.')
    note=f'Dữ liệu mô phỏng, {scope}; '+('theo chỉ số học kỳ của từng sinh viên (không phải kỳ lịch)' if historical else f'HK{term}' if term is not None else 'mốc '+str(engine.data_as_of))+'.'
    dims=[]
    if historical: dims.append('snapshot_term_index')
    for col,terms in [('major_code',('theo nganh','tung nganh','nganh nao','so sanh khoa','cac nganh')),('class_code',('theo lop','tung lop','lop nao','cac lop')),('cohort_year',('theo khoa','tung khoa','qua cac khoa','khoa tuyen sinh')),('education_mode',('theo he','so sanh he','sm va gm','gm va sm','sm voi gm','gm voi sm')),('degree_type',('theo bac','so sanh bac','cu nhan va ky su','ky su va cu nhan'))]:
        if has(q,terms): dims.append(col)
    if comparison and not dims: return response('Hãy nêu nhóm cần so sánh (ngành, lớp, khóa hoặc hệ đào tạo).','clarify')
    if len(dims)>2 or any(col not in frame for col in dims): return response('Chưa hỗ trợ tổ hợp phân nhóm này.','clarify')
    if course:
        if has(q,('so sinh vien','bao nhieu sinh vien','ty le sinh vien')) and has(q,('truot','rot','khong dat','diem f')):
            return response('Cần xác định tính sinh viên từng trượt ít nhất một lần hay kết quả lần học cuối. Có thể hỏi tỷ lệ lượt học phần không đạt.', 'clarify')
        a=getattr(engine,'df_attempts',None)
        if a is None or not {'record_uid','course_id','term_index','course_final_score10','grade_letter','is_passed'}.issubset(a.columns): return response('Chưa có dữ liệu điểm học phần đã công bố.')
        # Each attempt is scoped by a unique, authorized record. Historical grades
        # must also have been published before that specific snapshot cutoff.
        keys=['record_uid',*dict.fromkeys(dims)]
        if historical or term is not None:
            if 'snapshot_cutoff' not in frame or not {'available_at','score_published_at'}.issubset(a.columns): return response('Thiếu mốc công bố điểm lịch sử.')
            cols=list(dict.fromkeys(keys+['snapshot_term_index','snapshot_cutoff']))
            data=a.merge(frame[cols],left_on=['record_uid','term_index'],right_on=['record_uid','snapshot_term_index'],validate='many_to_one')
            cut=pd.to_datetime(data.snapshot_cutoff,errors='coerce',utc=True)
            data=data[pd.to_datetime(data.available_at,errors='coerce',utc=True).le(cut)&pd.to_datetime(data.score_published_at,errors='coerce',utc=True).le(cut)]
        else: data=a.merge(frame[keys],on='record_uid',validate='many_to_one')
        names=getattr(engine,'course_names',{})
        matches={str(cid) for cid,name in names.items() if has(q,(strip_accents(str(name)),strip_accents(str(cid))))}
        if matches: data=data[data.course_id.astype(str).isin(matches)]
        elif not has(q,('mon hoc','hoc phan','tung mon','cac mon','mon nao','pho diem','diem chu','diem f')): return response('Hãy nêu tên hoặc mã học phần.','clarify')
        by_course=has(q,('tung mon','cac mon','mon nao','tung hoc phan','hoc phan nao')) or bool(matches)
        if by_course: dims.append('course_id')
        if not matches and re.search(r'\b(?:mon|hoc phan)\s+(?!hoc\b|nao\b|co\b|truot\b|diem\b|trong\b|bi\b|theo\b|nay\b|do\b|da\b|can\b|nhieu\b|it\b|cao\b|thap\b)[a-z]+',q) and not by_course: return response('Chưa xác định được học phần; hãy nhập mã học phần.','clarify')
        if data.empty: return response('Không có lượt học phần phù hợp đã công bố điểm.')
        distribution=has(q,('pho diem','phan bo','diem chu'))
        fail=has(q,('truot','rot','diem f','khong dat'))
        def calc(g):
            if distribution:
                valid=g.grade_letter.dropna().astype(str); counts=valid.value_counts()
                return None, 'phổ điểm chữ ('+str(len(valid))+' lượt có điểm): '+', '.join(f'{k}: {v}' for k,v in counts.items())
            if fail:
                v=g.is_passed.astype(str).str.lower().map({'true':True,'false':False});v=v.dropna()
                k=int((~v.astype(bool)).sum());return ((k/len(v) if has(q,('ty le','phan tram')) else k) if len(v) else None),f'{k}/{len(v)} lượt không đạt'+(f' ({k/len(v):.1%})' if len(v) else '')
            v=pd.to_numeric(g.course_final_score10,errors='coerce');v=v[v.between(0,10)];return (float(v.mean()) if len(v) else None),f'điểm tổng kết trung bình {v.mean():.2f}/10 ({len(v)} lượt)' if len(v) else 'chưa có điểm hợp lệ'
        note+=' Thống kê lượt học phần đã công bố, có thể gồm học lại; không phải số sinh viên duy nhất.'
    else:
        data=frame
        if has(q,('canh bao',)):
            col='warning_current';kind='flag';label='cảnh báo học tập'
        elif has(q,('gpa','diem trung binh')):
            col='gpa4_cumulative' if has(q,('tich luy','toan khoa')) else 'gpa4_term';kind='mean';label='GPA /4'
        elif has(q,('chuyen can',)):
            col='attendance_rate_term';kind='mean';label='chuyên cần (%)'
        elif has(q,('qua han',)) and has(q,('hoc phi','cong no')):
            col='tuition_overdue_days_end';kind='positive';label='quá hạn học phí'
        elif has(q,('nguy co thap','nguy co trung binh')) and not historical and term is None:
            col='risk_probability';kind='risk';label='nguy cơ '+('thấp' if has(q,('nguy co thap',)) else 'trung bình'); data=data[data.state_as_of.eq('ENROLLED') & data.prediction_status.eq('VALID')]
            note+=' Chỉ sinh viên đang học có dự báo hợp lệ.'
        else: return None
        if col not in data: return response('Chưa có chỉ số cần thống kê.')
        def calc(g):
            if kind=='flag': v=g[col].astype(str).str.lower().map({'true':1,'false':0}).dropna()
            else: v=pd.to_numeric(g[col],errors='coerce').dropna()
            if kind=='risk':
                # Use deployed risk labels, not a guessed threshold for low/medium.
                from core.prediction import risk_level
                vv=v[v.between(0,1)].map(lambda x:strip_accents(risk_level(x,engine.meta['threshold'])))
                v=vv.isin(['low','thap'] if has(q,('nguy co thap',)) else ['medium','trung binh']).astype(int)
            if kind=='positive': v=v[v.ge(0)].gt(0).astype(int)
            if kind=='mean': v=v[v.between(0,4 if col.startswith('gpa') else 100)]
            if not len(v): return None,'chưa có giá trị hợp lệ'
            val=float(v.mean()); rank_value=val if kind=='mean' or has(q,('ty le','phan tram')) else int(v.sum());return rank_value,(f'{label}: {val:.2f} ({len(v)} hồ sơ)' if kind=='mean' else f'{int(v.sum())}/{len(v)} ({val:.1%}) {label}')
    if any(d not in data for d in dims): return response('Chưa có chiều phân nhóm yêu cầu.')
    groups=list(data.groupby(dims[0] if len(dims)==1 else dims,dropna=False)) if dims else [('Tổng',data)]
    result=[]
    for key,g in groups:
        val,msg=calc(g); key=key if isinstance(key,tuple) else (key,)
        title=' / '.join(str(names.get(k,k) if course and col=='course_id' else k) for col,k in zip(dims,key)) if dims else 'Tổng'
        result.append((val,title+': '+msg,len(g)))
    if comparison:
        if any(v is None for v,_,_ in result): return response('Không đủ dữ liệu hợp lệ cho tất cả nhóm để xếp hạng.',rows=[r[1] for r in result[:20]])
        if has(q,('cao nhat','thap nhat','nhieu nhat','it nhat')):
            result.sort(key=lambda x:x[0],reverse=not has(q,('thap nhat','it nhat'))); note+=' Xếp theo chỉ số được hỏi; xem mẫu số.';result=result[:5]
    if len(result)>30: return response('Có quá nhiều nhóm; hãy thu hẹp theo ngành, lớp hoặc kỳ.','clarify')
    return response(note,rows=[r[1] for r in result])


def personal_v3(engine,p,q,parsed):
    if p.role!='student' or not has(q,('em','toi','minh')): return None
    if has(q,('quy che','quy dinh','dieu kien','gia lap','neu','muc tieu','co duoc','phai','toi da','toi thieu','can dat','ky toi','ky sau','bao nhieu thi','co tinh','de tranh','de khong','nguong','nguy co','rui ro','so voi','thay doi','tang hay giam','muon dat','canh bao khong')): return None
    if not has(q,('may','bao nhieu','hien tai','da','roi','cho minh xem','cho em xem','cua em','cua toi','cua minh','ky nay','dang')): return None
    aliases=[(('diem tich luy','gpa tich luy'),'gpa4_cumulative','GPA tích lũy','{:.2f}/4'),(('canh bao may lan','canh bao may','bi canh bao may lan','canh bao bao nhieu lan'),'warnings_total_to_date','Số lần cảnh báo tích lũy','{:.0f}'),(('dang ky bao nhieu tin','dang ky bao nhieu tin chi','dang ky may tin chi'),'credits_registered_term','Tín chỉ đăng ký trong kỳ','{:.0f}'),(('nghi hoc tam thoi may hoc ky','da nghi hoc tam thoi','da nghi bao nhieu ky'),'leave_semesters_to_date','Số kỳ nghỉ tích lũy','{:.0f}'),(('gap co van may buoi','gap co van bao nhieu buoi'),'advisor_contacts_to_date','Số buổi gặp cố vấn','{:.0f}'),(('phu dao may buoi','phu dao bao nhieu buoi'),'tutoring_sessions_to_date','Số buổi phụ đạo','{:.0f}'),(('nop bai dung han','nop bai tre'),'assignment_on_time_rate_term','Tỷ lệ nộp bài đúng hạn','{:.1%}'),(('trinh do nam thu may','trinh do nam may'),'credits_earned_cumulative','Trình độ năm học','')]
    chosen=[(col,label,fmt) for phrases,col,label,fmt in aliases if has(q,phrases)]
    if len(chosen)!=1:return None
    col,label,fmt=chosen[0]; rec=permitted_student(engine,p,p.user_id); term=parsed['term_filter']
    if term is not None:rec=rec[rec.snapshot_term_index.eq(term)]
    if rec.empty:return response('Chưa có hồ sơ học kỳ được yêu cầu.','personal')
    row=rec.sort_values('snapshot_term_index').iloc[-1]; value=number(row.get(col))
    if value is None:return response('Chưa có dữ liệu: '+label+'.','personal')
    if not fmt: rendered='năm thứ nhất' if value<40 else 'năm thứ hai' if value<80 else 'năm thứ ba' if value<120 else 'diện tốt nghiệp (chưa xác nhận đủ điều kiện)'
    else: rendered=fmt.format(value)
    return response(f'{label} (HK{int(row.snapshot_term_index)}): {rendered}. Dữ liệu mô phỏng.','personal')


def numeric_policy(retrieval,q):
    if has(q,('va neu','dong thoi','ngoai ra','so voi','muc tieu','ky toi','ky sau','cai thien','hoc lai','thi lai','cua em','cua minh','cua toi')) or q.count('?')>1 or ';' in q:return None
    def answer(text, refs):
        evidence=[]; blocks=[]
        for a,k in refs:
            out=retrieval.answer(f'Điều {a} Khoản {k}',cloud=False)
            if not out['regulations']:return None
            evidence+=out['regulations'];blocks.append(out['reply'])
        return {'reply':'<p>'+html.escape(text)+'</p><p>Căn cứ: '+html.escape('; '.join(f'Điều {a}, khoản {k}' for a,k in refs))+'.</p><details><summary>Căn cứ quy chế chi tiết</summary>'+''.join(blocks)+'</details>','regulations':evidence,'retrieved':evidence,'intent':'regulation','method':'bm25','generation_status':'NOT_REQUESTED'}
    m=re.search(r'\bdang ky\s+(\d+)\s*(?:tin chi|tc)\s+(?:va\s+)?(?:truot|rot|khong dat)\s+(\d+)\s*(?:tin chi|tc)',q)
    if m and has(q,('50%','50 %','50 phan tram')) and not has(q,('gpa','buoc thoi hoc','no dong')):
        reg,failed=map(int,m.groups())
        if not reg or failed>reg:return answer('Số tín chỉ đăng ký/trượt không hợp lệ.',[(18,1)])
        return answer(f'{failed}/{reg} = {failed/reg:.1%}; '+('vượt' if 2*failed>reg else 'không vượt')+' 50%. Đây chỉ là tiêu chí tín chỉ không đạt; còn GPA và tín chỉ nợ.',[(18,1)])
    if has(q,('gpa','diem trung binh')) and has(q,('xep loai','loai gi','hoc luc gi')):
        m=re.search(r'\b(?:gpa|diem trung binh)(?:\s+tich luy)?\s*(?:la|dat|=)?\s*(\d+(?:[.,]\d+)?)',q)
        if m:
            v=number(m[1].replace(',','.'),0,4)
            if v is None:return answer('GPA phải trong khoảng 0–4.',[(17,3)])
            v=rounded(v); label=next(k for low,k in [(3.6,'Xuất sắc'),(3.2,'Giỏi'),(2.5,'Khá'),(2,'Trung bình'),(1,'Yếu'),(0,'Kém')] if v>=low)
            return answer(f'GPA {v:.2f}/4: {label}. Đây là xếp loại học lực, chưa phải hạng tốt nghiệp chính thức.',[(17,3)])
    if has(q,('thang 4','he 4','thang diem 4')) and has(q,('quy doi','tuong ung','bang bao nhieu','tinh may diem')):
        grades=re.findall(r'\bdiem\s+([abcd]\+?|f)(?![\w+])',q)
        m=re.search(r'\bdiem\s+(\d+(?:[.,]\d+)?)',q)
        if len(grades)==1 and not m and not has(q,('va','hoac')):return answer(f'{grades[0].upper()} = {GRADE_POINTS[grades[0].upper()]:g}/4.',[(17,1)])
        if m and not grades and not has(q,('va','hoac','diem liet','thi cuoi ky','diem qua trinh','diem thanh phan','diem giua ky')):
            v=number(m[1].replace(',','.'),0,10)
            if v is None:return answer('Điểm tổng kết phải trong khoảng 0–10.',[(17,1),(13,3)])
            grade=course_grade(v,10)
            return answer(f'Nếu đây là điểm tổng kết học phần tính GPA, không bị điểm liệt/cấm thi: {rounded(v,1):.1f}/10 tương ứng {grade}, {GRADE_POINTS[grade]:g}/4.',[(17,1),(13,3)])
    return None


def model_explanation(engine,q):
    if not has(q,('mo hinh','du bao','du doan','he thong')) or not has(q,('chinh xac','do tin cay','tin cay','tin du bao','bo sot','bao nham','tin mo hinh')):return None
    if has(q,('cua em','cua toi','cua minh','hoc phi','quy che','quy dinh','dieu kien','nganh nao','theo nganh','gioi tinh','lop nao')):return None
    vals=[]
    for k,label in [('test_recall','Recall'),('test_precision','Precision'),('test_pr_auc','Average Precision'),('test_roc_auc','ROC-AUC')]:
        v=number(engine.meta.get(k),0,1)
        if v is not None:vals.append(f'{label}: {v:.1%}')
    return response('Mô hình chỉ hỗ trợ sàng lọc nguy cơ trong hai học kỳ tiếp theo. '+('Kết quả tập test mô phỏng: '+', '.join(vals)+'. ' if vals else 'Chưa có số liệu kiểm định trong cấu hình này. ')+'Không phải xác suất chắc chắn, không thay quyết định học vụ; chưa xác thực trên sinh viên thật.','model')


def unavailable_aggregate(engine,p,q,parsed,classes):
    # High-confidence statistical questions that cannot be answered from present sources.
    if p.role not in ('advisor','admin') or not has(q,('bao nhieu','ty le','thong ke','so luong','phan bo','bao phu','hieu qua','anh huong','tac dong','co giam','giu chan','uoc tinh','xu huong','nhieu nhat','tong hop','so sanh')):return None
    if has(q,('quy che','quy dinh','dieu kien','duoc phep','phai','nguong')):return None
    if has(q,('gop y','phan anh','phan hoi sinh vien')):reason='Chưa có kho dữ liệu góp ý/ phản ánh sinh viên.'
    elif has(q,('anh huong','tac dong','hieu qua','co giam','giu chan')) and has(q,('hoc bong','phu dao','co van','can thiep','ho tro','tai chinh')):reason='Dữ liệu quan sát mô phỏng chưa đủ để xác định hiệu quả nhân quả của biện pháp hỗ trợ.'
    elif has(q,('that thu','chi phi','loi ich','ngan sach')):reason='Chưa có dữ liệu chi phí/doanh thu thực tế và giả định cần thiết để tính chỉ số này.'
    elif has(q,('thang nay','tuan nao','tung tuan','hom nay','thang truoc')) and has(q,('nguy co','thoi hoc','canh bao','hoc phi','gop y')):reason='Dữ liệu hiện tại theo snapshot học kỳ, không có chuỗi sự kiện theo tuần/tháng đáp ứng truy vấn này.'
    else:return None
    # Check all explicitly requested scope even when no data can be returned.
    _,scope=scope_frame(engine,p,q,parsed,classes,engine.latest_df)
    return response(reason+' Không suy diễn từ thống kê khác.','analytics')
