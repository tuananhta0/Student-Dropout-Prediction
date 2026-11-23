"""Chọn công cụ học vụ theo quyền; không chuyển câu hỏi cá nhân sang LLM."""
import html
import re
from fastapi import HTTPException
from core.chatbot_engine import row_to_clean_dict, risk_display, strip_accents
from core.security import permitted_student, require_role, require_major, require_class, scoped_classes, scoped_frame, assigned_classes, ADVISOR_MAJORS
from core.retrieval import numbers, public_focus, unsupported, normalize_query, span
from core.academic_rules import graduation_conditions, projection, number, whole


def reply(text, intent):
    return {'reply':text, 'regulations':[], 'intent':intent, 'method':'python'}


def extract_target_gpa(q):
    if not any(x in q for x in ('gpa', 'diem trung binh', 'dtb')):
        return None
    if any(x in q for x in ('xep loai', 'loai gi', 'canh bao', 'khoa luan', 'do an')) and not any(
        x in q for x in ('muc tieu', 'keo', 'nang', 'tang', 'muon dat', 'de dat', 'cai thien')
    ):
        return None
    goal_cues = (
        'muon', 'muc tieu', 'can dat', 'de dat', 'dat duoc',
        'keo', 'nang', 'tang', 'len', 'ky sau', 'ki sau', 'ky toi', 'ki toi', 'cai thien'
    )
    if not any(x in q for x in goal_cues):
        return None
    patterns = (
        r'(?:muc tieu(?:\s+(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?)?|(?:keo|nang|tang|cai thien|muon dat|can dat|de dat|dat duoc)\s+(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?)(?:\s+cua\s+(?:em|toi|minh))?\s*(?:(?:la|dat|duoc|len|thanh|toi|den|o muc|muc|it nhat|toi thieu|tu|tren|khoang|=)\s*){0,3}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?(?:\s+cua\s+(?:em|toi|minh))?\s*(?:muc tieu\s*)?(?:len|thanh|toi|den|o muc|muc)\s*(?:(?:duoc|it nhat|toi thieu|tu|tren|khoang)\s*){0,2}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(?:muc tieu|muon dat|can dat|de dat|dat duoc|dat muc|keo len|nang len|tang len|muon len|len muc|len|toi muc|den muc)\s+(?:(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?\s*)?(?:(?:la|dat|duoc|len|thanh|muc|it nhat|toi thieu|tu|tren|khoang|=)\s*){0,3}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(-?\d+(?:[.,]\d+)?)\s*(?:gpa|diem trung binh|dtb)\b',
        r'(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?\s*(?:(?:la|dat|it nhat|toi thieu|=)\s*){0,2}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tc|hoc ky|hk|mon|lan|%)\b)',
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match
    return None


def route(engine, retrieval, principal, message, context):
    q=re.sub(r"\s+", " ", strip_accents(message))
    simple=re.sub(r'[^a-z0-9 ]','',q).strip()
    if simple in ('xin chao','chao','hello','hi','chao ban','cam on','cam on ban','thanks','huong dan','ban lam duoc gi'):
        context.pop('article',None)
        return reply('<p>Xin chào! Tôi hỗ trợ tra cứu quy chế, bảng điểm, GPA và nguy cơ thôi học trong phạm vi tài khoản. Bạn muốn xem nội dung nào?</p>','greeting')
    parsed=engine.parse_flexible_query(message)
    ids=set(re.findall(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6})\b',message.upper()))
    classes=set(re.findall(r'\b[A-Z]{2}\d{2}-\d{2,3}\b',message.upper()))
    # Kiểm tra mọi mã, không chỉ mã đầu tiên do parser chọn.
    for uid in ids: permitted_student(engine,principal,uid)
    for cls in classes:
        require_role(principal,'advisor','admin')
        require_class(principal,cls)
    if len(ids)>1 or len(classes)>1 or (ids and classes):
        return reply('<p>Hãy tra cứu một sinh viên hoặc một lớp mỗi lần.</p>','clarify')
    arts,clauses=numbers(message)
    personal=bool(ids) or ('tot nghiep' in q and bool(re.search(r'\b(?:em|toi|minh)\b',q))) or any(x in q for x in ('cua em','cua toi','cua minh','ban than','em co','em da','em con','toi co','toi da','toi con','ho so cua'))
    aggregate=bool(classes) or any(x in q for x in ('lop manh nhat','lop yeu nhat','toan truong','tong ket khoa','tong so sinh vien','thong ke lop','xem lop','danh sach lop','top 5 lop','so sanh cac lop','hoc bong co giu chan','ty le thoi hoc','tong no hoc phi','thong ke theo','thong ke khoa','so sanh he','so sanh bac','so sanh khoa','ty le bo hoc'))
    admission='dau vao' in q or 'diem tuyen sinh' in q or 'phuong thuc tuyen sinh' in q
    risk=any(x in q for x in ('nguy co thoi hoc','xac suat thoi hoc','du bao thoi hoc','nguy co dropout'))
    target_match = extract_target_gpa(q)
    requested_registration = re.search(r'\bdang ky\s*(?:hoc\s*)?(-?\d+(?:[.,]\d+)?)\s*(?:tin chi|tc)\b',q)
    if requested_registration and principal.role=='student': personal=True
    retake_cmd = bool(re.search(r'(?:hoc lai|cai thien)\s+\S+\s+(?:dat|len)\s+(?:a\+|a|b\+|b|c\+|c|d\+|d|f)(?:\s|$)', q))
    has_scenario_cue = any(x in q for x in ('kich ban', 'du doan', 'gia lap', 'what-if', 'what if', 'ky sau', 'ki sau', 'ky toi', 'ki toi'))
    improve_policy_cues = ('quy che', 'quy dinh', 'dieu kien', 'lay diem', 'tinh diem', 'tinh vao gpa', 'diem cao nhat', 'lan sau', 'duoc phep', 'co duoc', 'gioi han', 'bao nhieu lan', 'diem d', 'diem c', 'khac nhau', 'nhu the nao', 'the nao')
    asking_improve_policy = (
        any(x in q for x in ('hoc cai thien', 'cai thien diem'))
        and not target_match
        and not retake_cmd
        and not any(x in q for x in ('kich ban', 'du doan', 'gia lap', 'what-if', 'what if'))
        and (any(x in q for x in improve_policy_cues) or not any(x in q for x in ('ky sau', 'ki sau', 'ky toi', 'ki toi')))
    )
    gpa_improve_scenario = (
        bool(target_match)
        or retake_cmd
        or ('cai thien' in q and not asking_improve_policy and (has_scenario_cue or any(x in q for x in ('gpa', 'diem trung binh', 'dtb'))))
    )
    scenario = gpa_improve_scenario or any(x in q for x in ('kich ban','du doan diem','du doan gpa','diem ky sau','gpa ky sau','gpa ky toi','what-if','what if'))
    course=parsed['is_course'] and (personal or q.startswith(('bang diem','xem diem','xem mon','xem hoc phan','diem mon')))
    model=any(x in q for x in ('roc-auc','pr-auc','average precision','recall','precision','brier','thong so mo hinh')) and not personal
    follow_clause=not arts and clauses and context.get('article') and not ids and not personal and not aggregate
    if follow_clause: message=f'{message} Điều {context["article"]}'; arts=[context['article']]
    context.pop('article',None)
    if follow_clause: pass
    elif clauses and not arts and not personal and not aggregate:
        return reply('<p>Khoản này thuộc Điều nào? Hãy nhập số Điều để tra cứu chính xác.</p>','clarify')
    if len(arts)>1 and clauses:
        return reply('<p>Hãy tra cứu một Điều và Khoản mỗi lần để tránh nhầm trích dẫn.</p>','clarify')
    if model:
        m=engine.meta
        metrics=[('ROC-AUC','test_roc_auc'),('Average Precision','test_pr_auc'),('Brier Score','test_brier'),('Recall','test_recall'),('Precision','test_precision'),('Recall@Top10%','test_recall_top10'),('Ngưỡng','threshold')]
        return reply('<p>Chỉ số mô hình '+html.escape(str(m.get('version','baseline')))+': '+ '; '.join(f'{label}: {html.escape(str(m[key]))}' for label,key in metrics if key in m)+'. Kết quả trên dữ liệu mô phỏng.</p>','model')
    if aggregate:
        require_role(principal,'advisor','admin')
        if classes:
            d=engine.get_class_detail(next(iter(classes)))
            if not d: raise HTTPException(404,'Không tìm thấy lớp đang học')
            return reply('<p>'+html.escape(d['summary_text'])+'</p>','class')
        if ('toan truong' in q or 'tong so sinh vien' in q or 'ty le thoi hoc' in q or 'ty le bo hoc' in q or 'tong no hoc phi' in q) and not any(x in q for x in ('thong ke theo','so sanh he','so sanh bac','so sanh khoa')):
            require_role(principal,'admin')
            active=engine.active_latest_df
            valid=active.prediction_status.eq('VALID').sum()
            debt=active.tuition_debt_end_vnd.sum()
            count=engine.latest_df.state_as_of.eq('DROPOUT').sum()
            total=len(engine.latest_df); rate=f'{count/total:.1%}' if total else 'Chưa có dữ liệu'
            return reply(f'<p>Toàn trường (mô phỏng, chỉ sinh viên đã có hồ sơ kết quả học kỳ tại mốc dữ liệu): {len(active)} đang học; {valid} có dự báo hợp lệ; {count}/{total} hồ sơ ở trạng thái thôi học ({rate}, tỷ lệ trạng thái tại mốc dữ liệu, không phải tỷ suất theo kỳ); công nợ nhóm đang học {debt:,.0f} đồng.</p>','analytics')
        major=parsed['major_code']
        if major: require_major(principal,major)
        if 'tong ket khoa' in q or 'thong ke khoa' in q:
            if not major: return reply('<p>Hãy nhập mã ngành cần tổng kết, ví dụ IT.</p>','clarify')
            if principal.role=='admin' or assigned_classes(principal) is None:
                d=engine.get_faculty_detail(major)
                return reply('<p>'+html.escape(d.get('summary_text','Không có dữ liệu khoa'))+'</p>','analytics')
            sub=scoped_frame(principal,engine.active_latest_df); sub=sub[sub.major_code.eq(major)]
            valid=int(sub.prediction_status.eq('VALID').sum())
            return reply(f'<p>Phạm vi lớp được phân công: {len(sub)} SV đang học; {valid}/{len(sub)} có dự báo hợp lệ.</p>','analytics')
        if any(x in q for x in ('thong ke theo','so sanh he','so sanh bac','so sanh khoa')):
            sub=engine.latest_df
            sub=scoped_frame(principal,sub)
            if major: sub=sub[sub.major_code.eq(major)]
            col='education_mode' if re.search(r'\bhe\b',q) else 'degree_type' if re.search(r'\b(?:bac|cu nhan|ky su)\b',q) else 'major_code' if 'so sanh khoa' in q else 'cohort_year'
            if col not in sub: return reply('<p>Chưa có dữ liệu nhóm.</p>','analytics')
            rows=[]
            for key,g in sub.groupby(col,dropna=False):
                active=g[g.state_as_of.eq('ENROLLED')]; valid=int(active.prediction_status.eq('VALID').sum()); dropout=int(g.state_as_of.eq('DROPOUT').sum())
                label={'Bachelor':'Cử nhân','Engineer':'Kỹ sư','SM':'Hệ tiêu chuẩn','GM':'Hệ song ngữ'}.get(str(key),str(key))
                rows.append(f'<li>{html.escape(label)}: {len(g)} hồ sơ; {dropout}/{len(g)} thôi học ({dropout/len(g):.1%}); {valid}/{len(active)} đang học có dự báo.</li>')
            return reply('<p>Thống kê trên hồ sơ đã có kết quả học kỳ; trạng thái tại mốc dữ liệu:</p><ul>'+''.join(rows)+'</ul>','analytics')
        rows=engine.get_all_classes_summary(major)
        rows=scoped_classes(principal,rows)
        if not rows: return reply('<p>Không có lớp phù hợp trong phạm vi truy cập.</p>','analytics')
        if 'hoc bong co giu chan' in q: return reply('<p>Chưa có bằng chứng nhân quả về tác động của học bổng.</p>','analytics')
        if 'lop manh nhat' in q or 'lop yeu nhat' in q or 'top 5 lop' in q:
            rows=sorted([r for r in rows if r['gpa_mean'] is not None],key=lambda r: r['gpa_mean'],reverse='lop yeu nhat' not in q)[:5]
        else: rows=rows[:15]
        return reply('<p>Các lớp trong phạm vi được phép (sinh viên đang học):</p><ul>'+''.join(f"<li>{html.escape(r['class_code'])}: {r['n_students']} SV, GPA TB {r['gpa_mean'] if r['gpa_mean'] is not None else 'Chưa có'}, cần hỗ trợ {r['high_risk_count']} SV.</li>" for r in rows)+'</ul>','analytics')
    # Điều hoặc khoản rõ ràng luôn tra cứu nguồn, trừ câu hỏi rõ ràng về hồ sơ cá nhân.
    # Các giả định có số rõ ràng đi qua Rule Engine; không suy diễn từ hồ sơ.
    calculation = engine.evaluate_regulation_trap(message) if not ids and not aggregate and not target_match else ''
    if unsupported(message) and not calculation and not arts:
        result=retrieval.answer(message,cloud=False); result['intent']='regulation';return result
    pq=normalize_query(message)
    policy_improve = not ids and not gpa_improve_scenario and any(x in pq for x in ('hoc cai thien', 'cai thien diem'))
    policy_only=not ids and not target_match and not gpa_improve_scenario and (policy_improve or any(x in pq for x in ('bao luu','nghi hoc tam thoi','hoc phan tien quyet','chuyen nganh','chuyen truong','phuc tra','thoi gian hoc toi da','dang ky tin chi','cam thi','dieu kien du thi')) or (not personal and bool(public_focus(message)) and not any((risk,admission,course)) and not any(x in q for x in ('kich ban','gpa hoc ky','gpa tich luy','gpa ky sau','gpa ky toi','hoc phi con no','tin chi da dat','tin chi no dong','so lan canh bao hoc tap','canh bao lien tiep','xem ho so','bang diem'))))
    if ('tot nghiep' in q or requested_registration) and personal and not arts:
        policy_only=False
    if re.search(r'(?:vang|nghi)\s*\d+(?:[.,]\d+)?\s*%',q) and not calculation and not arts:
        result=retrieval.answer('Điều 25 Khoản 1',cloud=False);result['intent']='regulation'
        result['reply']='<p>Cần biết tỷ lệ nghỉ không phép và tổng tỷ lệ nghỉ của từng học phần để kết luận.</p>'+result['reply']
        return result
    if (arts and (not personal or calculation)) or policy_only or calculation:
        if calculation:
            inferred=list(dict.fromkeys(re.findall(r'Điều (\d+)',calculation)))
            if not inferred: return reply(calculation,'regulation')
            wrong=bool(arts and set(arts)!=set(map(int,inferred)))
            message=' '.join('Điều '+x for x in inferred)
            if len(inferred)==1:
                references=re.findall(r'Điều (\d+)(?:\.(\d+))?',calculation)
                if references and all(k for _,k in references):
                    message+=' '+ ' và '.join('Khoản '+k for k in dict.fromkeys(k for _,k in references))
            result=retrieval.answer(message,cloud=False)
            result['reply']=('<p>Điều được nêu không phải căn cứ của phép tính; xem căn cứ đúng bên dưới.</p>' if wrong else '')+calculation+result['reply']
        else: result=retrieval.answer(message,cloud=True)
        result['intent']='regulation'
        found={c['article'] for c in result['regulations']}
        if len(found)==1 and not (set(arts)-found) and all(any(k in span(c['clause']) for c in result['regulations']) for k in clauses): context['article']=next(iter(found))
        return result
    personal_request=personal or (principal.role=='advisor' and bool(context.get('target')) and (risk or admission or scenario or course)) or (principal.role=='student' and (risk or admission or scenario or course or any(x in q for x in ('gio tu hoc','gio lam them','xem ho so','bang diem','gpa tich luy','gpa hoc ky','no hoc phi bao nhieu','tin chi da dat','so lan canh bao hoc tap','ty le chuyen can','hoc phi con no','tin chi no dong','tin chi tich luy','diem trung binh tich luy','dang nhap lms','buoi gap co van','buoi phu dao','so ngay qua han'))))
    if personal_request:
        target=next(iter(ids), principal.user_id if principal.role=='student' else context.get('target',''))
        if not target: return reply('<p>Hãy nhập mã sinh viên cần tra cứu.</p>','clarify')
        rec=permitted_student(engine,principal,target); context['target']=target
        row=row_to_clean_dict(rec.iloc[-1])
        if admission:
            score=row.get('admission_score'); method=row.get('admission_method')
            value='Chưa có hoặc không áp dụng' if score is None else str(score)
            method_code=str(method or '')
            methods={'CMC100':'Xét điểm thi THPT','CMC200':'Xét học bạ','CMC303':'Tuyển thẳng','CMC401':'Bài thi đánh giá năng lực CMC','CMC402':'Bài thi đánh giá năng lực'}
            method_label=methods.get(method_code.split('_')[0],'Chưa xác định phương thức')
            if method_code.startswith('CMC402_HSA'): method_label='Đánh giá năng lực ĐHQG Hà Nội'
            if method_code.startswith('CMC402_VNUHCM'): method_label='Đánh giá năng lực ĐHQG TP.HCM'
            scale=re.search(r'_(\d+)$',method_code)
            if score is not None and scale: value += ' / '+scale.group(1)
            return reply(f'<p>Điểm đầu vào: {html.escape(value)}. Phương thức: {html.escape(method_label)}. Dữ liệu mô phỏng.</p>','admission')
        if risk:
            term=parsed['term_filter']
            if term is not None:
                selected=rec[rec.snapshot_term_index.eq(term)]
                if selected.empty: return reply('<p>Chưa có hồ sơ học kỳ được yêu cầu.</p>','risk')
                row=row_to_clean_dict(selected.iloc[-1])
            label=risk_display(row,engine.meta['threshold'])
            prob=number(row.get('risk_probability'),0,1)
            if term is not None and prob is not None and (term!=int(rec.iloc[-1]['snapshot_term_index']) or row.get('prediction_status')!='VALID') and row.get('prediction_status') not in ('OUT_OF_SCOPE',):
                label=f'Dự báo lịch sử HK{term}: {prob*100:.1f}% (không dùng để can thiệp hiện tại)'
            return reply('<p>'+html.escape(label)+'. Chân trời: hai học kỳ tiếp theo tính từ snapshot; không phải quyết định học vụ.</p>','risk')
        if not target_match and 'tot nghiep' in q and any(x in q for x in ('dieu kien','du','thieu','duoc','chua','khi nao')):
            conditions=graduation_conditions(row)
            labels={'credits':'Tín chỉ','mandatory_courses':'Học phần bắt buộc','gpa':'GPA','english':'Ngoại ngữ','physical_education':'GDTC','national_defense':'GDQP-AN','financial':'Tài chính','program_outcomes':'Chuẩn đầu ra','discipline_and_criminal_clearance':'Kỷ luật / pháp lý','academic_debt':'Nợ học phần'}
            verdict='Chưa đủ điều kiện' if False in conditions.values() else 'Chưa đủ căn cứ xác nhận' if None in conditions.values() else 'Đạt các điều kiện có dữ liệu kiểm tra'
            result=retrieval.answer('Điều 20 Khoản 1',cloud=False);result['intent']='graduation'
            result['reply']='<p><b>'+verdict+'</b>; không thay thế quyết định xét tốt nghiệp.</p><ul>'+''.join(f'<li>{labels[k]}: '+('Đạt' if v is True else 'Chưa đạt' if v is False else 'Chưa xác minh')+'</li>' for k,v in conditions.items())+'</ul>'+result['reply']
            return result
        if requested_registration and not target_match:
            requested=whole(requested_registration.group(1).replace(',','.'),1)
            lo=whole(row.get('registration_min_credits'),1); hi=whole(row.get('registration_max_credits'),1)
            result=retrieval.answer('Điều 12 Khoản 2',cloud=False); result['intent']='registration'
            if requested is None: text='Số tín chỉ đăng ký phải là số nguyên dương.'
            elif lo is None or hi is None or lo>hi: text='Chưa đủ dữ liệu giới hạn đăng ký trong hồ sơ; cần biết khối lượng chuẩn của kỳ dự kiến.'
            elif requested>hi: text=f'{requested:.0f} tín chỉ vượt giới hạn tối đa {hi:.0f} tín chỉ theo giới hạn lưu trong hồ sơ.'
            elif requested<lo: text=f'{requested:.0f} tín chỉ dưới giới hạn tối thiểu {lo:.0f}; cần kiểm tra ngoại lệ cuối chương trình.'
            else: text=f'{requested:.0f} tín chỉ nằm trong khoảng {lo:.0f}–{hi:.0f} theo giới hạn lưu trong hồ sơ.'
            result['reply']='<p>'+text+' Chưa xác nhận đăng ký thực tế; cần đối chiếu khối lượng chuẩn kỳ tới, học phần và điều kiện tiên quyết.</p>'+result['reply']
            return result
        if target_match:
            goal=number(target_match.group(1).replace(',','.'),0,4)
            result=retrieval.answer('Điều 17 Khoản 2',cloud=False);result['intent']='gpa'
            def explain(text):
                result['reply']='<p>'+text+'</p>'+result['reply'];return result
            if goal is None: return explain('GPA mục tiêu phải từ 0 đến 4.')
            quantities=re.findall(r'(?<![\d.,-])(-?\d+(?:[.,]\d+)?)\s*(?:tin chi|tc)\b',q)
            # Không tự chọn giữa hai số tín chỉ khác nhau hoặc giả định cả học lại.
            if len(quantities)>1 or ('hoc lai' in q and 'khong hoc lai' not in q):
                return explain('Hãy nêu một số tín chỉ môn mới, không tính học lại, để tính GPA mục tiêu.')
            selected=whole(quantities[0].replace(',','.'),1) if quantities else None
            if quantities and selected is None: return explain('Số tín chỉ mới phải là số nguyên dương.')
            fields=[number(row.get(k)) for k in ('credits_required','credits_earned_cumulative','credits_owed_cumulative','new_course_credits_available','registration_max_credits')]
            if selected is not None and None not in fields:
                maximum=min(max(0,fields[0]-fields[1]-fields[2]),fields[3],fields[4])
                if selected>maximum: return explain(f'Yêu cầu {selected:.0f} tín chỉ mới vượt mức tối đa có thể giả định {maximum:.0f} tín chỉ (giới hạn hồ sơ, số môn mới và số tín chỉ còn lại). Không tự giảm số tín chỉ.')
            p=projection(row,target=goal,new_credits=selected)
            if p is None or row.get('state_as_of')!='ENROLLED': return explain('Chưa đủ dữ liệu hoặc không còn tín chỉ mới để tính mục tiêu GPA.')
            needed=p['req_gpa_next']
            result['reply']=f'<p>Mục tiêu GPA tích lũy {goal:.2f}; giả định {p["next_tc"]:.0f} tín chỉ mới, không học lại. '+(f'GPA kỳ tới tối thiểu {needed:.2f}/4.' if needed<=4 else f'Không khả thi chỉ bằng số tín chỉ mới này (cần {needed:.2f}/4).')+' Đây là phương án toán học, cần kiểm tra khối lượng đăng ký kỳ tới và khả năng mở lớp.</p>'+result['reply']
            return result
        if 'what-if' in q or 'what if' in q or 'thanh toan hoc phi' in q:
            return reply('<p>Mở Bảng điểm &amp; Kịch bản để nhập số tiền thanh toán giả định. Kết quả không chứng minh quan hệ nhân quả.</p>','payment')
        metric=engine.answer_personal_metric_query(message,rec)
        if metric: return reply(metric,'personal')
        if scenario:
            return reply(engine.answer_risk_and_whatif_query(message,rec) or '<p>Hãy nêu học phần hoặc kịch bản GPA cần tính.</p>','gpa')
        if course:
            courses=engine.get_student_courses(row['record_uid'],parsed['term_filter'],any(x in q for x in ('truot','rot','diem f')))
            return reply(engine.render_course_grades_html(target,courses,parsed['term_filter']),'courses')
        if any(x in q for x in ('ho so','bang diem','ket qua hoc tap','xem diem','gpa','tin chi','canh bao','tot nghiep')):
            return reply(engine.render_student_rich_html(rec,principal.role),'student')
        return reply('<p>Hãy nêu rõ chỉ số học vụ cần xem; tôi chỉ tra cứu dữ liệu theo quyền được cấp.</p>','clarify')
    if 'hoc phi' in q and any(x in q for x in ('bao nhieu tien','muc hoc phi','hoc phi mot tin')):
        return reply('<p>Quy chế này không cung cấp mức học phí cụ thể. Cần tra cứu thông báo học phí tương ứng khóa và chương trình.</p>','none')
    if any(x in q for x in ('quyen rieng tu','bao mat','gemini','lich su chat')):
        return reply('<p>Phiên đăng nhập và phân quyền được kiểm tra phía máy chủ. Hệ thống dùng tài khoản và dữ liệu mô phỏng; không dùng cho hồ sơ thật. Nếu bật Gemini, chỉ gửi bằng chứng quy chế công khai, không gửi hồ sơ hoặc câu hỏi gốc.</p>','privacy')
    # Phép tính giả định chỉ từ các số người dùng nêu, không truy cập dữ liệu.
    result=retrieval.answer(message,cloud=True);result['intent']='regulation'
    if calculation: result['reply']=calculation+result['reply']
    a={c['article'] for c in result['regulations']}
    if len(a)==1: context['article']=next(iter(a))
    else: context.pop('article',None)
    return result
