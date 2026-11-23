"""Chọn công cụ học vụ theo quyền; không chuyển câu hỏi cá nhân sang LLM."""
import html
import re
from fastapi import HTTPException
from core.chatbot_engine import row_to_clean_dict, risk_display, strip_accents
from core.security import permitted_student, require_role, require_major, require_class, scoped_classes, scoped_frame, assigned_classes, ADVISOR_MAJORS
from core.retrieval import numbers, public_focus, unsupported, normalize_query
from core.academic_rules import graduation_conditions, projection, number


def reply(text, intent):
    return {'reply':text, 'regulations':[], 'intent':intent, 'method':'python'}


def route(engine, retrieval, principal, message, context):
    q=strip_accents(message)
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
    target_match=re.search(r'(?:gpa(?: tich luy)?|muc tieu(?: gpa)?)\s*(?:la |dat |=\s*)?(-?\d+(?:[.,]\d+)?)',q) if any(x in q for x in ('muon dat','muc tieu','can dat','ky sau','ky toi')) and 'gpa' in q else None
    scenario=bool(target_match) or any(x in q for x in ('kich ban','du doan diem','du doan gpa','diem ky sau','gpa ky sau','gpa ky toi','what-if','what if','cai thien'))
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
    calculation = engine.evaluate_regulation_trap(message) if not ids and not aggregate else ''
    if unsupported(message) and not calculation and not arts:
        return reply('<p>Chưa tìm thấy căn cứ trong quy chế cho nội dung này.</p>','regulation')
    pq=normalize_query(message)
    policy_only=not ids and not target_match and (any(x in pq for x in ('bao luu','nghi hoc tam thoi','hoc phan tien quyet','chuyen nganh','chuyen truong','phuc tra','thoi gian hoc toi da','dang ky tin chi','hoc cai thien diem','cam thi','dieu kien du thi')) or (not personal and bool(public_focus(message)) and not any((risk,admission,course)) and not any(x in q for x in ('kich ban','gpa hoc ky','gpa tich luy','gpa ky sau','gpa ky toi','hoc phi con no','tin chi da dat','tin chi no dong','so lan canh bao hoc tap','canh bao lien tiep','xem ho so','bang diem'))))
    if 'tot nghiep' in q and personal and not ids and not arts:
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
            result=retrieval.answer(message,cloud=False)
            result['reply']=('<p>Điều được nêu không phải căn cứ của phép tính; xem căn cứ đúng bên dưới.</p>' if wrong else '')+calculation+result['reply']
        else: result=retrieval.answer(message,cloud=True)
        result['intent']='regulation'
        found={c['article'] for c in result['regulations']}
        if len(found)==1: context['article']=next(iter(found))
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
        if 'tot nghiep' in q and any(x in q for x in ('dieu kien','du','thieu','duoc','chua','khi nao')):
            conditions=graduation_conditions(row)
            labels={'credits':'Tín chỉ','mandatory_courses':'Học phần bắt buộc','gpa':'GPA','english':'Ngoại ngữ','physical_education':'GDTC','national_defense':'GDQP-AN','financial':'Tài chính','program_outcomes':'Chuẩn đầu ra','discipline_and_criminal_clearance':'Kỷ luật / pháp lý','academic_debt':'Nợ học phần'}
            verdict='Chưa đủ điều kiện' if False in conditions.values() else 'Chưa đủ căn cứ xác nhận' if None in conditions.values() else 'Đạt các điều kiện có dữ liệu kiểm tra'
            result=retrieval.answer('Điều 20 Khoản 1',cloud=False);result['intent']='graduation'
            result['reply']='<p><b>'+verdict+'</b>; không thay thế quyết định xét tốt nghiệp.</p><ul>'+''.join(f'<li>{labels[k]}: '+('Đạt' if v is True else 'Chưa đạt' if v is False else 'Chưa xác minh')+'</li>' for k,v in conditions.items())+'</ul>'+result['reply']
            return result
        if target_match:
            goal=number(target_match.group(1).replace(',','.'),0,4)
            if goal is None: return reply('<p>GPA mục tiêu phải từ 0 đến 4.</p>','gpa')
            p=projection(row,target=goal)
            if p is None or row.get('state_as_of')!='ENROLLED': return reply('<p>Chưa đủ dữ liệu hoặc không còn tín chỉ mới để tính mục tiêu GPA.</p>','gpa')
            needed=p['req_gpa_next']
            result=retrieval.answer('Điều 17 Khoản 2',cloud=False);result['intent']='gpa'
            result['reply']=f'<p>Mục tiêu GPA tích lũy {goal:.2f}; giả định {p["next_tc"]:.0f} tín chỉ mới, không học lại. '+(f'GPA kỳ tới tối thiểu {needed:.2f}/4.' if needed<=4 else f'Không khả thi chỉ bằng số tín chỉ mới này (cần {needed:.2f}/4).')+'</p>'+result['reply']
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
