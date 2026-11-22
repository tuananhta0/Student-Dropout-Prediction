"""Chọn công cụ học vụ theo quyền; không chuyển câu hỏi cá nhân sang LLM."""
import html
import re
from fastapi import HTTPException
from core.chatbot_engine import row_to_clean_dict, risk_display, strip_accents
from core.security import permitted_student, require_role, require_major, ADVISOR_MAJORS
from core.retrieval import numbers


def reply(text, intent):
    return {'reply':text, 'regulations':[], 'intent':intent, 'method':'python'}


def route(engine, retrieval, principal, message, context):
    q=strip_accents(message)
    parsed=engine.parse_flexible_query(message)
    ids=set(re.findall(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6})\b',message.upper()))
    classes=set(re.findall(r'\b[A-Z]{2}\d{2}-\d{2,3}\b',message.upper()))
    # Kiểm tra mọi mã, không chỉ mã đầu tiên do parser chọn.
    for uid in ids: permitted_student(engine,principal,uid)
    for cls in classes:
        require_role(principal,'advisor','admin')
        require_major(principal,cls[:2])
    if len(ids)>1 or len(classes)>1 or (ids and classes):
        return reply('<p>Hãy tra cứu một sinh viên hoặc một lớp mỗi lần.</p>','clarify')
    arts,clauses=numbers(message)
    personal=bool(ids) or any(x in q for x in ('cua em','cua toi','cua minh','ban than','em co','em da','em con','toi co','toi da','toi con','ho so cua'))
    aggregate=bool(classes) or any(x in q for x in ('lop manh nhat','lop yeu nhat','toan truong','tong ket khoa','tong so sinh vien','thong ke lop','xem lop','danh sach lop','top 5 lop','so sanh cac lop','hoc bong co giu chan','ty le thoi hoc','tong no hoc phi'))
    admission='dau vao' in q or 'diem tuyen sinh' in q or 'phuong thuc tuyen sinh' in q
    risk=any(x in q for x in ('nguy co thoi hoc','xac suat thoi hoc','du bao thoi hoc','nguy co dropout'))
    scenario=any(x in q for x in ('kich ban','du doan diem','du doan gpa','diem ky sau','gpa ky sau','gpa ky toi','what-if','what if','cai thien'))
    course=parsed['is_course'] and (personal or q.startswith(('bang diem','xem diem','xem mon','xem hoc phan','diem mon')))
    model=any(x in q for x in ('roc-auc','pr-auc','average precision','recall','precision','brier','thong so mo hinh')) and not personal
    follow_clause=not arts and clauses and context.get('article') and not ids and not personal and not aggregate
    if follow_clause: message=f'{message} Điều {context["article"]}'; arts=[context['article']]
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
        if 'toan truong' in q or 'tong so sinh vien' in q or 'ty le thoi hoc' in q or 'tong no hoc phi' in q:
            require_role(principal,'admin')
            active=engine.active_latest_df
            valid=active.prediction_status.eq('VALID').sum()
            debt=active.tuition_debt_end_vnd.sum()
            count=engine.latest_df.state_as_of.eq('DROPOUT').sum()
            return reply(f'<p>Toàn trường (mô phỏng): {len(active)} đang học; {valid} có dự báo hợp lệ; {count} hồ sơ ở trạng thái thôi học; công nợ của nhóm đang học {debt:,.0f} đồng.</p>','analytics')
        major=parsed['major_code']
        if major: require_major(principal,major)
        rows=engine.get_all_classes_summary(major)
        if principal.role=='advisor': rows=[r for r in rows if r['major_code'] in ADVISOR_MAJORS[principal.user_id]]
        if not rows: return reply('<p>Không có lớp phù hợp trong phạm vi truy cập.</p>','analytics')
        if 'hoc bong co giu chan' in q: return reply('<p>Chưa có bằng chứng nhân quả về tác động của học bổng.</p>','analytics')
        if 'lop manh nhat' in q or 'lop yeu nhat' in q or 'top 5 lop' in q:
            rows=sorted(rows,key=lambda r:r['gpa_mean'],reverse='lop yeu nhat' not in q)[:5]
        else: rows=rows[:15]
        return reply('<p>Các lớp trong phạm vi được phép (sinh viên đang học):</p><ul>'+''.join(f"<li>{html.escape(r['class_code'])}: {r['n_students']} SV, GPA TB {r['gpa_mean']:.2f}, cần hỗ trợ {r['high_risk_count']} SV.</li>" for r in rows)+'</ul>','analytics')
    # Điều hoặc khoản rõ ràng luôn tra cứu nguồn, trừ câu hỏi rõ ràng về hồ sơ cá nhân.
    # Các giả định có số rõ ràng đi qua Rule Engine; không suy diễn từ hồ sơ.
    calculation = engine.evaluate_regulation_trap(message) if not ids and not aggregate else ''
    if arts and (not personal or calculation):
        result=retrieval.answer(message,cloud=True); result['intent']='regulation'
        if calculation: result['reply']=calculation+result['reply']
        if result['regulations']: context['article']=arts[0] if len(arts)==1 else None
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
            return reply('<p>'+html.escape(risk_display(row,engine.meta['threshold']))+'. Dự báo không phải quyết định học vụ; căn cứ hồ sơ tại mốc dữ liệu.</p>','risk')
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
