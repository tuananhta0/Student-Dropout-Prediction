"""Local, authenticated conversational intents; never forward profile data to cloud."""
import html
import os
import re
from core.chatbot_engine import row_to_clean_dict
from core.security import permitted_student

_SELF = r'(?:toi|em|minh|tui)'
_POLITE = r'(?:\s+(?:a|nhi|nhe|ha|vay|the|khong|ko|hay khong))*'


def _match(pattern, text):
    return bool(re.fullmatch(pattern + _POLITE, text))


def _out(text, intent='greeting'):
    return {'reply': text, 'regulations': [], 'intent': intent, 'method': 'python',
            'generation_status': 'NOT_REQUESTED'}


def _clear(context):
    for key in ('article', 'v3_public', 'v3_metric', 'target', 'basic_leave'):
        context.pop(key, None)


def _policy(retrieval, query, conclusion, expected):
    out = retrieval.answer(query, cloud=False)
    if not expected.issubset({c['id'] for c in out['regulations']}):
        return _out('<p>Chưa tìm được đầy đủ căn cứ quy chế; cần kiểm tra văn bản gốc.</p>', 'clarify')
    out['reply'] = '<p>' + html.escape(conclusion) + '</p><details><summary>Căn cứ quy chế chi tiết</summary>' + out['reply'] + '</details>'
    out['intent'] = 'regulation'
    return out


def basic_conversation(engine, retrieval, principal, q, context):
    # Only bounded whole-question matches. Never consume additional queries,
    # explicit target IDs, scope requests or arbitrary instructions.
    text = re.sub(r'\s+', ' ', q).strip().rstrip('?!., ').strip()
    prior_leave = context.pop('basic_leave', False)
    if _match(r'(?:ban|bot|chatbot) (?:la ai|la gi|lam duoc gi|co the lam gi|giup duoc gi)', text):
        _clear(context)
        extra = ' Khi bật hội thoại AI V3, tôi cũng có thể trò chuyện và giải đáp kiến thức thông thường; không có khả năng tra cứu web trực tiếp.' if os.getenv('CMC_CHAT_AGENT')=='1' and os.getenv('CMC_RAG_CLOUD')=='1' and getattr(engine,'data_mode',None)=='DEMO' else ''
        return _out('<p>Tôi là trợ lý học vụ CMC Academic AI. Tôi giải thích quy chế, tra cứu hồ sơ, GPA và dự báo thôi học trong phạm vi quyền của tài khoản. Bạn có thể hỏi bằng tiếng Việt tự nhiên.'+extra+'</p>')
    if _match(r'(?:cam on(?: ban| bot)?|thanks|thank you)', text):
        _clear(context)
        return _out('<p>Không có gì! Bạn cứ hỏi thêm nếu cần.</p>')
    identity = _match(_SELF + r' (?:la ai|la sinh vien nao|dang hoc (?:nganh|lop) gi|hoc (?:nganh|lop) nao|co ma sinh vien gi)', text)
    concerns = _match(_SELF + r' (?:dang )?(?:mac loi gi|bi loi gi|gap van de gi(?: ve hoc tap)?|co van de gi(?: ve hoc tap)?|co diem gi can chu y|can chu y gi(?: ve hoc tap)?|hoc tap (?:ra sao|the nao)|co dang gap van de gi)', text) or _match(r'(?:tinh trang|tinh hinh|ket qua) hoc tap cua ' + _SELF + r' (?:the nao|ra sao|nhu the nao)', text)
    if identity or concerns:
        _clear(context)
        if principal.role != 'student':
            role = {'advisor':'Cố vấn học tập', 'admin':'Quản trị viên'}.get(principal.role, 'Tài khoản')
            intro = f'<p>Bạn đang đăng nhập với vai trò {html.escape(role)}, mã tài khoản <b>{html.escape(principal.user_id)}</b>.</p>'
            if concerns:
                return _out(intro + '<p>Bạn muốn kiểm tra vấn đề của sinh viên hay nhóm sinh viên nào? Hãy nêu phạm vi được cấp quyền.</p>', 'clarify')
            return _out(intro, 'personal')
        row = row_to_clean_dict(permitted_student(engine, principal, principal.user_id).iloc[-1])
        if identity:
            fields = [('Mã sinh viên', principal.user_id), ('Lớp', row.get('class_code')), ('Ngành', row.get('major_name') or row.get('major_code'))]
            values = '; '.join(f'{label}: <b>{html.escape(str(value))}</b>' for label,value in fields if value)
            return _out(f'<p>Bạn đang đăng nhập với tài khoản sinh viên. {values}.</p>', 'personal')
        term = row.get('snapshot_term_index')
        when = f' (học kỳ {html.escape(str(term))})' if term is not None else ''
        alerts = engine.evaluate_student_warnings(row)
        lines = ''.join(f'<li><b>{html.escape(str(a["title"]))}</b>: {html.escape(str(a["detail"]))}</li>' for a in alerts)
        body = '<ul>' + lines + '</ul>' if lines else '<p>Chưa có kết quả đánh giá đủ căn cứ để xác nhận tình trạng.</p>'
        return _out('<p><b>Những điểm cần chú ý trong hồ sơ gần nhất' + when + ':</b></p>' + body + '<p>Đây là đánh giá học vụ từ dữ liệu hiện có, không phải kết luận kỷ luật. Nguy cơ dự báo không đồng nghĩa vi phạm.</p>', 'personal')
    generic_leave = _match(r'(?:(?:' + _SELF + r')\s+)?(?:co the |muon |co duoc |xin |duoc )?nghi hoc(?: duoc)?', text)
    temporary = _match(r'(?:(?:' + _SELF + r')\s+)?(?:co the |muon |xin |co duoc |duoc )?(?:bao luu(?: ket qua hoc tap)?|nghi hoc tam thoi|nghi tam thoi)(?: duoc)?', text)
    permanent = _match(r'(?:(?:' + _SELF + r')\s+)?(?:co the |muon |xin |co duoc |duoc )?(?:nghi hoc han|thoi hoc(?: han)?|bo hoc han)(?: duoc)?', text)
    absence = _match(r'(?:(?:' + _SELF + r')\s+)?(?:co the |muon |xin |co duoc |duoc )?nghi(?: hoc)? (?:mot|1) buoi(?: hoc)?(?: duoc)?', text)
    if prior_leave:
        temporary = temporary or _match(r'(?:tam thoi|tam nghi|bao luu)', text)
        permanent = permanent or _match(r'(?:nghi han|thoi hoc han|han luon)', text)
        absence = absence or _match(r'(?:mot|1) buoi(?: hoc)?', text)
    if generic_leave:
        _clear(context)
        context['basic_leave'] = True
        return _out('<p>Có thể xin nghỉ học, nhưng điều kiện tùy hình thức nghỉ. Bạn muốn <b>nghỉ một buổi</b>, <b>tạm nghỉ/bảo lưu</b>, hay <b>thôi học hẳn</b>? Tôi sẽ hướng dẫn theo trường hợp của bạn.</p>', 'clarify')
    if temporary or permanent or absence:
        # Keep existing detailed policy questions on their normal path. The
        # whole-utterance forms also work outside a prior leave conversation.
        _clear(context)
        if temporary:
            return _policy(retrieval, 'Điều 21 Khoản 1', 'Bạn có thể xin nghỉ tạm thời và bảo lưu nếu đáp ứng Điều 21.1. Nếu vì lý do cá nhân: đã học ít nhất một học kỳ, không thuộc diện xem xét buộc thôi học hoặc kỷ luật, nộp đơn và minh chứng cho Phòng Đào tạo; tối đa ba học kỳ. Các lý do đặc biệt có điều kiện riêng.', {'D21.K1'})
        if permanent:
            return _policy(retrieval, 'Điều 21 Khoản 3', 'Bạn có thể xin thôi học vì lý do cá nhân nếu không thuộc diện xem xét buộc thôi học hoặc kỷ luật. Nếu muốn học lại, phải dự tuyển lại; kết quả đã tích lũy được xem xét công nhận theo quy định. Căn cứ: Điều 21.2–3.', {'D21.K2–3'})
        return _policy(retrieval, 'Điều 25 Khoản 1', 'Nghỉ một buổi không tự động đồng nghĩa bị cấm thi. Cần xin phép và theo dõi số giờ vắng của học phần: nghỉ không phép quá 20% hoặc tổng nghỉ quá 30% số giờ học thì không được dự thi và phải học lại. Chưa thể kết luận chỉ từ số buổi nghỉ. Căn cứ: Điều 25.1.', {'D25.K1'})
    return None
