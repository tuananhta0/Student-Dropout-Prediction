"""Công cụ học vụ cục bộ và bộ điều phối V3 tùy chọn cho DEMO.
Không gửi hồ sơ từ database hoặc kết quả công cụ cho LLM.
"""
import html
import re
from fastapi import HTTPException
from core.chatbot_engine import row_to_clean_dict, risk_display, strip_accents
from core.security import permitted_student, require_role, require_major, require_class, scoped_classes, scoped_frame, assigned_classes, ADVISOR_MAJORS
from core.retrieval import numbers, public_focus, unsupported, normalize_query, span
from core.academic_rules import graduation_conditions, projection, number, whole
from core.chat_query_v2 import scoped_analytics, focused_policy, student_risk
from core.chat_data_v3 import aggregate_v3, personal_v3, numeric_policy, model_explanation, unavailable_aggregate, simple_major_count
from core.policy_focus_v3 import normalize_chat, facet_ids, specific_facets, ambiguous_penalty
from core.conversation_basics import basic_conversation


def reply(text, intent):
    return {'reply':text, 'regulations':[], 'intent':intent, 'method':'python'}


def _fmt_num(val, fmt='{:.2f}', empty='Chưa có'):
    return empty if val is None else fmt.format(val)


def _render_model_html(m):
    specs = [
        ('ROC-AUC', 'test_roc_auc', 'Khả năng phân biệt nhóm nguy cơ và nhóm an toàn'),
        ('Average Precision', 'test_pr_auc', 'Độ chính xác trung bình trên nhóm có nguy cơ (PR-AUC)'),
        ('Brier Score', 'test_brier', 'Độ lệch hiệu chuẩn xác suất (càng nhỏ càng tốt)'),
        ('Recall', 'test_recall', 'Tỷ lệ phát hiện đúng sinh viên thuộc diện thôi học'),
        ('Precision', 'test_precision', 'Tỷ lệ cảnh báo đúng trong nhóm được gắn cờ'),
        ('Recall@Top10%', 'test_recall_top10', 'Tỷ lệ phát hiện trong nhóm 10% nguy cơ cao nhất'),
        ('Ngưỡng', 'threshold', 'Ngưỡng xác suất phân loại sinh viên cần hỗ trợ sớm'),
    ]
    rows = ''.join(
        f'<tr class="border-b border-[#2c2e30]">'
        f'<td class="py-2 px-2.5 font-semibold text-white">{label}</td>'
        f'<td class="py-2 px-2.5 font-mono text-blue-300 font-bold">{html.escape(str(m[key]))}</td>'
        f'<td class="py-2 px-2.5 text-[#c4c7c5]">{desc}</td></tr>'
        for label, key, desc in specs if key in m
    )
    ver = html.escape(str(m.get('version', 'baseline')))
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div class="text-sm font-bold text-blue-300">Chỉ số đánh giá mô hình AI ({ver})</div>'
        f'<div class="overflow-x-auto"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2.5">Chỉ số</th><th class="py-1.5 px-2.5">Giá trị</th><th class="py-1.5 px-2.5">Ý nghĩa</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
        f'<div class="text-[11px] text-[#9aa0a6]">Kết quả kiểm định trên tập dữ liệu mô phỏng.</div></div>'
    )


def _render_school_overview_html(active_cnt, valid_cnt, dropout_cnt, total_cnt, rate, debt):
    cov = f'{100 * valid_cnt / active_cnt:.1f}%' if active_cnt else '0.0%'
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-3 text-xs">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-white">Báo cáo tổng quan toàn trường</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">Phạm vi quản trị</span></div>'
        f'<div class="overflow-x-auto"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2.5">Chỉ tiêu thống kê</th><th class="py-1.5 px-2.5 text-right">Giá trị</th><th class="py-1.5 px-2.5">Ghi chú phạm vi</th>'
        f'</tr></thead><tbody>'
        f'<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-medium text-white">Sinh viên đang theo học</td><td class="py-2 px-2.5 text-right font-bold text-blue-300">{active_cnt:,} SV</td><td class="py-2 px-2.5 text-[#c4c7c5]">Chỉ tính hồ sơ đã có kết quả học kỳ tại mốc dữ liệu</td></tr>'
        f'<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-medium text-white">Có dự báo nguy cơ AI hợp lệ</td><td class="py-2 px-2.5 text-right font-bold text-emerald-300">{valid_cnt:,} / {active_cnt:,} SV ({cov})</td><td class="py-2 px-2.5 text-[#c4c7c5]">Hồ sơ đủ điều kiện chấm điểm dự báo còn hiệu lực</td></tr>'
        f'<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-medium text-white">Hồ sơ ở trạng thái thôi học</td><td class="py-2 px-2.5 text-right font-bold text-rose-400">{dropout_cnt:,} / {total_cnt:,} ({rate})</td><td class="py-2 px-2.5 text-[#c4c7c5]">Tỷ lệ trạng thái tại mốc dữ liệu, không phải tỷ suất theo kỳ</td></tr>'
        f'<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-medium text-white">Tổng công nợ học phí (nhóm đang học)</td><td class="py-2 px-2.5 text-right font-bold text-amber-300">{debt:,.0f} đồng</td><td class="py-2 px-2.5 text-[#c4c7c5]">Tổng công nợ cuối kỳ gần nhất trên dữ liệu mô phỏng</td></tr>'
        f'</tbody></table></div></div>'
    )


def _render_classes_table_html(title, rows, subtitle=''):
    trs = []
    for idx, r in enumerate(rows, 1):
        gpa = r.get('gpa_mean')
        owed = r.get('owed_mean')
        warn = r.get('warning_count')
        hr = r.get('high_risk_count', 0)
        valid = r.get('valid_predictions')
        n_stu = r.get('n_students', 0)
        gpa_cls = 'text-amber-300 font-semibold' if (gpa is not None and gpa < 2.5) else 'text-emerald-300 font-semibold'
        hr_cls = 'text-rose-400 font-bold' if hr > 0 else 'text-slate-400'
        cov_str = f'{valid}/{n_stu}' if valid is not None else '—'
        trs.append(
            f'<tr class="border-b border-[#2c2e30]">'
            f'<td class="py-2 px-2 text-[#9aa0a6]">{idx}</td>'
            f'<td class="py-2 px-2 font-bold text-blue-300 whitespace-nowrap">{html.escape(str(r["class_code"]))}</td>'
            f'<td class="py-2 px-2 text-center">{n_stu}</td>'
            f'<td class="py-2 px-2 text-center {gpa_cls}">{_fmt_num(gpa)}</td>'
            f'<td class="py-2 px-2 text-center">{_fmt_num(owed, "{:.1f}", "—")}</td>'
            f'<td class="py-2 px-2 text-center text-amber-300">{warn if warn is not None else "—"}</td>'
            f'<td class="py-2 px-2 text-center {hr_cls}">{hr}</td>'
            f'<td class="py-2 px-2 text-right text-[#c4c7c5]">{cov_str}</td>'
            f'</tr>'
        )
    sub_html = f'<div class="text-[11px] text-[#9aa0a6]">{html.escape(subtitle)}</div>' if subtitle else ''
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div><div class="text-sm font-bold text-blue-300">{html.escape(title)}</div>{sub_html}</div>'
        f'<div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2">#</th><th class="py-1.5 px-2">Mã lớp</th><th class="py-1.5 px-2 text-center">Sĩ số</th>'
        f'<th class="py-1.5 px-2 text-center">GPA TB</th><th class="py-1.5 px-2 text-center">Nợ TB (TC)</th>'
        f'<th class="py-1.5 px-2 text-center">Cảnh báo Đ18</th><th class="py-1.5 px-2 text-center">Cần hỗ trợ sớm</th>'
        f'<th class="py-1.5 px-2 text-right">Có dự báo AI</th>'
        f'</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'
        f'<div class="text-[11px] text-[#9aa0a6]">Ghi chú: "Cần hỗ trợ sớm" là số sinh viên vượt ngưỡng dự báo của mô hình AI, tách biệt với cảnh báo học vụ theo Điều 18.</div></div>'
    )


def _render_faculty_html(d):
    n_stu = d.get('n_students', 0)
    n_cls = d.get('n_classes', 0)
    valid = d.get('valid_predictions', 0)
    classes = d.get('classes') or []
    trs = []
    for idx, r in enumerate(classes[:15], 1):
        gpa = r.get('gpa_mean')
        owed = r.get('owed_mean')
        warn = r.get('warning_count', 0)
        hr = r.get('high_risk_count', 0)
        gpa_cls = 'text-amber-300 font-semibold' if (gpa is not None and gpa < 2.5) else 'text-emerald-300 font-semibold'
        hr_cls = 'text-rose-400 font-bold' if hr > 0 else 'text-slate-400'
        trs.append(
            f'<tr class="border-b border-[#2c2e30]">'
            f'<td class="py-2 px-2 text-[#9aa0a6]">{idx}</td>'
            f'<td class="py-2 px-2 font-bold text-blue-300 whitespace-nowrap">{html.escape(str(r["class_code"]))}</td>'
            f'<td class="py-2 px-2 text-center">{r.get("n_students", 0)}</td>'
            f'<td class="py-2 px-2 text-center {gpa_cls}">{_fmt_num(gpa)}</td>'
            f'<td class="py-2 px-2 text-center">{_fmt_num(owed, "{:.1f}", "—")}</td>'
            f'<td class="py-2 px-2 text-center text-amber-300">{warn}</td>'
            f'<td class="py-2 px-2 text-right {hr_cls}">{hr}</td></tr>'
        )
    table_html = (
        f'<div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2">#</th><th class="py-1.5 px-2">Mã lớp</th><th class="py-1.5 px-2 text-center">Sĩ số</th>'
        f'<th class="py-1.5 px-2 text-center">GPA TB</th><th class="py-1.5 px-2 text-center">Nợ TB (TC)</th>'
        f'<th class="py-1.5 px-2 text-center">Cảnh báo Đ18</th><th class="py-1.5 px-2 text-right">Cần hỗ trợ sớm</th>'
        f'</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'
    ) if trs else ''
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-white">Khoa {html.escape(str(d.get("major_name", "")))} ({html.escape(str(d.get("major_code", "")))})</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">{n_cls} lớp · {n_stu:,} SV</span></div>'
        f'<div class="text-[#c4c7c5]">GPA TB: <b class="text-emerald-300">{_fmt_num(d.get("gpa_mean"), "{:.2f}")}</b> · '
        f'Nợ TB: <b class="text-amber-300">{_fmt_num(d.get("owed_mean"), "{:.1f} TC")}</b> · '
        f'Cảnh báo Đ18: <b class="text-amber-300">{d.get("warning_count", 0):,} SV</b> · '
        f'Cần hỗ trợ sớm (AI): <b class="text-rose-400">{d.get("high_risk_count", 0):,} SV</b> '
        f'<span class="text-[#9aa0a6]">({valid:,}/{n_stu:,} có dự báo)</span></div>'
        f'{table_html}'
        f'<div class="text-[11px] text-[#9aa0a6]">Nhập mã lớp (ví dụ: IT25-12) để xem danh sách sinh viên trong lớp.</div></div>'
    )


def _render_class_detail_html(d):
    n_stu = d.get('n_students', 0)
    valid = d.get('valid_predictions', 0)
    students = d.get('students') or []
    rows = []
    for s in students:
        prob = s.get('risk_probability')
        status = s.get('prediction_status')
        risk_str = f'{prob * 100:.1f}%' if (status == 'VALID' and prob is not None) else html.escape(str(s.get('risk_display') or '—'))
        rows.append(
            f'<tr class="border-b border-[#2c2e30]">'
            f'<td class="py-2 px-2 font-mono font-bold text-blue-300">{html.escape(str(s.get("student_id", "")))}</td>'
            f'<td class="py-2 px-2 text-center">HK{s.get("snapshot_term_index", "")}</td>'
            f'<td class="py-2 px-2 text-center font-semibold">{_fmt_num(s.get("gpa4_cumulative"))}</td>'
            f'<td class="py-2 px-2 text-center">{_fmt_num(s.get("credits_owed_cumulative"), "{:.0f} TC", "—")}</td>'
            f'<td class="py-2 px-2 text-center">{_fmt_num(s.get("attendance_rate_term"), "{:.0f}%", "—")}</td>'
            f'<td class="py-2 px-2 text-right font-semibold text-rose-300">{risk_str}</td></tr>'
        )
    table_html = (
        f'<div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2">Mã SV</th><th class="py-1.5 px-2 text-center">Kỳ</th>'
        f'<th class="py-1.5 px-2 text-center">GPA TL</th><th class="py-1.5 px-2 text-center">TC Nợ</th>'
        f'<th class="py-1.5 px-2 text-center">Chuyên cần</th><th class="py-1.5 px-2 text-right">Nguy cơ AI</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    ) if rows else ''
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-white">Lớp {html.escape(str(d.get("class_code", "")))} · {html.escape(str(d.get("major_name", "")))}</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">{n_stu} SV đang học</span></div>'
        f'<div class="text-[#c4c7c5]">GPA TB: <b class="text-emerald-300">{_fmt_num(d.get("gpa_mean"), "{:.2f}")}</b> · '
        f'Nợ TB: <b class="text-amber-300">{_fmt_num(d.get("owed_mean"), "{:.1f} TC")}</b> · '
        f'Cảnh báo Đ18: <b class="text-amber-300">{d.get("warning_count", 0)} SV</b> · '
        f'Cần hỗ trợ sớm: <b class="text-rose-400">{d.get("high_risk_count", 0)} SV</b> '
        f'<span class="text-[#9aa0a6]">({valid}/{n_stu} có dự báo)</span></div>'
        f'{table_html}'
        f'<div class="text-[11px] text-[#9aa0a6]">Nhập mã sinh viên (ví dụ: BIT250093) để xem hồ sơ riêng của từng sinh viên.</div></div>'
    )


def _render_grouped_stats_html(title, group_col_title, rows_data):
    trs = ''.join(
        f'<tr class="border-b border-[#2c2e30]">'
        f'<td class="py-2 px-2.5 font-semibold text-white">{html.escape(label)}</td>'
        f'<td class="py-2 px-2.5 text-center">{total:,}</td>'
        f'<td class="py-2 px-2.5 text-center text-rose-400 font-semibold">{dropout:,}</td>'
        f'<td class="py-2 px-2.5 text-center font-bold text-amber-300">{rate}</td>'
        f'<td class="py-2 px-2.5 text-right text-blue-300">{valid:,} / {active:,} SV</td></tr>'
        for label, total, dropout, rate, valid, active in rows_data
    )
    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div class="text-sm font-bold text-blue-300">{html.escape(title)}</div>'
        f'<div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2.5">{html.escape(group_col_title)}</th>'
        f'<th class="py-1.5 px-2.5 text-center">Tổng hồ sơ</th>'
        f'<th class="py-1.5 px-2.5 text-center">Đã thôi học</th>'
        f'<th class="py-1.5 px-2.5 text-center">Tỷ lệ thôi học</th>'
        f'<th class="py-1.5 px-2.5 text-right">Đang học có dự báo AI</th>'
        f'</tr></thead><tbody>{trs}</tbody></table></div>'
        f'<div class="text-[11px] text-[#9aa0a6]">Thống kê trên hồ sơ đã có kết quả học kỳ tại mốc dữ liệu; tỷ lệ thôi học phản ánh trạng thái tích lũy, không phải tỷ suất theo kỳ.</div></div>'
    )


def _render_risk_html(target, label, row, term_idx):
    cutoff = row.get('snapshot_cutoff') or ''
    cutoff_note = f' (mốc dữ liệu {html.escape(str(cutoff))})' if cutoff else ''
    return (
        f'<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 space-y-2 text-xs leading-relaxed">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-blue-300">Đánh giá nguy cơ thôi học (Mô hình AI) — {html.escape(str(target))}</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">HK{term_idx}</span></div>'
        f'<div class="text-white font-semibold">{html.escape(label)}</div>'
        f'<div class="text-[#c4c7c5]"><b>Phạm vi dự báo:</b> Áp dụng cho hai học kỳ tiếp theo dựa trên kết quả học tập đã chốt đến hết HK{term_idx}{cutoff_note}.</div>'
        f'<div class="text-[11px] text-[#9aa0a6]"><b>Lưu ý học vụ:</b> Chỉ số này do mô hình học máy ước lượng nhằm hỗ trợ cảnh báo sớm và tư vấn học tập, không phải quyết định học vụ hay kết luận buộc thôi học.</div></div>'
    )


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
        r'(?:muc tieu(?:\s+(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?)?|(?:keo|nang|tang|cai thien|muon dat|can dat|de dat|dat duoc)\s+(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?)(?:\s+cua\s+(?:em|toi|minh))?\s*(?:(?:la|dat|duoc|len|thanh|toi|den|o muc|muc|it nhat|toi thieu|tu|tren|khoang|=)\s*){0,3}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tin|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?(?:\s+cua\s+(?:em|toi|minh))?\s*(?:muc tieu\s*)?(?:len|thanh|toi|den|o muc|muc)\s*(?:(?:duoc|it nhat|toi thieu|tu|tren|khoang)\s*){0,2}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tin|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(?:muc tieu|muon dat|can dat|de dat|dat duoc|dat muc|keo len|nang len|tang len|muon len|len muc|len|toi muc|den muc)\s+(?:(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?\s*)?(?:(?:la|dat|duoc|len|thanh|muc|it nhat|toi thieu|tu|tren|khoang|=)\s*){0,3}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tin|tc|hoc ky|hk|mon|lan|%)\b)',
        r'(-?\d+(?:[.,]\d+)?)\s*(?:gpa|diem trung binh|dtb)\b',
        r'(?:gpa|diem trung binh|dtb)(?:\s+tich luy)?\s*(?:(?:la|dat|it nhat|toi thieu|=)\s*){0,2}(-?\d+(?:[.,]\d+)?)(?!\s*(?:tin chi|tin|tc|hoc ky|hk|mon|lan|%)\b)',
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match
    return None


def _route_one(engine, retrieval, principal, message, context, allow_cloud=True):
    message=normalize_chat(message)
    q=re.sub(r"\s+", " ", strip_accents(message))
    # "Học lớp cải thiện" và "học cải thiện" là cùng ý định tra cứu Điều 15.
    q=re.sub(r'\bhoc\s+lop\s+cai\s+thien\b', 'hoc cai thien', q)
    q=re.sub(r'\blop\s+cai\s+thien\b', 'hoc cai thien', q)
    simple=re.sub(r'[^a-z0-9 ]','',q).strip()
    if simple in ('xin chao','chao','hello','hi','chao ban','cam on','cam on ban','thanks','huong dan','ban lam duoc gi'):
        context.pop('article',None)
        return reply('<p>Xin chào! Tôi hỗ trợ tra cứu quy chế, bảng điểm, GPA và nguy cơ thôi học trong phạm vi tài khoản. Bạn muốn xem nội dung nào?</p>','greeting')
    parsed=engine.parse_flexible_query(message)
    ids=set(re.findall(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6})\b',message.upper()))
    stripped_msg=re.sub(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6})\b',' ',message.upper())
    for m_code in re.finditer(r'(?<![A-Z0-9_])(?:SV|MSV)?(\d{6})(?!\s*(?:D|Đ|VND|DONG|TRIEU|NGHIN|K)\b)(?![A-Z0-9_])',stripped_msg):
        before_code=strip_accents(stripped_msg[:m_code.start()]).strip()
        if re.search(r'(?:thanh toan|nop|dong|tra|so tien|hoc phi|no)\s*$',before_code):
            continue
        code=m_code.group(1)
        matched_ids=sorted(set(engine.df['student_id'].astype(str).str.upper()[engine.df['student_id'].astype(str).str.upper().str.endswith(code)])) if hasattr(engine,'df') and 'student_id' in engine.df else []
        if len(matched_ids)==1:
            ids.add(matched_ids[0])
        elif principal.role=='student' and not principal.user_id.upper().endswith(code):
            raise HTTPException(403,'Sinh viên chỉ được phép xem hồ sơ của chính mình.')
        elif matched_ids:
            return reply(f'<p>Mã rút gọn {html.escape(code)} trùng nhiều sinh viên; hãy nhập mã đầy đủ (ví dụ {html.escape(matched_ids[0])}).</p>','clarify')
        else:
            return reply(f'<p>Không tìm thấy sinh viên có mã {html.escape(code)}. Hãy nhập mã sinh viên đầy đủ (ví dụ BIT250007).</p>','clarify')
    if principal.role=='student' and not ids and any(x in q for x in ('sinh vien khac','sv khac','nguoi khac','ban khac','ho so khac')):
        raise HTTPException(403,'Sinh viên chỉ được phép xem hồ sơ của chính mình.')
    classes=set(re.findall(r'\b[A-Z]{2}\d{2}-\d{2,3}\b',message.upper()))
    # Kiểm tra mọi mã, không chỉ mã đầu tiên do parser chọn.
    for uid in ids: permitted_student(engine,principal,uid)
    for cls in classes:
        require_role(principal,'advisor','admin')
        require_class(principal,cls)
    if len(ids)>1 or len(classes)>1 or (ids and classes):
        return reply('<p>Hãy tra cứu một sinh viên hoặc một lớp mỗi lần.</p>','clarify')
    arts,clauses=numbers(message)
    if not ids and not classes and not arts and not clauses:
        basic = basic_conversation(engine, retrieval, principal, q, context)
        if basic is not None:
            return basic
    context.pop('basic_leave',None)
    if not ids and not classes and ambiguous_penalty(message):
        context.pop('article',None)
        return retrieval.answer(message,cloud=False)
    if not arts and not clauses and not ids and not classes and principal.role in ('advisor','admin') and any(x in q for x in ('gop y qua chatbot','gop y cua sinh vien','gop y cua sv','sv gop y','gop y sinh vien','gop y nao','tong hop gop y','gop y co an danh','tra loi lai gop y')):
        context.pop('article',None)
        return reply('<p>Bản demo chưa có kho dữ liệu góp ý sinh viên; không thể thống kê, xác định người gửi hoặc hứa chức năng phản hồi chưa được triển khai.</p>','none')
    if not ids and not arts and not clauses:
        advanced=aggregate_v3(engine,principal,q,parsed,classes)
        if advanced is not None:
            context.pop('article',None)
            return advanced
    if not ids and not classes and not arts and not clauses:
        advanced=numeric_policy(retrieval,q)
        if advanced is not None:
            context.pop('article',None)
            return advanced
    # Dedicated queries use only authenticated, locally scoped data.
    if not ids and not arts and not clauses:
        if principal.role in ('advisor','admin') or not any(x in q for x in ('cua em','cua toi','cua minh','em co','em da','em con')):
            focused = scoped_analytics(engine, principal, q, parsed, classes)
            if focused is not None:
                context.pop('article',None)
                return focused
    if not ids and not classes:
        focused = focused_policy(retrieval, q, arts, clauses)
        if focused is not None:
            context.pop('article',None)
            found={c['article'] for c in focused['regulations']}
            if len(found)==1: context['article']=next(iter(found))
            return focused
        if not arts and not clauses and not unsupported(message) and specific_facets(message):
            out=retrieval.answer(message,cloud=allow_cloud)
            out['intent']='regulation'
            context.pop('article',None)
            found={c['article'] for c in out['regulations']}
            if len(found)==1: context['article']=next(iter(found))
            return out
        if not arts and not clauses:
            focused = student_risk(engine, principal, q, parsed, ids)
            if focused is not None:
                context.pop('article',None)
                return focused
    if not ids and not arts and not clauses and not unsupported(message):
        advanced=unavailable_aggregate(engine,principal,q,parsed,classes)
        if advanced is not None:
            context.pop('article',None)
            return advanced
    if not ids and not classes and not arts and not clauses and not unsupported(message):
        advanced=model_explanation(engine,q)
        if advanced is not None:
            context.pop('article',None)
            return advanced
        advanced=personal_v3(engine,principal,q,parsed)
        if advanced is not None:
            context.pop('article',None)
            return advanced
    personal=bool(ids) or ('tot nghiep' in q and (bool(re.search(r'\b(?:em|toi|minh)\b',q)) or any(x in q for x in ('du dieu kien tot nghiep chua','tot nghiep duoc chua','kiem tra tot nghiep')))) or ('tôi đã' in message.lower()) or ('toi da' in q and 'tối đa' not in message.lower() and not any(x in q for x in ('gioi han','thoi gian','so lan','tin chi','muc toi da','quy che','quy dinh','dieu kien','hoc phan'))) or any(x in q for x in ('cua em','cua toi','cua minh','ban than','em co','em da','em con','toi co','toi con','ho so cua'))
    dropout_rate_query = bool(re.search(r'\b(?:t[iy]\s+le|du\s*(?:bao|doan)\s+t[iy]\s+le|cach\s+tinh\s+t[iy]\s+le)\s*(?:sinh\s+vien\s*)?(?:thoi\s+hoc|nghi\s+hoc|bo\s+hoc|dropout)\b', q)) or (bool(re.search(r'\bt[iy]\s+le\b', q)) and any(x in q for x in ('thoi hoc', 'nghi hoc', 'bo hoc', 'dropout')))
    faculty_stats_query = bool(re.search(r'\b(?:thong ke|tong ket|bao cao|tinh hinh)\s+(?:cua\s+|ve\s+|toan\s+)?(?:khoa|nganh)\b(?!\s+luan\b)', q)) or any(x in q for x in ('tong ket khoa','thong ke khoa','tong ket nganh','thong ke nganh','bao cao khoa','tinh hinh khoa'))
    aggregate = bool(classes) or dropout_rate_query or faculty_stats_query or bool(re.search(r'\b(?:thong ke|tong ket|bao cao|tinh hinh)\s+(?:cua\s+|ve\s+|toan\s+)?(?:lop|truong|sinh vien)\b', q)) or any(x in q for x in ('lop manh nhat','lop yeu nhat','toan truong','tong so sinh vien','thong ke lop','xem lop','danh sach lop','cac lop','top 5 lop','top lop','lop nguy co','lop can ho tro','xep hang lop','so sanh cac lop','hoc bong co giu chan','ty le thoi hoc','ti le thoi hoc','ty le bo hoc','ti le bo hoc','ty le nghi hoc','ti le nghi hoc','tong no hoc phi','thong ke theo','so sanh he','so sanh bac','so sanh khoa','so sanh cac khoa','so sanh nganh','thong ke cac khoa','tong quan truong'))
    admission='dau vao' in q or 'diem tuyen sinh' in q or 'phuong thuc tuyen sinh' in q
    risk=any(x in q for x in ('nguy co thoi hoc','nguy co nghi hoc','nguy co bo hoc','xac suat thoi hoc','xac suat nghi hoc','xac suat bo hoc','du bao thoi hoc','du bao nghi hoc','du bao bo hoc','du doan thoi hoc','du doan nghi hoc','du doan bo hoc','nguy co dropout','du bao dropout'))
    payment_query = any(x in q for x in ('thanh toan hoc phi','dong hoc phi','nop hoc phi','hoc phi phai tra','hoc phi phai dong','tien hoc phi phai tra','tien hoc phi phai dong','what-if','what if')) and not unsupported(message)
    target_match = extract_target_gpa(q)
    requested_registration = re.search(r'\bdang ky\s*(?:hoc\s*)?(-?\d+(?:[.,]\d+)?)\s*(?:tin chi|tin(?!\s*(?:tuc|nhan|dung))|tc)\b',q)
    if requested_registration and principal.role=='student': personal=True
    retake_cmd = bool(re.search(r'(?:hoc lai|cai thien)\s+\S+\s+(?:dat|len)\s+(?:a\+|a|b\+|b|c\+|c|d\+|d|f)(?:\s|$)', q))
    has_scenario_cue = any(x in q for x in ('kich ban', 'du doan', 'gia lap', 'what-if', 'what if', 'ky sau', 'ki sau', 'ky toi', 'ki toi'))
    improve_policy_cues = ('quy che', 'quy dinh', 'dieu kien', 'lay diem', 'tinh diem', 'tinh vao gpa', 'diem cao nhat', 'lan sau', 'duoc phep', 'co duoc', 'gioi han', 'bao nhieu lan', 'diem d', 'diem c', 'khac nhau', 'nhu the nao', 'the nao')
    passed_retake_policy = bool(
        'thi lai' in q and 'cai thien' in q
        and re.search(r'\bdiem\s*[abcd]\+?(?=\s|[.,!?;]|$)',q)
    )
    asking_improve_policy = (
        (any(x in q for x in ('hoc cai thien', 'cai thien diem')) or passed_retake_policy)
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
    scenario = gpa_improve_scenario or any(x in q for x in ('kich ban','du doan diem','du doan gpa','du bao gpa','du bao diem','tinh gpa','diem ky sau','diem ki sau','diem ky toi','diem ki toi','gpa ky sau','gpa ki sau','gpa ky toi','gpa ki toi','what-if','what if'))
    course_policy_guard = any(x in q for x in ('dang ky','gioi han','toi da','toi thieu','khoi luong','tien quyet','bat buoc','tu chon','rut hoc phan','rut mon','hoc cai thien','quy che','quy dinh','dieu kien','diem liet','thi lai','cam thi','phuc tra','phuc khao','luu tru','khoa luan','do an','cong nhan','bao luu','chuyen nganh','chuyen truong','canh bao','thoi hoc'))
    short_course_cue = any(x in q for x in ('bang diem','diem chi tiet','chi tiet mon','chi tiet hoc phan','cac hoc phan','cac mon','diem mon','mon truot','mon rot','mon no','diem f'))
    course = not course_policy_guard and not arts and ((parsed['is_course'] and (personal or q.startswith(('bang diem','xem bang diem','xem diem','xem mon','xem hoc phan','diem mon')))) or (short_course_cue and (personal or principal.role=='student' or bool(context.get('target')))))
    model=any(x in q for x in ('roc-auc','pr-auc','average precision','recall','precision','brier','thong so mo hinh','chi so mo hinh','danh gia mo hinh','do chinh xac mo hinh','hieu nang mo hinh','ket qua mo hinh')) and not personal
    follow_clause=not arts and clauses and context.get('article') and not ids and not personal and not aggregate
    if follow_clause: message=f'{message} Điều {context["article"]}'; arts=[context['article']]
    context.pop('article',None)
    if follow_clause: pass
    elif clauses and not arts and not personal and not aggregate:
        return reply('<p>Khoản này thuộc Điều nào? Hãy nhập số Điều để tra cứu chính xác.</p>','clarify')
    if len(arts)>1 and clauses:
        return reply('<p>Hãy tra cứu một Điều và Khoản mỗi lần để tránh nhầm trích dẫn.</p>','clarify')
    if model:
        return reply(_render_model_html(engine.meta),'model')
    if aggregate:
        require_role(principal,'advisor','admin')
        is_grouped=any(x in q for x in ('thong ke theo','so sanh he','so sanh bac','so sanh khoa','so sanh cac khoa','so sanh nganh','thong ke cac khoa'))
        if dropout_rate_query and not is_grouped:
            if any(x in q for x in ('toan truong','tong quan truong','cua truong')):
                require_role(principal,'admin')
            major_q=parsed['major_code']
            if major_q: require_major(principal,major_q)
            sub=scoped_frame(principal,engine.latest_df)
            if classes:
                cls_code=next(iter(classes))
                sub=sub[sub['class_code'].astype(str).str.upper().eq(cls_code)]
                if sub.empty: raise HTTPException(404,'Không tìm thấy lớp trong phạm vi quản lý')
                scope_title=f'Lớp {cls_code}'
            elif major_q:
                sub=sub[sub['major_code'].astype(str).str.upper().eq(major_q)]
                m_name=str(sub.iloc[0]['major_name']) if not sub.empty and 'major_name' in sub.columns else major_q
                scope_title=f'Khoa {m_name} ({major_q})'
            elif principal.role=='admin':
                scope_title='Toàn trường'
            else:
                m_cfg=ADVISOR_MAJORS.get(principal.user_id)
                if isinstance(m_cfg,str): scope_title=f'Phạm vi quản lý — Ngành {m_cfg}'
                elif m_cfg and len(m_cfg)==1: scope_title=f'Phạm vi quản lý — Ngành {next(iter(m_cfg))}'
                else: scope_title='Phạm vi lớp được phân công'
            return reply(_render_dropout_rate_analysis_html(sub,engine.meta,scope_title),'analytics')
        if classes:
            d=engine.get_class_detail(next(iter(classes)))
            if not d: raise HTTPException(404,'Không tìm thấy lớp đang học')
            return reply(_render_class_detail_html(d),'class')
        if any(x in q for x in ('toan truong','tong quan truong','tong so sinh vien','ty le thoi hoc','ty le bo hoc','ty le nghi hoc','tong no hoc phi')) and not is_grouped:
            require_role(principal,'admin')
            active=engine.active_latest_df
            valid=int(active.prediction_status.eq('VALID').sum())
            debt=float(active.tuition_debt_end_vnd.sum())
            count=int(engine.latest_df.state_as_of.eq('DROPOUT').sum())
            total=len(engine.latest_df); rate=f'{count/total:.1%}' if total else 'Chưa có dữ liệu'
            return reply(_render_school_overview_html(len(active),valid,count,total,rate,debt),'analytics')
        major=parsed['major_code']
        if major: require_major(principal,major)
        if faculty_stats_query:
            if not major and principal.role=='advisor':
                m_cfg=ADVISOR_MAJORS.get(principal.user_id)
                if isinstance(m_cfg,str): major=m_cfg
                elif m_cfg and len(m_cfg)==1: major=next(iter(m_cfg))
            if not major: return reply('<p>Hãy nhập mã ngành cần tổng kết, ví dụ: Tổng kết khoa IT.</p>','clarify')
            if principal.role=='admin' or assigned_classes(principal) is None:
                d=engine.get_faculty_detail(major)
                if not d: return reply('<p>Không có dữ liệu khoa phù hợp.</p>','analytics')
                return reply(_render_faculty_html(d),'analytics')
            sub=scoped_frame(principal,engine.active_latest_df); sub=sub[sub.major_code.eq(major)]
            valid=int(sub.prediction_status.eq('VALID').sum())
            scoped_rows=scoped_classes(principal,engine.get_all_classes_summary(major))
            header=f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-3.5 text-xs"><div class="text-sm font-bold text-blue-300">Tổng hợp phạm vi lớp được phân công — Ngành {html.escape(major)}</div><div class="text-[#e3e3e3] mt-1">Đang quản lý <b>{len(sub)} SV</b> đang học; có <b>{valid}/{len(sub)} SV</b> có dự báo AI hợp lệ.</div></div>'
            return reply(('<div class="space-y-4">'+header+_render_classes_table_html('Chi tiết các lớp phụ trách',scoped_rows)+'</div>') if scoped_rows else header,'analytics')
        if is_grouped:
            sub=engine.latest_df
            sub=scoped_frame(principal,sub)
            if major: sub=sub[sub.major_code.eq(major)]
            if re.search(r'\bhe\b',q): col,g_title,col_lbl='education_mode','Thống kê theo hệ đào tạo','Hệ đào tạo'
            elif re.search(r'\b(?:bac|cu nhan|ky su)\b',q): col,g_title,col_lbl='degree_type','Thống kê theo bậc đào tạo','Bậc đào tạo'
            elif any(x in q for x in ('so sanh khoa','so sanh cac khoa','so sanh nganh','thong ke cac khoa')): col,g_title,col_lbl='major_code','So sánh thống kê giữa các khoa/ngành','Khoa / Ngành'
            else: col,g_title,col_lbl='cohort_year','Thống kê theo khóa tuyển sinh','Khóa'
            if major: g_title += f' — Ngành {major}'
            if col not in sub: return reply('<p>Chưa có dữ liệu nhóm.</p>','analytics')
            rows_data=[]
            for key,g in sub.groupby(col,dropna=False):
                active=g[g.state_as_of.eq('ENROLLED')]; valid=int(active.prediction_status.eq('VALID').sum()); dropout=int(g.state_as_of.eq('DROPOUT').sum())
                label={'Bachelor':'Cử nhân','Engineer':'Kỹ sư','SM':'Hệ tiêu chuẩn (SM)','GM':'Hệ song ngữ (GM)'}.get(str(key),str(key))
                rows_data.append((label,len(g),dropout,f'{dropout/len(g):.1%}',valid,len(active)))
            return reply(_render_grouped_stats_html(g_title,col_lbl,rows_data),'analytics')
        if 'hoc bong co giu chan' in q: return reply('<p>Chưa có bằng chứng nhân quả về tác động của học bổng trên dữ liệu quan sát hiện có.</p>','analytics')
        rows=engine.get_all_classes_summary(major)
        rows=scoped_classes(principal,rows)
        if not rows: return reply('<p>Không có lớp phù hợp trong phạm vi truy cập.</p>','analytics')
        if 'lop manh nhat' in q or 'lop yeu nhat' in q or 'top 5' in q or 'top lop' in q or 'xep hang lop' in q:
            asc='lop yeu nhat' in q
            rows=sorted([r for r in rows if r['gpa_mean'] is not None],key=lambda r: r['gpa_mean'],reverse=not asc)[:5]
            tbl_title='Top 5 lớp có GPA trung bình thấp nhất' if asc else 'Top 5 lớp có GPA trung bình cao nhất'
            tbl_sub='Xếp hạng dựa trên sinh viên đang học trong phạm vi được phân công.'
        else:
            rows=rows[:15]
            tbl_title=f'Danh sách lớp ngành {major} (ưu tiên hỗ trợ sớm)' if major else 'Danh sách lớp trong phạm vi quản lý (ưu tiên hỗ trợ sớm)'
            tbl_sub=f'Hiển thị {len(rows)} lớp có số sinh viên cần hỗ trợ sớm cao nhất.'
        return reply(_render_classes_table_html(tbl_title,rows,tbl_sub),'analytics')
    # Điều hoặc khoản rõ ràng luôn tra cứu nguồn, trừ câu hỏi rõ ràng về hồ sơ cá nhân.
    # Các giả định có số rõ ràng đi qua Rule Engine; không suy diễn từ hồ sơ.
    calculation = engine.evaluate_regulation_trap(message) if not ids and not aggregate and not target_match else ''
    if unsupported(message) and not calculation and not arts:
        result=retrieval.answer(message,cloud=False); result['intent']='regulation';return result
    pq=normalize_query(message)
    policy_improve = not ids and not gpa_improve_scenario and (
        any(x in pq for x in ('hoc cai thien', 'cai thien diem')) or passed_retake_policy
    )
    # Hỏi cách tính GPA theo quy chế là câu lý thuyết, không phải yêu cầu
    # đọc GPA cá nhân chỉ vì trong câu chứa 'GPA tích lũy'.
    # 'Cho em biết số lần cảnh báo ...' là yêu cầu giải thích ngưỡng theo
    # quy chế, không phải xem lịch sử cảnh báo của tài khoản đang đăng nhập.
    # Yêu cầu lịch sử cá nhân rõ ràng vẫn đi qua phân quyền/hồ sơ như cũ.
    warning_policy_query = bool(
        re.search(r'\b(?:so|may|bao nhieu|qua|tren)\s+lan\s+(?:bi\s+)?canh bao\b',pq)
        and any(x in pq for x in ('buoc thoi hoc','duoi hoc','bi thoi hoc','bao nhieu','quy dinh','quy che','gioi han','so lan'))
        and not any(x in pq for x in ('cua em','cua toi','cua minh','em da bi','toi da bi','em hien co','lich su canh bao','ho so cua'))
    )
    if warning_policy_query:
        pq += ' canh bao hoc tap buoc thoi hoc'
    policy_gpa_explanation = ('gpa tich luy' in pq and
        any(x in pq for x in ('cong thuc gpa', 'tinh tu', 'duoc tinh', 'tinh nhu the nao')) and
        not any(x in pq for x in ('gpa cua em', 'gpa cua toi', 'gpa cua minh', 'ho so cua')))
    policy_only=not ids and not target_match and not gpa_improve_scenario and not scenario and not payment_query and not risk and not course and (policy_improve or any(x in pq for x in ('bao luu','nghi hoc tam thoi','hoc phan tien quyet','chuyen nganh','chuyen truong','phuc tra','thoi gian hoc toi da','dang ky tin chi','cam thi','dieu kien du thi')) or (not personal and bool(public_focus(message)) and not admission and not any(x in q for x in ('kich ban','gpa hoc ky','gpa tich luy','gpa hien tai','gpa ky sau','gpa ki sau','gpa ky toi','gpa ki toi','du doan gpa','du bao gpa','hoc phi con no','no hoc phi','cong no hoc phi','tin chi da dat','tin chi no dong','tin chi no','so lan canh bao hoc tap','canh bao lien tiep','xem ho so','ho so','bang diem','ket qua hoc tap'))))
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
        else:
            # Intent đã được xác nhận theo ngưỡng quy chế, nên truy xuất theo
            # chủ đề chuẩn hóa thay vì chuyển nhầm thành truy vấn hồ sơ cá nhân.
            query = message + ' cảnh báo học tập buộc thôi học' if warning_policy_query else message
            result=retrieval.answer(query,cloud=allow_cloud)
        result['intent']='regulation'
        found={c['article'] for c in result['regulations']}
        if len(found)==1 and not (set(arts)-found) and all(any(k in span(c['clause']) for c in result['regulations']) for k in clauses): context['article']=next(iter(found))
        return result
    if not ids and not classes and not arts and not clauses and not personal and not scenario and not aggregate and not unsupported(message):
        facets=facet_ids(message)
        if facets:
            result=retrieval.answer(message,cloud=allow_cloud); result['intent']='regulation'
            if result['regulations']:
                found={c['article'] for c in result['regulations']}
                if len(found)==1: context['article']=next(iter(found))
                return result
    personal_request=personal or (principal.role=='advisor' and bool(context.get('target')) and (risk or admission or scenario or course or payment_query)) or (principal.role=='student' and (risk or admission or scenario or course or payment_query or any(x in q for x in ('gio tu hoc','gio lam them','xem ho so','ho so','ket qua hoc tap','xem diem','bang diem','gpa tich luy','gpa hoc ky','gpa hien tai','diem trung binh hien tai','no hoc phi','cong no hoc phi','tin chi da dat','so lan canh bao hoc tap','ty le chuyen can','chuyen can','hoc phi con no','tin chi no dong','tin chi no','tin chi tich luy','diem trung binh tich luy','dang nhap lms','buoi gap co van','buoi phu dao','so ngay qua han','du dieu kien tot nghiep chua','tot nghiep duoc chua','kiem tra tot nghiep'))))
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
            term_idx=int(row.get('snapshot_term_index') or 1)
            if term is not None and prob is not None and (term!=int(rec.iloc[-1]['snapshot_term_index']) or row.get('prediction_status')!='VALID') and row.get('prediction_status') not in ('OUT_OF_SCOPE',):
                label=f'Dự báo lịch sử HK{term}: {prob*100:.1f}% (không dùng để can thiệp hiện tại)'
            return reply(_render_risk_html(target,label,row,term_idx),'risk')
        if not target_match and 'tot nghiep' in q and any(x in q for x in ('dieu kien','du','thieu','duoc','chua','khi nao','kiem tra','xet')):
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
            quantities=re.findall(r'(?<![\d.,-])(-?\d+(?:[.,]\d+)?)\s*(?:tin chi|tin(?!\s*(?:tuc|nhan|dung))|tc)\b',q)
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
        if payment_query:
            return reply(_render_tuition_payment_html(target, row), 'payment')
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
    result=retrieval.answer(message,cloud=allow_cloud);result['intent']='regulation'
    if calculation: result['reply']=calculation+result['reply']
    a={c['article'] for c in result['regulations']}
    if len(a)==1: context['article']=next(iter(a))
    else: context.pop('article',None)
    return result


# Only store public topic IDs or a metric enum, never raw questions or results.
_FOLLOW_PUBLIC = {
    16: ('Điều 16 Khoản 1', 'Phúc tra: xem thời hạn và quy trình trong căn cứ.'),
    21: ('Điều 21 Khoản 1', 'Nghỉ học tạm thời: thời hạn tùy lý do nghỉ.'),
    12: ('Điều 12', 'Đăng ký, rút học phần và khóa luận có thời hạn riêng.'),
}
_FOLLOW_METRIC = {'gpa tich luy':'GPA tích lũy của em', 'gpa hoc ky':'GPA học kỳ của em', 'tin chi tich luy':'Tín chỉ tích lũy của em'}


def _legacy_route(engine, retrieval, principal, message, context, allow_cloud=True):
    q=strip_accents(normalize_chat(message))
    topic=context.pop('v3_public',None)
    metric=context.pop('v3_metric',None)
    # Strictly bounded elliptical questions; a new subject never inherits context.
    if topic in _FOLLOW_PUBLIC and re.fullmatch(r'(?:con |the |vay )?(?:thoi han|han nop don|bao lau)(?: thi sao)?[?.! ]*',q):
        query,intro=_FOLLOW_PUBLIC[topic]
        out=retrieval.answer(query,cloud=False)
        out['reply']='<p>'+html.escape(intro)+'</p>'+out['reply'];out['intent']='regulation'
        context['article']=topic;context['v3_public']=topic
        return out
    follow=re.fullmatch(r'(?:con |the |vay )?(?:hoc ky|ky|hk)\s*(\d{1,2})(?: thi sao)?[?.! ]*',q)
    if follow and metric in _FOLLOW_METRIC and principal.role=='student':
        # _route_one always checks current authentication and record scope again.
        out=_route_one(engine,retrieval,principal,_FOLLOW_METRIC[metric]+' học kỳ '+follow[1],context,allow_cloud=allow_cloud)
        if out.get('intent')=='personal': context['v3_metric']=metric
        return out
    # Separate only clearly independent public policy questions. Existing
    # mixed-topic and private questions keep the original authorization path.
    parts=[p.strip() for p in re.split(r'[?;]+',message) if p.strip()]
    if (2<=len(parts)<=3 and not re.search(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6}|[A-Z]{2}\d{2}-\d{2,3})\b',message.upper())
        and all(facet_ids(p) and not unsupported(p) and not numbers(p)[0] and not numbers(p)[1] for p in parts)):
        outs=[_route_one(engine,retrieval,principal,p,{},allow_cloud=False) for p in parts]
        if all(o.get('intent')=='regulation' for o in outs):
            evidence=list({c['id']:c for o in outs for c in o['regulations']}.values())
            context.pop('article',None)
            return {'reply':''.join(f'<p><b>Ý {i+1}</b></p>'+o['reply'] for i,o in enumerate(outs)), 'regulations':evidence,'retrieved':evidence,'intent':'regulation','method':'bm25','generation_status':'NOT_REQUESTED'}
    out=_route_one(engine,retrieval,principal,message,context,allow_cloud=allow_cloud)
    arts={c['article'] for c in out.get('regulations',[])}
    if out.get('intent')=='regulation' and len(arts)==1: context['v3_public']=next(iter(arts))
    if principal.role=='student' and out.get('intent')=='personal':
        selected=[k for k in _FOLLOW_METRIC if k in q]
        if len(selected)==1:context['v3_metric']=selected[0]
    return out


def _render_tuition_payment_html(target, row):
    debt = number(row.get('tuition_debt_end_vnd'))
    overdue = number(row.get('tuition_overdue_days_end'))
    term_idx = int(row.get('snapshot_term_index') or 1)
    if debt is None:
        body = '<div class="text-[#c4c7c5]">Chưa có dữ liệu công nợ học phí trong hồ sơ.</div>'
    elif debt > 0:
        overdue_html = (
            f'<div class="text-amber-300">Thời gian quá hạn: <b>{overdue:.0f} ngày</b></div>'
            if overdue is not None and overdue > 0 else ''
        )
        body = (
            f'<div class="text-white">Số tiền học phí cần thanh toán: <b class="text-amber-300">{debt:,.0f} VNĐ</b></div>'
            f'{overdue_html}'
        )
    else:
        body = '<div class="text-emerald-300 font-semibold">Hiện tại không có học phí cần thanh toán (0 VNĐ).</div>'
    return (
        f'<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 text-xs space-y-1.5 leading-relaxed">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-blue-300">Thông tin thanh toán học phí — {html.escape(str(target))}</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">HK{term_idx}</span></div>'
        f'{body}</div>'
    )


def _render_dropout_rate_analysis_html(sub, meta, scope_title):
    total = len(sub)
    if total == 0:
        return '<p>Chưa có dữ liệu sinh viên trong phạm vi tra cứu.</p>'

    threshold = float(meta.get('threshold', 0.12))
    thr_pct = threshold * 100.0
    n_drop = int(sub['state_as_of'].eq('DROPOUT').sum())
    active = sub[sub['state_as_of'].eq('ENROLLED')]
    valid = active[active['prediction_status'].eq('VALID') & active['risk_probability'].notna()]
    n_valid = len(valid)
    hist_rate = n_drop / total * 100.0

    if n_valid > 0:
        probs = valid['risk_probability'].astype(float)
        expected_rate = float(probs.mean() * 100.0)
        n_flagged = int((probs >= threshold).sum())
        flagged_rate = n_flagged / n_valid * 100.0
        exp_str = f'{expected_rate:.1f}%'
        flag_str = f'{n_flagged:,} SV ({flagged_rate:.1f}%)'
    else:
        exp_str = '—'
        flag_str = '—'

    if 'major_code' in sub.columns and sub['major_code'].nunique(dropna=True) > 1:
        group_col, col_title = 'major_code', 'Khoa / Ngành'
    elif 'cohort_year' in sub.columns:
        group_col, col_title = 'cohort_year', 'Khóa'
    else:
        group_col, col_title = None, 'Phạm vi'

    rows_html = []
    if group_col:
        for key, g in sub.groupby(group_col, dropna=False):
            g_tot = len(g)
            g_drop = int(g['state_as_of'].eq('DROPOUT').sum())
            g_act = g[g['state_as_of'].eq('ENROLLED')]
            g_val = g_act[g_act['prediction_status'].eq('VALID') & g_act['risk_probability'].notna()]
            if len(g_val) > 0:
                gp = g_val['risk_probability'].astype(float)
                g_exp = f'{float(gp.mean() * 100.0):.1f}%'
                g_fl = int((gp >= threshold).sum())
                g_fl_s = f'{g_fl:,} ({g_fl / len(g_val) * 100.0:.1f}%)'
            else:
                g_exp, g_fl_s = '—', '—'
            lbl = f'Khóa {int(key)}' if (group_col == 'cohort_year' and str(key).isdigit()) else str(key)
            rows_html.append(
                f'<tr class="border-b border-[#2c2e30]">'
                f'<td class="py-2 px-2.5 font-semibold text-white">{html.escape(lbl)}</td>'
                f'<td class="py-2 px-2.5 text-center">{g_tot:,}</td>'
                f'<td class="py-2 px-2.5 text-center text-rose-400 font-semibold">{g_drop:,} ({g_drop / g_tot * 100.0:.1f}%)</td>'
                f'<td class="py-2 px-2.5 text-center text-amber-300 font-bold">{g_exp}</td>'
                f'<td class="py-2 px-2.5 text-right text-rose-300 font-semibold">{g_fl_s}</td></tr>'
            )
    else:
        rows_html.append(
            f'<tr class="border-b border-[#2c2e30]">'
            f'<td class="py-2 px-2.5 font-semibold text-white">{html.escape(scope_title)}</td>'
            f'<td class="py-2 px-2.5 text-center">{total:,}</td>'
            f'<td class="py-2 px-2.5 text-center text-rose-400 font-semibold">{n_drop:,} ({hist_rate:.1f}%)</td>'
            f'<td class="py-2 px-2.5 text-center text-amber-300 font-bold">{exp_str}</td>'
            f'<td class="py-2 px-2.5 text-right text-rose-300 font-semibold">{flag_str}</td></tr>'
        )

    return (
        f'<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-2.5 text-xs">'
        f'<div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">'
        f'<div class="text-sm font-bold text-white">Thống kê &amp; Dự báo tỷ lệ thôi học — {html.escape(scope_title)}</div>'
        f'<span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">Tỷ lệ dự báo AI: {exp_str}</span></div>'
        f'<div class="text-[#c4c7c5]">Đã nghỉ thực tế: <b class="text-white">{n_drop:,}/{total:,} ({hist_rate:.1f}%)</b> · '
        f'Dự báo kỳ vọng (AI): <b class="text-blue-300">{exp_str}</b> · '
        f'Cần hỗ trợ sớm (≥{thr_pct:.0f}%): <b class="text-rose-400">{flag_str}</b></div>'
        f'<div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs">'
        f'<thead><tr class="border-b border-[#333537] text-[#9aa0a6]">'
        f'<th class="py-1.5 px-2.5">{col_title}</th>'
        f'<th class="py-1.5 px-2.5 text-center">Tổng SV</th>'
        f'<th class="py-1.5 px-2.5 text-center">Đã nghỉ thực tế</th>'
        f'<th class="py-1.5 px-2.5 text-center">Tỷ lệ dự báo AI</th>'
        f'<th class="py-1.5 px-2.5 text-right">Cần hỗ trợ sớm</th>'
        f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>'
        f'<div class="text-[11px] text-[#9aa0a6]">Dự báo từ mô hình Random Forest + HistGradientBoosting (học từ dữ liệu khóa trước; tính trung bình xác suất trên {n_valid:,} SV đang học). Gõ "danh sách lớp" hoặc mã lớp để xem chi tiết.</div></div>'
    )


def route(engine, retrieval, principal, message, context):
    # Simple, fully specified counts are business queries, not LLM guesses.
    local=simple_major_count(engine,principal,strip_accents(normalize_chat(message)))
    if local is not None:
        context.clear()
        return local
    from core.chat_agent import enabled, route_agent
    if enabled(engine):
        return route_agent(_legacy_route, engine, retrieval, principal, message, context)
    context.pop('agent_context', None)
    return _legacy_route(engine, retrieval, principal, message, context)
