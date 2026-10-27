import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import google.generativeai as genai
from src.preprocess import StudentDataManager
from src.utils import (
    build_grouped_horizontal_tables,
    evaluate_academic_rules,
    classify_academic_standing,
    RegulationRetriever
)

st.set_page_config(page_title="Cổng Thông Tin Học Vụ - ĐH CMC", layout="wide")

st.markdown("""
<style>
    .section-header {
        font-weight: 700;
        font-size: 1.15rem;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        color: #f8fafc;
    }
    div[data-testid="stTable"] table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stTable"] th {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        text-align: center !important;
        padding: 8px 10px !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stTable"] td {
        text-align: center !important;
        padding: 10px 10px !important;
        font-size: 0.92rem !important;
    }
</style>
""", unsafe_allow_html=True)

def get_services():
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = StudentDataManager()
    if "reg_retriever" not in st.session_state:
        st.session_state.reg_retriever = RegulationRetriever()
    return st.session_state.data_manager, st.session_state.reg_retriever

data_manager, reg_retriever = get_services()


def render_student_charts(metrics: dict, dropout_prob: float):
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#0f172a",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#f8fafc",
        "text.color": "#f8fafc",
        "font.size": 9.5
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.patch.set_facecolor("#0f172a")

    gpa_4 = metrics["gpa_4"]
    benchmarks = [
        "Ngưỡng Cảnh báo HK1\n(Điều 18: 0.8)",
        "Ngưỡng Cảnh báo HK2+\n(Điều 18: 1.0)",
        "Xếp loại Trung bình\n(ĐK tốt nghiệp: 2.0)",
        "ĐK Làm Khóa luận\n(Loại Khá: 2.5)",
        "Xếp loại Giỏi\n(Điều 17: 3.2)",
        f"GPA CỦA BẠN\n({gpa_4:.2f} / 4.0)"
    ]
    values = [0.8, 1.0, 2.0, 2.5, 3.2, gpa_4]
    colors = ["#ef4444", "#f97316", "#eab308", "#38bdf8", "#818cf8", "#10b981" if gpa_4 >= 2.0 else "#f43f5e"]

    bars1 = ax1.barh(benchmarks, values, color=colors, height=0.55)
    ax1.set_xlim(0, 4.3)
    ax1.set_xlabel("Thang điểm 4.0")
    ax1.set_title("So sánh GPA Cá nhân với Ngưỡng Quy chế ĐH CMC", fontsize=10.5, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.25, color="#94a3b8")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.08, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontweight="bold")

    reg_c = max(1.0, metrics["reg_credits"])
    pass_rate = min(100.0, (metrics["passed_credits"] / reg_c) * 100.0)
    att_rate = min(100.0, metrics["attendance_pct"])
    risk_pct = (dropout_prob * 100.0) if dropout_prob is not None else 15.0
    safety_pct = max(0.0, 100.0 - risk_pct)

    categories = [
        "Tỷ lệ Đi học\n(Chuyên cần)",
        "Tỷ lệ Tín chỉ Đạt\ntrong học kỳ",
        "Mức độ Ổn định\nHọc vụ",
        "Nguy cơ Bỏ học /\nCảnh báo"
    ]
    cat_vals = [att_rate, pass_rate, safety_pct, risk_pct]
    cat_colors = ["#3b82f6", "#10b981", "#22c55e", "#f43f5e" if risk_pct >= 50 else "#f59e0b"]

    bars2 = ax2.bar(categories, cat_vals, color=cat_colors, width=0.48)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Tỷ lệ (%)")
    ax2.set_title("Biểu đồ Chuyên cần, Tín chỉ & Mức độ Rủi ro Học vụ", fontsize=10.5, fontweight="bold", pad=10)
    ax2.axhline(80, color="#38bdf8", linestyle=":", alpha=0.6, label="Ngưỡng chuyên cần tối thiểu (80%)")
    ax2.axhline(50, color="#ef4444", linestyle="--", alpha=0.6, label="Ngưỡng cảnh báo rớt tín chỉ (50%)")
    ax2.legend(loc="upper right", fontsize=8, facecolor="#1e293b", edgecolor="#334155")
    ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{h:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    return fig


def generate_deep_analysis(student_id: str, profile: dict, rule_eval: dict, dropout_prob: float) -> str:
    m = rule_eval["metrics"]
    gpa_4 = m["gpa_4"]
    gpa_10 = m["gpa_10"]
    standing = classify_academic_standing(gpa_4)
    prob_pct = (dropout_prob * 100.0) if dropout_prob is not None else 0.0

    major_name = profile.get("major_name", "Công nghệ thông tin")
    edu_mode = str(profile.get("education_mode", "GM")).upper()
    req_credits = int(m["req_credits"])

    if gpa_4 >= 3.2:
        gpa_comment = (
            f"Điểm trung bình đạt **{gpa_4:.2f}/4.0** (**{gpa_10:.2f}/10**), xếp loại **{standing}** (Điều 17). "
            f"Bạn đủ điều kiện đăng ký làm Khóa luận tốt nghiệp (yêu cầu >= 2.5) và nằm trong nhóm xét Học bổng."
        )
    elif gpa_4 >= 2.5:
        gpa_comment = (
            f"Điểm trung bình đạt **{gpa_4:.2f}/4.0** (**{gpa_10:.2f}/10**), xếp loại **{standing}**. "
            f"Bạn đủ điều kiện làm Khóa luận tốt nghiệp (Điều 12, Khoản 3: GPA >= 2.5) và cần tăng thêm **{max(0.0, 3.2 - gpa_4):.2f} điểm** (hệ 4) để đạt loại Giỏi."
        )
    elif gpa_4 >= 2.0:
        gpa_comment = (
            f"Điểm trung bình đạt **{gpa_4:.2f}/4.0** (**{gpa_10:.2f}/10**), xếp loại **{standing}**. "
            f"Bạn đủ điều kiện xét tốt nghiệp (>= 2.0 theo Điều 20), nhưng cần cải thiện thêm **{2.5 - gpa_4:.2f} điểm** để đạt loại Khá và đủ điều kiện làm Khóa luận."
        )
    else:
        gpa_comment = (
            f"Điểm trung bình hiện tại là **{gpa_4:.2f}/4.0** (**{gpa_10:.2f}/10**), xếp loại **{standing}**. "
            f"Điểm số đang gần ngưỡng Cảnh báo học tập (0.8 ở HK1 và 1.0 ở các kỳ sau theo Điều 18). Bạn nên ưu tiên học cải thiện các môn điểm D/D+ và học lại các môn điểm F."
        )

    pass_c = int(m["passed_credits"])
    reg_c = int(m["reg_credits"])
    fail_c = int(m["failed_credits"])
    debt_c = int(m["debt_credits"])
    credit_comment = (
        f"Trong học kỳ, bạn đạt **{pass_c}/{reg_c} tín chỉ** (không đạt **{fail_c} TC**, tổng nợ đọng **{debt_c}/24 TC** cho phép). "
        f"Tổng khối lượng toàn khóa ngành **{major_name}** là **{req_credits} tín chỉ**."
    )

    att_pct = m["attendance_pct"]
    eng_req = "B2 (Bậc 4/6)" if edu_mode == "GM" else "B1 (Bậc 3/6)"
    att_comment = (
        f"Tỷ lệ chuyên cần đạt **{att_pct:.1f}%** (quy chế yêu cầu tối thiểu >= 80% số giờ không phép và >= 70% tổng số giờ để đủ điều kiện dự thi theo Điều 25). "
        f"Chuẩn đầu ra Tiếng Anh yêu cầu đối với hệ **{edu_mode}** là **{eng_req}** (Điều 19, Khoản 6)."
    )

    if prob_pct >= 60:
        ai_comment = (
            f"Xác suất rủi ro bỏ học/vi phạm quy chế ở mức **Cao ({prob_pct:.1f}%)**. "
            f"**Khuyến nghị:** Liên hệ trực tiếp với **Giảng viên Cố vấn học tập** (Điều 7) để điều chỉnh kế hoạch học tập học kỳ tới và ưu tiên trả nợ các học phần chưa đạt."
        )
    elif prob_pct >= 35:
        ai_comment = (
            f"Xác suất rủi ro ở mức **Trung bình ({prob_pct:.1f}%)**. "
            f"**Khuyến nghị:** Đảm bảo tỷ lệ chuyên cần trên 80%, tránh điểm liệt cuối kỳ (<= 2.0/10) và hoàn thành đầy đủ các bài kiểm tra thành phần."
        )
    else:
        ai_comment = (
            f"Hồ sơ học vụ ở mức **An toàn ({prob_pct:.1f}% rủi ro — {100-prob_pct:.1f}% ổn định)**. "
            f"**Khuyến nghị:** Duy trì tiến độ tích lũy tín chỉ hiện tại và hoàn thiện sớm chứng chỉ Tiếng Anh đầu ra."
        )

    return (
        f"#### 4. Nhận xét & Phân tích Học vụ Chuyên sâu — Sinh viên `{student_id}`\n\n"
        f"1. **Đánh giá Năng lực Điểm số (GPA):** {gpa_comment}\n"
        f"2. **Tiến độ Tích lũy Tín chỉ:** {credit_comment}\n"
        f"3. **Chuyên cần & Điều kiện Đào tạo:** {att_comment}\n"
        f"4. **Đánh giá Rủi ro & Khuyến nghị:** {ai_comment}"
    )


def render_personal_report_view(student_id: str, profile: dict, rule_eval: dict, dropout_prob: float):
    df_t1, df_t2, df_t3 = build_grouped_horizontal_tables(profile)

    st.markdown(f"### BẢNG HỒ SƠ & CHỈ SỐ HỌC TẬP CÁ NHÂN — {student_id}")

    st.markdown("**1.1. Thông tin Cá nhân & Chương trình Đào tạo**")
    st.table(df_t1.assign(hack="").set_index("hack"))

    st.markdown("**1.2. Kết quả Điểm số & Tín chỉ Học tập**")
    st.table(df_t2.assign(hack="").set_index("hack"))

    st.markdown("**1.3. Chuyên cần, Ngoại ngữ & Trạng thái Quy chế**")
    st.table(df_t3.assign(hack="").set_index("hack"))

    st.markdown("#### 3. Biểu đồ Phân tích Điểm số, Chuyên cần & Rủi ro Học vụ")
    fig = render_student_charts(rule_eval["metrics"], dropout_prob)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    analysis_md = generate_deep_analysis(student_id, profile, rule_eval, dropout_prob)
    st.markdown(analysis_md)


if "logged_in_student" not in st.session_state:
    st.session_state.logged_in_student = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# MÀN HÌNH 1: ĐĂNG NHẬP
# ==========================================
if st.session_state.logged_in_student is None:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("## Cổng Đăng Nhập Sinh Viên - ĐH CMC")
        st.info("Đăng nhập bằng Mã sinh viên để tra cứu hồ sơ học tập cá nhân và quy chế đào tạo.")
        sample_ids = data_manager.get_sample_student_ids(5)
        st.caption(f"Mã SV mẫu: `{', '.join(sample_ids)}` | Mật khẩu: `cmc@123` (hoặc nhập lại Mã SV)")

        with st.form("login_form"):
            username = st.text_input("Mã sinh viên")
            password = st.text_input("Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)
            if submitted:
                if data_manager.authenticate(username, password):
                    st.session_state.logged_in_student = username.strip()
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": (
                                f"Xin chào **{username.strip()}**! Hệ thống Trợ lý Học vụ Trường Đại học CMC sẵn sàng hỗ trợ bạn.\n\n"
                                "Bạn có thể nhập câu hỏi trực tiếp:\n"
                                "- **Tra cứu thông tin học tập cá nhân** *(xem 3 bảng ngang gọn gàng, biểu đồ và nhận xét chi tiết)*\n"
                                "- **Dự đoán nguy cơ bỏ học của tôi**\n"
                                "- **Hỏi đáp quy chế & thủ tục:** *Tôi muốn phúc khảo điểm giữa kì môn tiếng anh / Bấm báo cháy có bị sao không / Làm hỏng đồ đền bù như nào / Điều kiện bảo lưu, học lại, chuyển ngành...*"
                            )
                        }
                    ]
                    st.rerun()
                else:
                    st.error("Sai Mã sinh viên hoặc Mật khẩu!")

# ==========================================
# MÀN HÌNH 2: GIAO DIỆN TRA CỨU & CHATBOT
# ==========================================
else:
    student_id = st.session_state.logged_in_student
    student_data = data_manager.get_personal_profile_and_prediction(student_id)
    profile = student_data.get("profile", {})
    dropout_prob = student_data.get("dropout_probability", None)
    rule_eval = evaluate_academic_rules(profile)
    df_t1_s, df_t2_s, df_t3_s = build_grouped_horizontal_tables(profile)

    with st.sidebar:
        st.markdown(f"### Sinh viên: `{student_id}`")
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.logged_in_student = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("#### Dự báo Học vụ")
        if dropout_prob is not None:
            pct = dropout_prob * 100
            if pct >= 60:
                st.error(f"Nguy cơ bỏ học: **{pct:.1f}%** (Cao)")
            elif pct >= 35:
                st.warning(f"Nguy cơ bỏ học: **{pct:.1f}%** (Trung bình)")
            else:
                st.success(f"Nguy cơ bỏ học: **{pct:.1f}%** (Thấp - An toàn)")

        st.markdown(f"**Trạng thái Quy chế:** `{rule_eval['risk_level']}`")
        for w in rule_eval["warnings"]:
            st.warning(w)

        st.divider()
        st.markdown("#### Tóm tắt Hồ sơ Cá nhân")
        for col_name in df_t1_s.columns:
            st.markdown(f"- **{col_name}:** `{df_t1_s[col_name].iloc[0]}`")

        with st.expander("Xem nhanh Điểm & Tín chỉ"):
            for col_name in df_t2_s.columns:
                st.markdown(f"- **{col_name}:** `{df_t2_s[col_name].iloc[0]}`")
            for col_name in df_t3_s.columns:
                st.markdown(f"- **{col_name}:** `{df_t3_s[col_name].iloc[0]}`")

        st.divider()
        api_key = st.text_input("Gemini API Key (Tùy chọn):", type="password")

    st.markdown("### Hệ thống Tư vấn Học vụ & Tra cứu Quy chế - Trường Đại học CMC")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "personal_dashboard":
                render_personal_report_view(student_id, profile, rule_eval, dropout_prob)
            else:
                st.markdown(msg["content"])

    if user_query := st.chat_input("Nhập câu hỏi (VD: Tôi muốn phúc khảo điểm giữa kì môn tiếng anh / Tra cứu thông tin cá nhân)..."):
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        intent, confidence = reg_retriever.predict_intent(user_query)

        with st.chat_message("assistant"):
            if intent == "personal":
                render_personal_report_view(student_id, profile, rule_eval, dropout_prob)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "personal_dashboard",
                    "content": "Bảng thông tin học tập cá nhân."
                })
            elif intent == "risk":
                analysis_md = generate_deep_analysis(student_id, profile, rule_eval, dropout_prob)
                fig = render_student_charts(rule_eval["metrics"], dropout_prob)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(analysis_md)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": analysis_md
                })
            else:
                reg_answer = reg_retriever.get_answer_for_intent(intent, user_query, profile)
                if api_key:
                    genai.configure(api_key=api_key)
                    llm = genai.GenerativeModel("gemini-2.5-flash")
                    prompt = f"""
Bạn là Cố vấn Học tập của Trường Đại học CMC.
YÊU CẦU: Không dùng icon/emoji ở đầu dòng. Trả lời trực tiếp, rõ ràng, đúng trọng tâm câu hỏi của sinh viên dựa trên tài liệu quy chế dưới đây.

Tài liệu tham chiếu:
{reg_answer}

Câu hỏi của sinh viên: {user_query}
"""
                    final_ans = llm.generate_content(prompt).text
                else:
                    final_ans = reg_answer

                st.markdown(final_ans)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": final_ans
                })
