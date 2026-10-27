import os
import re
import joblib
import pandas as pd
from pypdf import PdfReader
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

INTENT_MODEL_PATH = "models/intent_classifier.joblib"
QA_DATASET_PATH = "data/processed/chatbot_qa_training.csv"

# Các cột kỹ thuật của dataset mô phỏng cần ẩn hoàn toàn khỏi giao diện sinh viên
IGNORED_TECHNICAL_COLUMNS = {
    "record_uid", "enrollment_age", "feature_snapshot_semester", "status_date",
    "feature_snapshot_date", "dataset_version", "regulation_version",
    "generator_seed", "is_counterfactual_program", "academic_status_detail",
    "label_observed", "dropout_flag", "target", "row_key"
}

# Từ điển dịch chính xác từng cột sang tiếng Việt
FIELD_LABELS_VI = {
    "student_id": "Mã sinh viên",
    "birth_year": "Năm sinh",
    "gender": "Giới tính",
    "home_province": "Tỉnh/Thành phố",
    "living_status": "Hình thức cư trú",
    "cohort_year": "Khóa tuyển sinh",
    "enrollment_date": "Ngày nhập học",
    "campus": "Cơ sở đào tạo",
    "major_code": "Mã ngành",
    "major_name": "Ngành đào tạo",
    "class_code": "Lớp hành chính",
    "degree_type": "Bậc đào tạo",
    "education_mode": "Hệ đào tạo",
    "curriculum_year": "Khung chương trình",
    "designed_semesters": "Số học kỳ thiết kế",
    "max_semesters": "Số học kỳ tối đa",
    "credits_required": "Tín chỉ yêu cầu toàn khóa",
    "admission_method": "Phương thức xét tuyển",
    "admission_score": "Điểm trúng tuyển",
    "scholarship_percent": "Mức học bổng (%)",
    "semester_current": "Học kỳ hiện tại",
    "semester_last_completed": "Học kỳ vừa hoàn thành",
    "semesters_on_leave": "Số học kỳ đã bảo lưu",
    "academic_status": "Trạng thái học tập",
    "gpa_hk1_10": "Điểm TB HK1 (Hệ 10)",
    "gpa_hk1_4": "Điểm TB HK1 (Hệ 4)",
    "gpa_10": "Điểm TB (Hệ 10)",
    "gpa_4": "Điểm TB (Hệ 4)",
    "gpa": "Điểm TB tích lũy (Hệ 4)",
    "cum_gpa": "Điểm TB tích lũy",
    "credits_registered_hk1": "Tín chỉ đăng ký HK1",
    "registered_credits_sem": "Tín chỉ đăng ký trong kỳ",
    "credits_passed_hk1": "Tín chỉ đạt HK1",
    "passed_credits": "Tín chỉ tích lũy trong kỳ",
    "credits_failed_hk1": "Tín chỉ không đạt HK1",
    "failed_credits_sem": "Tín chỉ không đạt",
    "total_debt_credits": "Tổng tín chỉ nợ đọng",
    "accumulated_credits": "Tổng tín chỉ tích lũy",
    "attendance_rate": "Tỷ lệ chuyên cần",
    "attendance_rate_hk1": "Tỷ lệ chuyên cần HK1",
    "absent_unexcused_pct": "Tỷ lệ nghỉ không phép",
    "absent_total_pct": "Tổng tỷ lệ vắng học",
    "part_time_work_hours_week": "Giờ làm thêm / tuần",
    "english_requirement_status": "Chuẩn Tiếng Anh",
    "warning_current": "Cảnh báo kỳ hiện tại",
    "warning_total": "Tổng số lần cảnh báo",
    "warnings_consecutive": "Số lần cảnh báo liên tiếp",
    "discipline_status": "Kỷ luật học đường"
}

def format_field_value(col: str, val: Any) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return "Không có"
    s = str(val).strip()
    col_l = col.lower()

    if col_l == "campus":
        return "Hà Nội" if s == "HN" else s
    if col_l == "degree_type":
        if s.lower() == "engineer":
            return "Kỹ sư (11 kỳ)"
        if s.lower() == "bachelor":
            return "Cử nhân (9 kỳ)"
    if col_l == "education_mode":
        if s.upper() == "GM":
            return "Song ngữ GM (Chuẩn B2)"
        if s.upper() == "SM":
            return "Chuẩn SM (Chuẩn B1)"
    if col_l in {"warning_current"}:
        return "Có cảnh báo" if s.lower() in {"true", "1", "yes"} else "Không"
    if col_l == "academic_status":
        mapping = {
            "enrolled": "Đang theo học",
            "on_leave": "Đang bảo lưu",
            "dropped_out": "Đã thôi học",
            "graduated": "Đã tốt nghiệp",
            "warning": " diện Cảnh báo"
        }
        return mapping.get(s.lower(), s)
    if col_l == "scholarship_percent":
        try:
            return f"{float(val):.0f}%"
        except ValueError:
            return s
    if col_l == "part_time_work_hours_week":
        try:
            return f"{float(val):.0f} giờ/tuần"
        except ValueError:
            return s
    if isinstance(val, float):
        if "rate" in col_l or "pct" in col_l or "ratio" in col_l:
            return f"{val*100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
        return f"{val:.2f}" if not val.is_integer() else f"{int(val)}"
    return s


def build_grouped_horizontal_tables(profile: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia hồ sơ thành 3 bảng ngang (1 dòng) gọn gàng, loại bỏ toàn bộ cột kỹ thuật thừa."""
    def get_v(col_list, default="Không có"):
        for c in col_list:
            for k, v in profile.items():
                if k.lower() == c.lower() and pd.notna(v) and str(v).strip() != "":
                    return format_field_value(k, v)
        return default

    # Bảng 1: Thông tin Cá nhân & Chương trình Đào tạo
    t1_data = {
        "Mã sinh viên": [get_v(["student_id"])],
        "Năm sinh": [get_v(["birth_year"])],
        "Tỉnh/Thành phố": [get_v(["home_province"])],
        "Cư trú": [get_v(["living_status"])],
        "Ngành học": [f"{get_v(['major_name'])} ({get_v(['major_code'])})"],
        "Lớp hành chính": [get_v(["class_code"])],
        "Bậc đào tạo": [get_v(["degree_type"])],
        "Hệ đào tạo": [get_v(["education_mode"])],
        "Khóa / Học kỳ hiện tại": [f"Khóa {get_v(['cohort_year'])} - Kỳ {get_v(['semester_current'])}"],
        "Trạng thái": [get_v(["academic_status"])]
    }

    # Bảng 2: Kết quả Điểm số & Tín chỉ
    m = extract_academic_metrics(profile)
    standing = classify_academic_standing(m["gpa_4"])
    t2_data = {
        "Điểm đầu vào": [get_v(["admission_score"])],
        "Điểm TB Hệ 4": [f"{m['gpa_4']:.2f} / 4.0"],
        "Điểm TB Hệ 10": [f"{m['gpa_10']:.2f} / 10"],
        "Xếp loại học lực": [standing],
        "Tín chỉ đăng ký kỳ": [f"{int(m['reg_credits'])} TC"],
        "Tín chỉ đạt": [f"{int(m['passed_credits'])} TC"],
        "Tín chỉ không đạt": [f"{int(m['failed_credits'])} TC"],
        "Tổng tín chỉ nợ đọng": [f"{int(m['debt_credits'])} / 24 TC"],
        "Yêu cầu toàn khóa": [f"{int(m['req_credits'])} TC"]
    }

    # Bảng 3: Chuyên cần, Ngoại ngữ & Trạng thái Quy chế
    t3_data = {
        "Tỷ lệ chuyên cần": [f"{m['attendance_pct']:.1f}%"],
        "Chuẩn Tiếng Anh": [get_v(["english_requirement_status"])],
        "Mức học bổng": [get_v(["scholarship_percent"], "0%")],
        "Số kỳ đã bảo lưu": [f"{get_v(['semesters_on_leave'], '0')} học kỳ"],
        "Làm thêm ngoài giờ": [get_v(["part_time_work_hours_week"], "0 giờ/tuần")],
        "Cảnh báo kỳ hiện tại": [get_v(["warning_current"], "Không")],
        "Tổng số lần cảnh báo": [get_v(["warning_total"], "0")],
        "Cảnh báo liên tiếp": [get_v(["warnings_consecutive"], "0")],
        "Kỷ luật học đường": [get_v(["discipline_status"], "Không")]
    }

    return pd.DataFrame(t1_data), pd.DataFrame(t2_data), pd.DataFrame(t3_data)


def extract_academic_metrics(student_record: Dict[str, Any]) -> Dict[str, float]:
    def find_num(keywords: List[str], exclude: List[str] = None, default: float = None):
        exclude = exclude or []
        for kw in keywords:
            for col, val in student_record.items():
                cl = col.lower()
                if kw in cl and not any(ex in cl for ex in exclude):
                    if val is not None and pd.notna(val):
                        try:
                            return float(val)
                        except ValueError:
                            pass
        return default

    gpa_4 = find_num(["gpa_hk1_4", "gpa_4", "diem_tb_he_4", "gpa4", "cum_gpa_4"], default=None)
    gpa_10 = find_num(["gpa_hk1_10", "gpa_10", "diem_tb_he_10", "gpa10", "avg_score"], default=None)
    if gpa_4 is None and gpa_10 is not None:
        gpa_4 = round((gpa_10 / 10.0) * 4.0, 2)
    elif gpa_4 is None:
        raw_gpa = find_num(["gpa"], default=2.13)
        gpa_4 = raw_gpa if raw_gpa <= 4.0 else round((raw_gpa / 10.0) * 4.0, 2)
    if gpa_10 is None:
        gpa_10 = round((gpa_4 / 4.0) * 10.0, 2)

    reg_credits = find_num(["credits_registered", "registered_credits", "tc_dang_ky", "enrolled_credits"], default=18.0)
    passed_credits = find_num(["credits_passed", "passed_credits", "tc_dat", "earned_credits"], default=None)
    failed_credits = find_num(["credits_failed", "failed_credits", "tc_khong_dat", "failed"], default=0.0)
    if passed_credits is None:
        passed_credits = max(0.0, reg_credits - failed_credits)

    debt_credits = find_num(["debt_credit", "tc_no", "no_dong", "backlog"], default=failed_credits)
    req_credits = find_num(["credits_required"], default=164.0)
    attendance = find_num(["attendance_rate", "attendance", "chuyen_can"], default=0.743)
    attendance_pct = round(attendance * 100, 1) if attendance <= 1.0 else round(attendance, 1)

    absent_pct = find_num(["absent_total_pct", "absent_rate", "absence"], default=max(0.0, 100.0 - attendance_pct))
    if 0 < absent_pct <= 1.0:
        absent_pct = round(absent_pct * 100, 1)

    return {
        "gpa_4": round(gpa_4, 2),
        "gpa_10": round(gpa_10, 2),
        "reg_credits": reg_credits,
        "passed_credits": passed_credits,
        "failed_credits": failed_credits,
        "debt_credits": debt_credits,
        "req_credits": req_credits,
        "attendance_pct": attendance_pct,
        "absent_pct": absent_pct
    }


def classify_academic_standing(gpa_4: float) -> str:
    if gpa_4 >= 3.6:
        return "Xuất sắc (3,6 - 4,0)"
    if gpa_4 >= 3.2:
        return "Giỏi (3,2 - cận 3,6)"
    if gpa_4 >= 2.5:
        return "Khá (2,5 - cận 3,2)"
    if gpa_4 >= 2.0:
        return "Trung bình (2,0 - cận 2,5)"
    if gpa_4 >= 1.0:
        return "Yếu (1,0 - cận 2,0)"
    return "Kém (Dưới 1,0)"


def evaluate_academic_rules(student_record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = extract_academic_metrics(student_record)
    warnings: List[str] = []
    risk_level = "An toàn (Bình thường)"

    gpa_4 = metrics["gpa_4"]
    failed_credits = metrics["failed_credits"]
    reg_credits = metrics["reg_credits"]
    debt_credits = metrics["debt_credits"]
    absent_pct = metrics["absent_pct"]

    if absent_pct > 20:
        warnings.append(
            f"Tỷ lệ vắng học ({absent_pct:.1f}%) vượt ngưỡng 20% không phép (hoặc tiệm cận ngưỡng 30% tổng số giờ) — Nguy cơ bị cấm thi học phần (Điều 25, Khoản 1a)."
        )
        risk_level = "Cảnh báo Chuyên cần"

    is_warning = False
    if gpa_4 < 0.8:
        warnings.append(f"Điểm TB đạt {gpa_4:.2f}/4.0 (< 0.8 đối với HK đầu) — Cảnh báo học tập (Điều 18, Khoản 1a).")
        is_warning = True
    elif gpa_4 < 1.0:
        warnings.append(f"Điểm TB đạt {gpa_4:.2f}/4.0 (< 1.0 đối với các kỳ tiếp theo) — Cảnh báo học tập (Điều 18, Khoản 1a).")
        is_warning = True

    if reg_credits > 0 and (failed_credits / reg_credits) > 0.5:
        warnings.append(
            f"Số tín chỉ không đạt ({failed_credits:.0f}/{reg_credits:.0f} TC) vượt 50% khối lượng đăng ký trong kỳ — Cảnh báo học tập (Điều 18, Khoản 1b)."
        )
        is_warning = True

    if debt_credits > 24:
        warnings.append(
            f"Tổng tín chỉ nợ đọng ({debt_credits:.0f} TC) vượt quá 24 tín chỉ — Cảnh báo học tập (Điều 18, Khoản 1b)."
        )
        is_warning = True

    if is_warning:
        risk_level = "Cảnh báo Học tập"

    return {
        "risk_level": risk_level,
        "is_academic_warning": is_warning,
        "warnings": warnings,
        "metrics": metrics
    }


# ==============================================================================
# KHO TRI THỨC & DỮ LIỆU HUẤN LUYỆN MÔ HÌNH NLP PHÂN LOẠI CÂU HỎI (16 CHỦ ĐỀ)
# ==============================================================================
KNOWLEDGE_BASE = {
    "personal": {
        "title": "Tra cứu Hồ sơ & Bảng điểm Cá nhân",
        "samples": [
            "tra cứu thông tin học tập cá nhân", "xem hồ sơ cá nhân của tôi", "bảng điểm của tôi",
            "cho tôi xem kết quả học tập", "thông tin sinh viên của mình", "điểm trung bình gpa của tôi là bao nhiêu",
            "tôi đã tích lũy được bao nhiêu tín chỉ", "xem thông tin cá nhân", "kiểm tra hồ sơ học vụ",
            "tình hình học tập hiện tại của tôi", "cho xem bảng điểm và biểu đồ", "tra cứu điểm cá nhân"
        ],
        "response": ""
    },
    "risk": {
        "title": "Dự báo Nguy cơ Bỏ học & Phân tích Rủi ro Học vụ",
        "samples": [
            "dự đoán tỷ lệ bỏ học của tôi", "nguy cơ bỏ học của tôi là bao nhiêu phần trăm",
            "tôi có nguy cơ bị cảnh báo học tập hay buộc thôi học không", "dự báo học vụ của tôi",
            "phân tích rủi ro bỏ học", "tỉ lệ thôi học của mình cao hay thấp",
            "kiểm tra mức độ an toàn học tập", "xác suất bỏ học của tôi"
        ],
        "response": ""
    },
    "phuc_khao_diem": {
        "title": "Quy định về Phúc tra / Phúc khảo & Khiếu nại Điểm số (Điều 16)",
        "samples": [
            "tôi muốn phúc khảo điểm giữa kì môn tiếng anh", "phúc khảo điểm giữa kỳ như thế nào",
            "phúc tra điểm thi cuối kỳ ở đâu", "chấm lại bài thi kết thúc học phần",
            "điểm thành phần bị sai thì khiếu nại với ai", "thủ tục phúc khảo bài thi",
            "khiếu nại điểm đánh giá thường xuyên", "bao lâu thì có kết quả phúc tra điểm",
            "em muốn xin chấm phúc khảo môn học", "thắc mắc điểm kiểm tra quá trình giữa kỳ"
        ],
        "response": (
            "**Quy định về Khiếu nại và Phúc tra kết quả thi (Điều 16 - QĐ 22/2023/QĐ-ĐHCMC-HĐT):**\n\n"
            "1. **Đối với điểm giữa kỳ, điểm đánh giá thường xuyên và điểm thành phần (Điều 16, Khoản 1):**\n"
            "- Nếu thấy điểm chưa thỏa đáng, sinh viên **khiếu nại trực tiếp với giảng viên giảng dạy** ngay sau khi điểm được công bố (trước khi giảng viên nộp bảng điểm chính thức cho Trường).\n"
            "- Trường hợp giảng viên đã nộp điểm cho Trường, việc khiếu nại điều chỉnh điểm được thực hiện thông qua **Phòng Đào tạo**.\n\n"
            "2. **Đối với điểm thi kết thúc học phần / cuối kỳ (Điều 16, Khoản 2):**\n"
            "- Sinh viên làm đơn đề nghị phúc tra bài thi gửi qua **Phòng Đảm bảo chất lượng** theo mẫu đơn quy định của Trường và nộp lệ phí phúc tra (nếu có) trong thời hạn thông báo của từng học kỳ.\n"
            "- **Thời gian trả kết quả:** Muộn nhất **01 tuần** sau khi kết thúc nhận đơn phúc tra. Kết quả phúc tra là kết quả cuối cùng."
        )
    },
    "canh_bao_hoc_tap": {
        "title": "Quy định về Cảnh báo Học tập (Điều 18, Khoản 1)",
        "samples": [
            "khi nào sinh viên bị cảnh báo học tập", "điều kiện bị cảnh báo học tập là gì",
            "điểm gpa dưới bao nhiêu thì bị cảnh báo", "rớt bao nhiêu tín chỉ thì bị cảnh báo học tập",
            "nợ quá 24 tín chỉ có bị sao không", "quy định cảnh báo học vụ điều 18",
            "điểm trung bình học kỳ dưới 1.0 có bị cảnh báo không", "cảnh báo học tập tính như thế nào"
        ],
        "response": (
            "**Quy định về Cảnh báo học tập cuối mỗi học kỳ (Điều 18, Khoản 1):**\n\n"
            "Sinh viên bị cảnh báo học tập nếu vi phạm **một trong các điều kiện** sau:\n"
            "1. **Điểm trung bình học kỳ (thang điểm 4):** Đạt dưới **0,8** đối với học kỳ đầu tiên của khóa học; hoặc đạt dưới **1,0** đối với các học kỳ tiếp theo.\n"
            "2. **Tín chỉ không đạt trong học kỳ:** Tổng số tín chỉ không đạt trong học kỳ vượt quá **50%** khối lượng tín chỉ đã đăng ký học trong kỳ đó.\n"
            "3. **Tín chỉ nợ đọng từ đầu khóa:** Tổng số tín chỉ nợ đọng từ đầu khóa học vượt quá **24 tín chỉ** (không tính các môn chứng chỉ Giáo dục thể chất và Giáo dục Quốc phòng - An ninh)."
        )
    },
    "buoc_thoi_hoc": {
        "title": "Quy định về Buộc thôi học & Thời gian học tối đa (Điều 18, Khoản 2-4)",
        "samples": [
            "khi nào bị buộc thôi học", "bị cảnh báo học tập mấy lần thì bị đuổi học",
            "cảnh báo 2 lần liên tiếp có bị thôi học không", "thời gian học tối đa ở trường là bao lâu",
            "sinh viên bị đuổi học trong trường hợp nào", "quá 3 lần cảnh báo học tập thì sao",
            "bị buộc thôi học có được bảo lưu môn đã học khi thi lại không"
        ],
        "response": (
            "**Quy định về Buộc thôi học đối với sinh viên (Điều 18, Khoản 2 - 4):**\n\n"
            "1. **Các trường hợp bị buộc thôi học:**\n"
            "- Số lần bị cảnh báo học tập **vượt quá 2 lần liên tiếp** hoặc **vượt quá 3 lần không liên tiếp**.\n"
            "- Vượt quá thời gian tối đa được phép học tại Trường (Hệ Cử nhân: tối đa **18 học kỳ**; Hệ Kỹ sư: tối đa **22 học kỳ** — Khoản 5 Điều 3).\n"
            "- Vi phạm kỷ luật mức cao nhất: Sử dụng tài liệu trong phòng thi lần 3, thi hộ/nhờ thi hộ lần 2, hoặc sử dụng văn bằng/hồ sơ giả (Điều 25).\n\n"
            "2. **Thông báo & Quy định tuyển sinh lại:**\n"
            "- Phòng Công tác sinh viên gửi thông báo Quyết định buộc thôi học tới sinh viên và gia đình.\n"
            "- Nếu sinh viên dự thi tuyển sinh lại và trúng tuyển thì **không được bảo lưu hoặc miễn học phần cũ**, trừ các học phần đã cấp chứng chỉ riêng (GDTC, GDQP-AN)."
        )
    },
    "rut_hoc_phan_bo_hoc": {
        "title": "Quy định về Rút bớt Học phần & Tự ý Bỏ học (Điều 12, Khoản 4)",
        "samples": [
            "em muốn rút bớt môn học đã đăng ký", "thời hạn rút học phần là khi nào",
            "hủy môn học sau 1 tuần có được không", "tự ý nghỉ học không đi học môn đã đăng ký thì sao",
            "bỏ học giữa chừng có phải đóng học phí không", "rút học phần điều 12"
        ],
        "response": (
            "**Quy định về Rút bớt học phần đã đăng ký và Tự ý bỏ học (Điều 12, Khoản 2c & Khoản 4):**\n\n"
            "- **Thời hạn rút/hủy học phần:** Chỉ được thực hiện trong vòng **01 tuần** kể từ khi học kỳ chính thức bắt đầu và phải có xác nhận của Phòng Đào tạo.\n"
            "- **Hệ quả nếu tự ý bỏ học sau thời hạn 01 tuần:** Sinh viên tự ý không đi học sẽ bị tính là bỏ học, phải nhận **điểm F (0 điểm hệ 4)** cho học phần đó và vẫn bị tính **100% học phí** của học phần bỏ học."
        )
    },
    "bao_luu_tam_nghi": {
        "title": "Quy định về Nghỉ học tạm thời / Bảo lưu & Tự xin thôi học (Điều 21)",
        "samples": [
            "thủ tục xin bảo lưu kết quả học tập", "nghỉ học tạm thời được tối đa bao lâu",
            "em muốn xin nghỉ học tạm thời vì lý do cá nhân", "sinh viên năm nhất học kỳ 1 có được bảo lưu không",
            "bị ốm nằm viện xin bảo lưu có tính vào thời gian học không", "muốn quay lại học sau khi bảo lưu cần làm gì",
            "tự xin thôi học vì lý do cá nhân"
        ],
        "response": (
            "**Quy định về Nghỉ học tạm thời (Bảo lưu) và Tự xin thôi học (Điều 21):**\n\n"
            "1. **Các trường hợp được nghỉ học tạm thời và bảo lưu kết quả:**\n"
            "- Được điều động vào lực lượng vũ trang hoặc đại diện quốc gia dự thi, giải đấu quốc tế.\n"
            "- **Ốm, thai sản hoặc tai nạn điều trị dài ngày:** Có chứng nhận của cơ sở y tế có thẩm quyền. Thời gian nghỉ không tính vào thời gian chậm tiến độ nếu tổng cộng **không quá 6 học kỳ chính**.\n"
            "- **Vì lý do cá nhân khác:** Phải học tối thiểu **01 học kỳ** tại Trường, không thuộc diện bị xem xét buộc thôi học hoặc kỷ luật. Thời gian nghỉ tối đa là **03 học kỳ (01 năm)** và được tính vào tổng thời gian học chính thức.\n\n"
            "2. **Thủ tục trở lại học tiếp:**\n"
            "- Sinh viên phải viết đơn gửi **Phòng Đào tạo ít nhất 02 tuần** trước khi bắt đầu học kỳ mới."
        )
    },
    "hoc_lai_diem_liet": {
        "title": "Quy định về Điểm liệt, Học lại, Học cải thiện & Thi lại (Điều 13, 14, 15)",
        "samples": [
            "điểm liệt cuối kỳ là bao nhiêu", "thi cuối kỳ được 2 điểm có qua môn không",
            "khi nào phải học lại môn học", "điểm tổng kết dưới 4.0 là điểm gì",
            "học cải thiện điểm thì lấy điểm nào", "thi lại điểm tối đa được bao nhiêu",
            "cách quy đổi điểm chữ A B C D F sang thang 4", "học lại có mất tiền học phí không"
        ],
        "response": (
            "**Quy định về Đánh giá học phần, Điểm liệt và Học lại (Điều 13, 14, 15, 17):**\n\n"
            "1. **Điểm liệt & Điểm không đạt (Điều 13):**\n"
            "- Điểm thi cuối kỳ **nhỏ hơn hoặc bằng 2,0 (thang 10)** bị tính là **điểm liệt** và điểm tổng kết học phần sẽ nhận **điểm F**.\n"
            "- Điểm tổng kết học phần **< 4,0 (thang 10)** xếp loại **F (Không đạt - 0 điểm hệ 4)**.\n\n"
            "2. **Học lại & Học cải thiện điểm (Điều 14, 15):**\n"
            "- Sinh viên bắt buộc phải học lại khi nhận điểm F, bị hủy kết quả thi do kỷ luật, hoặc bị cấm thi do nghỉ quá số tiết quy định.\n"
            "- **Học cải thiện:** Sinh viên đã đạt học phần được phép đăng ký học lại để cải thiện điểm. Điểm chính thức được công nhận là **điểm cao nhất** giữa các lần học.\n"
            "- **Thi lại (nếu có tổ chức theo quyết định Hiệu trưởng):** Điểm học phần sau khi thi lại bị giới hạn tối đa ở mức **điểm C (2,0 hệ 4)** trở xuống.\n"
            "- Học lại hoặc làm lại khóa luận phải nộp học phí theo số tín chỉ của học phần."
        )
    },
    "chuyen_can_cam_thi": {
        "title": "Quy định về Chuyên cần, Nghỉ học & Đi thi muộn (Điều 15 & Điều 25)",
        "samples": [
            "nghỉ học bao nhiêu buổi thì bị cấm thi", "vắng quá 20% số tiết có được thi cuối kỳ không",
            "nghỉ học có phép quá 30% có bị học lại không", "đi thi muộn 15 phút có được vào phòng thi không",
            "vắng thi cuối kỳ không lý do bị điểm mấy", "ốm không đi thi cuối kỳ được có được thi bù không"
        ],
        "response": (
            "**Quy định về Chuyên cần và Thời gian dự thi (Điều 14, 15 & Điều 25):**\n\n"
            "1. **Điều kiện chuyên cần để dự thi kết thúc học phần (Điều 15 Khoản 1a & Điều 25 Khoản 1a):**\n"
            "- Sinh viên **không được dự thi và phải học lại học phần** nếu:\n"
            "  - Nghỉ học **không phép quá 20%** tổng số giờ của môn học; HOẶC\n"
            "  - Tổng số giờ nghỉ (**gồm cả có phép và không phép**) **quá 30%** tổng số giờ của môn học.\n\n"
            "2. **Quy định giờ giấc và vắng mặt trong buổi thi (Điều 14 Khoản 2e & Điều 25 Khoản 2a):**\n"
            "- **Đến muộn quá 15 phút** sau khi đã phát đề thi: Sinh viên không được dự thi và nhận **điểm 0** cho bài thi.\n"
            "- **Vắng thi không có lý do chính đáng:** Nhận **điểm 0**.\n"
            "- **Vắng thi có lý do chính đáng:** Được phép dự thi bù ở đợt kế tiếp (nhận điểm tạm thời là `I`)."
        )
    },
    "ky_luat_thi_cu": {
        "title": "Xử lý Kỷ luật Gian lận Thi cử, Điện thoại, Tài liệu & Học hộ/Thi hộ (Điều 25)",
        "samples": [
            "mang điện thoại vào phòng thi bị xử lý như thế nào", "sử dụng tài liệu trong phòng thi bị phạt gì",
            "nhìn bài hoặc trao đổi bài trong giờ thi bị trừ bao nhiêu điểm", "đi học hộ bạn bị kỷ luật thế nào",
            "nhờ người khác thi hộ bị phạt sao", "làm hộ bài kiểm tra giữa kỳ bị xử lý ra sao",
            "dùng bằng cấp chứng chỉ giả bị xử lý thế nào"
        ],
        "response": (
            "**Khung xử lý kỷ luật đối với hành vi vi phạm trong học tập và thi cử (Điều 25):**\n\n"
            "- **Nhìn bài hoặc trao đổi bài:** Lần 1 khiển trách, trừ **25%** điểm bài thi; Lần 2 cảnh cáo, trừ **50%** điểm bài thi; Nhắc nhở đến lần thứ 3 nhận **điểm 0**.\n"
            "- **Sử dụng tài liệu trong phòng thi:** Lần 1 hủy kết quả học phần; Lần 2 hủy toàn bộ kết quả học tập của học kỳ; Lần 3 **buộc thôi học**.\n"
            "- **Mang điện thoại, đồng hồ thông minh, thiết bị truyền tin (không tắt nguồn / để sai nơi quy định / sử dụng):** Tùy mức độ bị hủy kết quả thi học phần đến hủy toàn bộ kết quả kỳ thi; Vi phạm lần 2 hủy toàn bộ kết quả học kỳ và **đình chỉ học 01 học kỳ kế tiếp**.\n"
            "- **Học hộ / Nhờ học hộ:** Người đi học hộ bị đình chỉ học tập đến buộc thôi học; Người nhờ học hộ bị hủy kết quả môn học (tái phạm lần 2 đình chỉ học 01 học kỳ).\n"
            "- **Làm hộ bài kiểm tra thường xuyên/quá trình:** Đình chỉ học **01 học kỳ kế tiếp** (cả người làm hộ và người nhờ làm hộ).\n"
            "- **Thi hộ hoặc nhờ người thi hộ:** Vi phạm lần 1 **đình chỉ học tập 01 năm**; Vi phạm lần 2 **buộc thôi học**.\n"
            "- **Sử dụng hồ sơ, văn bằng, chứng chỉ giả:** **Buộc thôi học** và thu hồi văn bằng (nếu đã cấp)."
        )
    },
    "tieng_anh_ngoai_ngu": {
        "title": "Quy định về Chuẩn đầu ra Tiếng Anh & Ngoại ngữ (Điều 19, 20)",
        "samples": [
            "chuẩn đầu ra tiếng anh của trường là gì", "hệ gm cần tiếng anh b1 hay b2",
            "hệ sm chuẩn tiếng anh đầu ra bậc mấy", "chưa có chứng chỉ tiếng anh có được xét tốt nghiệp không",
            "hết thời gian học tối đa mà nợ tiếng anh hoặc thể chất quốc phòng thì sao"
        ],
        "response": (
            "**Quy định về Điều kiện Tiếng Anh và Hoàn thiện Chuẩn đầu ra (Điều 19 Khoản 6 & Điều 20 Khoản 5):**\n\n"
            "1. **Chuẩn đầu ra Tiếng Anh theo hệ đào tạo:**\n"
            "- **Hệ Đào tạo Chuẩn (Standard Mode - SM):** Đạt trình độ Tiếng Anh **Bậc 3/6** theo Khung năng lực ngoại ngữ 6 bậc Việt Nam hoặc **B1 (CEFR)** hoặc tương đương.\n"
            "- **Hệ Đào tạo Song ngữ (Global Mode - GM):** Đạt trình độ Tiếng Anh **Bậc 4/6** theo Khung năng lực ngoại ngữ 6 bậc Việt Nam hoặc **B2 (CEFR)** hoặc tương đương.\n\n"
            "2. **Gia hạn hoàn thiện Ngoại ngữ / GDTC / GDQP-AN (Điều 20, Khoản 5):**\n"
            "- Sinh viên đã hết thời gian học tập tối đa nhưng chưa đủ điều kiện tốt nghiệp do chưa hoàn thành học phần GDQP-AN, GDTC hoặc chưa đạt chuẩn đầu ra ngoại ngữ được phép hoàn thiện các nội dung còn thiếu trong thời hạn **03 năm tính từ khi thôi học** để đề nghị xét công nhận tốt nghiệp."
        )
    },
    "khoa_luan_tot_nghiep": {
        "title": "Quy định về Khóa luận Tốt nghiệp, Xét Tốt nghiệp & Hạ bậc Bằng (Điều 12, 20)",
        "samples": [
            "điều kiện để được làm khóa luận đồ án tốt nghiệp là gì", "điểm gpa bao nhiêu thì được làm khóa luận",
            "khi nào sinh viên đủ điều kiện xét tốt nghiệp", "tại sao bằng giỏi xuất sắc lại bị hạ bậc xuống khá",
            "học lại quá 5% số tín chỉ có bị hạ bằng tốt nghiệp không", "thời gian làm khóa luận tốt nghiệp là mấy tháng"
        ],
        "response": (
            "**Quy định về Khóa luận tốt nghiệp và Xếp hạng tốt nghiệp (Điều 12 Khoản 3 & Điều 20):**\n\n"
            "1. **Điều kiện đăng ký làm Khóa luận / Đồ án tốt nghiệp (Điều 12, Khoản 3):**\n"
            "- Sinh viên phải đạt điểm trung bình tích lũy tối thiểu **2,5/4.0 (loại Khá trở lên)**. Thời gian thực hiện khóa luận từ **3 đến 4 tháng (01 học kỳ)**. Nếu không đủ điều kiện hoặc khóa luận không đạt, sinh viên học các học phần thay thế theo quy định của Khoa.\n\n"
            "2. **Các trường hợp bị giảm 01 mức hạng tốt nghiệp đối với loại Xuất sắc và Giỏi (Điều 20, Khoản 4):**\n"
            "- Khối lượng của các học phần phải học lại **vượt quá 5%** so với tổng số tín chỉ quy định cho toàn chương trình; HOẶC\n"
            "- Sinh viên đã bị kỷ luật từ mức **Cảnh cáo trở lên** trong thời gian học."
        )
    },
    "chuyen_nganh_song_bang": {
        "title": "Quy định về Chuyển ngành, Chuyển trường & Học Song bằng (Điều 22, 23)",
        "samples": [
            "điều kiện xin chuyển ngành học là gì", "sinh viên năm nhất hoặc năm cuối có được chuyển ngành không",
            "được chuyển ngành mấy lần trong khóa học", "điều kiện đăng ký học cùng lúc 2 chương trình song bằng",
            "tích lũy bao nhiêu tín chỉ thì được học thêm ngành thứ 2"
        ],
        "response": (
            "**Quy định về Chuyển ngành và Học cùng lúc hai chương trình (Điều 22 & Điều 23):**\n\n"
            "1. **Chuyển ngành đào tạo (Điều 22):**\n"
            "- **Điều kiện:** Không đang là sinh viên trình độ năm thứ nhất hoặc năm cuối khóa, không thuộc diện bị xem xét buộc thôi học, đạt điều kiện trúng tuyển của ngành chuyển đến trong cùng khóa tuyển sinh và được sự đồng ý của Giám đốc CTĐT, Trưởng khoa và Hiệu trưởng.\n"
            "- Sinh viên chỉ được chuyển ngành **một lần duy nhất** trong cả khóa học.\n\n"
            "2. **Học cùng lúc hai chương trình / Song bằng (Điều 23):**\n"
            "- **Điều kiện:** Đã tích lũy được **>= 40 tín chỉ**; có học lực từ **Khá trở lên** (đáp ứng ngưỡng đảm bảo chất lượng ngành 2) hoặc có học lực **Trung bình** và đáp ứng điều kiện trúng tuyển của ngành thứ 2 trong năm tuyển sinh.\n"
            "- Sinh viên chỉ được xét tốt nghiệp ngành thứ hai khi đã đạt điều kiện tốt nghiệp ngành thứ nhất."
        )
    },
    "den_bu_hong_do_csvc": {
        "title": "Quy định về Bảo vệ Tài sản & Đền bù Hư hỏng Cơ sở vật chất",
        "samples": [
            "làm hỏng đồ thì phải đền bù như nào", "lỡ làm vỡ thiết bị trong phòng thực hành có bị sao không",
            "làm hư hỏng máy tính bàn ghế của trường đền bù bao nhiêu", "quy định bồi thường tài sản cơ sở vật chất",
            "làm hỏng thiết bị phòng lab xử lý thế nào"
        ],
        "response": (
            "**Quy định về Trách nhiệm Bảo vệ Tài sản và Đền bù Hư hỏng Thiết bị tại Trường Đại học CMC:**\n\n"
            "1. **Quy trình xử lý khi xảy ra hư hỏng đồ dùng, thiết bị:**\n"
            "- Sinh viên cần báo ngay cho giảng viên phụ trách lớp học phần, cán bộ kỹ thuật phòng máy/phòng thực hành hoặc bộ phận Quản trị thiết bị để lập biên bản ghi nhận sự việc.\n"
            "- **Nếu hư hỏng do sự cố khách quan / vô ý trong quá trình thực hành đúng hướng dẫn:** Nhà trường xem xét mức bồi thường theo **chi phí sửa chữa thực tế hoặc giá trị khấu hao còn lại** của thiết bị.\n"
            "- **Nếu tự ý tháo lắp, sử dụng sai mục đích hoặc cố ý làm hư hỏng tài sản:** Sinh viên phải **bồi thường 100% giá trị thiết bị/chi phí khắc phục** và bị đưa ra Hội đồng xem xét kỷ luật thông qua Phòng Công tác sinh viên (Điều 7, Khoản 1f) từ mức **Khiển trách, Cảnh cáo** đến **Đình chỉ học tập**.\n\n"
            "2. **Lưu ý về điều kiện tốt nghiệp (Điều 20, Khoản 1e & Khoản 4b):**\n"
            "- Sinh viên phải hoàn thành đầy đủ nghĩa vụ tài chính và khoản bồi thường (nếu có) mới được xét công nhận tốt nghiệp.\n"
            "- Nếu bị kỷ luật từ mức **Cảnh cáo trở lên**, sinh viên xếp loại Xuất sắc/Giỏi sẽ bị **hạ 01 bậc hạng tốt nghiệp**."
        )
    },
    "bam_bao_chay_noi_quy": {
        "title": "Quy định về An toàn PCCC & Xử lý Hành vi Tự ý Bấm Chuông Báo Cháy",
        "samples": [
            "bấm báo cháy có bị sao không", "tự ý bấm nút chuông báo cháy giả ở trường bị phạt thế nào",
            "nghịch chuông báo cháy trong tòa nhà có bị kỷ luật không", "vi phạm an toàn phòng cháy chữa cháy pccc trong trường",
            "gây rối mất trật tự an toàn trong trường bị xử lý ra sao"
        ],
        "response": (
            "**Quy định về Xử lý Hành vi Tự ý Bấm Chuông Báo Cháy & Vi phạm Nội quy An toàn Học đường:**\n\n"
            "1. **Tự ý bấm báo cháy khi không có sự cố có bị sao không?**\n"
            "- **CÓ — Đây là hành vi vi phạm nghiêm trọng nội quy an toàn của Nhà trường và quy định Phòng cháy chữa cháy (PCCC).**\n"
            "- Việc kích hoạt báo cháy giả gây hoảng loạn, làm gián đoạn trực tiếp các lớp học và phòng thi trong tòa nhà (vi phạm **Điều 6 Quy chế đào tạo** về trách nhiệm thực hiện nội quy, quy định của Trường).\n\n"
            "2. **Hình thức xử lý kỷ luật:**\n"
            "- **Kỷ luật cấp Trường:** Hệ thống camera an ninh và bộ phận Quản lý tòa nhà sẽ trích xuất dữ liệu, chuyển Phòng Công tác sinh viên và Giáo viên chủ nhiệm họp xét kỷ luật (Điều 7, Khoản 1f). Sinh viên vi phạm sẽ bị trừ điểm rèn luyện và nhận quyết định kỷ luật từ **Khiển trách, Cảnh cáo** đến **Đình chỉ học tập có thời hạn**.\n"
            "- **Hệ quả học vụ (Điều 20, Khoản 4b):** Nếu nhận mức kỷ luật từ **Cảnh cáo trở lên**, sinh viên có học lực Xuất sắc hoặc Giỏi sẽ bị **giảm 01 mức hạng tốt nghiệp** khi ra trường.\n"
            "- **Bồi thường & Chế tài hành chính:** Phải đền bù chi phí lắp đặt lại hộp kính/nút nhấn báo cháy (nếu làm vỡ) và có thể bị xử phạt hành chính về hành vi báo cháy giả theo quy định pháp luật."
        )
    },
    "dang_ky_tin_chi_co_van": {
        "title": "Quy định về Đăng ký Tín chỉ, Khối lượng Học tập & Cố vấn Học tập (Điều 5, 7, 12)",
        "samples": [
            "mỗi học kỳ phải đăng ký tối thiểu và tối đa bao nhiêu tín chỉ", "lớp học phần bị hủy do ít sinh viên thì làm sao",
            "cố vấn học tập và giáo viên chủ nhiệm hỗ trợ những gì", "thời gian các tiết học sáng chiều tối của trường",
            "tổng thời lượng học online trực tuyến tối đa là bao nhiêu phần trăm", "cộng điểm thưởng nghiên cứu khoa học olympic"
        ],
        "response": (
            "**Quy định về Đăng ký học tập, Thời gian học và Cố vấn học tập (Điều 5, 7, 11, 12, 13):**\n\n"
            "- **Giới hạn khối lượng tín chỉ đăng ký mỗi kỳ (Điều 12, Khoản 2e):** Khối lượng tối thiểu không ít hơn **2/3** và tối đa không vượt quá **3/2** khối lượng trung bình một học kỳ theo kế hoạch học tập chuẩn.\n"
            "- **Lớp học phần bị hủy do ít sinh viên (Điều 12, Khoản 2c):** Nếu thời hạn đăng ký trực tuyến đã hết, sinh viên liên hệ **Cố vấn học tập** để được hỗ trợ đăng ký lớp học phần thay thế.\n"
            "- **Thời lượng học trực tuyến (Điều 11, Khoản 2):** Không vượt quá **30%** tổng khối lượng của chương trình đào tạo.\n"
            "- **Khung giờ học (Điều 5, Khoản 2 - 50 phút/tiết):** Buổi sáng (Tiết 1–6: 7h00 – 12h25); Buổi chiều (Tiết 7–12: 13h00 – 18h25); Buổi tối (Tiết 13–15: 18h50 – 21h30).\n"
            "- **Điểm thưởng thành tích (Điều 13, Khoản 4):** Sinh viên tham gia nghiên cứu khoa học, thi Olympic và các cuộc thi chuyên môn cấp Trường/Quốc gia/Quốc tế được xem xét cộng điểm thưởng vào học phần chuyên môn tương ứng và ưu tiên xét học bổng."
        )
    }
}


def generate_and_train_intent_model() -> Pipeline:
    """Sinh file dataset câu hỏi huấn luyện (CSV) và train mô hình NLP phân loại ý định người dùng."""
    rows = []
    for intent_key, data in KNOWLEDGE_BASE.items():
        for sample in data["samples"]:
            rows.append({"question": sample, "intent": intent_key, "topic_title": data["title"]})

    df_qa = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(QA_DATASET_PATH), exist_ok=True)
    df_qa.to_csv(QA_DATASET_PATH, index=False, encoding="utf-8-sig")

    # Kết hợp TF-IDF (cả cấp độ từ và cụm ký tự) + LogisticRegression để nhận diện chính xác cả câu có lỗi chính tả/viết tắt
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
        ("clf", LogisticRegression(C=5.0, max_iter=500, class_weight="balanced", random_state=42))
    ])
    pipeline.fit(df_qa["question"], df_qa["intent"])

    os.makedirs(os.path.dirname(INTENT_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, INTENT_MODEL_PATH)
    return pipeline


class RegulationRetriever:
    def __init__(self):
        self.chunks: List[Tuple[str, str]] = []
        self._load_pdfs()
        if os.path.exists(INTENT_MODEL_PATH):
            try:
                self.intent_model = joblib.load(INTENT_MODEL_PATH)
            except Exception:
                self.intent_model = generate_and_train_intent_model()
        else:
            self.intent_model = generate_and_train_intent_model()

    def _load_pdfs(self):
        for root, _, files in os.walk("data"):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    fpath = os.path.join(root, fname)
                    reader = PdfReader(fpath)
                    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    parts = re.split(r"(?=\nĐiều \d+\.)", full_text)
                    for p in parts:
                        clean_p = re.sub(r"\s+", " ", p).strip()
                        if len(clean_p) > 60:
                            self.chunks.append((fname, clean_p))

    def predict_intent(self, query: str) -> Tuple[str, float]:
        q = query.lower().strip()
        # Ưu tiên bắt chính xác các từ khóa đặc thù trước khi qua xác suất mô hình NLP
        direct_rules = {
            "phuc_khao_diem": ["phúc khảo", "phúc tra", "chấm lại", "khiếu nại điểm", "sai điểm"],
            "bam_bao_chay_noi_quy": ["báo cháy", "chuông báo cháy", "pccc", "cứu hỏa", "gây rối"],
            "den_bu_hong_do_csvc": ["hỏng đồ", "làm hỏng", "đền bù", "bồi thường", "làm vỡ", "hư hỏng"],
            "ky_luat_thi_cu": ["thi hộ", "học hộ", "mang điện thoại", "tài liệu vào phòng thi", "nhìn bài", "gian lận"],
            "bao_luu_tam_nghi": ["bảo lưu", "nghỉ học tạm thời", "tạm nghỉ"],
            "rut_hoc_phan_bo_hoc": ["rút học phần", "rút môn", "hủy học phần", "hủy môn"],
            "chuyen_nganh_song_bang": ["chuyển ngành", "chuyển trường", "song bằng", "hai chương trình", "ngành thứ hai"],
            "khoa_luan_tot_nghiep": ["khóa luận", "đồ án tốt nghiệp", "hạ bậc", "giảm hạng tốt nghiệp"],
            "tieng_anh_ngoai_ngu": ["chuẩn tiếng anh", "chuẩn ngoại ngữ", "b1", "b2", "6 bậc"],
            "chuyen_can_cam_thi": ["cấm thi", "nghỉ quá", "vắng quá", "muộn 15 phút", "đi thi muộn"],
            "hoc_lai_diem_liet": ["điểm liệt", "học cải thiện", "thi lại", "điểm f"],
            "buoc_thoi_hoc": ["buộc thôi học", "đuổi học", "quá 2 lần", "quá 3 lần"],
            "canh_bao_hoc_tap": ["điều kiện cảnh báo", "bị cảnh báo học tập", "nợ quá 24"]
        }
        for intent_key, kw_list in direct_rules.items():
            if any(kw in q for kw in kw_list):
                return intent_key, 0.99

        probs = self.intent_model.predict_proba([q])[0]
        classes = self.intent_model.classes_
        best_idx = probs.argmax()
        return str(classes[best_idx]), float(probs[best_idx])

    def get_answer_for_intent(self, intent: str, query: str, student_profile: Dict[str, Any] = None) -> str:
        if intent in KNOWLEDGE_BASE and KNOWLEDGE_BASE[intent]["response"]:
            topic = KNOWLEDGE_BASE[intent]
            ans = f"#### {topic['title']}\n\n{topic['response']}"
            # Liên hệ thực tế với hồ sơ của sinh viên đang đăng nhập nếu câu hỏi có liên quan
            if student_profile:
                if intent == "phuc_khao_diem" and "tiếng anh" in query.lower():
                    eng_st = student_profile.get("english_requirement_status", "Đang học")
                    edu_m = student_profile.get("education_mode", "GM")
                    ans += (
                        f"\n\n---\n**Thông tin liên hệ theo hồ sơ của bạn:** Hiện tại trạng thái chuẩn Tiếng Anh của bạn là **{eng_st}** "
                        f"(Hệ đào tạo **{edu_m}**). Vì đây là **điểm giữa kỳ (điểm thành phần)**, bạn hãy liên hệ trực tiếp với giảng viên dạy lớp Tiếng Anh của mình ngay sau khi công bố điểm để được kiểm tra và đối chiếu bài làm trước khi giảng viên khóa bảng điểm nộp về Phòng Đào tạo."
                    )
                elif intent == "tieng_anh_ngoai_ngu":
                    eng_st = student_profile.get("english_requirement_status", "Đang học")
                    edu_m = str(student_profile.get("education_mode", "GM")).upper()
                    req_lv = "B2 (Bậc 4/6)" if edu_m == "GM" else "B1 (Bậc 3/6)"
                    ans += f"\n\n---\n**Đối chiếu hồ sơ của bạn:** Bạn đang theo học hệ **{edu_m}** (yêu cầu đầu ra **{req_lv}**), trạng thái hiện tại: **{eng_st}**."
            return ans

        # Tìm kiếm trực tiếp trên văn bản 2 file PDF nếu câu hỏi hỏi về Điều khoản cụ thể khác
        q_lower = query.lower()
        dieu_match = re.search(r"điều\s+(\d+)", q_lower)
        if dieu_match:
            dieu_num = dieu_match.group(1)
            for fname, chunk in self.chunks:
                if chunk.lower().startswith(f"điều {dieu_num}."):
                    return f"#### Trích dẫn {fname}\n\n{chunk}"

        stop_words = {"như", "thế", "nào", "sao", "không", "có", "bị", "thì", "phải", "làm", "gì", "của", "trường", "cho", "hỏi", "về", "tôi", "muốn"}
        query_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2 and w not in stop_words]
        scored = []
        for fname, chunk in self.chunks:
            clower = chunk.lower()
            score = sum(2 if w in clower else 0 for w in query_words)
            scored.append((score, fname, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored and scored[0][0] > 0:
            return f"#### Trích dẫn Quy chế Đào tạo ({scored[0][1]})\n\n{scored[0][2][:750]}"

        return (
            "#### Hướng dẫn Tra cứu Quy chế & Học vụ ĐH CMC\n\n"
            "Bạn có thể hỏi trực tiếp một trong các nội dung sau:\n"
            "- **Hồ sơ & Điểm số:** *Tra cứu thông tin học tập cá nhân, dự đoán tỷ lệ bỏ học của tôi*\n"
            "- **Điểm & Thi cử:** *Phúc khảo điểm giữa kỳ/cuối kỳ, điểm liệt, học lại, học cải thiện, nghỉ quá 20% số tiết*\n"
            "- **Học vụ & Thủ tục:** *Điều kiện cảnh báo học tập, buộc thôi học, xin bảo lưu, rút môn học, chuyển ngành, học song bằng, chuẩn tiếng Anh B1/B2, làm khóa luận*\n"
            "- **Nội quy & Kỷ luật:** *Mang điện thoại/tài liệu vào phòng thi, học hộ/thi hộ, làm hỏng đồ đền bù như nào, bấm báo cháy có bị sao không*"
        )
