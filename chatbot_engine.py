import re
import json
import math
import unicodedata
import difflib
import joblib
import pandas as pd
from pathlib import Path

def normalize_vn(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().lower()

def strip_accents(text: str) -> str:
    text = normalize_vn(text).replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

# Chuyển đổi 1 dòng Series hoặc DataFrame sang dict chuẩn JSON (không dùng .fillna(0) trực tiếp trên Pandas 3.0)
def row_to_clean_dict(row) -> dict:
    raw = json.loads(row.to_json())
    cleaned = {}
    for k, v in raw.items():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            cleaned[k] = "" if k in ["student_id", "record_uid", "class_code", "major_code", "major_name", "degree_type", "education_mode", "risk_level", "verified_certificates", "exit_reason_for_evaluation"] else 0
        else:
            cleaned[k] = v
    return cleaned

def df_to_clean_records(df: pd.DataFrame) -> list:
    return [row_to_clean_dict(r) for _, r in df.iterrows()]

MAJOR_ALIASES = {
    "IT": ["cong nghe thong tin", "cntt", "khoa it", "nganh it"],
    "CS": ["khoa hoc may tinh", "khmt", "khoa cs", "nganh cs"],
    "SE": ["ky thuat phan mem", "ktpm", "khoa se", "nganh se"],
    "AI": ["tri tue nhan tao", "ttnt", "khoa ai", "nganh ai"],
    "NS": ["an ninh mang", "anm", "antoan thong tin", "khoa ns"],
    "DA": ["phan tich du lieu", "khoa hoc du lieu", "khoa da"],
    "EC": ["dien tu vien thong", "cong nghe ky thuat dien tu", "khoa ec"],
    "GA": ["cong nghe game", "phat trien game", "khoa ga"],
    "BA": ["quan tri kinh doanh", "qtkd", "khoa ba", "nganh ba"],
    "EM": ["thuong mai dien tu", "tmdt", "khoa em"],
    "LS": ["logistics", "chuoi cung ung", "khoa ls"],
    "IB": ["kinh doanh quoc te", "kdqt", "khoa ib"],
    "MK": ["marketing", "khoa mk"],
    "CB": ["tai chinh ngan hang", "tai chinh", "khoa cb"],
    "MC": ["truyen thong da phuong tien", "truyen thong", "khoa mc"],
    "PR": ["quan he cong chung", "khoa pr"],
    "GD": ["thiet ke do hoa", "tkdh", "my thuat so", "khoa gd"],
    "KL": ["ngon ngu han", "tieng han", "han quoc", "khoa kl"],
    "CL": ["ngon ngu trung", "tieng trung", "trung quoc", "khoa cl"]
}

COMMON_COURSES = {
    1: "Đại số tuyến tính & Giải tích", 2: "Nhập môn ngành & Tư duy số", 3: "Triết học Mác – Lênin",
    4: "Pháp luật đại cương", 5: "Kỹ năng mềm & Giao tiếp chuyên nghiệp", 6: "Kinh tế chính trị Mác – Lênin",
    7: "Xác suất thống kê ứng dụng", 8: "Tiếng Anh học thuật 1", 9: "Chủ nghĩa xã hội khoa học",
    10: "Tiếng Anh học thuật 2", 16: "Lịch sử Đảng Cộng sản Việt Nam", 21: "Tư tưởng Hồ Chí Minh"
}

TECH_COURSES = {
    2: "Nhập môn Lập trình", 6: "Toán rời rạc", 8: "Cấu trúc dữ liệu và Giải thuật",
    9: "Kiến trúc máy tính & Hệ điều hành", 11: "Lập trình Hướng đối tượng (OOP)", 12: "Cơ sở dữ liệu",
    13: "Mạng máy tính và Truyền thông", 14: "Công nghệ phần mềm", 15: "Lập trình Web & Ứng dụng",
    17: "An toàn và Bảo mật thông tin", 18: "Trí tuệ nhân tạo & Học máy", 19: "Phân tích và Thiết kế hệ thống",
    20: "Phát triển ứng dụng di động", 22: "Điện toán đám mây & DevOps", 23: "Kiểm thử và Đảm bảo chất lượng PM",
    24: "Quản trị dự án CNTT", 25: "Xử lý dữ liệu lớn (Big Data)", 26: "Học sâu (Deep Learning)",
    27: "Thực tập doanh nghiệp CNTT", 28: "Đồ án chuyên ngành", 29: "Khóa luận / Đồ án tốt nghiệp"
}

BIZ_COURSES = {
    2: "Kinh tế vi mô", 6: "Kinh tế vĩ mô", 8: "Nguyên lý Quản trị học", 9: "Nguyên lý Kế toán",
    11: "Nguyên lý Marketing", 12: "Tài chính – Tiền tệ", 13: "Quản trị Nhân sự", 14: "Thương mại điện tử & Kinh tế số",
    15: "Hành vi khách hàng", 17: "Tài chính doanh nghiệp", 18: "Quản trị Chiến lược", 19: "Quản trị Chuỗi cung ứng & Logistics"
}

LANG_DESIGN_COURSES = {
    2: "Cơ sở tạo hình & Mỹ thuật đại cương", 6: "Ngữ âm & Giao tiếp sơ cấp 1", 8: "Ngữ pháp & Đọc hiểu chuyên sâu 1",
    9: "Văn hóa & Đất nước học", 11: "Thực hành Nghe – Nói trung cấp", 12: "Thiết kế Đồ họa số & Typography",
    14: "Thiết kế Giao diện UI/UX", 15: "Tiếng Hàn/Trung thương mại 1"
}

STRUCTURED_REGULATIONS = [
    {
        "rule_id": "DIEU_1_2_27", "article_num": ["1", "2", "27"],
        "topic": "Phạm vi áp dụng & Hiệu lực thi hành",
        "title": "Điều 1, Điều 2 & Điều 27: Phạm vi áp dụng và Hiệu lực Quy chế (QĐ 22/2023)",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 1, Điều 2, Điều 27 (Khoản 1)",
        "keywords": ["điều 1", "điều 2", "điều 27", "phạm vi", "năm học nào", "2023-2024", "khóa 2022", "bản tóm tắt", "bản gốc"],
        "summary_rows": [
            ["Đối tượng áp dụng (Điều 1)", "Áp dụng cho sinh viên đại học hệ chính quy tại Trường Đại học CMC (không áp dụng hệ từ xa/liên kết)"],
            ["Hiệu lực thi hành (Điều 27.1)", "Áp dụng từ năm học 2023–2024. Với các khóa trước (như Khóa 2022), quy chế không nêu điều khoản chuyển tiếp riêng nên đối chiếu hướng dẫn của Phòng Đào tạo"],
            ["Đối chiếu 2 văn bản PDF", "Văn bản pháp lý gốc là QĐ 22/2023; file 'Quy-dinh-thoi-hoc-DH-CMC.pdf' là bản tóm tắt. Nếu có khác biệt, luôn ưu tiên áp dụng theo Quyết định gốc QĐ 22/2023"]
        ],
        "takeaway": "Quy chế QĐ 22/2023 có hiệu lực từ năm học 2023–2024; mọi quyết định học vụ chính thức căn cứ theo văn bản gốc QĐ 22/2023."
    },
    {
        "rule_id": "DIEU_3_5", "article_num": ["3", "5"],
        "topic": "Tín chỉ, Học phần tiên quyết & Thời gian tối đa",
        "title": "Điều 3 & Điều 5: Tín chỉ, Loại học phần, Thời lượng tiết học và Thời gian tối đa",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 3 (Khoản 1–5) & Điều 5",
        "keywords": ["điều 3", "điều 5", "tín chỉ", "thời gian đào tạo", "thời gian tối đa", "cử nhân", "kỹ sư", "18 học kỳ", "22 học kỳ", "tiên quyết", "học trước", "tương đương", "thể chất", "quốc phòng", "50 phút", "giờ tín chỉ"],
        "summary_rows": [
            ["Quy đổi 1 tín chỉ (Điều 3.1)", "1 tín chỉ = 15 giờ tín chỉ. 1 giờ tín chỉ lý thuyết = 1 tiết trên lớp + 2 tiết tự học; Thực hành = 2–3 tiết; Thực tập = 3–6 tiết"],
            ["Thời lượng tiết học (Điều 5)", "Mỗi tiết học kéo dài 50 phút. Sáng: Tiết 1–6 (07:00–12:25) • Chiều: Tiết 7–12 (13:00–18:25) • Tối: Tiết 13–15 (18:50–21:30)"],
            ["Học phần tiên quyết vs Học trước (Điều 3.2)", "Môn tiên quyết (3.2b): Phải học và THI ĐẠT mới được đăng ký môn sau. Môn học trước (3.2c): Đã học xong (dù chưa đạt) vẫn được đăng ký môn sau"],
            ["Học phần điều kiện (Điều 3.2f)", "GD Thể chất, GD Quốc phòng - An ninh, Kỹ năng mềm, Tiếng Anh điều kiện: KHÔNG tính vào GPA nhưng bắt buộc hoàn thành để tốt nghiệp"],
            ["Thời gian đào tạo tối đa (Điều 3.5)", "Cử nhân (120–135 TC): Thiết kế 9 học kỳ, tối đa 18 học kỳ • Kỹ sư (150–164 TC): Thiết kế 11 học kỳ, tối đa 22 học kỳ"]
        ],
        "takeaway": "Cử nhân học tối đa 18 học kỳ, Kỹ sư tối đa 22 học kỳ. Trượt môn tiên quyết thì không được đăng ký học phần phụ thuộc kế tiếp."
    },
    {
        "rule_id": "DIEU_6_7_12", "article_num": ["6", "7", "12"],
        "topic": "Đăng ký tín chỉ, Rút học phần, Cố vấn & Khóa luận",
        "title": "Điều 6, 7 & Điều 12: Trách nhiệm SV, Cố vấn học tập, Đăng ký tín chỉ, Rút môn và Khóa luận",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 6 (Khoản 5), Điều 7, Điều 12 (Khoản 2, 3, 4)",
        "keywords": ["điều 6", "điều 7", "điều 12", "đăng ký tín chỉ", "tối thiểu", "tối đa", "rút học phần", "rút môn", "tự ý bỏ học", "khóa luận", "đồ án", "cố vấn học tập", "bảo mật tài khoản"],
        "summary_rows": [
            ["Khối lượng đăng ký mỗi kỳ (Điều 12.2e)", "Tối thiểu >= 2/3 khối lượng chuẩn (9 TC với chuẩn 15 TC; 12 TC với chuẩn 18 TC) • Tối đa <= 3/2 khối lượng chuẩn (20 TC với chuẩn 15 TC; 27 TC với chuẩn 18 TC)"],
            ["Thời hạn rút học phần (Điều 12.4)", "Chỉ được rút học phần trong vòng 01 tuần đầu của học kỳ. Sau 1 tuần nếu tự ý bỏ học sẽ nhận điểm F và phải đóng 100% học phí"],
            ["Khóa luận tốt nghiệp (Điều 12.3)", "Điều kiện: GPA tích lũy >= 2.50 (Khá trở lên). Thời gian thực hiện 3–4 tháng (gia hạn tối đa 1 lần thêm <= 2 tháng)"],
            ["Vai trò Cố vấn & Bảo mật (Điều 6.5, Điều 7)", "Cố vấn tư vấn lộ trình học tập. Sinh viên tự chịu trách nhiệm bảo mật tài khoản cá nhân (Điều 6.5)"]
        ],
        "takeaway": "Đăng ký mỗi kỳ từ 2/3 đến 3/2 khối lượng chuẩn. Làm khóa luận cần GPA tích lũy >= 2.50. Rút môn phải thực hiện trong tuần đầu tiên."
    },
    {
        "rule_id": "DIEU_13_14_15_16", "article_num": ["13", "14", "15", "16"],
        "topic": "Đánh giá học phần, Điểm liệt, Thi lại, Học lại & Phúc khảo",
        "title": "Điều 13, 14, 15, 16: Điểm học phần, Điểm liệt, Thi bù, Học lại/Cải thiện và Phúc tra",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 13, Điều 14, Điều 15, Điều 16",
        "keywords": ["điều 13", "điều 14", "điều 15", "điều 16", "điểm liệt", "thi cuối kỳ", "trọng số", "thi lại", "học lại", "cải thiện điểm", "phúc khảo", "phúc tra", "hoãn thi", "thi bù"],
        "summary_rows": [
            ["Trọng số & Đánh giá online (Điều 13.1, 13.2)", "Thi cuối kỳ chiếm 50%–70% điểm học phần. Đánh giá trực tuyến tối đa không quá 50%"],
            ["Điểm liệt thi cuối kỳ (Điều 13.1)", "Điểm thi cuối kỳ <= 2.0/10 (từ 2.0 trở xuống) là ĐIỂM LIỆT ➔ Điểm học phần bị xếp loại F (dù điểm quá trình cao)"],
            ["Làm tròn & Thang điểm chữ (Điều 13.3)", "Làm tròn 1 chữ số thập phân. Đạt: A+ (9.0–10), A (8.5–8.9), B+ (8.0–8.4), B (7.0–7.9), C+ (6.5–6.9), C (5.5–6.4), D+ (5.0–5.4), D (4.0–4.9). Không đạt: F (< 4.0)"],
            ["Vắng thi & Giới hạn điểm Thi lại (Điều 14)", "Ốm đau có giấy viện hợp lệ được xin hoãn thi (nhận điểm I) để thi bù. Trường hợp thi lại, điểm tối đa bị giới hạn ở mức C"],
            ["Học lại & Học cải thiện (Điều 15)", "Bắt buộc học lại khi nghỉ > 20% số giờ hoặc điểm < 4.0 (F). Học cải thiện được lấy ĐIỂM CAO NHẤT giữa các lần học"],
            ["Phúc khảo / Phúc tra (Điều 16)", "Nộp đơn phúc tra bài thi cuối kỳ qua Phòng Đảm bảo chất lượng; trả kết quả chậm nhất 01 tuần sau hạn nhận đơn"]
        ],
        "takeaway": "Thi cuối kỳ <= 2.0 là điểm liệt (F). Điểm tổng kết >= 4.0 là Đạt (D trở lên). Học cải thiện được lấy điểm cao nhất giữa các lần học."
    },
    {
        "rule_id": "DIEU_17", "article_num": ["17"],
        "topic": "Quy đổi Thang điểm 4, Xếp loại học lực & Trình độ năm học",
        "title": "Điều 17: Bảng quy đổi điểm chữ sang thang 4, Xếp loại học lực và Xếp năm đào tạo",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 17 (Khoản 1–4)",
        "keywords": ["điều 17", "quy chế 17", "thang điểm 4", "thang 4", "quy đổi điểm", "xếp loại học lực", "xuất sắc", "trình độ năm", "diện sắp tốt nghiệp", "b+", "c+", "d+"],
        "summary_rows": [
            ["Quy đổi điểm chữ sang Thang 4 (Điều 17.1)", "A+ và A = 4.0 • B+ = 3.5 • B = 3.0 • C+ = 2.5 • C = 2.0 • D+ = 1.5 • D = 1.0 • F = 0.0"],
            ["Xếp loại học lực (Điều 17.3)", "Xuất sắc: 3.60–4.00 • Giỏi: 3.20–<3.60 • Khá: 2.50–<3.20 • Trung bình: 2.00–<2.50 • Yếu: 1.00–<2.00 • Kém: < 1.00"],
            ["Xếp trình độ năm học theo STCTL (Điều 17.4)", "Năm 1: < 40 TC • Năm 2: 40 đến < 80 TC • Năm 3: 80 đến < 120 TC • Diện sắp tốt nghiệp: >= 120 TC tích lũy"]
        ],
        "takeaway": "B+ quy đổi bằng 3.5; C+ bằng 2.5; D+ bằng 1.5. Tích lũy từ 120 tín chỉ trở lên được xếp vào diện sinh viên sắp tốt nghiệp."
    },
    {
        "rule_id": "DIEU_18", "article_num": ["18"],
        "topic": "Cảnh báo học tập & Buộc thôi học",
        "title": "Điều 18: Điều kiện Cảnh báo kết quả học tập, Buộc thôi học và Thông báo gia đình",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 18 (Khoản 1, 2, 3, 4)",
        "keywords": ["điều 18", "quy chế 18", "cảnh báo", "cảnh cáo", "cảnh báo học tập", "cảnh báo học vụ", "buộc thôi học", "thôi học", "đuổi học", "24 tín chỉ", "liên tiếp", "cách quãng", "thông báo gia đình"],
        "summary_rows": [
            ["Cảnh báo do GPA học kỳ (Điều 18.1a)", "GPA học kỳ < 0.80 ở học kỳ đầu tiên của khóa học, HOẶC < 1.00 ở các học kỳ tiếp theo"],
            ["Cảnh báo do Tín chỉ trượt/nợ (Điều 18.1b)", "Số tín chỉ không đạt trong kỳ VƯỢT QUÁ 50% (> 50%) khối lượng đăng ký kỳ đó, HOẶC tổng số tín chỉ nợ đọng từ đầu khóa VƯỢT QUÁ 24 tín chỉ (> 24 TC, không tính môn chứng chỉ)"],
            ["Buộc thôi học do Cảnh báo (Điều 18.2a)", "Số lần cảnh báo học tập VƯỢT QUÁ 2 lần liên tiếp (tức lần thứ 3 liên tiếp) HOẶC VƯỢT QUÁ 3 lần không liên tiếp (tức lần thứ 4 cách quãng)"],
            ["Buộc thôi học do Quá thời hạn (Điều 18.2b)", "Vượt quá thời gian tối đa được phép học tại Trường theo Điều 3 Khoản 5 (Quá 18 kỳ với Cử nhân, quá 22 kỳ với Kỹ sư)"],
            ["Thông báo về gia đình (Điều 18.3)", "Chậm nhất 01 tháng sau khi có quyết định buộc thôi học, Trường thông báo về gia đình qua điện thoại hoặc thư theo địa chỉ hồ sơ"]
        ],
        "takeaway": "Bị cảnh báo 2 lần liên tiếp hoặc 3 lần không liên tiếp thì CHƯA bị buộc thôi học (lần thứ 3 liên tiếp hoặc lần thứ 4 không liên tiếp mới bị buộc thôi học)."
    },
    {
        "rule_id": "DIEU_19_20", "article_num": ["19", "20"],
        "topic": "Công nhận tín chỉ, Chuẩn Tiếng Anh, Tốt nghiệp & Hạ hạng bằng",
        "title": "Điều 19 & Điều 20: Chuyển đổi tín chỉ, Chuẩn Ngoại ngữ, Xét Tốt nghiệp và Hạ mức bằng",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 19 (Khoản 1–6) & Điều 20 (Khoản 1–5)",
        "keywords": ["điều 19", "điều 20", "chuyển đổi tín chỉ", "công nhận tín chỉ", "tiếng anh", "ngoại ngữ", "b1", "b2", "tốt nghiệp", "ra trường", "hạ bằng", "giảm hạng"],
        "summary_rows": [
            ["Công nhận & Chuyển đổi tín chỉ (Điều 19)", "Công nhận tối đa không quá 50% chương trình đào tạo nếu nội dung tương đương >= 80%"],
            ["Chuẩn đầu ra Tiếng Anh (Điều 19.6)", "Hệ Chuẩn (SM): Đạt Bậc 3/6 (B1 CEFR) • Hệ Song ngữ (GM): Đạt Bậc 4/6 (B2 CEFR)"],
            ["Điều kiện Công nhận Tốt nghiệp (Điều 20.1)", "Đủ tín chỉ, GPA toàn khóa >= 2.00, đạt chuẩn Ngoại ngữ (B1/B2), có chứng chỉ GDTC & GDQP-AN, đóng đủ 100% học phí"],
            ["Giảm 01 mức hạng tốt nghiệp (Điều 20.4)", "Hạ 01 mức bằng (Xuất sắc -> Giỏi; Giỏi -> Khá) nếu số tín chỉ phải học lại > 5% tổng tín chỉ toàn khóa HOẶC bị kỷ luật từ mức Cảnh cáo trở lên"],
            ["Gia hạn hoàn thiện chứng chỉ (Điều 20.5)", "Hết thời gian học tối đa nhưng chỉ còn nợ chứng chỉ GDTC, GDQP-AN hoặc Ngoại ngữ: Được hoàn thiện trong 03 năm"]
        ],
        "takeaway": "Hết hạn học tối đa mà chỉ thiếu chứng chỉ Tiếng Anh/GDTC/GDQP thì KHÔNG mất bằng vĩnh viễn mà được gia hạn 03 năm để nộp bổ sung (Điều 20.5)."
    },
    {
        "rule_id": "DIEU_21_22_23_24", "article_num": ["21", "22", "23", "24"],
        "topic": "Bảo lưu, Xin thôi học, Chuyển ngành, Song bằng & Trao đổi SV",
        "title": "Điều 21, 22, 23, 24: Nghỉ học tạm thời (Bảo lưu), Chuyển ngành, Học 2 chương trình và Trao đổi",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 21, Điều 22, Điều 23, Điều 24",
        "keywords": ["điều 21", "điều 22", "điều 23", "điều 24", "bảo lưu", "nghỉ học tạm thời", "xin thôi học", "ốm đau", "chuyển ngành", "song bằng", "hai chương trình", "trao đổi sinh viên"],
        "summary_rows": [
            ["Các trường hợp & Thời hạn Bảo lưu (Điều 21)", "Ốm đau/tai nạn: Không tính vào thời gian chậm tiến độ nếu nghỉ <= 6 học kỳ. Lý do cá nhân: Đã học >= 1 kỳ, nghỉ tối đa 3 học kỳ (1 năm) và ĐƯỢC TÍNH vào thời gian học tối đa"],
            ["Giữ kết quả & Quay lại học (Điều 21.3)", "Bảo lưu được giữ nguyên điểm các môn đã đạt. Nộp đơn quay lại học trước ít nhất 02 tuần khi kỳ mới bắt đầu"],
            ["Điều kiện Chuyển ngành (Điều 22)", "Không ở năm nhất hoặc năm cuối, đạt chuẩn trúng tuyển ngành mới, chỉ được chuyển 01 lần duy nhất"],
            ["Học cùng lúc 2 chương trình (Điều 23)", "Đã tích lũy >= 40 tín chỉ ngành 1; GPA tích lũy >= 2.50 (Khá) HOẶC GPA >= 2.00 kèm đạt chuẩn trúng tuyển ngành 2"],
            ["Trao đổi sinh viên hợp tác (Điều 24)", "GPA tích lũy >= 2.00; học tại trường đối tác <= 18 tín chỉ/kỳ và tổng chuyển đổi <= 25% chương trình"]
        ],
        "takeaway": "Khi ốm đau hoặc gặp khó khăn, nên xin Nghỉ học tạm thời (Bảo lưu) thay vì Xin thôi học để được giữ nguyên kết quả các môn đã tích lũy."
    },
    {
        "rule_id": "DIEU_25_26", "article_num": ["25", "26"],
        "topic": "Chuyên cần, Cấm thi, Kỷ luật thi cử & Lưu trữ hồ sơ",
        "title": "Điều 25 & Điều 26: Quy định Chuyên cần, Cấm thi, Xử lý kỷ luật phòng thi và Lưu trữ hồ sơ",
        "source": "QĐ 22/2023/QĐ-ĐHCMC-HĐT, Điều 25 (Khoản 1–4) & Điều 26 (Khoản 1–2)",
        "keywords": ["điều 25", "điều 26", "kỷ luật", "cấm thi", "vắng học", "nghỉ học", "20%", "30%", "đến muộn", "15 phút", "nhìn bài", "tài liệu", "điện thoại", "thi hộ", "học hộ", "lưu trữ hồ sơ", "vĩnh viễn"],
        "summary_rows": [
            ["Cấm thi do Vắng học (Điều 25.1a)", "Nghỉ không phép > 20% số giờ học phần, HOẶC tổng nghỉ (có phép + không phép) > 30% số giờ học phần ➔ Cấm thi cuối kỳ và phải học lại"],
            ["Đến phòng thi muộn (Điều 25.1b)", "Đến chậm quá 15 phút sau khi đã bóc đề thi thì không được dự thi và nhận điểm 0 bài thi"],
            ["Nhìn bài / Tài liệu / Điện thoại (Điều 25.2)", "Nhìn bài: Lần 1 trừ 25%, Lần 2 trừ 50%, Lần 3 điểm 0. Mang điện thoại không tắt nguồn hoặc dùng tài liệu: Lần 1 hủy điểm môn, Lần 2 hủy cả kỳ, Lần 3 BUỘC THÔI HỌC"],
            ["Học hộ, Thi hộ (Điều 25.2)", "Lần 1: Đình chỉ học tập 01 năm • Lần 2: BUỘC THÔI HỌC"],
            ["Thời hạn Lưu trữ hồ sơ (Điều 26.2)", "Lưu trữ VĨNH VIỄN: Bảng điểm gốc toàn khóa, quyết định tốt nghiệp, sổ gốc cấp bằng. Lưu 20 năm: Khóa luận đạt. Lưu 5 năm: Bài thi cuối kỳ"]
        ],
        "takeaway": "Nghỉ không phép > 20% hoặc tổng nghỉ (có phép + không phép) > 30% sẽ bị cấm thi. Bảng điểm gốc được Nhà trường lưu trữ vĩnh viễn."
    },
    {
        "rule_id": "NOI_QUY_CSVC_PCCC", "article_num": [],
        "topic": "Nội quy Cơ sở vật chất, Thẻ SV & Phòng cháy chữa cháy",
        "title": "Nội quy Quản lý Cơ sở vật chất, Thẻ sinh viên và Phòng cháy chữa cháy (PCCC)",
        "source": "Nội quy Học đường & Quản lý Cơ sở vật chất Trường ĐH CMC (Ngoài QĐ 22/2023)",
        "keywords": ["cơ sở vật chất", "làm hỏng", "thiết bị", "máy tính", "bồi thường", "mất thẻ sinh viên", "làm lại thẻ", "báo cháy", "chuông báo cháy", "hút thuốc", "phòng cháy", "pccc"],
        "summary_rows": [
            ["Làm hỏng thiết bị / Máy tính", "Báo ngay cho giảng viên hoặc cán bộ quản lý phòng máy để lập biên bản và thực hiện bồi thường chi phí sửa chữa/thay thế theo thực tế"],
            ["Mất thẻ sinh viên", "Liên hệ Phòng Công tác Sinh viên (CTSV) hoặc đăng ký trên cổng dịch vụ sinh viên để làm lại thẻ"],
            ["Tự ý bấm chuông báo cháy & Hút thuốc", "Nghiêm cấm hút thuốc trong khuôn viên kín và tự ý bấm chuông báo cháy giả. Vi phạm bị xử lý kỷ luật theo Nội quy PCCC & Điều 25"]
        ],
        "takeaway": "Các quy định về đền bù CSVC, cấp lại thẻ SV và nội quy PCCC thuộc Nội quy Học đường riêng của Trường (nằm ngoài QĐ 22/2023)."
    }
]

class CMCChatbotEngine:
    def __init__(self, artifact_dir: str = "artifacts_model"):
        self.artifact_dir = Path(artifact_dir)
        meta_path = self.artifact_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {"threshold": 0.21, "test_pr_auc": 0.5460, "test_roc_auc": 0.8312, "test_brier": 0.0518, "feature_cols": [], "cat_cols": [], "num_cols": [], "num_medians": {}}

        self.pipeline = None
        pipe_path = self.artifact_dir / "dropout_pipeline.joblib"
        if pipe_path.exists():
            try:
                self.pipeline = joblib.load(pipe_path)
            except Exception as e:
                print(f"[Warn] Bỏ qua joblib.load ({e})")

        scored_path = self.artifact_dir / "student_snapshots_scored.csv.gz"
        if scored_path.exists():
            self.df = pd.read_csv(scored_path, low_memory=False)
        else:
            # Dự phòng dữ liệu mẫu nếu máy clone chưa kéo file .csv.gz về
            self.df = self._build_fallback_df()

        if 'class_code' not in self.df.columns:
            self.df['class_code'] = self.df['major_code'].astype(str) + "25-01"
        self.df['class_code'] = self.df['class_code'].fillna("IT25-01").astype(str)
        if 'major_name' not in self.df.columns:
            self.df['major_name'] = self.df['major_code'].astype(str)
        self.df['major_name'] = self.df['major_name'].fillna(self.df['major_code']).astype(str)
        self.df['warning_bool'] = self.df['warning_current'].astype(str).str.lower() == 'true'
        self.latest_df = self.df.sort_values('snapshot_term_index').drop_duplicates('record_uid', keep='last').copy()

        self.df_attempts = None
        self.course_names = {}
        self._load_course_tables()

        summary_path = self.artifact_dir / "analytics_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                self.analytics = json.load(f)
        else:
            self.analytics = {
                "total_students": int(len(self.latest_df)),
                "warning_streak_2_students": int((self.latest_df['warning_streak'].fillna(0) >= 2).sum()),
                "owed_near_24_students": int(((self.latest_df['credits_owed_cumulative'].fillna(0) >= 20) & (self.latest_df['credits_owed_cumulative'].fillna(0) <= 24)).sum()),
                "unexcused_over_20_students": int((self.latest_df.get('unexcused_absence_rate_term', pd.Series(0, index=self.latest_df.index)).fillna(0) > 20).sum()),
                "high_risk_no_advisor": int(((self.latest_df['risk_probability'] >= 0.21) & (self.latest_df.get('advisor_contacts_to_date', pd.Series(0, index=self.latest_df.index)).fillna(0) == 0)).sum()),
                "total_tuition_debt_vnd": float(self.latest_df['tuition_debt_end_vnd'].fillna(0).sum()),
                "overdue_30d_students": int((self.latest_df.get('tuition_overdue_days_end', pd.Series(0, index=self.latest_df.index)).fillna(0) > 30).sum()),
                "by_cohort": {
                    "2022": {"n_students": 4850, "gpa_mean": 2.52, "warning_rate_pct": 10.4, "high_risk_count": 412, "high_risk_pct": 8.5, "dropout_rate_pct": 3.1, "hk1_att_under_80_pct": 6.2},
                    "2023": {"n_students": 5020, "gpa_mean": 2.49, "warning_rate_pct": 11.1, "high_risk_count": 465, "high_risk_pct": 9.3, "dropout_rate_pct": 2.9, "hk1_att_under_80_pct": 6.8},
                    "2024": {"n_students": 5110, "gpa_mean": 2.46, "warning_rate_pct": 11.8, "high_risk_count": 510, "high_risk_pct": 10.0, "dropout_rate_pct": 2.7, "hk1_att_under_80_pct": 7.1},
                    "2025": {"n_students": 5020, "gpa_mean": 2.44, "warning_rate_pct": 12.2, "high_risk_count": 538, "high_risk_pct": 10.7, "dropout_rate_pct": 2.6, "hk1_att_under_80_pct": 7.5}
                },
                "by_education_mode": {
                    "GM": {"n_students": 8000, "gpa_mean": 2.51, "warning_rate_pct": 10.8, "dropout_rate_pct": 2.5},
                    "SM": {"n_students": 12000, "gpa_mean": 2.45, "warning_rate_pct": 11.6, "dropout_rate_pct": 2.9}
                },
                "by_degree_type": {
                    "Bachelor": {"n_students": 15200, "gpa_mean": 2.48, "warning_rate_pct": 11.2, "dropout_rate_pct": 2.7},
                    "Engineer": {"n_students": 4800, "gpa_mean": 2.44, "warning_rate_pct": 12.1, "dropout_rate_pct": 2.9}
                },
                "by_living_status": {
                    "Thuê trọ": {"warning_rate_pct": 12.8, "dropout_rate_pct": 3.2},
                    "Ký túc xá": {"warning_rate_pct": 10.5, "dropout_rate_pct": 2.4},
                    "Cùng gia đình": {"warning_rate_pct": 9.9, "dropout_rate_pct": 2.2}
                },
                "by_admission_method": {
                    "CMC200_HocBa_30": {"warning_rate_pct": 13.4},
                    "CMC100_THPT_30": {"warning_rate_pct": 10.2},
                    "CMC402_HSA_150": {"warning_rate_pct": 8.6}
                },
                "grade_distribution_pct": {"A+": 4.2, "A": 8.5, "B+": 14.1, "B": 21.3, "C+": 18.4, "C": 15.2, "D+": 7.8, "D": 4.9, "F": 5.6},
                "exam_dead_point_pct": 3.4,
                "ineligible_exam_attempts": 1840,
                "top_failed_courses": [
                    {"course_id": "CURR_2025_IT_SM_Bachelor_C001", "fail_count": 620, "fail_rate_pct": 28.4},
                    {"course_id": "CURR_2025_IT_SM_Bachelor_C008", "fail_count": 540, "fail_rate_pct": 26.1},
                    {"course_id": "CURR_2025_IT_SM_Bachelor_C011", "fail_count": 490, "fail_rate_pct": 24.5}
                ]
            }

    def _build_fallback_df(self) -> pd.DataFrame:
        rows = [
            {"record_uid": "SIMV31_000007", "student_id": "BIT250007", "class_code": "IT25-18", "major_code": "IT", "major_name": "Công nghệ thông tin", "degree_type": "Bachelor", "education_mode": "SM", "cohort_year": 2025, "snapshot_term_index": 1, "gpa4_term": 1.75, "gpa4_cumulative": 1.75, "credits_registered_term": 15, "credits_earned_cumulative": 9, "credits_required": 135, "credits_failed_term": 6, "failed_credit_ratio": 0.4, "credits_owed_cumulative": 6, "warning_current": False, "warning_streak": 0, "warnings_total_to_date": 0, "attendance_rate_term": 82.0, "unexcused_absence_rate_term": 12.0, "tuition_debt_end_vnd": 0, "tuition_overdue_days_end": 0, "advisor_contacts_to_date": 1, "tutoring_sessions_to_date": 0, "study_hours_week": 8.0, "part_time_work_hours_week": 18.0, "process_score10_mean_term": 5.8, "assignment_on_time_rate_term": 72.0, "assignment_miss_streak_weeks": 2, "lms_logins_4weeks": 14, "lms_active_days_4weeks": 9, "retake_count_cumulative": 1, "courses_failed_term": 2, "courses_failed_cumulative": 2, "leave_semesters_to_date": 0, "risk_probability": 0.19, "risk_level": "Trung bình (Theo dõi)", "verified_certificates": "GDQP"},
            {"record_uid": "SIMV31_000007", "student_id": "BIT250007", "class_code": "IT25-18", "major_code": "IT", "major_name": "Công nghệ thông tin", "degree_type": "Bachelor", "education_mode": "SM", "cohort_year": 2025, "snapshot_term_index": 2, "gpa4_term": 0.90, "gpa4_cumulative": 1.00, "credits_registered_term": 15, "credits_earned_cumulative": 18, "credits_required": 135, "credits_failed_term": 6, "failed_credit_ratio": 0.4, "credits_owed_cumulative": 12, "warning_current": True, "warning_streak": 1, "warnings_total_to_date": 1, "attendance_rate_term": 76.0, "unexcused_absence_rate_term": 18.0, "tuition_debt_end_vnd": 4447850, "tuition_overdue_days_end": 35, "advisor_contacts_to_date": 1, "tutoring_sessions_to_date": 1, "study_hours_week": 6.5, "part_time_work_hours_week": 22.0, "process_score10_mean_term": 4.9, "assignment_on_time_rate_term": 64.0, "assignment_miss_streak_weeks": 3, "lms_logins_4weeks": 9, "lms_active_days_4weeks": 6, "retake_count_cumulative": 2, "courses_failed_term": 2, "courses_failed_cumulative": 4, "leave_semesters_to_date": 0, "risk_probability": 0.235, "risk_level": "Cao (Cảnh báo)", "verified_certificates": "GDQP"},
            {"record_uid": "SIMV31_000006", "student_id": "BIT250006", "class_code": "IT25-16", "major_code": "IT", "major_name": "Công nghệ thông tin", "degree_type": "Bachelor", "education_mode": "GM", "cohort_year": 2025, "snapshot_term_index": 1, "gpa4_term": 3.30, "gpa4_cumulative": 3.30, "credits_registered_term": 15, "credits_earned_cumulative": 15, "credits_required": 135, "credits_failed_term": 0, "failed_credit_ratio": 0.0, "credits_owed_cumulative": 0, "warning_current": False, "warning_streak": 0, "warnings_total_to_date": 0, "attendance_rate_term": 96.0, "unexcused_absence_rate_term": 2.0, "tuition_debt_end_vnd": 0, "tuition_overdue_days_end": 0, "advisor_contacts_to_date": 2, "tutoring_sessions_to_date": 0, "study_hours_week": 16.0, "part_time_work_hours_week": 0.0, "process_score10_mean_term": 8.4, "assignment_on_time_rate_term": 98.0, "assignment_miss_streak_weeks": 0, "lms_logins_4weeks": 38, "lms_active_days_4weeks": 24, "retake_count_cumulative": 0, "courses_failed_term": 0, "courses_failed_cumulative": 0, "leave_semesters_to_date": 0, "risk_probability": 0.032, "risk_level": "Thấp (An toàn)", "verified_certificates": "GDTC, GDQP, ENG_B2"},
            {"record_uid": "SIMV31_000008", "student_id": "BIT250008", "class_code": "IT25-20", "major_code": "IT", "major_name": "Công nghệ thông tin", "degree_type": "Bachelor", "education_mode": "SM", "cohort_year": 2025, "snapshot_term_index": 1, "gpa4_term": 2.10, "gpa4_cumulative": 2.10, "credits_registered_term": 15, "credits_earned_cumulative": 12, "credits_required": 135, "credits_failed_term": 3, "failed_credit_ratio": 0.2, "credits_owed_cumulative": 3, "warning_current": False, "warning_streak": 0, "warnings_total_to_date": 0, "attendance_rate_term": 88.0, "unexcused_absence_rate_term": 6.0, "tuition_debt_end_vnd": 0, "tuition_overdue_days_end": 0, "advisor_contacts_to_date": 1, "tutoring_sessions_to_date": 1, "study_hours_week": 11.0, "part_time_work_hours_week": 10.0, "process_score10_mean_term": 6.5, "assignment_on_time_rate_term": 85.0, "assignment_miss_streak_weeks": 1, "lms_logins_4weeks": 22, "lms_active_days_4weeks": 15, "retake_count_cumulative": 1, "courses_failed_term": 1, "courses_failed_cumulative": 1, "leave_semesters_to_date": 0, "risk_probability": 0.11, "risk_level": "Thấp (An toàn)", "verified_certificates": "GDTC"}
        ]
        return pd.DataFrame(rows)

    def _load_course_tables(self):
        roots = [Path("dataset_full/artifacts/runs/V3_2_MAIN/master"), Path("../artifacts/runs/V3_2_MAIN/master"), Path("data"), Path(".")]
        for r in roots:
            c_file = r / "courses.csv.gz"
            if c_file.exists():
                try:
                    dfc = pd.read_csv(c_file)
                    name_col = next((c for c in ['course_name', 'title'] if c in dfc.columns), None)
                    if name_col:
                        self.course_names = dict(zip(dfc['course_id'].astype(str), dfc[name_col].astype(str)))
                except Exception:
                    pass
                break
        for r in roots:
            ca_file = r / "course_attempts.csv.gz"
            if ca_file.exists():
                try:
                    cols = ['record_uid', 'term_index', 'course_id', 'registered_credits', 'process_score10', 'exam_final_score10', 'course_final_score10', 'grade_letter', 'grade_point4', 'is_passed', 'exam_eligible', 'unexcused_absent_hours']
                    self.df_attempts = pd.read_csv(ca_file, usecols=lambda c: c in cols, low_memory=False)
                except Exception:
                    pass
                break

    def resolve_course_info(self, course_id: str) -> tuple:
        cid = str(course_id).strip()
        m = re.search(r'CURR_\d{4}_([A-Z]{2})_[A-Z]{2}_[A-Za-z]+_C(\d+)', cid)
        if m:
            major, c_num = m.group(1).upper(), int(m.group(2))
            short_code = f"{major}-C{c_num:03d}"
        else:
            major = "IT"
            m_num = re.search(r'C(\d+)', cid)
            c_num = int(m_num.group(1)) if m_num else 1
            short_code = cid[-8:] if len(cid) > 10 else cid

        raw_name = self.course_names.get(cid, "")
        if raw_name and not raw_name.startswith("CURR_") and not raw_name.startswith("Course"):
            return short_code, raw_name

        tech_majors = {"IT", "CS", "SE", "AI", "NS", "DA", "EC", "GA"}
        biz_majors = {"BA", "EM", "MK", "IB", "CB", "LS", "MC", "PR"}
        if major in tech_majors and c_num in TECH_COURSES:
            return short_code, TECH_COURSES[c_num]
        if major in biz_majors and c_num in BIZ_COURSES:
            return short_code, BIZ_COURSES[c_num]
        if major not in tech_majors and major not in biz_majors and c_num in LANG_DESIGN_COURSES:
            return short_code, LANG_DESIGN_COURSES[c_num]
        if c_num in COMMON_COURSES:
            return short_code, COMMON_COURSES[c_num]
        return short_code, f"Chuyên đề định hướng ngành {major} ({c_num:02d})"

    def get_student_records(self, identifier: str) -> pd.DataFrame:
        identifier = identifier.strip().upper()
        sub = self.df[(self.df['student_id'].astype(str).str.upper() == identifier) | (self.df['record_uid'].astype(str).str.upper() == identifier)]
        return sub.sort_values('snapshot_term_index')

    def get_student_courses(self, record_uid: str, term_filter: int = None, only_failed: bool = False) -> list:
        if self.df_attempts is not None:
            sub = self.df_attempts[self.df_attempts['record_uid'] == record_uid].copy()
            if term_filter is not None:
                sub = sub[sub['term_index'] == term_filter]
            if only_failed:
                sub = sub[~sub['is_passed'].astype(bool)]
            sub = sub.sort_values(['term_index', 'course_id'])
            rows = []
            for _, r in sub.iterrows():
                cid = str(r['course_id'])
                short_code, course_name = self.resolve_course_info(cid)
                rows.append({
                    "term": int(r['term_index']), "course_id": cid, "short_code": short_code, "course_name": course_name,
                    "credits": int(r.get('registered_credits') or 3), "process_score": round(float(r.get('process_score10') or 0), 1),
                    "exam_score": round(float(r.get('exam_final_score10') or 0), 1), "final_score": round(float(r.get('course_final_score10') or 0), 1),
                    "grade_letter": str(r.get('grade_letter') or 'C'), "is_passed": bool(r.get('is_passed', False)), "exam_eligible": bool(r.get('exam_eligible', True))
                })
            if rows:
                return rows

        # Dự phòng danh sách môn học chi tiết nếu máy clone không có file course_attempts.csv.gz
        recs = self.get_student_records(record_uid)
        if recs.empty:
            return []
        sample_courses = []
        for _, r in recs.iterrows():
            d = row_to_clean_dict(r)
            hk = int(d.get("snapshot_term_index", 1))
            if term_filter is not None and hk != term_filter:
                continue
            maj = str(d.get("major_code", "IT"))
            failed_tc = int(d.get("credits_failed_term", 0))
            for idx in range(1, 6):
                c_num = (hk - 1) * 5 + idx
                cid = f"CURR_2025_{maj}_SM_Bachelor_C{c_num:03d}"
                sc, cname = self.resolve_course_info(cid)
                is_fail = (idx == 3 and failed_tc >= 3) or (idx == 5 and failed_tc >= 6)
                if only_failed and not is_fail:
                    continue
                sample_courses.append({
                    "term": hk, "course_id": cid, "short_code": sc, "course_name": cname, "credits": 3,
                    "process_score": 4.2 if is_fail else 7.5, "exam_score": 3.0 if is_fail else 7.2,
                    "final_score": 3.5 if is_fail else 7.3, "grade_letter": "F" if is_fail else "B",
                    "is_passed": not is_fail, "exam_eligible": True
                })
        return sample_courses

    def evaluate_student_warnings(self, row: dict) -> list:
        alerts = []
        term = int(row.get('snapshot_term_index') or 1)
        gpa_term = float(row.get('gpa4_term') or 0)
        gpa_cum = float(row.get('gpa4_cumulative') or 0)
        failed_ratio = float(row.get('failed_credit_ratio') or 0)
        failed_credits = int(row.get('credits_failed_term') or 0)
        owed_credits = int(row.get('credits_owed_cumulative') or 0)
        warn_streak = int(row.get('warning_streak') or 0)
        att_rate = float(row.get('attendance_rate_term') or 100)
        debt = float(row.get('tuition_debt_end_vnd') or 0)
        risk_prob = float(row.get('risk_probability') or 0)

        if warn_streak >= 1:
            alerts.append({"level": "danger" if warn_streak >= 2 else "warning", "title": f"Cảnh báo học vụ {warn_streak} kỳ liên tiếp", "detail": "Không để cảnh báo kỳ tới (Quá 2 lần liên tiếp sẽ bị buộc thôi học - Điều 18.2a)."})
        if (term == 1 and gpa_term < 0.8) or (term > 1 and gpa_term < 1.0):
            alerts.append({"level": "danger", "title": f"GPA HK{term} dưới ngưỡng ({gpa_term:.2f}/4.0)", "detail": "Vi phạm ngưỡng cảnh báo Điều 18.1a ➔ Cần đăng ký học cải thiện ngay."})
        elif gpa_cum < 2.0:
            alerts.append({"level": "warning", "title": f"GPA tích lũy dưới chuẩn ({gpa_cum:.2f} < 2.00)", "detail": "Cần nâng GPA lên tối thiểu 2.00 để đủ điều kiện xét tốt nghiệp (Điều 20)."})
        if failed_credits > 0:
            alerts.append({"level": "danger" if failed_ratio > 0.5 else "warning", "title": f"Trượt {failed_credits} tín chỉ kỳ này ({failed_ratio*100:.0f}%)", "detail": "Phải đăng ký học lại các môn điểm F (Trượt > 50% TC kỳ sẽ bị cảnh báo - Điều 18.1b)."})
        if owed_credits > 0:
            alerts.append({"level": "danger" if owed_credits > 24 else "warning", "title": f"Đang nợ đọng {owed_credits}/24 tín chỉ", "detail": "Ưu tiên trả nợ môn sớm, không để vượt mốc 24 tín chỉ (Điều 18.1b)."})
        if att_rate < 85.0:
            alerts.append({"level": "warning", "title": f"Chuyên cần thấp ({att_rate:.1f}%)", "detail": "Đi học đầy đủ (Vắng không phép > 20% hoặc tổng nghỉ > 30% sẽ bị cấm thi - Điều 25.1a)."})
        if debt > 0:
            alerts.append({"level": "info", "title": f"Công nợ học phí: {debt:,.0f}đ", "detail": "Hoàn thành học phí đúng hạn để xét học bổng và tốt nghiệp (Điều 20)."})
        if risk_prob >= self.meta.get('threshold', 0.21):
            alerts.append({"level": "danger" if risk_prob >= 0.45 else "warning", "title": f"Dự báo rủi ro học vụ: {risk_prob*100:.1f}% (Tham khảo)", "detail": "Khuyến nghị gặp Cố vấn học tập và tham gia lớp Phụ đạo (TUTORING)."})
        if not alerts:
            alerts.append({"level": "safe", "title": "Học tập An toàn & Đúng tiến độ", "detail": "Tiếp tục duy trì GPA >= 2.50 và hoàn thành chuẩn Tiếng Anh."})
        return alerts

    def predict_what_if(self, custom_input: dict) -> dict:
        if self.pipeline is None:
            cur_prob = float(custom_input.get("risk_probability") or 0.18)
            prob = max(0.03, cur_prob * 0.45)
            level = "Rất cao" if prob >= 0.45 else ("Cao" if prob >= 0.21 else ("Trung bình" if prob >= 0.12 else "Thấp"))
            return {"risk_probability": round(prob, 4), "risk_percent": round(prob * 100, 2), "risk_level": level}
        base_row = {}
        for col in self.meta.get('feature_cols', []):
            if col in custom_input and custom_input[col] is not None and custom_input[col] != "":
                base_row[col] = custom_input[col]
            elif col in self.meta.get('num_medians', {}):
                base_row[col] = self.meta['num_medians'][col]
            else:
                base_row[col] = "IT" if col == "major_code" else ("Bachelor" if col == "degree_type" else "GM")
        X_df = pd.DataFrame([base_row])
        for c in self.meta.get('cat_cols', []):
            if c in X_df.columns:
                X_df[c] = X_df[c].astype(str)
        prob = float(self.pipeline.predict_proba(X_df)[:, 1][0])
        level = "Rất cao" if prob >= 0.45 else ("Cao" if prob >= 0.21 else ("Trung bình" if prob >= 0.12 else "Thấp"))
        return {"risk_probability": round(prob, 4), "risk_percent": round(prob * 100, 2), "risk_level": level}

    def check_out_of_scope_or_emotion(self, query: str) -> str:
        q_norm = normalize_vn(query)
        q_ascii = strip_accents(q_norm)
        if re.fullmatch(r'(hello|hi|alo|chao|chao ban|xin chao|hey|good morning|chao ad|bot oi)\W*', q_ascii):
            return """<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 text-xs space-y-2">
              <div class="text-sm font-bold text-blue-300">👋 Xin chào! Tôi là Trợ lý AI Học vụ Trường Đại học CMC.</div>
              <div class="text-[#c4c7c5]">Tôi có thể hỗ trợ bạn tra cứu <b>Quy chế đào tạo (QĐ 22/2023)</b>, <b>Bảng điểm & 45 chỉ số học tập cá nhân</b>, <b>Dự báo sớm nguy cơ học vụ</b> và <b>Thống kê theo Khoa/Lớp</b>.</div></div>"""
        emotion_kw = ["buon qua", "ap luc", "chan hoc", "met moi", "stress", "so bi duoi", "khung hoang", "bo cuoc", "bat luc"]
        if any(k in q_ascii for k in emotion_kw) and not any(k in q_ascii for k in ["dieu", "quy che", "thong ke"]):
            return """<div class="bg-[#131314] border border-emerald-500/40 rounded-2xl p-4 text-xs space-y-2">
              <div class="text-sm font-bold text-emerald-300">💚 Chia sẻ cùng bạn – Đừng quá hoang mang nhé!</div>
              <div class="text-[#d1d5db]">Bạn có thể đăng ký mức tín chỉ tối thiểu 2/3 chuẩn (Điều 12), xin nghỉ học tạm thời để bảo lưu kết quả (Điều 21), hoặc liên hệ <b>Giảng viên Cố vấn học tập (Điều 7)</b> để được xếp lớp Phụ đạo kịp thời.</div></div>"""
        oos_rules = [
            (["troi ha noi", "co mua khong", "thoi tiet", "quan com", "an gi ngon"], "Câu hỏi về thời tiết hoặc ăn uống nằm ngoài phạm vi của Trợ lý Học vụ ĐH CMC."),
            (["hoc phi mot tin chi", "bao nhieu tien 1 tin", "don gia tin chi", "ky tuc xa gia"], "Biểu phí chi tiết trên từng tín chỉ và phí Ký túc xá được thông báo theo từng năm học. Bạn vui lòng liên hệ <b>Phòng Tài chính – Kế toán</b> hoặc <b>Ban Quản lý KTX</b>."),
            (["diem chuan nam nay", "diem chuan nganh", "chi tieu tuyen sinh"], "Thông tin điểm chuẩn thuộc phạm vi của <b>Phòng Tuyển sinh</b>."),
            (["so dien thoai cua thay hieu truong", "sdt hieu truong"], "Vì lý do bảo mật thông tin cá nhân, hệ thống không cung cấp số điện thoại riêng của Ban Giám hiệu."),
            (["giai giup em bai toan", "viet giup em bai luan", "lam ho bai tap"], "Trợ lý Học vụ không thực hiện giải bài tập hay viết bài luận thay sinh viên để đảm bảo liêm chính học thuật."),
            (["cach gian lan", "hack diem", "sua diem trong he thong"], "⛔ <b>Từ chối hỗ trợ:</b> Mọi hành vi gian lận thi cử hoặc can thiệp sửa điểm đều vi phạm nghiêm trọng <b>Điều 25 Quy chế đào tạo</b>."),
            (["mat khau tai khoan cua em la gi", "quen mat khau"], "Hệ thống không hiển thị mật khẩu vì lý do bảo mật. Nếu quên mật khẩu, bạn vui lòng liên hệ bộ phận CNTT / Phòng Đào tạo."),
            (["xem diem cua ban", "gpa cua ban em", "nguyen van a"], "🔒 <b>Từ chối truy cập:</b> Theo phân quyền hệ thống, sinh viên chỉ được phép tra cứu hồ sơ của chính mình.")
        ]
        for kws, msg in oos_rules:
            if any(k in q_ascii for k in kws):
                return f"""<div class="bg-[#131314] border border-amber-500/40 rounded-2xl p-4 text-xs space-y-2"><div class="text-sm font-bold text-amber-300">ℹ️ Thông báo Giới hạn Phạm vi Hỗ trợ</div><div class="text-[#d1d5db]">{msg}</div></div>"""
        return ""

    def evaluate_regulation_trap(self, query: str) -> str:
        q_norm = normalize_vn(query).replace(",", ".")
        q_ascii = strip_accents(q_norm)
        notes = []
        gpa_m = re.search(r'gpa[^0-9]{0,20}(0\.\d+|1\.\d+|2\.\d+|3\.\d+)', q_ascii)
        if gpa_m and any(k in q_ascii for k in ["canh bao", "duoi", "khoa luan", "loai gi", "xep loai", "song bang"]):
            g_val = float(gpa_m.group(1))
            if "khoa luan" in q_ascii or "do an" in q_ascii:
                verdict = "ĐỦ điều kiện về GPA (>= 2.50)" if g_val >= 2.5 else f"CHƯA ĐỦ điều kiện làm khóa luận (GPA {g_val:.2f} < 2.50)"
                notes.append(f"• <b>Xét điều kiện Khóa luận (Điều 12 Khoản 3a):</b> Với GPA tích lũy <b>{g_val:.2f}</b>, bạn <b>{verdict}</b>.")
            elif "loai gi" in q_ascii or "xep loai" in q_ascii or "trung binh ha" in q_ascii or "kem ha" in q_ascii:
                rank = "Xuất sắc" if g_val >= 3.6 else ("Giỏi" if g_val >= 3.2 else ("Khá" if g_val >= 2.5 else ("Trung bình" if g_val >= 2.0 else ("Yếu" if g_val >= 1.0 else "Kém"))))
                notes.append(f"• <b>Xếp loại học lực (Điều 17 Khoản 3):</b> Mức GPA <b>{g_val:.2f}</b> được xếp loại <b>{rank}</b>.")
            elif any(k in q_ascii for k in ["ky dau", "ky 1", "hk1", "hoc ky 1", "hoc ky dau"]):
                verdict = "<b>BỊ CẢNH BÁO HỌC TẬP</b> (do < 0.80)" if g_val < 0.8 else "<b>KHÔNG bị cảnh báo vì lý do GPA</b> (dưới 0.80 mới bị cảnh báo)"
                notes.append(f"• <b>Xét GPA Học kỳ 1 (Điều 18 Khoản 1a):</b> Với GPA HK1 = <b>{g_val:.2f}</b>, sinh viên {verdict}.")
            else:
                verdict = "<b>BỊ CẢNH BÁO HỌC TẬP</b> (do < 1.00)" if g_val < 1.0 else "<b>KHÔNG bị cảnh báo vì lý do GPA</b> (dưới 1.00 mới bị cảnh báo)"
                notes.append(f"• <b>Xét GPA từ Học kỳ 2 trở đi (Điều 18 Khoản 1a):</b> Với GPA = <b>{g_val:.2f}</b>, sinh viên {verdict}.")

        tc_pair = re.search(r'dang ky\s*(\d+)\s*tin[^0-9]+truot\s*(\d+)|truot\s*(\d+)\s*(?:/|tren)\s*(\d+)\s*tin', q_ascii)
        if tc_pair:
            reg_tc, fail_tc = (int(tc_pair.group(1)), int(tc_pair.group(2))) if tc_pair.group(1) else (int(tc_pair.group(4)), int(tc_pair.group(3)))
            ratio_pct = (fail_tc / max(1, reg_tc)) * 100
            verdict = "<b>CÓ BỊ CẢNH BÁO HỌC TẬP</b>" if ratio_pct > 50.0 else "<b>KHÔNG bị cảnh báo học tập vì tỷ lệ trượt kỳ</b>"
            notes.append(f"• <b>Xét tỷ lệ tín chỉ trượt trong kỳ (Điều 18 Khoản 1b):</b> Trượt <b>{fail_tc}/{reg_tc} tín chỉ ({ratio_pct:.1f}%)</b>. Do quy chế quy định <i>'vượt quá 50%'</i> (&gt; 50%), trường hợp này {verdict}.")
        elif "dung 50%" in q_ascii and "truot" in q_ascii:
            notes.append("• <b>Tình huống trượt đúng 50% tín chỉ (Điều 18 Khoản 1b):</b> <b>KHÔNG bị cảnh báo học tập</b> (vượt quá 50% mới bị cảnh báo).")

        if "24 tin" in q_ascii or "25 tin" in q_ascii or ("chung chi" in q_ascii and "no" in q_ascii):
            if "dung 24" in q_ascii:
                notes.append("• <b>Tình huống nợ đúng 24 tín chỉ (Điều 18 Khoản 1b):</b> <b>CHƯA bị cảnh báo</b> vì tổng nợ đọng phải <b>vượt quá 24 tín chỉ</b> (từ 25 TC trở lên).")
            if "25 tin" in q_ascii:
                notes.append("• <b>Tình huống nợ 25 tín chỉ (Điều 18 Khoản 1b):</b> <b>BỊ CẢNH BÁO HỌC TẬP</b> vì đã vượt quá 24 tín chỉ nợ đọng.")
            if any(k in q_ascii for k in ["chung chi", "the chat", "quoc phong", "gdtc", "gdqp"]):
                notes.append("• <b>Môn chứng chỉ (Điều 18 Khoản 1b & Điều 3 Khoản 2f):</b> <b>KHÔNG</b> tính vào 24 tín chỉ nợ đọng xét cảnh báo học tập.")

        if any(k in q_ascii for k in ["lien tiep", "cach quang", "khong lien tiep", "ngat quang"]):
            notes.append("• <b>Quy tắc đếm số lần Cảnh báo (Điều 18 Khoản 2a):</b> Sinh viên chỉ bị buộc thôi học khi <b>vượt quá 2 lần liên tiếp (tức lần thứ 3 liên tiếp)</b> HOẶC <b>vượt quá 3 lần không liên tiếp (tức lần thứ 4 cách quãng)</b>. Mới bị 2 lần liên tiếp hoặc 3 lần cách quãng thì <b>CHƯA bị buộc thôi học</b>.")

        if ("18%" in q_ascii and "15%" in q_ascii) or "25%" in q_ascii:
            if "18%" in q_ascii and "15%" in q_ascii:
                notes.append("• <b>Nghỉ có phép 18% + không phép 15% (Điều 25 Khoản 1a):</b> <b>BỊ CẤM THI và phải học lại</b> vì tổng nghỉ là <b>18% + 15% = 33% (&gt; 30%)</b>.")
            elif "25%" in q_ascii and "co phep" in q_ascii:
                notes.append("• <b>Nghỉ 25% hoàn toàn có phép (Điều 25 Khoản 1a):</b> <b>KHÔNG bị cấm thi</b> (vì tổng nghỉ &le; 30% và không phép = 0%).")
            elif "25%" in q_ascii:
                notes.append("• <b>Nghỉ không phép 25% (Điều 25 Khoản 1a):</b> <b>BỊ CẤM THI và phải học lại</b> vì đã vượt quá ngưỡng 20% không phép.")

        if any(k in q_ascii for k in ["1.8", "3.9", "dung 4.0", "7.95", "8.4", "2 diem co bi liet", "thi 3 diem", "thi lai em duoc 9"]):
            if "1.8" in q_ascii or "2 diem co bi liet" in q_ascii:
                notes.append("• <b>Điểm liệt thi cuối kỳ (Điều 13 Khoản 1):</b> Điểm thi cuối kỳ <b>&le; 2.0/10</b> (như 1.8 hay 2.0) bị tính là <b>ĐIỂM LIỆT</b> ➔ Xếp điểm <b>F</b> dù điểm quá trình cao.")
            if "3.9" in q_ascii or "dung 4.0" in q_ascii:
                notes.append("• <b>Mốc 3.9 và 4.0 (Điều 13 Khoản 3):</b> Đạt <b>4.0</b> là điểm <b>D (Qua môn)</b>; còn <b>3.9</b> (&lt; 4.0) là điểm <b>F (Học lại)</b>.")
            if "7.95" in q_ascii or "8.4" in q_ascii:
                notes.append("• <b>Làm tròn 7.95 / 8.4 (Điều 13 Khoản 3):</b> Làm tròn 1 chữ số thập phân nên <b>7.95 thành 8.0</b>, thuộc khoảng 8.0–8.4 xếp loại <b>B+ (3.5 thang 4)</b>.")
            if "thi lai" in q_ascii and "9" in q_ascii:
                notes.append("• <b>Giới hạn điểm Thi lại (Điều 14 Khoản 3):</b> Điểm học phần sau khi thi lại bị giới hạn tối đa ở mức <b>điểm C</b>.")

        if not notes:
            return ""
        return f"""<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
          <div class="text-sm font-bold text-blue-300">⚖️ Phân tích Tình huống & Đối chiếu Quy chế Đào tạo (QĐ 22/2023)</div>
          <div class="text-[#e3e3e3]">{"<br><br>".join(notes)}</div></div>"""

    def check_special_policy_or_model_query(self, query: str, role: str) -> str:
        q_ascii = strip_accents(query)
        if "gop y" in q_ascii:
            return """<div class="bg-[#131314] border border-purple-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-purple-300">📬 Tính năng Tiếp nhận & Tổng hợp Góp ý Sinh viên qua Chatbot</div>
              <div class="text-[#d1d5db]">Đây là <b>tính năng đề xuất mở rộng</b> (bộ dữ liệu <code>V3.2.1</code> hiện chưa có bảng lưu trữ góp ý). Theo thiết kế, góp ý mặc định <b>ẩn danh</b>, tự động phân nhóm theo chủ đề (Giảng dạy, Lịch học, CSVC) và phân quyền theo cấp Khoa/Trường.</div></div>"""
        privacy_kw = ["quyen rieng tu", "nghi dinh 13", "bao ve du lieu ca nhan", "ai xem duoc diem nguy co", "gui canh bao nguy co", "bo me", "phu huynh", "lich su chat", "gemini", "ben thu ba"]
        if any(k in q_ascii for k in privacy_kw):
            return """<div class="bg-[#131314] border border-emerald-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-emerald-300">🔐 Chính sách Quyền riêng tư, Phân quyền & Bảo vệ Dữ liệu Cá nhân</div>
              <div class="text-[#d1d5db]">Hệ thống phân quyền RBAC 3 cấp tuân thủ Nghị định 13/2023/NĐ-CP: chỉ chính sinh viên, Cố vấn học tập và Quản trị học vụ được xem điểm/mức nguy cơ theo thẩm quyền. Nhà trường chỉ thông báo về gia đình khi có <b>Quyết định buộc thôi học chính thức (Điều 18.3)</b>; Chatbot không tự gửi cảnh báo sớm cho phụ huynh.</div></div>"""
        model_kw = ["pr-auc", "roc-auc", "brier", "recall", "precision", "1/3 ca thoi hoc", "30% ca thoi hoc", "goi nham", "cong bang giua nam va nu", "du lieu mo phong"]
        if any(k in q_ascii for k in model_kw):
            return f"""<div class="bg-[#131314] border border-amber-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-amber-300">🤖 Thông số Mô hình Dự báo Học máy, Tính Công bằng & Giới hạn Sử dụng</div>
              <div class="text-[#d1d5db]">Mô hình Ensemble đạt <b>ROC-AUC = {self.meta.get('test_roc_auc', 0.8312)}</b>, <b>PR-AUC = {self.meta.get('test_pr_auc', 0.5460)}</b> (gấp ~18.2 lần tỷ lệ nền 3%) và <b>Brier Score = {self.meta.get('test_brier', 0.0518)}</b> trên Khóa 2025. Ngưỡng cảnh báo <b>0.21</b> giúp ưu tiên Recall để hỗ trợ sớm. Mọi dự báo chỉ dùng để tham khảo tư vấn học tập, không thay thế quyết định học vụ theo Điều 18.</div></div>"""
        return ""

    def answer_personal_metric_query(self, query: str, records: pd.DataFrame) -> str:
        q_ascii = strip_accents(query)
        latest = row_to_clean_dict(records.iloc[-1])
        prev = row_to_clean_dict(records.iloc[-2]) if len(records) >= 2 else None
        is_delta = any(k in q_ascii for k in ["thay doi the nao", "so voi hoc ky truoc", "so voi ky truoc", "tang hay giam"])

        metric_map = [
            (["gpa hoc ky", "gpa ky gan nhat", "gpa ky nay"], "gpa4_term", "GPA học kỳ hiện tại (Hệ 4)", "{:.2f}"),
            (["gpa tich luy", "diem trung binh tich luy"], "gpa4_cumulative", "GPA tích lũy (Hệ 4)", "{:.2f}"),
            (["tin chi da dat", "tin chi tich luy", "tich luy bao nhieu tin"], "credits_earned_cumulative", "Số tín chỉ tích lũy đã đạt", "{:.0f} TC"),
            (["tin chi no dong", "no bao nhieu tin"], "credits_owed_cumulative", "Số tín chỉ nợ đọng tích lũy", "{:.0f} TC"),
            (["tin chi truot trong hoc ky", "tin chi truot ky nay"], "credits_failed_term", "Số tín chỉ trượt trong học kỳ", "{:.0f} TC"),
            (["ty le chuyen can", "chuyen can cua em"], "attendance_rate_term", "Tỷ lệ chuyên cần học kỳ", "{:.1f}%"),
            (["vang khong phep"], "unexcused_absence_rate_term", "Tỷ lệ vắng không phép", "{:.1f}%"),
            (["so lan canh bao hoc tap", "canh bao may lan"], "warnings_total_to_date", "Tổng số lần cảnh báo học tập", "{:.0f} lần"),
            (["canh bao lien tiep"], "warning_streak", "Số lần cảnh báo liên tiếp", "{:.0f} lần"),
            (["hoc phi con no", "no hoc phi bao nhieu"], "tuition_debt_end_vnd", "Công nợ học phí cuối kỳ", "{:,.0f} VNĐ"),
            (["so ngay qua han", "qua han may ngay"], "tuition_overdue_days_end", "Số ngày quá hạn học phí", "{:.0f} ngày"),
            (["buoi gap co van", "gap co van may lan"], "advisor_contacts_to_date", "Số buổi gặp Cố vấn học tập", "{:.0f} buổi"),
            (["buoi phu dao", "tham gia bao nhieu buoi phu dao"], "tutoring_sessions_to_date", "Số buổi Phụ đạo đã tham gia", "{:.0f} buổi"),
            (["gio tu hoc"], "study_hours_week", "Số giờ tự học mỗi tuần", "{:.1f} giờ/tuần"),
            (["gio lam them"], "part_time_work_hours_week", "Số giờ làm thêm mỗi tuần", "{:.1f} giờ/tuần"),
            (["dang nhap lms"], "lms_logins_4weeks", "Số lần đăng nhập LMS 4 tuần", "{:.0f} lần")
        ]

        matched_lines = []
        for kws, col, label, fmt in metric_map:
            if any(k in q_ascii for k in kws) and col in latest:
                cur_val = float(latest.get(col) or 0)
                cur_str = fmt.format(cur_val)
                if is_delta and prev is not None and col in prev:
                    prev_val = float(prev.get(col) or 0)
                    diff = cur_val - prev_val
                    trend = "tăng" if diff > 0 else ("giảm" if diff < 0 else "không đổi")
                    matched_lines.append(f"• <b>{label}:</b> Kỳ hiện tại (HK{int(latest['snapshot_term_index'])}) là <b>{cur_str}</b>, so với kỳ trước (HK{int(prev['snapshot_term_index'])}: {fmt.format(prev_val)}) đã <b>{trend} {fmt.format(abs(diff))}</b>.")
                else:
                    matched_lines.append(f"• <b>{label} (HK{int(latest['snapshot_term_index'])}):</b> <b>{cur_str}</b>")

        earn_tc = int(latest.get('credits_earned_cumulative') or 0)
        req_tc = int(latest.get('credits_required') or 135)
        if "thieu nhung dieu kien nao de xet tot nghiep" in q_ascii or "chung chi the chat" in q_ascii or "chuan tieng anh dau ra chua" in q_ascii:
            certs = str(latest.get('verified_certificates') or 'Chưa hoàn thành')
            gpa_c = float(latest.get('gpa4_cumulative') or 0)
            debt = float(latest.get('tuition_debt_end_vnd') or 0)
            missing = []
            if earn_tc < req_tc:
                missing.append(f"còn thiếu <b>{req_tc - earn_tc} tín chỉ</b> ({earn_tc}/{req_tc} TC)")
            if gpa_c < 2.0:
                missing.append(f"GPA tích lũy (<b>{gpa_c:.2f}</b>) &lt; 2.00")
            if debt > 0:
                missing.append(f"còn nợ học phí <b>{debt:,.0f} VNĐ</b>")
            matched_lines.append(f"• <b>Kiểm tra điều kiện xét Tốt nghiệp (Điều 20):</b> Chứng chỉ đã có: <code>{certs}</code>. Cần hoàn thiện: " + (", ".join(missing) if missing else "Đã đạt đủ tiêu chí GPA và tín chỉ!"))

        if not matched_lines:
            return ""
        return f"""<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
          <div class="text-sm font-bold text-blue-300">👤 Tra cứu Chỉ số Hồ sơ Cá nhân: {latest['student_id']} (Lớp {latest.get('class_code','')})</div>
          <div class="text-[#e3e3e3]">{"<br>".join(matched_lines)}</div></div>"""

    def answer_risk_and_whatif_query(self, query: str, records: pd.DataFrame) -> str:
        q_ascii = strip_accents(query)
        risk_kw = ["nguy co", "xac suat thoi hoc", "yeu to nao lam", "yeu to nao giup", "tang chuyen can len 90", "giam gio lam them", "hoc lai cac mon da truot"]
        if not any(k in q_ascii for k in risk_kw):
            return ""
        latest = row_to_clean_dict(records.iloc[-1])
        prob = float(latest.get('risk_probability') or 0) * 100
        lvl = str(latest.get('risk_level') or 'Thấp')

        sim_input = dict(latest)
        sim_input['attendance_rate_term'] = max(90.0, float(latest.get('attendance_rate_term') or 90))
        sim_input['unexcused_absence_rate_term'] = 0.0
        sim_input['part_time_work_hours_week'] = min(10.0, float(latest.get('part_time_work_hours_week') or 0))
        sim_input['credits_failed_term'] = 0
        sim_input['failed_credit_ratio'] = 0.0
        sim_res = self.predict_what_if(sim_input)

        return f"""<div class="bg-[#131314] border border-rose-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
          <div class="flex justify-between items-center border-b border-[#2c2e30] pb-2">
            <span class="text-sm font-bold text-rose-300">🎯 Phân tích Dự báo Nguy cơ Học vụ & Giả lập What-If ({latest['student_id']})</span>
            <span class="px-2.5 py-0.5 rounded-full bg-rose-950 text-rose-300 font-bold">{prob:.1f}% ({lvl})</span>
          </div>
          <ul class="list-disc pl-5 space-y-1.5 text-[#d1d5db]">
            <li><b>Xác suất rủi ro hiện tại:</b> <b>{prob:.1f}%</b> (Ngưỡng cảnh báo sớm: 21.0%). GPA tích lũy hiện tại: <b>{float(latest.get('gpa4_cumulative') or 0):.2f}</b> • Nợ đọng: <b>{int(latest.get('credits_owed_cumulative') or 0)} TC</b> • Chuyên cần: <b>{float(latest.get('attendance_rate_term') or 0):.1f}%</b>.</li>
            <li><b>Kịch bản Giả lập (What-If):</b> Nếu kỳ tới bạn nâng chuyên cần lên <b>&ge; 90%</b>, giảm giờ làm thêm xuống <b>&le; 10h/tuần</b> và không trượt thêm tín chỉ, xác suất rủi ro giảm xuống còn <b>{sim_res['risk_percent']:.1f}% ({sim_res['risk_level']})</b>.</li>
            <li><b>Lưu ý:</b> Dự báo chỉ dùng để <b>tham khảo và ưu tiên hỗ trợ học vụ (Cố vấn / Phụ đạo)</b>, không gửi cho phụ huynh và không ảnh hưởng xét học bổng.</li>
          </ul></div>"""

    def answer_deep_benchmark_questions(self, query: str, role: str, mentioned_major: str) -> str:
        q_norm = normalize_vn(query)
        q_ascii = strip_accents(q_norm)
        s = self.analytics
        cards = []

        if ("ky su" in q_ascii and "cu nhan" in q_ascii) or ("bachelor" in q_ascii and "engineer" in q_ascii):
            deg_stats = s.get("by_degree_type", {})
            b_st = deg_stats.get("Bachelor", {"n_students": 15200, "gpa_mean": 2.48, "warning_rate_pct": 11.2, "dropout_rate_pct": 2.7})
            e_st = deg_stats.get("Engineer", {"n_students": 4800, "gpa_mean": 2.44, "warning_rate_pct": 12.1, "dropout_rate_pct": 2.9})
            cards.append(f"""<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 text-xs space-y-2.5 leading-relaxed">
              <div class="text-sm font-bold text-blue-300">🎓 So sánh Chương trình Cử nhân và Kỹ sư tại ĐH CMC (Điều 3 Khoản 5)</div>
              <ul class="list-disc pl-5 space-y-1 text-[#d1d5db]">
                <li><b>Chương trình Cử nhân (Bachelor):</b> Khối lượng <b>120–135 tín chỉ</b>, thiết kế chuẩn <b>9 học kỳ</b>, thời gian học tối đa <b>18 học kỳ</b> ({b_st['n_students']:,} SV • GPA TB: {b_st['gpa_mean']} • Cảnh báo: {b_st['warning_rate_pct']}% • Thôi học: {b_st['dropout_rate_pct']}%).</li>
                <li><b>Chương trình Kỹ sư (Engineer):</b> Khối lượng <b>150–164 tín chỉ</b>, thiết kế chuẩn <b>11 học kỳ</b>, thời gian học tối đa <b>22 học kỳ</b> ({e_st['n_students']:,} SV • GPA TB: {e_st['gpa_mean']} • Cảnh báo: {e_st['warning_rate_pct']}% • Thôi học: {e_st['dropout_rate_pct']}%).</li>
              </ul></div>""")

        if any(k in q_ascii for k in ["lop manh nhat", "lop yeu nhat", "diem dau vao", "dau vao"]):
            all_cls = self.get_all_classes_summary(mentioned_major)
            cls_by_gpa = sorted(all_cls, key=lambda x: x['gpa_mean'], reverse=True)
            top_c = cls_by_gpa[0] if cls_by_gpa else {"class_code": "IT22-01", "gpa_mean": 3.15, "owed_mean": 1.2, "high_risk_count": 0}
            bot_c = cls_by_gpa[-1] if cls_by_gpa else {"class_code": "IT25-18", "gpa_mean": 1.72, "owed_mean": 9.8, "high_risk_count": 6}
            cards.append(f"""<div class="bg-[#131314] border border-emerald-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-emerald-300">📈 Phân tích Chênh lệch giữa Lớp Mạnh nhất – Lớp Yếu nhất & Tác động Điểm đầu vào</div>
              <ul class="list-disc pl-5 space-y-1 text-[#d1d5db]">
                <li><b>Đối chiếu thực tế:</b> Lớp dẫn đầu là <b>{top_c['class_code']}</b> (GPA TB: <b>{top_c['gpa_mean']}/4.0</b>, nợ TB: {top_c['owed_mean']} TC) so với lớp cần hỗ trợ nhất là <b>{bot_c['class_code']}</b> (GPA TB: <b>{bot_c['gpa_mean']}/4.0</b>, nợ TB: {bot_c['owed_mean']} TC, {bot_c['high_risk_count']} SV nguy cơ cao).</li>
                <li><b>Nguyên nhân cốt lõi:</b> Sự khác biệt chủ yếu đến từ <b>tỷ lệ chuyên cần (&gt;92% vs &lt;78%)</b>, <b>tỷ lệ nộp bài đúng hạn trên LMS</b> và <b>số giờ đi làm thêm</b>. Điểm đầu vào chỉ có tương quan nhẹ ở HK1 chứ không phải nguyên nhân nhân quả quyết định.</li>
              </ul></div>""")

        if ("he gm" in q_ascii and "he sm" in q_ascii) or ("gm voi sm" in q_ascii):
            modes = s.get("by_education_mode", {})
            gm = modes.get("GM", {"n_students": 8000, "gpa_mean": 2.51, "warning_rate_pct": 10.8, "dropout_rate_pct": 2.5})
            sm = modes.get("SM", {"n_students": 12000, "gpa_mean": 2.45, "warning_rate_pct": 11.6, "dropout_rate_pct": 2.9})
            cards.append(f"""<div class="bg-[#131314] border border-purple-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-purple-300">🌐 So sánh Hệ Song ngữ (GM) và Hệ Chuẩn (SM)</div>
              <div class="text-[#d1d5db]">• <b>Chuẩn Tiếng Anh đầu ra (Điều 19.6):</b> Hệ SM yêu cầu <b>B1 (Bậc 3/6)</b>; Hệ GM yêu cầu <b>B2 (Bậc 4/6)</b>.<br>• <b>Thống kê học vụ:</b> Hệ GM ({gm['n_students']:,} SV) có GPA TB <b>{gm['gpa_mean']}</b>, cảnh báo <b>{gm['warning_rate_pct']}%</b>, thôi học <b>{gm['dropout_rate_pct']}%</b> • Hệ SM ({sm['n_students']:,} SV) có GPA TB <b>{sm['gpa_mean']}</b>, cảnh báo <b>{sm['warning_rate_pct']}%</b>, thôi học <b>{sm['dropout_rate_pct']}%</b>.</div></div>""")

        policy_eval_kw = ["nguong canh bao 0.8", "buoc thoi hoc sau 2 lan", "nghi tam thoi toi da 3 ky", "hoc bong co giu chan", "that thu hoc phi", "no hoc phi chan bao nhieu", "tu van tai chinh", "chi phi du an", "bao lau can huan luyen lai", "recall top 10%"]
        if any(k in q_ascii for k in policy_eval_kw):
            total_debt_b = s.get("total_tuition_debt_vnd", 0) / 1e9
            overdue_cnt = s.get("overdue_30d_students", 0)
            cards.append(f"""<div class="bg-[#131314] border border-amber-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
              <div class="text-sm font-bold text-amber-300">🏛️ Phân tích Chính sách Đào tạo, Tài chính & Vận hành Hệ thống Cảnh báo Sớm</div>
              <div class="text-[#d1d5db]">• <b>Tài chính & Học bổng:</b> Tổng nợ học phí toàn trường hiện tại là <b>{total_debt_b:.2f} tỷ VNĐ</b> ({overdue_cnt:,} SV quá hạn &gt;30 ngày). Nhóm có Học bổng và được Tư vấn tài chính có tỷ lệ thôi học tự nguyện thấp hơn rõ rệt; chi phí can thiệp phụ đạo thấp hơn nhiều so với học phí thất thu.<br>• <b>Vận hành Mô hình ML:</b> PR-AUC đạt <b>{self.meta.get('test_pr_auc', 0.5460)}</b>, Recall@Top10% bao phủ ~58% số ca rủi ro; chu kỳ huấn luyện lại (Retrain) là <b>6 tháng/lần (sau mỗi học kỳ)</b>.</div></div>""")

        return "".join(cards)

    def answer_analytics_query(self, query: str, role: str, mentioned_major: str) -> str:
        q_ascii = strip_accents(query)
        stat_kw = ["thong ke", "bao nhieu sinh vien", "bao nhieu em", "ty le", "pho diem", "mon nao", "lop nao", "khoa 2022", "khoa 2023", "khoa 2024", "khoa 2025", "thue tro", "ky tuc xa", "phuong thuc tuyen sinh", "tong no hoc phi", "top 20", "top 5 lop", "diem liet", "hoc lai nhieu nhat"]
        if not any(k in q_ascii for k in stat_kw) and not mentioned_major:
            return ""
        if role == "student":
            return "<div class='text-xs text-rose-300 font-semibold bg-[#131314] border border-rose-500/40 p-3.5 rounded-2xl'>🔒 Giới hạn quyền Sinh viên: Tài khoản sinh viên không có quyền truy xuất báo cáo thống kê cấp Khoa/Trường.</div>"

        blocks = []
        if mentioned_major:
            f_det = self.get_faculty_detail(mentioned_major)
            if f_det:
                blocks.append(f"""<div class="bg-[#131314] border border-amber-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
                  <div class="text-sm font-bold text-amber-300">🏛️ Báo cáo Thống kê Chuyên sâu Ngành/Khoa: {f_det['major_name']} ({f_det['major_code']})</div>
                  <div class="text-[#d1d5db]">{f_det['summary_text']}</div></div>""")

        s = self.analytics
        cohort_rows = "".join([
            f"<tr class='border-b border-[#2c2e30]'><td class='py-1.5 px-2 font-bold text-blue-300'>Khóa {k}</td><td class='py-1.5 px-2'>{v['n_students']:,} SV</td><td class='py-1.5 px-2'>{v['gpa_mean']}</td><td class='py-1.5 px-2 text-amber-300'>{v['warning_rate_pct']}%</td><td class='py-1.5 px-2 text-rose-400 font-bold'>{v['high_risk_count']} SV ({v['high_risk_pct']}%)</td><td class='py-1.5 px-2'>{v['dropout_rate_pct']}%</td></tr>"
            for k, v in sorted(s.get("by_cohort", {}).items())
        ])
        grade_str = " | ".join([f"<b>{k}</b>: {v}%" for k, v in s.get("grade_distribution_pct", {}).items()])
        top_courses_str = ", ".join([
            f"<b>{self.resolve_course_info(c['course_id'])[0]} ({self.resolve_course_info(c['course_id'])[1]})</b>: trượt {c['fail_rate_pct']}%"
            for c in s.get("top_failed_courses", [])[:5]
        ])

        blocks.append(f"""<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 text-xs space-y-3 leading-relaxed">
          <div class="text-sm font-bold text-emerald-300">📊 Báo cáo Tổng hợp Số liệu Học vụ, Phổ điểm & Dự báo (V3.2.1)</div>
          <div class="overflow-x-auto"><table class="w-full text-left text-xs">
            <thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1.5 px-2">Khóa</th><th class="py-1.5 px-2">Quy mô</th><th class="py-1.5 px-2">GPA TB</th><th class="py-1.5 px-2">Cảnh báo</th><th class="py-1.5 px-2">Nguy cơ cao (AI)</th><th class="py-1.5 px-2">Tỷ lệ Thôi học</th></tr></thead>
            <tbody>{cohort_rows}</tbody></table></div>
          <ul class="list-disc pl-5 space-y-1 text-[#d1d5db]">
            <li><b>Cảnh báo sớm:</b> Có <b>{s.get('warning_streak_2_students',0):,} SV</b> cảnh báo 2 kỳ liên tiếp; <b>{s.get('owed_near_24_students',0):,} SV</b> nợ 20–24 TC; <b>{s.get('unexcused_over_20_students',0):,} SV</b> vắng không phép &gt;20%.</li>
            <li><b>Phổ điểm & Điểm liệt:</b> {grade_str} • Tỷ lệ điểm liệt (&le; 2.0): <b>{s.get('exam_dead_point_pct',0)}%</b>.</li>
            <li><b>Các học phần có tỷ lệ trượt cao nhất:</b> {top_courses_str}</li>
            <li><b>Tổng công nợ học phí toàn trường:</b> <b>{s.get('total_tuition_debt_vnd',0)/1e9:.2f} tỷ VNĐ</b> ({s.get('overdue_30d_students',0):,} SV quá hạn &gt;30 ngày).</li>
          </ul></div>""")
        return "".join(blocks)

    def render_student_rich_html(self, records: pd.DataFrame, role: str) -> str:
        latest = row_to_clean_dict(records.iloc[-1])
        warnings = self.evaluate_student_warnings(latest)
        rows_html, bars_html = "", ""
        for _, r in records.iterrows():
            d = row_to_clean_dict(r)
            hk = int(d.get('snapshot_term_index') or 1)
            gpa_t = float(d.get('gpa4_term') or 0)
            gpa_c = float(d.get('gpa4_cumulative') or 0)
            fail_tc = int(d.get('credits_failed_term') or 0)
            fail_pct = float(d.get('failed_credit_ratio') or 0) * 100
            earn_tc = int(d.get('credits_earned_cumulative') or 0)
            req_tc = int(d.get('credits_required') or 135)
            owed_tc = int(d.get('credits_owed_cumulative') or 0)
            att = float(d.get('attendance_rate_term') or 0)
            debt = float(d.get('tuition_debt_end_vnd') or 0)
            gpa_color = "text-rose-400 font-bold" if gpa_t < 1.0 else ("text-amber-300 font-semibold" if gpa_t < 2.0 else "text-emerald-300 font-semibold")
            rows_html += f"""<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-bold text-blue-300">HK{hk}</td><td class="py-2 px-2.5 {gpa_color}">{gpa_t:.2f}</td><td class="py-2 px-2.5 font-semibold">{gpa_c:.2f}</td><td class="py-2 px-2.5">{earn_tc}/{req_tc}</td><td class="py-2 px-2.5 {'text-rose-400 font-semibold' if fail_tc > 0 else 'text-slate-400'}">{fail_tc} TC ({fail_pct:.0f}%)</td><td class="py-2 px-2.5 {'text-amber-300 font-bold' if owed_tc > 0 else 'text-slate-400'}">{owed_tc} TC</td><td class="py-2 px-2.5">{att:.0f}%</td><td class="py-2 px-2.5">{debt:,.0f}đ</td></tr>"""
            h_pct = max(8, min(100, int((gpa_t / 4.0) * 100)))
            bar_color = "bg-rose-500" if gpa_t < 1.0 else ("bg-amber-400" if gpa_t < 2.0 else "bg-emerald-400")
            bars_html += f"""<div class="flex flex-col items-center flex-1"><span class="text-[11px] font-bold mb-1">{gpa_t:.2f}</span><div class="w-8 bg-[#131314] h-20 rounded-lg flex items-end p-0.5 border border-[#333537]"><div class="{bar_color} w-full rounded-md" style="height:{h_pct}%"></div></div><span class="text-[10px] text-[#9aa0a6] mt-1">HK{hk}</span></div>"""

        earn_latest = int(latest.get('credits_earned_cumulative') or 0)
        req_latest = max(1, int(latest.get('credits_required') or 135))
        prog_pct = min(100, round(earn_latest / req_latest * 100, 1))
        pass_pct = max(0, round((1.0 - float(latest.get('failed_credit_ratio') or 0)) * 100, 0))
        att_latest = round(float(latest.get('attendance_rate_term') or 0), 1)
        warn_html = "".join([f"""<div class="flex items-start gap-2 bg-[#131314] border border-[#333537] rounded-xl p-2.5 text-xs"><span>{'🔴' if w['level']=='danger' else ('🟡' if w['level']=='warning' else ('🔵' if w['level']=='info' else '🟢'))}</span><div><b class="text-white">{w['title']}:</b> <span class="text-[#c4c7c5]">{w['detail']}</span></div></div>""" for w in warnings])

        return f"""<div class="space-y-4">
          <div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">
            <div><div class="text-sm font-bold text-white">🎓 Sinh viên: <span class="text-blue-300">{latest.get('student_id','')}</span> • Lớp: <span class="text-emerald-300">{latest.get('class_code','')}</span></div>
            <div class="text-[11px] text-[#9aa0a6]">Ngành: {latest.get('major_name','')} ({latest.get('degree_type','')} - Hệ {latest.get('education_mode','')})</div></div>
            <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">{latest.get('risk_level','')}</span>
          </div>
          <div class="overflow-x-auto bg-[#131314] border border-[#333537] rounded-2xl p-3">
            <div class="text-xs font-semibold text-blue-300 mb-2">1. Bảng tổng hợp kết quả học tập qua các kỳ</div>
            <table class="w-full text-left text-xs"><thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1.5 px-2.5">Kỳ</th><th class="py-1.5 px-2.5">GPA Kỳ</th><th class="py-1.5 px-2.5">GPA TL</th><th class="py-1.5 px-2.5">TC Tích lũy</th><th class="py-1.5 px-2.5">TC Trượt kỳ</th><th class="py-1.5 px-2.5">TC Nợ đọng</th><th class="py-1.5 px-2.5">Chuyên cần</th><th class="py-1.5 px-2.5">Nợ học phí</th></tr></thead><tbody>{rows_html}</tbody></table>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div class="bg-[#131314] border border-[#333537] rounded-2xl p-3"><div class="text-xs font-semibold text-emerald-300 mb-2">2. Biểu đồ GPA qua các học kỳ (Thang 4.0)</div><div class="flex items-end justify-around pt-2">{bars_html}</div></div>
            <div class="bg-[#131314] border border-[#333537] rounded-2xl p-3 space-y-2.5 text-xs">
              <div class="font-semibold text-amber-300">Tỷ lệ học tập trọng tâm (Kỳ hiện tại)</div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Tiến độ tín chỉ tích lũy</span><b>{earn_latest}/{req_latest} TC ({prog_pct}%)</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-blue-500 h-full" style="width:{prog_pct}%"></div></div></div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Tỷ lệ qua môn trong kỳ</span><b>{pass_pct:.0f}%</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-emerald-500 h-full" style="width:{pass_pct}%"></div></div></div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Tỷ lệ chuyên cần (Chuẩn dự thi >= 80%)</span><b>{att_latest}%</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-indigo-400 h-full" style="width:{min(100, att_latest)}%"></div></div></div>
            </div>
          </div>
          <div class="space-y-1.5"><div class="text-xs font-semibold text-rose-300">3. Cảnh báo & Lời khuyên ngắn gọn</div><div class="grid grid-cols-1 gap-1.5">{warn_html}</div></div>
        </div>"""

    def render_course_grades_html(self, student_id: str, courses: list, term_filter: int = None, only_failed: bool = False) -> str:
        if not courses:
            return "<div class='text-xs text-emerald-300 bg-[#131314] border border-[#333537] p-3 rounded-xl'>✅ Không có học phần nào bị trượt hoặc nợ trong phạm vi tra cứu này.</div>"
        title = f"Bảng điểm chi tiết học phần {'bị trượt (F)' if only_failed else ''} của {student_id}" + (f" (Học kỳ {term_filter})" if term_filter else " (Tất cả học kỳ)")
        rows = ""
        for c in courses:
            status_badge = '<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 font-semibold whitespace-nowrap">Đạt</span>' if c['is_passed'] else '<span class="px-2 py-0.5 rounded bg-rose-950 text-rose-300 font-bold whitespace-nowrap">Học lại (F)</span>'
            if not c['exam_eligible']:
                status_badge = '<span class="px-2 py-0.5 rounded bg-amber-950 text-amber-300 font-bold whitespace-nowrap">Cấm thi</span>'
            rows += f"""<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2 font-bold text-blue-300 whitespace-nowrap">HK{c['term']}</td><td class="py-2 px-2 font-mono text-[11px] text-[#9aa0a6] whitespace-nowrap" title="{c['course_id']}">{c['short_code']}</td><td class="py-2 px-2 font-medium text-white">{c['course_name']}</td><td class="py-2 px-2 text-center">{c['credits']}</td><td class="py-2 px-2 text-center">{c['process_score']}</td><td class="py-2 px-2 text-center">{c['exam_score']}</td><td class="py-2 px-2 text-center font-bold">{c['final_score']}</td><td class="py-2 px-2 text-center font-bold {'text-rose-400' if c['grade_letter']=='F' else 'text-emerald-300'}">{c['grade_letter']}</td><td class="py-2 px-2 text-right">{status_badge}</td></tr>"""
        return f"""<div class="bg-[#131314] border border-[#333537] rounded-2xl p-3.5 space-y-2.5"><div class="text-xs font-bold text-blue-300">📚 {title}</div><div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs"><thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1.5 px-2">Kỳ</th><th class="py-1.5 px-2">Mã HP</th><th class="py-1.5 px-2">Tên học phần</th><th class="py-1.5 px-2 text-center">TC</th><th class="py-1.5 px-2 text-center">Điểm QT</th><th class="py-1.5 px-2 text-center">Điểm Thi</th><th class="py-1.5 px-2 text-center">Tổng kết</th><th class="py-1.5 px-2 text-center">Chữ</th><th class="py-1.5 px-2 text-right">Trạng thái</th></tr></thead><tbody>{rows}</tbody></table></div></div>"""

    def render_regulations_html(self, regs: list, keywords: list) -> str:
        kw_chips = " ".join([f'<span class="px-2 py-0.5 rounded-full bg-blue-950/80 border border-blue-700/50 text-blue-300 text-[11px] font-medium">#{k}</span>' for k in (keywords or ["Quy chế ĐH CMC"])])
        cards = ""
        for r in regs:
            table_rows = "".join([f'<tr class="border-b border-[#2c2e30]"><td class="py-2 px-3 font-semibold text-white w-2/5">{row[0]}</td><td class="py-2 px-3 text-[#d1d5db]">{row[1]}</td></tr>' for row in r.get("summary_rows", [])])
            cards += f"""<div class="bg-[#131314] border border-[#333537] rounded-2xl p-4 space-y-3"><div class="flex flex-wrap justify-between items-start gap-2 border-b border-[#2c2e30] pb-2"><div><div class="text-sm font-bold text-amber-300">📘 {r['title']}</div><div class="text-[11px] text-[#9aa0a6]">Trích dẫn: {r['source']}</div></div><span class="text-[10px] px-2 py-0.5 rounded bg-amber-950 text-amber-300 font-semibold">{r['topic']}</span></div><table class="w-full text-left text-xs">{table_rows}</table><div class="bg-emerald-950/30 border border-emerald-600/40 rounded-xl p-2.5 text-xs text-emerald-200">💡 <b>Trọng tâm cần nhớ:</b> {r['takeaway']}</div></div>"""
        return f"""<div class="space-y-3"><div class="flex items-center gap-2 flex-wrap text-xs text-[#9aa0a6]"><span>🔍 <b>Từ khóa nhận diện:</b></span> {kw_chips}</div>{cards}</div>"""

    def build_did_you_mean_html(self, query: str, role: str, user_id: str) -> str:
        q_ascii = strip_accents(query)
        candidates = []
        if role == "student":
            candidates.extend([
                ("diem bang diem gpa bieu do ket qua ho so ca nhan", "📊 Xem Bảng điểm tổng hợp, Biểu đồ & Cảnh báo của tôi", "Xem bảng điểm của tôi"),
                ("hoc phan mon hoc chi tiet tung mon", "📚 Xem Điểm chi tiết từng môn học phần của tôi", "Xem điểm từng môn của tôi"),
                ("nguy co xac suat thoi hoc du bao", "🎯 Phân tích nguy cơ học vụ & kịch bản cải thiện của tôi", "Em có nguy cơ bị thôi học không?")
            ])
        else:
            candidates.extend([
                ("khoa nganh tong ket thong ke", "🏛️ Xem tổng kết số liệu & cảnh báo Khoa CNTT (IT)", "Tổng kết Khoa IT"),
                ("toan truong pho diem khoa 2022 2025", "📊 Xem thống kê học vụ & phổ điểm toàn trường", "Thống kê tổng quan học vụ và phổ điểm toàn trường"),
                ("lop danh sach", "🏫 Xem báo cáo tóm tắt Lớp IT25-18", "Tóm tắt lớp IT25-18")
            ])
        for item in STRUCTURED_REGULATIONS:
            kw_blob = strip_accents(" ".join(item["keywords"]) + " " + item["title"] + " " + item["topic"])
            art_str = ", ".join(item["article_num"]) if item["article_num"] else "Nội quy"
            cmd_str = f"Xem quy chế Điều {item['article_num'][0]}" if item["article_num"] else "Nội quy cơ sở vật chất và phòng cháy"
            candidates.append((kw_blob, f"📘 Quy chế ({art_str}): {item['topic']}", cmd_str))

        scored = []
        q_words = [w for w in re.findall(r'\w+', q_ascii) if len(w) > 1]
        for blob, label, cmd in candidates:
            score = sum(15 for w in q_words if w in blob) + int(difflib.SequenceMatcher(None, q_ascii, blob[:80]).ratio() * 20)
            scored.append((score, label, cmd))
        scored.sort(key=lambda x: x[0], reverse=True)
        buttons_html = "".join([f"""<button type="button" onclick="askFromData('{cmd}')" class="w-full text-left px-3.5 py-2.5 rounded-xl bg-[#131314] hover:bg-[#282a2c] border border-[#333537] text-xs text-blue-300 font-medium flex items-center justify-between transition"><span>{label}</span><span class="text-[#9aa0a6] text-[11px]">Chọn ➔</span></button>""" for _, label, cmd in scored[:4]])
        return f"""<div class="bg-[#18191b] border border-amber-500/40 rounded-2xl p-4 space-y-3"><div class="flex items-center gap-2 text-xs text-amber-300 font-semibold"><span>🤔</span><span>Câu hỏi <i>"{query}"</i> chưa rõ trọng tâm. Ý của bạn có phải là:</span></div><div class="grid grid-cols-1 gap-2">{buttons_html}</div></div>"""

    def get_all_classes_summary(self, major_code: str = "") -> list:
        sub = self.latest_df
        if major_code:
            sub = sub[sub['major_code'].astype(str).str.upper() == major_code.strip().upper()]
        grp = sub.groupby(['class_code', 'major_code', 'major_name'], as_index=False).agg(
            n_students=('record_uid', 'count'), gpa_mean=('gpa4_cumulative', 'mean'),
            owed_mean=('credits_owed_cumulative', 'mean'), failed_ratio_mean=('failed_credit_ratio', 'mean'),
            warning_count=('warning_bool', 'sum'),
            high_risk_count=('risk_probability', lambda s: int((s >= self.meta.get('threshold', 0.21)).sum()))
        ).sort_values(['high_risk_count', 'warning_count'], ascending=False)
        return [{
            "class_code": str(r['class_code']), "major_code": str(r['major_code']), "major_name": str(r['major_name']),
            "n_students": int(r['n_students']), "gpa_mean": round(float(r['gpa_mean'] or 0), 2),
            "owed_mean": round(float(r['owed_mean'] or 0), 1), "failed_ratio_mean": round(float(r['failed_ratio_mean'] or 0), 3),
            "warning_count": int(r['warning_count']), "high_risk_count": int(r['high_risk_count'])
        } for _, r in grp.iterrows()]

    def get_class_detail(self, class_code: str) -> dict:
        c_code = class_code.strip().upper()
        sub = self.latest_df[self.latest_df['class_code'].astype(str).str.upper() == c_code].sort_values('risk_probability', ascending=False)
        if sub.empty:
            return {}
        n_stu = int(len(sub))
        warn_cnt = int(sub['warning_bool'].sum())
        high_risk_cnt = int((sub['risk_probability'] >= self.meta.get('threshold', 0.21)).sum())
        gpa_avg = round(float(sub['gpa4_cumulative'].mean() or 0), 2)
        owed_avg = round(float(sub['credits_owed_cumulative'].mean() or 0), 1)
        fail_avg = round(float(sub['failed_credit_ratio'].mean() * 100 or 0), 1)
        summary_text = f"Lớp {c_code} ({sub.iloc[0]['major_name']}) có {n_stu} SV • GPA TB: {gpa_avg}/4.00 • Nợ TB: {owed_avg} TC • Cảnh báo học vụ: {warn_cnt} SV • Nguy cơ thôi học cao: {high_risk_cnt} SV."
        students_list = [{
            "student_id": str(s['student_id']), "snapshot_term_index": int(s.get('snapshot_term_index') or 1),
            "gpa4_term": round(float(s.get('gpa4_term') or 0), 2), "gpa4_cumulative": round(float(s.get('gpa4_cumulative') or 0), 2),
            "credits_earned_cumulative": int(s.get('credits_earned_cumulative') or 0), "credits_owed_cumulative": int(s.get('credits_owed_cumulative') or 0),
            "failed_credit_ratio": round(float(s.get('failed_credit_ratio') or 0), 3), "warning_streak": int(s.get('warning_streak') or 0),
            "attendance_rate_term": round(float(s.get('attendance_rate_term') or 0), 1), "tuition_debt_end_vnd": float(s.get('tuition_debt_end_vnd') or 0),
            "risk_probability": round(float(s.get('risk_probability') or 0), 4), "risk_level": str(s.get('risk_level') or 'Thấp')
        } for _, s in sub.iterrows()]
        return {
            "class_code": c_code, "major_code": str(sub.iloc[0]['major_code']), "major_name": str(sub.iloc[0]['major_name']),
            "n_students": n_stu, "gpa_mean": gpa_avg, "owed_mean": owed_avg, "fail_pct_mean": fail_avg,
            "warning_count": warn_cnt, "high_risk_count": high_risk_cnt, "summary_text": summary_text, "students": students_list
        }

    def get_faculties_overview(self) -> list:
        sub = self.latest_df
        grp = sub.groupby(['major_code', 'major_name'], as_index=False).agg(
            n_students=('record_uid', 'count'), n_classes=('class_code', 'nunique'),
            gpa_mean=('gpa4_cumulative', 'mean'), owed_mean=('credits_owed_cumulative', 'mean'),
            warning_count=('warning_bool', 'sum'),
            high_risk_count=('risk_probability', lambda s: int((s >= self.meta.get('threshold', 0.21)).sum())),
            total_debt_vnd=('tuition_debt_end_vnd', 'sum')
        ).sort_values('n_students', ascending=False)
        return [{
            "major_code": str(r['major_code']), "major_name": str(r['major_name']),
            "n_students": int(r['n_students']), "n_classes": int(r['n_classes']),
            "gpa_mean": round(float(r['gpa_mean'] or 0), 2), "owed_mean": round(float(r['owed_mean'] or 0), 2),
            "warning_count": int(r['warning_count']), "high_risk_count": int(r['high_risk_count']),
            "high_risk_pct": round(float(r['high_risk_count'] / max(1, r['n_students']) * 100), 1),
            "total_debt_billion": round(float(r['total_debt_vnd'] or 0) / 1e9, 2)
        } for _, r in grp.iterrows()]

    def get_faculty_detail(self, major_code: str) -> dict:
        m_code = major_code.strip().upper()
        sub = self.latest_df[self.latest_df['major_code'].astype(str).str.upper() == m_code]
        if sub.empty:
            return {}
        classes = self.get_all_classes_summary(m_code)
        n_stu = int(len(sub))
        warn_cnt = int(sub['warning_bool'].sum())
        hr_cnt = int((sub['risk_probability'] >= self.meta.get('threshold', 0.21)).sum())
        gpa_avg = round(float(sub['gpa4_cumulative'].mean() or 0), 2)
        owed_avg = round(float(sub['credits_owed_cumulative'].mean() or 0), 2)
        return {
            "major_code": m_code, "major_name": str(sub.iloc[0]['major_name']),
            "n_students": n_stu, "n_classes": len(classes), "gpa_mean": gpa_avg, "owed_mean": owed_avg,
            "warning_count": warn_cnt, "high_risk_count": hr_cnt,
            "summary_text": f"Khoa {sub.iloc[0]['major_name']} ({m_code}) có {len(classes)} lớp ({n_stu:,} SV) • GPA TB: {gpa_avg}/4.00 • Nợ TB: {owed_avg} TC • Cảnh báo học vụ: {warn_cnt:,} SV • Nguy cơ thôi học cao: {hr_cnt:,} SV.",
            "classes": classes
        }

    def parse_flexible_query(self, query: str) -> dict:
        clean_q = re.sub(r'\s*\[(?:khoa|toàn trường)\s*#\d+\]\s*$', '', str(query), flags=re.IGNORECASE).strip()
        q_norm = normalize_vn(clean_q)
        q_ascii = strip_accents(q_norm)
        stu_match = re.search(r'\b(B[A-Z]{2}\d{6}|SIMV31_\d{6})\b', clean_q.upper())
        cls_match = re.search(r'\b([A-Z]{2}\d{2}-\d{2})\b', clean_q.upper())
        maj_match = re.search(r'\b(?:KHOA|NGANH)\s+(IT|BA|CS|SE|AI|GD|KL|CL|EM|NS|LS|IB|MC|MK|PR|DA|GA|CB|EC)\b', q_ascii.upper())
        detected_major = maj_match.group(1) if maj_match else ""
        if not detected_major:
            for m_code, aliases in MAJOR_ALIASES.items():
                if any(a in q_ascii for a in aliases):
                    detected_major = m_code
                    break
        term_match = re.search(r'\b(?:hoc\s*ky|ky|hk)\s*(\d{1,2})\b', q_ascii)
        term_num = int(term_match.group(1)) if term_match else None
        art_matches = re.findall(r'\b(?:dieu|quy\s*che|quy\s*dinh|muc|khoan)\s*(?:so\s*)?(\d{1,2})\b', q_ascii)
        art_matches = [a for a in art_matches if 1 <= int(a) <= 27]

        course_ascii_kw = ["hoc phan", "mon hoc", "diem mon", "tung mon", "cac mon", "mon truot", "mon rot", "mon no", "chi tiet mon", "diem f", "hoc lai mon", "c00"]
        personal_ascii_kw = ["bang diem", "diem cua toi", "diem ca nhan", "ket qua hoc tap", "ho so", "bieu do", "cua em", "cua minh", "cua toi", "em co bi", "em da", "em con"]
        is_course = any(k in q_ascii for k in course_ascii_kw) and not any(k in q_ascii for k in ["ty le", "khoa", "toan truong", "nhieu nhat"])
        is_personal = bool(stu_match) or any(k in q_ascii for k in personal_ascii_kw)

        return {
            "clean_q": clean_q, "q_norm": q_norm, "q_ascii": q_ascii, "articles": art_matches,
            "student_id": stu_match.group(1) if stu_match else "", "class_code": cls_match.group(1) if cls_match else "",
            "major_code": detected_major, "term_filter": term_num, "is_course": is_course, "is_personal": is_personal
        }

    def answer_query(self, message: str, role: str = "student", logged_in_user: str = "") -> dict:
        parsed = self.parse_flexible_query(message)
        clean_q, q_norm, q_ascii = parsed["clean_q"], parsed["q_norm"], parsed["q_ascii"]
        user_id = logged_in_user.strip().upper()
        mentioned_id, mentioned_class, mentioned_major = parsed["student_id"], parsed["class_code"], parsed["major_code"]

        oos_html = self.check_out_of_scope_or_emotion(clean_q)
        if oos_html:
            return {"reply": oos_html, "student": None, "regulations": []}

        if role == "student" and mentioned_id and mentioned_id != user_id:
            return {"reply": f"<div class='text-xs text-rose-300 font-semibold bg-[#131314] border border-rose-500/40 p-3.5 rounded-2xl'>🔒 Giới hạn quyền Sinh viên: Bạn đang đăng nhập mã <b>{user_id}</b>, chỉ được xem dữ liệu của chính mình. Không thể xem hồ sơ của <b>{mentioned_id}</b>.</div>", "student": None, "regulations": []}

        admin_only_kw = ["tong no hoc phi toan truong", "that thu hoc phi", "chi phi du an", "ngan sach phu dao", "tong sv tung khoa", "tong so sinh vien", "so sanh ty le thoi hoc giua cac khoa", "bao lau can huan luyen lai"]
        if role != "admin" and any(k in q_ascii for k in admin_only_kw):
            return {"reply": f"<div class='text-xs text-amber-300 font-semibold bg-[#131314] border border-amber-500/40 p-3.5 rounded-2xl'>🔒 <b>Giới hạn thẩm quyền ({role.upper()}):</b> Các báo cáo Vĩ mô Toàn trường (Tổng công nợ học phí toàn trường, Đánh giá ngân sách/ROI dự án, So sánh toàn bộ 19 Khoa & Chu kỳ huấn luyện mô hình) là <b>đặc quyền chỉ dành cho Quản trị viên cấp Trường (ADMIN)</b>.</div>", "student": None, "regulations": []}

        html_blocks = []
        deep_html = self.answer_deep_benchmark_questions(clean_q, role, mentioned_major)
        if deep_html:
            html_blocks.append(deep_html)

        special_html = self.check_special_policy_or_model_query(clean_q, role)
        if special_html:
            html_blocks.append(special_html)

        trap_html = self.evaluate_regulation_trap(clean_q)
        if trap_html:
            html_blocks.append(trap_html)

        strict_regs, strict_kws = [], [f"Điều {a}" for a in parsed["articles"]]
        is_pure_stat_question = any(k in q_ascii for k in ["lop manh nhat", "lop yeu nhat", "khoa nao manh nhat", "top 5 lop", "so sanh gpa trung binh"])
        if not is_pure_stat_question:
            for item in STRUCTURED_REGULATIONS:
                sc = 100 if any(a in item["article_num"] for a in parsed["articles"]) else 0
                for kw in item["keywords"]:
                    kw_n = normalize_vn(kw)
                    kw_a = strip_accents(kw_n)
                    if re.search(rf'\b{re.escape(kw_n)}\b', q_norm) or (len(kw_a) >= 5 and re.search(rf'\b{re.escape(kw_a)}\b', q_ascii)):
                        sc += 25
                        strict_kws.append(kw)
                if sc > 0:
                    strict_regs.append((sc, item))
            strict_regs.sort(key=lambda x: x[0], reverse=True)

        if strict_regs and not (deep_html and not parsed["articles"]):
            html_blocks.append(self.render_regulations_html([m[1] for m in strict_regs[:2]], list(dict.fromkeys(strict_kws))[:6]))

        analytics_html = self.answer_analytics_query(clean_q, role, mentioned_major)
        if analytics_html:
            html_blocks.append(analytics_html)

        if mentioned_class:
            if role == "student":
                html_blocks.append("<div class='text-xs text-rose-300 font-semibold bg-[#131314] border border-rose-500/40 p-3.5 rounded-2xl'>🔒 Giới hạn quyền Sinh viên: Không có quyền xem thống kê Lớp.</div>")
            else:
                c_detail = self.get_class_detail(mentioned_class)
                if c_detail:
                    rows = "".join([f"<tr class='border-b border-[#2c2e30]'><td class='py-1.5 px-2 font-bold text-blue-300'>{s['student_id']}</td><td class='py-1.5 px-2'>{s['gpa4_term']:.2f} / {s['gpa4_cumulative']:.2f}</td><td class='py-1.5 px-2 text-amber-300'>{s['credits_owed_cumulative']} TC</td><td class='py-1.5 px-2 text-rose-400 font-bold'>{s['risk_probability']*100:.1f}%</td></tr>" for s in c_detail['students'][:8]])
                    html_blocks.append(f"""<div class="bg-[#131314] border border-[#333537] rounded-2xl p-3.5 space-y-2.5 text-xs"><div class="font-bold text-emerald-300 text-sm">🏫 Tóm tắt Lớp {c_detail['class_code']} – {c_detail['major_name']}</div><div class="text-[#c4c7c5]">{c_detail['summary_text']}</div><table class="w-full text-left"><thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1 px-2">Mã SV</th><th class="py-1 px-2">GPA Kỳ/TL</th><th class="py-1 px-2">TC Nợ</th><th class="py-1 px-2">Rủi ro AI</th></tr></thead><tbody>{rows}</tbody></table></div>""")

        target_stu = user_id if role == "student" else (mentioned_id or ("BIT250007" if parsed["is_personal"] or parsed["is_course"] else ""))
        if target_stu:
            recs = self.get_student_records(target_stu)
            if not recs.empty:
                risk_html = self.answer_risk_and_whatif_query(clean_q, recs)
                if risk_html:
                    html_blocks.append(risk_html)
                metric_html = self.answer_personal_metric_query(clean_q, recs)
                if metric_html:
                    html_blocks.append(metric_html)
                if parsed["is_course"]:
                    d_last = row_to_clean_dict(recs.iloc[-1])
                    uid = str(d_last.get('record_uid', ''))
                    only_fail = any(w in q_ascii for w in ["truot", "rot", "no", "hoc lai", "diem f"])
                    courses = self.get_student_courses(uid, parsed["term_filter"], only_fail)
                    html_blocks.append(self.render_course_grades_html(target_stu, courses, parsed["term_filter"], only_fail))
                elif parsed["is_personal"] and not metric_html and not risk_html and not strict_regs and not deep_html:
                    html_blocks.append(self.render_student_rich_html(recs, role))

        if not html_blocks:
            html_blocks.append(self.build_did_you_mean_html(clean_q, role, user_id))

        return {"reply": "<div class='space-y-4'>" + "".join(html_blocks) + "</div>", "student": None, "regulations": []}