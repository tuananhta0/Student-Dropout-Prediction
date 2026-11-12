import json
import pypdf
from pathlib import Path

#duong dan thu muc
REG_DIR = Path("regulations")
OUT_DIR = Path("artifacts_model")
OUT_DIR.mkdir(exist_ok=True)

#du lieu day du 27 dieu tu ban scan Quy-che-dao-tao-trinh-do-dai-hoc.pdf va Quy-dinh-thoi-hoc-DH-CMC.pdf
FULL_CMC_REGULATIONS = [
    {
        "rule_id": "QD22_DIEU_1_2",
        "title": "Điều 1 & Điều 2: Phạm vi áp dụng và Mục tiêu đào tạo",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 5 - Chương I)",
        "keywords": ["điều 1", "điều 2", "phạm vi", "đối tượng áp dụng", "mục tiêu đào tạo", "chính quy", "từ xa"],
        "content": "Áp dụng cho sinh viên đại học hệ chính quy tại Trường Đại học CMC từ năm học 2023–2024 (theo QĐ số 22/2023/QĐ-ĐHCMC-HĐT ngày 01/12/2023, không áp dụng hệ từ xa/liên kết). Mục tiêu đào tạo đại học đa ngành, công nghệ mũi nhọn, sử dụng tốt tiếng Anh (khuyến khích ngoại ngữ 2 Nhật/Hàn)."
    },
    {
        "rule_id": "QD22_DIEU_3",
        "title": "Điều 3: Tín chỉ, Chương trình đào tạo (GM/SM) và Thời gian học tối đa",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 5–7) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 3)",
        "keywords": ["điều 3", "tín chỉ", "thời gian đào tạo", "tối đa", "cử nhân", "kỹ sư", "hệ gm", "hệ sm", "tiên quyết", "học trước", "thay thế"],
        "content": "1 tín chỉ = 15 giờ tín chỉ (1 giờ lý thuyết = 1 tiết lớp + 2 tiết tự học; 1 giờ thực hành = 2-3 tiết thực hành). Trường có 2 hệ: Song ngữ (Global Mode - GM) và Chuẩn (Standard Mode - SM). Học phần điều kiện (GD Thể chất, GD Quốc phòng - An ninh) không tính vào GPA tích lũy nhưng là điều kiện xét tốt nghiệp. Thời gian đào tạo: Hệ Cử nhân (120–135 tín chỉ) thiết kế 9 học kỳ, tối đa không quá 18 học kỳ; Hệ Kỹ sư (150–164 tín chỉ) thiết kế 11 học kỳ, tối đa không quá 22 học kỳ."
    },
    {
        "rule_id": "QD22_DIEU_4_5",
        "title": "Điều 4 & Điều 5: Phương thức đào tạo và Khung giờ 15 tiết học trong ngày",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 7–8)",
        "keywords": ["điều 4", "điều 5", "tiết học", "thời gian học", "buổi sáng", "buổi chiều", "buổi tối", "mấy phút", "học lại bắt buộc"],
        "content": "Đào tạo theo tín chỉ; trượt học phần bắt buộc phải học lại đúng học phần đó, trượt học phần tự chọn có thể học lại hoặc đổi sang học phần tự chọn khác. Mỗi tiết học 50 phút: Buổi sáng (Tiết 1: 7:00–7:50 đến Tiết 6: 11:35–12:25); Buổi chiều (Tiết 7: 13:00–13:50 đến Tiết 12: 17:35–18:25); Buổi tối (Tiết 13: 18:50–19:40 đến Tiết 15: 20:40–21:30)."
    },
    {
        "rule_id": "QD22_DIEU_6_7",
        "title": "Điều 6 & Điều 7: Trách nhiệm Sinh viên, Giáo viên chủ nhiệm và Cố vấn học tập",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 8–10)",
        "keywords": ["điều 6", "điều 7", "trách nhiệm sinh viên", "cố vấn học tập", "giáo viên chủ nhiệm", "bảo mật tài khoản", "đóng học phí"],
        "content": "Sinh viên có trách nhiệm nắm vững quy chế, đăng ký học đúng hạn, đi học/thi đầy đủ, bảo mật tài khoản và đóng học phí đúng hạn. Giáo viên chủ nhiệm theo dõi lớp, tổ chức họp xét khen thưởng/kỷ luật và báo cáo Phòng CTSV. Giảng viên Cố vấn học tập hướng dẫn sinh viên chọn học phần, lập kế hoạch học tập từng kỳ, phương pháp tự học, chọn nơi thực tập và đề tài khóa luận."
    },
    {
        "rule_id": "QD22_DIEU_8_9_10_11",
        "title": "Điều 8, 9, 10, 11: Kế hoạch đào tạo, Tổ chức lớp, Quy mô lớp và Học trực tuyến",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 11–12 - Chương II)",
        "keywords": ["điều 8", "điều 9", "điều 10", "điều 11", "quy mô lớp", "trực tuyến", "online", "hủy lớp", "thời khóa biểu"],
        "content": "Thời khóa biểu thông báo trước 03 tuần khi bắt đầu kỳ mới. Quy mô lớp kiến thức chung: 30–100 SV/lớp; lớp đặc thù (Ngôn ngữ, Mỹ thuật, Thiết kế): 25–40 SV/lớp; lớp chuyên ngành: 25–80 SV/lớp (nếu dưới mức tối thiểu lớp sẽ bị hủy và SV được chuyển lớp). Dạy học trực tuyến không vượt quá 30% tổng khối lượng chương trình đào tạo."
    },
    {
        "rule_id": "QD22_DIEU_12",
        "title": "Điều 12: Đăng ký học tập, Khóa luận tốt nghiệp, Rút học phần và Tự ý bỏ học",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 12–14) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 5)",
        "keywords": ["điều 12", "đăng ký học tập", "tối thiểu", "tối đa", "2/3", "3/2", "khóa luận", "đồ án", "rút học phần", "tự ý bỏ học", "100% học phí"],
        "content": "Đăng ký muộn nhất 01 tuần trước kỳ mới. Khối lượng mỗi học kỳ: tối thiểu không ít hơn 2/3 (9 TC) và tối đa không vượt quá 3/2 (20 TC) khối lượng trung bình học kỳ chuẩn (15 TC). Điều kiện làm Khóa luận/Đồ án tốt nghiệp: GPA tích lũy tối thiểu đạt 2.50 (loại Khá trở lên), thời gian làm 3–4 tháng (1 học kỳ), chỉ được gia hạn tối đa 1 lần. Rút học phần chỉ thực hiện trong vòng 01 tuần kể từ khi kỳ học bắt đầu; sau hạn này nếu không đi học bị tính là tự ý bỏ học, nhận điểm F và phải đóng 100% học phí."
    },
    {
        "rule_id": "QD22_DIEU_13",
        "title": "Điều 13: Đánh giá học phần, Điểm liệt (<= 2.0) và Bảng quy đổi điểm chữ",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 15–16) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 6)",
        "keywords": ["điều 13", "điểm liệt", "điểm thành phần", "trọng số", "điểm chữ", "thang 10", "điểm i", "điểm r", "điểm p"],
        "content": "Điểm thi cuối kỳ <= 2.0 (thang 10) là ĐIỂM LIỆT và học phần bị xếp điểm F dù điểm quá trình cao. Trọng số thi kết thúc học phần chiếm từ 50% đến 70%; đánh giá trực tuyến không quá 50%. Quy đổi thang 10 sang chữ: 9.0–10 (A+), 8.5–8.9 (A), 8.0–8.4 (B+), 7.0–7.9 (B), 6.5–6.9 (C+), 5.5–6.4 (C), 5.0–5.4 (D+), 4.0–4.9 (D), dưới 4.0 (F - không đạt). Học phần chỉ yêu cầu đạt: P (>= 5.0). Điểm đặc biệt: I (hoãn thi), X (thiếu dữ liệu), R (miễn học/công nhận tín chỉ)."
    },
    {
        "rule_id": "QD22_DIEU_14_15_16",
        "title": "Điều 14, 15, 16: Thi kết thúc học phần, Thi lại (giới hạn điểm C), Học lại và Phúc tra",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 16–18) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 6)",
        "keywords": ["điều 14", "điều 15", "điều 16", "thi lại", "giới hạn điểm c", "học lại", "cải thiện điểm", "nghỉ quá 20%", "phúc tra"],
        "content": "Điều 14: Vắng thi không lý do chính đáng nhận điểm 0; điểm học phần sau khi THI LẠI / đánh giá lại bị giới hạn tối đa ở mức điểm C. Điều 15: Sinh viên phải HỌC LẠI khi: (a) nghỉ học vượt quá 20% kế hoạch; (b) bị hủy kết quả thi do kỷ luật; (c) điểm tổng kết < 4.0 (điểm F). Học cải thiện điểm được lấy điểm cao nhất giữa các lần học; học lại phải đóng học phí theo tín chỉ. Điều 16: Phúc tra bài thi kết thúc học phần nộp đơn qua Phòng Đảm bảo chất lượng, thông báo kết quả muộn nhất 01 tuần sau hạn nhận đơn."
    },
    {
        "rule_id": "QD22_DIEU_17",
        "title": "Điều 17: Cách tính GPA thang 4, Xếp loại học lực và Xếp trình độ năm học",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 18–19) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 6)",
        "keywords": ["điều 17", "quy đổi thang 4", "xếp loại học lực", "xuất sắc", "giỏi", "khá", "yếu", "kém", "trình độ năm học", "stctl"],
        "content": "Quy đổi điểm chữ sang thang 4: A+/A = 4.0; B+ = 3.5; B = 3.0; C+ = 2.5; C = 2.0; D+ = 1.5; D = 1.0; F = 0 (làm tròn 2 chữ số thập phân, không tính môn chứng chỉ). Xếp loại học lực: Xuất sắc (3.60–4.00), Giỏi (3.20 đến <3.60), Khá (2.50 đến <3.20), Trung bình (2.00 đến <2.50), Yếu (1.00 đến <2.00), Kém (<1.00). Xếp trình độ năm học theo Số tín chỉ tích lũy (STCTL): Năm thứ nhất (STCTL < 40); Năm thứ hai (40 <= STCTL < 80); Năm thứ ba (80 <= STCTL < 120); Diện sắp tốt nghiệp (STCTL >= 120)."
    },
    {
        "rule_id": "QD22_DIEU_18",
        "title": "Điều 18: Cảnh báo học tập và Buộc thôi học do kết quả học tập",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 19–20) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 2, 3)",
        "keywords": ["điều 18", "cảnh báo học tập", "cảnh báo học vụ", "cảnh cáo", "buộc thôi học", "0.8", "1.0", "50%", "24 tín chỉ", "2 lần liên tiếp", "3 lần"],
        "content": "Khoản 1 (Cảnh báo học tập): Sinh viên bị cảnh báo cuối mỗi học kỳ nếu vi phạm: (a) GPA học kỳ < 0.80 đối với học kỳ đầu của khóa học, hoặc < 1.00 đối với các học kỳ tiếp theo; HOẶC (b) Tổng tín chỉ không đạt (trượt) trong kỳ > 50% khối lượng đã đăng ký trong kỳ, hoặc tổng số tín chỉ nợ đọng từ đầu khóa > 24 tín chỉ (không kể môn chứng chỉ). Khoản 2 (Buộc thôi học): Sinh viên bị buộc thôi học nếu: (a) Số lần cảnh báo học tập vượt quá 2 lần liên tiếp (tức từ lần thứ 3 liên tiếp) hoặc vượt quá 3 lần không liên tiếp; (b) Vượt quá thời gian tối đa được phép học tại Trường (18 kỳ Cử nhân, 22 kỳ Kỹ sư). Phòng CTSV thông báo về gia đình."
    },
    {
        "rule_id": "QD22_DIEU_19",
        "title": "Điều 19: Chuyển đổi tín chỉ (<= 50%) và Chuẩn đầu ra Tiếng Anh (Hệ SM: B1, Hệ GM: B2)",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 20–21) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 8)",
        "keywords": ["điều 19", "chuyển đổi tín chỉ", "công nhận tín chỉ", "50%", "chuẩn tiếng anh", "ngoại ngữ", "b1", "b2", "cefr", "olympic"],
        "content": "Khối lượng tối đa được công nhận, chuyển đổi tín chỉ không vượt quá 50% khối lượng học tập tối thiểu của chương trình (học phần phải đạt trở lên và số tín chỉ >= học phần tại Trường). Điều kiện Tiếng Anh khi tốt nghiệp: Hệ Chuẩn (SM) phải đạt bậc 3/6 Khung năng lực ngoại ngữ Việt Nam (B1 CEFR hoặc tương đương); Hệ Song ngữ (GM) phải đạt bậc 4/6 (B2 CEFR hoặc tương đương)."
    },
    {
        "rule_id": "QD22_DIEU_20",
        "title": "Điều 20: Công nhận tốt nghiệp, Hạ hạng bằng tốt nghiệp và Gia hạn hoàn thiện chứng chỉ 3 năm",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 21–22) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 8)",
        "keywords": ["điều 20", "tốt nghiệp", "ra trường", "hạ bằng", "giảm hạng tốt nghiệp", "5%", "3 năm", "chứng chỉ", "nghĩa vụ tài chính", "công nợ"],
        "content": "Khoản 1 (Điều kiện tốt nghiệp): (a) Tích lũy đủ học phần và số tín chỉ; (b) GPA tích lũy toàn khóa đạt từ Trung bình (>= 2.00) trở lên; (c) Đạt chuẩn đầu ra ngoại ngữ (B1 với SM, B2 với GM); (d) Có chứng chỉ GD Quốc phòng - An ninh và GD Thể chất; (e) Hoàn thành đầy đủ nghĩa vụ tài chính (công nợ = 0); (f) Không bị truy cứu hình sự hoặc kỷ luật từ mức đình chỉ học tập. Khoản 4 (Hạ hạng bằng): Sinh viên loại Xuất sắc và Giỏi sẽ bị giảm đi 01 mức nếu: (a) Khối lượng học phần phải học lại vượt quá 5% tổng số tín chỉ toàn chương trình; hoặc (b) Đã bị kỷ luật từ mức cảnh cáo trở lên. Khoản 5: Nếu hết thời gian học tối đa nhưng chỉ còn thiếu GDQP-AN, GDTC hoặc chuẩn ngoại ngữ thì được phép hoàn thiện trong thời hạn 03 năm kể từ khi thôi học để xét tốt nghiệp."
    },
    {
        "rule_id": "QD22_DIEU_21",
        "title": "Điều 21: Nghỉ học tạm thời (Bảo lưu kết quả) và Tự xin thôi học",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 23 - Chương IV) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 4)",
        "keywords": ["điều 21", "nghỉ học tạm thời", "bảo lưu", "ốm đau", "thai sản", "6 học kỳ", "3 học kỳ", "lí do cá nhân", "tự xin thôi học"],
        "content": "Sinh viên được nghỉ học tạm thời và bảo lưu khi: (a, b) Điều động vào lực lượng vũ trang hoặc đại diện quốc gia thi quốc tế; (c) Ốm, thai sản, tai nạn điều trị dài ngày có xác nhận y tế (không tính vào thời gian chậm tiến độ nếu tổng cộng không quá 6 học kỳ chính); (d) Lý do cá nhân khác: phải học tối thiểu 01 học kỳ tại Trường, không thuộc diện xem xét buộc thôi học/kỷ luật, nghỉ tối đa 03 học kỳ (01 năm) và thời gian nghỉ cá nhân ĐƯỢC TÍNH vào thời gian học chính thức tối đa. Khi quay lại phải nộp đơn trước ít nhất 02 tuần trước kỳ mới. Nếu tự xin thôi học hẳn, muốn quay lại phải dự tuyển đầu vào như thí sinh mới."
    },
    {
        "rule_id": "QD22_DIEU_22_23_24",
        "title": "Điều 22, 23, 24: Chuyển ngành, Học cùng lúc hai chương trình (Song bằng) và Trao đổi sinh viên",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 23–25)",
        "keywords": ["điều 22", "điều 23", "điều 24", "chuyển ngành", "chuyển trường", "học hai chương trình", "song bằng", "ngành thứ hai", "trao đổi sinh viên", "25%"],
        "content": "Điều 22 (Chuyển ngành): Không đang ở năm thứ nhất hoặc năm cuối khóa, không thuộc diện buộc thôi học, đạt điều kiện trúng tuyển ngành chuyển đến và chỉ được chuyển ngành 01 lần duy nhất. Điều 23 (Học cùng lúc 2 chương trình): Phải tích lũy >= 40 tín chỉ, học lực từ Khá trở lên (hoặc Trung bình nhưng đạt ngưỡng trúng tuyển ngành 2); chỉ được xét tốt nghiệp ngành 2 khi đã đủ điều kiện tốt nghiệp ngành 1. Điều 24 (Trao đổi sinh viên): Tín chỉ tích lũy tại trường đối tác không vượt quá 25% tổng khối lượng chương trình đào tạo."
    },
    {
        "rule_id": "QD22_DIEU_25",
        "title": "Điều 25: Xử lý kỷ luật sinh viên (Vắng quá 20%, Học hộ, Thi hộ, Tài liệu, Hồ sơ giả)",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 25–26) & Quy-dinh-thoi-hoc-DH-CMC.pdf (Mục 7)",
        "keywords": ["điều 25", "kỷ luật", "vi phạm", "nghỉ quá 20%", "30%", "cấm thi", "học hộ", "thi hộ", "tài liệu", "điện thoại", "đến muộn 15 phút", "văn bằng giả"],
        "content": "Chi tiết mức kỷ luật Điều 25: (1) Nghỉ không phép > 20% số giờ môn học hoặc tổng nghỉ (có phép + không phép) > 30%: Không được dự thi (cấm thi) và phải học lại. (2) Đi học hộ: Từ đình chỉ đến buộc thôi học; Nhờ học hộ: Lần 1 hủy điểm môn, Lần 2 đình chỉ 01 học kỳ. (3) Làm hộ/nhờ làm hộ bài kiểm tra quá trình: Đình chỉ 01 học kỳ; lần 2 hủy kết quả cả học kỳ và đình chỉ 01 kỳ tiếp. (4) Nhìn bài, trao đổi: Lần 1 khiển trách (-25% điểm), Lần 2 cảnh cáo (-50% điểm), Lần 3 nhận điểm 0. (5) Đến phòng thi muộn quá 15 phút sau khi phát đề: Không được dự thi (điểm 0). (6) Dùng tài liệu trong phòng thi: Lần 1 hủy điểm môn, Lần 2 hủy toàn bộ kết quả học kỳ, Lần 3 BUỘC THÔI HỌC. (7) Mang điện thoại/thiết bị truyền tin không tắt nguồn: Lần 1 hủy kết quả môn/kỳ thi, Lần 2 hủy kết quả học kỳ và đình chỉ 01 kỳ. (8) Thi hộ hoặc nhờ thi hộ: Lần 1 đình chỉ 01 năm, Lần 2 BUỘC THÔI HỌC. (9) Dùng hồ sơ/chứng chỉ giả: Buộc thôi học và thu hồi văn bằng."
    },
    {
        "rule_id": "QD22_DIEU_26_27",
        "title": "Điều 26 & Điều 27: Lưu trữ hồ sơ (Vĩnh viễn / 20 năm / 5 năm) và Hiệu lực thi hành",
        "source": "Quy-che-dao-tao-trinh-do-dai-hoc.pdf (Trang 27 - Chương V)",
        "keywords": ["điều 26", "điều 27", "lưu trữ", "bảng điểm gốc", "khóa luận", "20 năm", "hiệu lực"],
        "content": "Điều 26: Quyết định trúng tuyển, Bảng điểm gốc, Quyết định tốt nghiệp và Sổ gốc cấp bằng được bảo quản VĨNH VIỄN. Khóa luận đạt yêu cầu lưu trữ 20 năm; khóa luận không đạt lưu trữ 05 năm. Điều 27: Quy chế có hiệu lực từ năm học 2023–2024 theo Quyết định số 22/2023/QĐ-ĐHCMC-HĐT do Chủ tịch Hội đồng trường Nguyễn Trung Chính ký."
    }
]

def scan_and_save_regulations() -> list:
    kb = list(FULL_CMC_REGULATIONS)
    REG_DIR.mkdir(exist_ok=True)

    #quet truc tiep cac trang pdf co text trong thu muc regulations
    for pdf_file in sorted(REG_DIR.glob("*.pdf")):
        try:
            reader = pypdf.PdfReader(str(pdf_file))
            for i, page in enumerate(reader.pages):
                raw_txt = (page.extract_text() or "").strip()
                if len(raw_txt) > 80:
                    kb.append({
                        "rule_id": f"PDF_{pdf_file.stem}_TRANG_{i+1}",
                        "title": f"Quét trực tiếp {pdf_file.name} – Trang {i+1}",
                        "source": f"regulations/{pdf_file.name} (Trang {i+1})",
                        "keywords": [w for w in ["điều", "khoản", "cảnh báo", "thôi học", "tín chỉ", "học phí", "tốt nghiệp", "kỷ luật", "bảo lưu", "điểm", "thi hộ", "tài liệu", "khóa luận"] if w in raw_txt.lower()],
                        "content": " ".join(raw_txt.split())
                    })
        except Exception as e:
            print(f"Skip {pdf_file.name}: {e}")

    out_path = OUT_DIR / "regulations_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"Da quet va luu {len(kb)} dieu khoan quy che vao {out_path}")
    return kb

if __name__ == "__main__":
    scan_and_save_regulations()