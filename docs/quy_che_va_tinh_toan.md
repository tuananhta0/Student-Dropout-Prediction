# Quy chế học vụ và công thức tính toán

Căn cứ QĐ 22/2023/QĐ-ĐHCMC-HĐT (áp dụng từ năm học 2023–2024). Dùng nhất quán cho các khóa mô phỏng là giả định nghiên cứu, không phải xác minh lịch sử chính thức.

## Quy tắc

- Điều 12.2e: khối lượng đăng ký từ 2/3 đến 3/2 chuẩn, làm tròn cận dưới lên và cận trên xuống. Ví dụ chuẩn 15 tín chỉ: 10–22.
- Điều 13–15: thi cuối kỳ ≤ 2 là F; làm tròn điểm học phần một chữ số thập phân trước quy đổi; thi lại tối đa C, khác với học lại/cải thiện lấy điểm tốt nhất.
- Điều 15, 17: GPA bằng tổng điểm quy đổi nhân tín chỉ chia tổng tín chỉ tính điểm. Một học phần học lại không cộng trùng tín chỉ; F đóng góp 0 điểm, vẫn tính tín chỉ. Loại học phần không tính GPA và I/X/R/P.
- Điều 18: cảnh báo nếu GPA học kỳ đầu < 0,8, kỳ sau < 1,0, trượt > 50% số tín chỉ đăng ký hoặc nợ > 24 tín chỉ (trừ môn chứng chỉ). Xem xét buộc thôi học khi quá 2 lần liên tiếp hoặc quá 3 lần cảnh báo, hoặc vượt thời gian đào tạo tối đa 18/22 học kỳ theo hệ. Không tự ra quyết định hành chính.
- Điều 25.1a: vắng không phép > 20% hoặc tổng vắng > 30% của từng môn. Không dùng chuyên cần trung bình toàn kỳ để kết luận cấm thi.
- Điều 19–20: tín chỉ, học phần bắt buộc, GPA ≥ 2, ngoại ngữ B1/B2 theo hệ, thể chất, quốc phòng, nghĩa vụ tài chính, kỷ luật và các chuẩn đầu ra. Dùng bảng xét tốt nghiệp nếu có; thiếu thông tin giữ trạng thái chưa xác định.

## Kịch bản GPA

Biểu đồ lấy kỳ học đã công bố mới nhất và lập kịch bản kỳ kế tiếp, không coi đó là dự báo ML. Mặc định tối đa 15 tín chỉ mới và không học lại. Kịch bản học lại dùng mã môn đầy đủ: `Kịch bản học lại CURR_... đạt B`. Có thể chọn nút giả định trong bảng điểm rồi hỏi chatbot. Chỉ thay điểm cao hơn, không cộng trùng tín chỉ. Không áp dụng dự báo can thiệp cho hồ sơ không còn đang học.

## Xử lý dữ liệu cần đối chiếu

- 4.338 trường hợp đăng ký ít tín chỉ do cuối chương trình: giả định ngoại lệ mô phỏng, không xác nhận phê duyệt của trường. 87 trường hợp còn lại: thiếu căn cứ. Không tự sửa khối lượng đăng ký.
- 2 lượt thi có số giờ vắng đúng ngưỡng nhưng hồ sơ ghi không đủ điều kiện: hiển thị cần đối chiếu thay vì suy ra cấm thi do chuyên cần. Giữ kết quả lịch sử.
- 269 hồ sơ thiếu điểm ngoài tuyển thẳng giữ giá trị thiếu với lý do SYNTHETIC_SOURCE_MISSING; 756 hồ sơ tuyển thẳng không áp dụng điểm. 4 điểm thấp hơn ngưỡng đã đối chiếu không được dùng làm bằng chứng đủ điều kiện tuyển sinh. Không tự bịa điểm hoặc quyết định.
- 37.938 lượt học phần khác quy tắc làm tròn. Chưa sửa nguồn vì 2.621 lượt có thể đổi trạng thái F thành D và ảnh hưởng lịch sử học lại. Phải tái tạo đồng bộ GPA, tín chỉ, cảnh báo, feature và kiểm chứng nhãn trước khi huấn luyện. Báo cáo và bảng điểm đánh dấu sai lệch, không coi là đã sửa nguồn.

## Nguồn đối chiếu tuyển sinh

- 2022: https://fad.cmc-u.edu.vn/phuong-thuc-tuyen-sinh-2022/
- 2023: https://cmcu.edu.vn/phuong-thuc-tuyen-sinh-2023/ và https://cmcu.edu.vn/diem-san-va-du-bao-diem-chuan-xet-tuyen-truong-dai-hoc-cmc-nam-2023/
- 2024: https://fict.cmc-u.edu.vn/cau-hoi-thuong-gap/ và https://cmcu.edu.vn/diem-san-nop-ho-so-xet-tuyen-truong-dai-hoc-cmc-theo-phuong-thuc-xet-diem-thi-tot-nghiep-thpt-nam-2024/
- 2025: https://cmcu.edu.vn/truong-dai-hoc-cmc-cong-bo-4-phuong-thuc-xet-tuyen-nam-2025/ và https://cmcu.edu.vn/thong-bao-quy-doi-diem-tuong-duong-giua-cac-phuong-thuc-xet-tuyen-truong-dai-hoc-cmc-nam-2025/

Chỉ dùng các ngưỡng đã công bố có thể so sánh. Thiếu tổ hợp, ưu tiên hoặc hồ sơ đặc biệt nên không kết luận trúng tuyển cá nhân. Toàn bộ báo cáo sinh ra nằm ngoài Git.

## Kịch bản bổ sung

Giao diện cho phép chọn nhiều học phần đã có điểm để học lại hoặc cải thiện, nhập số tín chỉ môn mới và GPA giả định. Backend kiểm tra tổng tín chỉ kỳ mới và không cộng trùng tín chỉ đã tính. Kết quả xét tốt nghiệp kèm ngày bản ghi xét. Với model nghiên cứu mới, kịch bản thanh toán lấy đầu vào từ snapshot ML hợp lệ, không phải hồ sơ học vụ thiếu feature. Đây là phép thử độ nhạy chứ không phải bằng chứng nhân quả. Xem [hướng dẫn thực nghiệm](huan_luyen_va_thuc_nghiem.md).
