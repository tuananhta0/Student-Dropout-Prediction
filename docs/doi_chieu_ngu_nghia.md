# Đối chiếu trực tiếp câu hỏi với PDF quy chế

**Phạm vi:** rà soát thủ công độc lập với mã nguồn trích dẫn, thực hiện trên bản PDF 27 trang `regulations/Quy-che-dao-tao-trinh-do-dai-hoc.pdf` bằng cách xem ảnh trang 17 và 23. Không sử dụng OCR hoặc Gemini để suy ra điều khoản. Không phải đánh giá mù bởi thành viên thứ ba; kết quả Gemini thực chưa được chấm trong tài liệu này.

| Câu hỏi kiểm tra | Căn cứ trực tiếp trong PDF | Nội dung phải thể hiện hoặc giới hạn |
|---|---|---|
| Điều kiện học lớp cải thiện khi đạt C? | PDF trang 17, Điều 15 khoản 2 | Môn đã đạt được đăng ký học lại để cải thiện; điểm chính thức cao nhất giữa các lần |
| Em có thể học lớp cải thiện với điểm D? | PDF trang 17, Điều 15 khoản 2–3 | Đã đạt vẫn được cải thiện; có học phí học lại theo tín chỉ |
| Học cải thiện lấy điểm lần sau hay điểm cao hơn? | PDF trang 17, Điều 15 khoản 2 | Điểm cao nhất của các lần học, không mặc định điểm mới nhất |
| Học phí cao quá em muốn nghỉ học một năm được không? | PDF trang 23, Điều 21 khoản 1(d) | Không tự động cho phép vì lý do tài chính; lý do cá nhân phải đã học tối thiểu một kỳ, không thuộc diện xem xét buộc thôi học/kỷ luật; nộp đơn và minh chứng; tối đa 3 học kỳ |
| Mức học phí một tín chỉ bao nhiêu và bảo lưu thế nào? | PDF trang 23, Điều 21 khoản 1 | Chỉ giải thích điều kiện bảo lưu; không suy đoán con số học phí vì PDF này không nêu mức thu cụ thể |
| Nghỉ học do ốm có tính thời gian đào tạo không? | PDF trang 23, Điều 21 khoản 1(c) | Tổng thời gian được miễn tính tối đa 6 học kỳ chính, phần vượt tính vào tiến độ |
| Sau khi bảo lưu trở lại cần làm gì? | PDF trang 23, Điều 21 khoản 1(d) | Nộp đơn ít nhất hai tuần trước học kỳ mới trong trường hợp nêu tại điểm d |
| Xin thôi học cá nhân rồi muốn quay lại? | PDF trang 23, Điều 21 khoản 3 | Muốn trở lại phải tuyển sinh lại; kết quả đã tích lũy được công nhận theo quy định |

**Kiểm tra tự động:** `tests/test_delivery_regressions.py` kiểm thử các cách diễn đạt chính qua router và API; `scripts.benchmark_rag --all --e2e` kiểm chứng Điều mong đợi được hiển thị. Các phép thử này không đánh giá độ đúng mọi câu sinh của LLM.

**Đánh giá độc lập còn cần thực hiện:** một thành viên không sửa retrieval chuẩn bị trước tập câu mới, chạy `scripts.benchmark_rag --cases ...` và (nếu có khóa mới) `--cloud`, đọc câu trả lời và nguồn PDF, ghi mức đúng/đủ vào CSV rồi chạy `scripts.review_rag`. Không tự nhận PASS cho bài kiểm tra này trước khi người đó xác nhận.

## Đối chiếu ca truy vấn bổ sung với quy chế gốc (bản sửa tiếp theo)

Đã đối chiếu phạm vi với Điều 14, 15, 18, 21 và 25 trong PDF quy chế 27 trang; đây là rà soát của người sửa mã, **không phải chấm mù độc lập**.

| Câu hỏi | Nguồn quy chế | Kết luận được phép đưa ra |
|---|---|---|
| Mức học phí một tín chỉ bao nhiêu? Có được nghỉ học do khó khăn tài chính? | Điều 21.1d (trang PDF 23) | Trả căn cứ nghỉ cá nhân; không dự đoán mức tiền học phí, không coi khó khăn tài chính là căn cứ tự động được chấp thuận |
| Em có thể nghỉ 3 kỳ vì lý do cá nhân? | Điều 21.1d (trang PDF 23) | 3 kỳ không vượt giới hạn; còn phải học ít nhất một kỳ, không thuộc diện buộc thôi học/kỷ luật, nộp đơn và minh chứng |
| Em xin nghỉ 4 kỳ vì lý do cá nhân? | Điều 21.1d (trang PDF 23) | 4 kỳ vượt giới hạn tối đa 3 học kỳ; không kéo Điều 25 về vi phạm thi vào trả lời |
| Em xin nghỉ 1 học kỳ vì ốm? | Điều 21.1c (trang PDF 23) | Nêu chứng nhận y tế/điều trị dài ngày; không tự áp giới hạn 3 kỳ dành cho lý do cá nhân |
| Điểm C hoặc B muốn thi lại để cải thiện? | Điều 15.2 (trang PDF 17) và Điều 14.3 (trang PDF 16) | Phân biệt học lại để cải thiện môn đã đạt (lấy điểm cao nhất) với thi lại theo quy định riêng; không dùng Điều 14 thay cho Điều 15 |
| GPA học kỳ thứ hai 0,99 có bị cảnh báo? | Điều 18.1a (trang PDF 20) | Rule Engine kiểm tra GPA học kỳ sau đầu < 1,0; kết luận chỉ về tiêu chí GPA, không quyết định buộc thôi học hay suy diễn từ GPA hồ sơ |

Các test liên quan nằm trong `tests/test_delivery_regressions.py`, bộ `scripts.benchmark_rag --all --e2e` được mở rộng thêm 7 câu. Các test mới là **hồi quy do người phát triển soạn**, không thay cho người thứ ba đánh giá độ đúng ngữ nghĩa/Gemini trên câu hỏi giữ kín.
