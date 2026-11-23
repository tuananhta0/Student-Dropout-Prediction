# Đánh giá bổ sung model cmc-research-v1

Dữ liệu mô phỏng MAIN, tập Test khóa 2025, ngưỡng đã khóa 0,12. Các giá trị dưới đây lấy từ kết quả nghiệm thu trên máy nhóm trước commit 4097616; không phải thực nghiệm mới trên dữ liệu thực tế. Không chỉnh ngưỡng theo tập Test.

| Chỉ số | Snapshot Test | Snapshot đầu tiên mỗi sinh viên trong Test |
|---|---:|---:|
| Số mẫu | 6012 | 2986 |
| Dương tính | 859 | 338 |
| AP | 0,6291 | 0,6565 |
| ROC-AUC | 0,8293 | 0,8582 |
| Recall | 0,5774 | 0,6302 |
| Precision | 0,5077 | 0,4374 |

## Calibration
ECE theo 10 bin bằng nhau 0,03894 trên Test. Cả 10 bin có tỷ lệ dương tính quan sát lớn hơn xác suất trung bình: ví dụ bin 0–0,1 (4870 mẫu): 3,89% so với 6,82%; bin 0,1–0,2 (475 mẫu): 13,69% so với 20,42%. Xác suất chưa được xác nhận ngoài bộ mô phỏng, không phải quyết định buộc thôi học. Chỉ đánh giá phương pháp calibration mới với validation độc lập, không fit trên test.

## Fairness
Recall Nam 59,51% (489 snapshot dương), Nữ 55,03% (358 dương), Khác 66,67% (12 dương). Ngành KL 75% (36 dương), GD 37,04% (27 dương). Mẫu nhỏ và phụ thuộc nhiều snapshot của cùng người; chênh lệch không đủ để kết luận thiên lệch hoặc quan hệ nhân quả. Không đặt riêng ngưỡng bằng kết quả Test.

## Thời gian và censoring
HK3: 2960 mẫu có 333 dương; HK4: 2763 có 237 dương; HK5: 192/192 dương; HK6: 97/97 dương. Hai kỳ cuối chỉ có một lớp nhãn trong phần quan sát được; không suy diễn toàn bộ sinh viên thôi học. Nhãn không quan sát đủ chân trời bị loại khỏi đánh giá có giám sát, không gán âm tính. Snapshot đầu tiên trong Test không đồng nghĩa kỳ đầu nhập học. Model cần dữ liệu thật và validation triển khai trước khi áp dụng vào hồ sơ thật.

## Tái lập
`python -B -m scripts.review_experiment --bootstrap 200` đọc lại xác suất, không fit lại. Báo cáo thêm khoảng tin cậy percentile theo cụm sinh viên; không xem các snapshot cùng sinh viên là độc lập. Chỉ báo cáo CI khi đã chạy trên release thật. What-If GPA là phép tính học vụ; What-If học phí không khẳng định hiệu quả can thiệp.
