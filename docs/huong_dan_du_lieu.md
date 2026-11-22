# Hướng dẫn dữ liệu

Sử dụng duy nhất V3.2.1 MAIN (20.000 sinh viên mô phỏng). Không trộn BASE, STRESS hoặc biến ẩn. Installer kiểm tra manifest, SHA256, kích thước và schema. Thực hiện từ thư mục gốc repository theo README. `--supplement` chỉ thêm bảng `graduation_clearance`, không ghi đè dữ liệu đã cài. Nếu cài ở vị trí khác, đặt `CMC_DATASET_ROOT` trỏ tới thư mục chứa `artifacts/runs/V3_2_MAIN`.

## Các mốc dữ liệu

| Mã nội bộ | Mốc | Mục đích |
|---|---|---|
| DEMO | 03/02/2027 | Trình diễn, có HK3 khóa 2025 |
| ASOF_2026_DEMO | 01/10/2026 | Nghiên cứu tại mốc gốc của bộ dữ liệu |
| FULL_SIMULATION | 31/12/2028 | Quan sát toàn bộ lịch sử mô phỏng |

Thiết lập bằng `$env:CMC_DATA_MODE = 'DEMO'`. Không đổi nhãn hoặc mốc quan sát nghiên cứu gốc. Model huấn luyện trên FULL_SIMULATION, không phải kiểm chứng triển khai theo thời gian thực.

Hồ sơ học vụ lấy mọi học kỳ đã công bố từ `term_academics`; `course_attempts` cũng lọc ngày công bố và ngày khả dụng. Snapshot ML lưu riêng trong `engine.ml_df`. Không ghi đè feature bằng học kỳ mới. Nếu hồ sơ đã có kỳ mới hơn hoặc đã hết cửa sổ hai kỳ, dự báo cũ là lịch sử, không dùng xếp hạng can thiệp. Điều kiện tốt nghiệp lấy bản ghi khả dụng gần nhất; chưa có bằng chứng không coi là đạt. Không đưa nhãn đánh giá vào API.

## Kiểm tra

Chạy `python -m scripts.audit_dataset` và `python -m scripts.review_policy` để xuất báo cáo vào `data/kiem_tra_du_lieu/` và `data/doi_chieu_quy_che/`. Giữ nguyên mã khóa JSON/CSV để các script nghiên cứu đọc được; tên thông báo và tài liệu dùng tiếng Việt.

| Kiểm tra nguồn | Số lượng |
|---|---:|
| Bản ghi học kỳ | 149.196 |
| Lượt học phần | 729.416 |
| Snapshot ML toàn bộ | 57.569 |
| Snapshot tại mốc 01/10/2026 | 54.609 |
| Nhãn chưa quan sát tại mốc 01/10/2026 | 9.970 |
| Bản ghi xét tốt nghiệp | 10.959 |

Cohort split không trùng sinh viên nhưng thời gian có thể chồng lấn; cần thực nghiệm rolling-origin. Báo cáo kiểm tra tính nhất quán nguồn không phải bằng chứng tuân thủ tất cả quy chế.
