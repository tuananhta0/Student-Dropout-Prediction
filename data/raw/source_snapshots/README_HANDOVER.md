# Bàn giao CMC Dataset V2.2 · AUDIT READY

**Mục tiêu:** xây dựng trợ lý học vụ CMC gồm tra cứu văn bản quy chế (RAG) và cảnh báo sớm thôi học (Dropout Prediction). CSV ở đây phục vụ mô-đun Dropout Prediction, **không thay tài liệu nguồn của RAG**.

## File cốt lõi
- `dataset_cleaned.csv`: 10,000 dòng, 81 cột, bảo toàn đúng bytes nguồn. SHA-256: `d1e08eb0c252db7fadd37f46b438e70c8f0948afa0c1b3db33a054eca2e78436`.
- `data_dictionary.csv`: 81 dòng mô tả, bổ sung `hk1_feature_policy` và giải thích chống rò rỉ.
- `quality_report.md`: kết quả audit, giới hạn và hướng dẫn bàn giao.

## File hỗ trợ thực nghiệm
- `feature_manifest_hk1.csv`: whitelist/denylist 81 cột.
- `hk1_preprocessing_input.csv`: 9,887 dòng quan sát được; `record_uid` chỉ để nối dữ liệu, 18 đặc trưng HK1 và nhãn ở cuối. **Chưa impute/scale/encode**.
- `hk1_temporal_split_manifest.csv`: phương án Train khóa 2022–2023, Validation 2024, Test 2025.
- `curriculum_credit_mapping.csv`: 200 mức tín chỉ cố định giả lập, không phải bảng tín chỉ CMC chính thức.
- `validation_summary.json`: số lượt kiểm tra, lỗi và thống kê có thể đọc bằng máy.
- `scripts/validate_handover.py`: kiểm tra nhanh lại 3 file cốt lõi và tập ML an toàn.

## Cảnh báo
Không train trực tiếp mọi cột của bảng 81 cột. Nhất là `academic_status`, `academic_status_detail`, `exit_reason`, `warning_total`, GPA/tín chỉ tại cutoff, `label_observed`, `generator_seed` đều bị loại khỏi X. Tách Train/Val/Test **trước** impute/encode/scale hoặc resampling. Cần đánh giá PR-AUC, Recall, Precision, F1 và calibration do nhãn dương chỉ 272 trường hợp.

## Tái kiểm định
```bash
python scripts/validate_handover.py .
```

**Phạm vi:** Đầu vào cung cấp chỉ chứa 3 file CLEANED. Kiểm toán chéo lịch sử được thực hiện với các bảng sự thật từ bộ sinh có trong môi trường làm việc (372.053 lượt học phần). Các bảng gốc và mã nguồn sinh lịch sử cần được lưu cùng phiên bản nếu nhóm muốn tái lập toàn bộ *synthetic generation*, không chỉ tái kiểm tra bảng CLEANED.

## Tái lập từ nguồn (chỉ có trong Full Package)

Tệp ZIP mở rộng đi kèm `provenance/raw/`, `provenance/master/`, mã nguồn `provenance/scripts/generate.py` và cấu hình bộ sinh. Dựng lại bản Cleaning độc lập từ RAW + lịch sử HK1:

```bash
python scripts/rebuild_cleaned_from_raw.py --raw provenance/raw/student_dropout_cmc_v2_2_raw.csv --semesters provenance/master/semester_history_clean.csv --output recreated_cleaned.csv
```

SHA-256 `recreated_cleaned.csv` phải là `d1e08eb0c252db7fadd37f46b438e70c8f0948afa0c1b3db33a054eca2e78436`. Để sinh toàn bộ synthetic từ đầu, dùng `provenance/scripts/generate.py`, seed `221026`, cấu hình đi kèm và thư viện `numpy`. Không trộn dữ liệu từ seed khác.
