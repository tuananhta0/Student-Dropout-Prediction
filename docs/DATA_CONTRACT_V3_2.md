# CMC Synthetic Dataset V3.2: Data Contract

**Trạng thái:** Phase 1, nghiệm thu kỹ thuật E6 PASS (26/26); chờ rà soát và commit.
**Ngày cập nhật:** 02/11/2026
**Dataset:** `3.2.0` · **Scenario:** `V3_2_MAIN` · **Synthetic:** `true`
**Phạm vi:** Ứng dụng Student Dropout Prediction, không phải hệ thống quyết định học vụ chính thức của Trường Đại học CMC.

## 1. Mục tiêu và giới hạn

Dataset là dữ liệu **mô phỏng** (`RESEARCH_CANDIDATE`), phục vụ nghiên cứu, demo và kiểm thử kỹ thuật. Các mốc lịch sử được giả lập không được trình bày là lịch sử thực của trường. Không diễn giải tỷ lệ nguy cơ hay kết quả mô hình thành quyết định thôi học chính thức.

Bài toán ML chính có nhãn `dropout_next_2_terms`: dự báo sự kiện thôi học trong **hai học kỳ tiếp theo**. Không dùng `academic_status`, `warning_current` hay `exit_reason_for_evaluation` thay cho nhãn này.

## 2. Nguồn và provenance

| Trường | Quy ước |
|---|---|
| Source run | `artifacts/runs/V3_2_MAIN` trong gói bàn giao |
| Local source root | `data/external/v3_2_main/` (Git-ignored) |
| Manifest | `manifests/generation_manifest.json` |
| `generator_version` | `3.2.0` (được dùng để xác minh phiên bản generator) |
| `scenario_id` | `V3_2_MAIN` |
| `is_synthetic` | `true` |
| `demo_as_of` | `2026-10-01` |
| `simulation_as_of` | `2028-12-31` |
| Serving catalog | `data/processed/v3_2/dataset_catalog.json` (Git-ignored) |
| SHA-256 của ZIP V3.2 bàn giao | `5DF81F995347B167C4996F3A83C63C11637F78674834F76DD3D28A0548E6ECD6` (đã được ghi nhận lúc Phase 0) |

**Phân biệt checksum:** `source_zip_sha256` trong *generation manifest* là checksum **ZIP đầu vào của generator**, không phải checksum ZIP V3.2 bàn giao ở bảng trên. Catalog lưu `generation_manifest_sha256` và `generator_source_zip_sha256` để truy vết, không dùng hai giá trị đó thay thế checksum gói bàn giao.

Script xác minh: `scripts/verify_v32_manifest.py`, `scripts/prepare_v32_data.py`, `scripts/check_v32_provenance.py`.

## 3. Khối lượng dữ liệu và khóa

| Bảng / phân vùng | Số dòng | Khóa / quy tắc |
|---|---:|---|
| `master/students.csv.gz` | 20.000 | `record_uid` duy nhất; `student_id` cũng duy nhất, chỉ dùng để hiển thị/demo |
| `features/prediction_snapshots.csv.gz` | 57.569 | (`record_uid`, `snapshot_term_index`) duy nhất; HK1–HK3 |
| `labels/snapshot_labels_full_simulation.csv.gz` | 57.569 | Join 1–1 theo (`record_uid`, `snapshot_term_index`) |
| Train 2022–2023 | 34.304 | `TRAIN_2022_23` |
| Validation 2024 | 13.981 | `VALIDATION_2024` |
| Test 2025 | 9.284 | `TEST_2025` |
| `master/term_academics.csv.gz` | 149.196 | (`record_uid`, `term_index`) |
| `master/engagement_weekly.csv.gz` | 1.790.352 | (`record_uid`, `term_index`, `teaching_week`) |
| `labels/snapshot_labels_asof_2026_demo.csv.gz` | 54.609 | Demo view; bao gồm 9.970 nhãn right-censored |

`record_uid` là định danh nội bộ ổn định. Không join theo vị trí dòng (`.iloc`) và không thay `record_uid` bằng `student_id`. Các bảng con phải có foreign key hợp lệ tới bảng sinh viên hoặc bảng tham chiếu tương ứng.

## 4. Hai observation views

### `FULL_SIMULATION`

- Phục vụ **training/evaluation** có kiểm soát, sử dụng nhãn đã quan sát của mô phỏng đầy đủ.
- Chia Train/Validation/Test theo cohort nêu ở mục 3; mọi snapshot của một UID nằm trong cùng split.
- Không dùng nhãn và sự kiện tương lai để giải thích một hồ sơ tại một thời điểm lịch sử.
- Không tune mô hình hoặc threshold dựa trên Test 2025.

### `ASOF_2026_DEMO`

- Mốc quan sát tối đa: **01/10/2026**.
- Có 54.609 snapshot đủ điều kiện theo thời điểm, trong đó 9.970 nhãn demo chưa quan sát hết horizon (`right-censored`).
- **Không** coi giá trị nhãn bị censor là `0` và không dùng tập demo này thay Test full-simulation để tính PR-AUC.
- Data Access phục vụ lịch sử mặc định giới hạn `asof <= 2026-10-01` và chỉ xuất bản các trường đã sẵn có tại thời điểm được chọn.

## 5. Feature contract và chống leakage

- Snapshot gốc có **65 cột** trong `docs/feature_registry.csv`; registry đánh dấu **60** feature được dùng theo as-of.
- Baseline **B5 Logistic Regression** có **58 feature được phép**, đóng băng trong `src/features.py` (`B5_FEATURES`).
- Mỗi bảng `X_*` full simulation có **62 cột**, bao gồm **58 B5 feature + 4 cột metadata**. Không đưa cả 62 cột trực tiếp vào estimator.
- Loại khỏi feature: `record_uid`, các trường định danh/metadata, `split`, `dropout_next_2_terms`, `label_observed`, `censor_reason`, `first_dropout_event_id`, `exit_reason_for_evaluation` và mọi latent hoặc kết quả hậu nghiệm.
- Bắt buộc kiểm tra `feature_source_latest_at <= snapshot_cutoff`.
- `audit_only_hidden_latents/` và `V3_2_STRESS` không thuộc nguồn inference/training chính.
- Chỉ fit imputation, encoding và estimator bằng Train; chọn threshold/calibration theo quy trình không nhìn Test.

Script kiểm chứng: `scripts/validate_v32_data.py`, `scripts/audit_v32_temporal_integrity.py`, `scripts/check_v32_feature_contract.py`.

## 6. Data Access Layer (read-only)

Module: `src/data_access.py`, lớp `V32DataAccess`.

| API | Hợp đồng |
|---|---|
| `get_student(uid, asof=None)` | Tra cứu đúng `record_uid`, trả **16 trường hồ sơ theo `PROFILE_FIELDS`**; trả `None` nếu UID không tồn tại hoặc chưa nhập học tại `asof`. Từ chối `asof` sau `2026-10-01`. |
| `list_snapshots(uid, asof)` | Trả snapshot sẵn có trước hoặc tại `asof`, thứ tự theo học kỳ. |
| `get_snapshot(uid, term, asof)` | Trả đúng một snapshot theo khóa tổng hợp, hoặc `None`. |
| `get_term_history(uid, asof)` | Chỉ trả học kỳ có cả `available_at` và `snapshot_cutoff` không vượt `asof`. |
| `get_weekly_history(uid, term, asof)` | Chỉ trả hoạt động tuần có cả `observed_at` và `available_at` không vượt `asof`. |

Catalog cho phép đúng **11 bảng**: `students`, `term_academics`, `course_attempts`, `engagement_weekly`, `finance_terms`, `student_events`, `enrollment_terms`, `state_transitions`, `support_events`, `term_calendar`, `prediction_snapshots`. Bảng nhãn và hidden latents không được đăng ký trong serving catalog. Các bảng khác dù có trong nguồn không mặc nhiên được phép hiển thị hoặc truy vấn trên UI.

**Đã hoàn thành ở D4:** `get_student()` chỉ trả 16 trường trong `PROFILE_FIELDS`, kiểm tra ngày nhập học theo `asof` và loại trường không cần thiết khỏi payload mặc định. D4 đã được kiểm chứng cho cohort 2022–2025 và kiểm thử hồi quy B1–B5. **Giới hạn này chưa thay thế xác thực hoặc phân quyền theo người dùng**; phần phân quyền UI thuộc Phase 3. Các API lịch sử khác cũng phải được kiểm soát khi tích hợp giao diện.

## 7. Ngoại lệ integrity: giao dịch tài chính

- `master/finance_transactions.csv.gz` nguồn có **303.375 dòng** và **241 `transaction_id` bị lặp**, tổng cộng **482 dòng liên quan**.
- Các cặp này có một giao dịch giá trị **0 VND** và một giao dịch giá trị khác 0; nằm sau mốc demo 01/10/2026.
- Bản dẫn xuất `data/processed/v3_2/finance_transactions_resolved.csv.gz` có **303.134 dòng**, `transaction_id` duy nhất, **tổng tiền không đổi**. Báo cáo đi kèm: `finance_transactions_resolution.json`.
- **Không** sửa dữ liệu nguồn/manifest để làm đẹp kết quả. Kiểm toán nguồn vẫn phải báo ngoại lệ này. Bảng giao dịch gốc không thuộc 11 bảng serving catalog.

Script: `scripts/audit_v32_master_integrity.py` và `scripts/resolve_v32_finance_exception.py`.

## 8. Quy chế học vụ và hạn chế diễn giải

- Đây là dữ liệu synthetic, **không** thay văn bản quy chế trường hoặc phê duyệt ngoại lệ thực tế.
- Theo manifest/plan, có **4.425 học kỳ liên quan Điều 12** mang trạng thái `REVIEW_REQUIRED`; không tự quy đổi thành `APPROVED` hoặc kết luận vi phạm.
- `policy_certification`: `PENDING_EXPLICIT_REGISTRATION_EXCEPTION_REVIEW`. Phải giữ nguyên ý nghĩa chưa xác minh và nêu rõ khi dùng để demo/quyết định nghiên cứu.

## 9. Quy trình nghiệm thu và điều kiện chưa đóng

**Đã kiểm tra trên máy phát triển theo log Phase 1:** manifest/checksum 32 bảng + 2 tài liệu; các khóa snapshot và split; core schema; toàn bộ 17 bảng master (có ngoại lệ giao dịch tài chính đã ghi nhận); thời gian các nguồn; 58-feature B5; Data Access B1–B5; catalog/provenance; D4 whitelist 16 trường và enrollment `asof`; E1–E5 gồm ba snapshot, thôi học sớm, NA, right-censoring và cohort 2025.

**Kết quả E6:** Nghiệm thu kỹ thuật Phase 1 PASS (26/26). Ngoại lệ 241 mã giao dịch trùng ở nguồn vẫn được ghi nhận; bản dẫn xuất đã được xác minh và không thay đổi tổng tiền. Đang chờ rà soát và commit mã nguồn.

**Ngoài phạm vi Phase 1:** Huấn luyện và kiểm định model bundle thuộc Phase 2; xác thực, phân quyền và kiểm thử UI thuộc Phase 3. Chưa tuyên bố các giai đoạn này hoàn thành.

**Gate Phase 1:** 20.000 UID, 57.569 snapshot, không có khóa trùng ở snapshot, không có future features, `FULL_SIMULATION`/`ASOF_2026_DEMO` tách biệt, và bảo vệ dữ liệu phục vụ đã qua smoke tests. Chỉ đóng gate khi các mục còn lại đều đạt.

**Git:** `data/external/`, `data/processed/` và artifact runtime không commit. Chỉ commit scripts, modules và tài liệu sau review.
