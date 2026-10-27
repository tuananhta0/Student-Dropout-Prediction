# BÁO CÁO DATA PREPROCESSING · CMC V2.2

**Phạm vi:** Bài toán phân loại Dropout trong hai học kỳ sau khi đã hoàn thành HK1, sử dụng dữ liệu **synthetic**, không phải hồ sơ sinh viên thực tế.
**Kết quả kiểm tra:** PASS · 52 checks · 0 lỗi.

## 1. Tách tập theo thời gian

| Tập | Khóa | Số dòng | Dropout dương (tỷ lệ) |
|---|---|---:|---:|
| train | 2022–2023 | 5,892 | 161 (2.73%) |
| validation | 2024 | 2,391 | 62 (2.59%) |
| test | 2025 | 1,604 | 49 (3.05%) |

Tổng: 9.887 quan sát có nhãn, 272 dương. 113 bản ghi thiếu nhãn quan sát được **không** được đưa vào ML. Không shuffle giữa các khóa và không lấy thông tin nhãn để tạo feature.

## 2. Chính sách đặc trưng và chống leakage

- 18 cột HK1 được cho phép từ `feature_manifest_hk1.csv`; có **23** cột kỹ thuật/tổng hợp trong dataframe sau ghép và engineering (bao gồm ID, nhãn, split); tạo **48** đặc trưng sau mã hóa.
- Không dùng ID, tình trạng học tập sau HK1, thời điểm thoát học, GPA và tín chỉ cập nhật hậu nghiệm, nhãn hoặc biến của generator làm đầu vào X.
- Toàn bộ imputation, thống kê chuẩn hóa và danh mục One-Hot **chỉ fit trên Train**. Validation/Test chỉ `.transform()`.
- Lưu hai pipeline: `preprocessor_linear.joblib` (median + StandardScaler + OHE), và `preprocessor_tree.joblib` (median + OHE, không scale).
- Không resample/SMOTE và không dùng Validation/Test để cân bằng dữ liệu. Nếu cần `class_weight`, chọn trong quá trình train, chỉ dựa vào Train.
- Với `admission_score`, trích mẫu `_30`, `_40`, `_80`, `_150`, `_1200` từ phương thức; tính `admission_score_ratio = score/maximum`. Giữ cờ `admission_score_is_missing`, trong đó tuyển thẳng là thiếu có cấu trúc, **không đồng nghĩa điểm 0**.
- Chuẩn hóa phương thức bỏ hậu tố thang, ví dụ `CMC100_THPT_30` và `CMC100_THPT_40` cùng `CMC100_THPT`. Không ép giả định hai thang có độ khó hay ý nghĩa thống kê giống hệt nhau.
- Tạo cờ thiếu cho LMS; giữ `first_sem_warning_flag` ở 0/1. Giữ `cohort_year` từ whitelist, khuyến nghị kiểm tra ablation không có cohort ở giai đoạn Model Training.

## 3. Thiếu dữ liệu trước khi xử lý

| Split | Cột | Thiếu | Tỷ lệ |
|---|---|---:|---:|
| train | `admission_score` | 309 | 5.24% |
| train | `living_status` | 162 | 2.75% |
| train | `first_sem_lms_logins_4weeks` | 76 | 1.29% |
| validation | `admission_score` | 126 | 5.27% |
| validation | `living_status` | 60 | 2.51% |
| validation | `first_sem_lms_logins_4weeks` | 32 | 1.34% |
| test | `admission_score` | 59 | 3.68% |
| test | `living_status` | 36 | 2.24% |
| test | `first_sem_lms_logins_4weeks` | 13 | 0.81% |

Sau preprocessing, ma trận X của cả ba tập đều không chứa NaN hoặc vô cực. Missing không bị điền trước khi chia tập.

## 4. Data drift và unseen categories

| Split | Feature | Số trường hợp chưa xuất hiện ở Train | Tỷ lệ | Ví dụ |
|---|---|---:|---:|---|
| test | `admission_method_family` | 284 | 17.71% | `CMC401_CMCTEST` |

Chính sách `handle_unknown="ignore"` tránh crash nhưng các category mới có vector one-hot toàn 0. Đây là **hạn chế tổng quát hóa**, không phải bằng chứng mô hình đã học được hiệu ứng của phương thức mới.

Phương thức gốc 2025 có `_40` không xuất hiện ở Train, nhưng đã được gom cùng họ tuyển sinh với `_30`, đồng thời điểm được biểu diễn thành tỷ lệ theo thang riêng. `CMC401_CMCTEST` vẫn hoàn toàn mới so với Train và cần đánh giá riêng.

## 5. File bàn giao

- `data/model_ready/X_{train,validation,test}_linear.csv`: ma trận mã hóa + scale, sẵn sàng cho Logistic Regression, SVM, KNN.
- `data/model_ready/X_{train,validation,test}_tree.csv`: ma trận mã hóa không scale, dùng Random Forest, Decision Tree, boosting.
- `data/model_ready/y_{train,validation,test}.csv`: nhãn tách riêng, không xuất hiện trong X.
- `data/model_ready/row_keys_{train,validation,test}.csv`: ID và thông tin khóa để truy ngược, **không đưa vào model**.
- `data/model_ready/semantic_features_{train,validation,test}.csv`: input có ý nghĩa nghiệp vụ để inference hoặc diễn giải; cần transform qua preprocessor tương ứng.
- `artifacts/preprocessor_{linear,tree}.joblib`: pipeline đã fit Train; phải dùng cùng phiên bản thư viện và chỉ nạp artifact tin cậy.
- `artifacts/encoded_feature_names.csv`, `feature_lineage.csv`, `preprocessing_schema.json`: schema và provenance.
- `reports/validation_results.json`, `data_drift_report.csv`, `missingness_report.csv`, `output_fingerprints.json`: báo cáo kiểm định và hash.

## 6. Giới hạn nghiên cứu

1. Dữ liệu 100% synthetic. Độ chính xác sau huấn luyện chỉ phản ánh quy luật do generator tạo, không được diễn giải thành hiệu quả dự đoán sinh viên thật tại CMC.
2. Một số chương trình chưa tồn tại trong lịch sử của từng khóa, `credits_required` là mapping mô phỏng, MSSV không phải định dạng chính thức; theo giả định đã công khai từ bước Cleaning.
3. Nhãn dương rất ít (2,75%): khi huấn luyện cần tập trung PR-AUC, Recall, Precision, F1, confusion matrix, calibration và khoảng tin cậy; không chỉ accuracy.
4. Tập 2025 có drift rõ rệt, nhất là phương thức CMC-TEST; cần ghi kết quả riêng theo khóa, ngành, phương thức. Không điều chỉnh pipeline theo kết quả Test.
5. Nhãn `dropout_next_2_terms` cần được mô tả là thôi học quan sát trong HK2–HK3 sau thời điểm HK1, không dùng nhầm làm nhãn `academic_status` tại ngày chốt.

## 7. Kiểm chứng tạo lại

Lệnh: `python scripts/run_preprocessing.py` và `python scripts/validate_outputs.py` từ thư mục gói.
Nguồn cleaned SHA-256: `d1e08eb0c252db7fadd37f46b438e70c8f0948afa0c1b3db33a054eca2e78436`.
Phiên bản dùng để tạo artifact: Python 3.13.5, scikit-learn 1.8.0, pandas 2.2.3, numpy 2.3.5.

**Kết luận:** PASS kiểm tra dữ liệu và pipeline; đủ điều kiện bàn giao sang **Model Training trên dữ liệu synthetic**, chưa đủ để triển khai cảnh báo học vụ thực tế.

## 8. Kiểm định độc lập và kiểm thử tái lập

- `python scripts/validate_outputs.py`: **106/106 checks PASS**, đối chiếu ngược dữ liệu X đã xuất với kết quả `.transform()` từ joblib và ước lượng thống kê fit trên Train.
- `python scripts/test_preprocessing_rules.py`: **5/5 unit tests PASS**, gồm xử lý `_30/_40`, tuyển thẳng thiếu điểm, chặn điểm vượt thang, chặn mã phương thức sai và thử unseen categories.
- Chạy `python scripts/run_preprocessing.py` hai lần từ cùng nguồn: **tất cả hash SHA-256 của ma trận model-ready và artifacts khớp tuyệt đối**.
- Chạy `python scripts/example_model_training.py` trên Train/Validation: quá trình `.fit()`/`.predict_proba()` hoạt động; đây chỉ là kiểm thử tương thích, không phát hành mô hình nào trong gói Preprocessing.
- Chạy `scripts/transform_for_inference.py` trên một bản ghi nguồn độc lập: dữ liệu xuất khớp ma trận của cùng bản ghi đã bàn giao.

## 9. Rủi ro tiếp tục theo dõi khi Model Training

- `living_status` và `scholarship_percent` có thể gián tiếp phản ánh hoàn cảnh kinh tế; cần kiểm tra tác động bất bình đẳng, độ ổn định theo phân nhóm và ngưỡng can thiệp trước khi đưa hệ thống vào sử dụng thực tế.
- `first_sem_warning_flag` được suy ra từ kết quả học vụ HK1. Hợp lệ với điểm cutoff, nhưng thông tin có thể trùng với GPA/tín chỉ; nên thực hiện ablation bỏ cờ cảnh báo để xác định đóng góp thực sự.
- Đánh giá riêng nhóm CMC-TEST 2025 bị unseen, hạn chế diễn giải dự đoán ở phương thức chưa từng có trong Train.
- Không thể dùng kết quả của dữ liệu synthetic để kết luận mô hình dự đoán chính xác tỷ lệ thôi học hay nguyên nhân thôi học của sinh viên thật.
