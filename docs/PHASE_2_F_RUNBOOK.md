# Phase 2-F: So sánh ML trên Train/Validation

**Bản triển khai:** F v1.0. **Phạm vi:** F1–F5 bắt buộc; F6 ablation là mục nâng cao và đã có báo cáo bổ sung từ Phase E nhưng chưa được coi là F6 PASS.

## Điều kiện đầu vào

- Nhánh `Dung`, Phase A–E đã PASS, `src/features.py`, `src/ml_v32.py` đã có bản vá pandas 3.x.
- Các split `V3_2_MAIN` trên máy cục bộ, các báo cáo Phase E.
- Model V2.2 và artifact B5 đã lưu không được sửa. Script không đọc `TEST_2025` hoặc `ASOF_2026_DEMO`.

## File mới

- `scripts/compare_phase2_F.py`: so sánh **7 cấu hình** đã khóa trong `TRIALS`, 58 feature và chung Train/Validation, không ghi model artifact.
- `scripts/test_phase2_F.py`: 9 unit tests (budget, schema, top-k, selection).
- `docs/PHASE_2_F_RUNBOOK.md`: hướng dẫn này.

## Chạy trên Windows PowerShell

```powershell
cd C:\Code\AI\CMC_Dropout_Prediction
.\.venv\Scripts\python.exe -B -m unittest scripts.test_phase2_F -v
.\.venv\Scripts\python.exe -B scripts/compare_phase2_F.py --mode smoke
.\.venv\Scripts\python.exe -B scripts/compare_phase2_F.py --mode full --confirm-full

Import-Csv reports/ml_v3_2/model_comparison.csv |
  Format-Table trial,hk1_pr_auc,hk1_recall_at_top10,pooled_pr_auc,pooled_brier,train_sec
Get-Content reports/ml_v3_2/model_comparison.md
Get-Content reports/ml_v3_2/model_comparison.json | Select-Object -Last 30
git status -sb
```

**Dự kiến:** `Ran 9 tests / OK`, `PHASE_2_F_SMOKE_PASS`, `PHASE_2_F_FULL_PASS; trials=7; Test accessed=NO`.

**Chính sách an toàn:** Mặc định `--mode smoke`; chạy full chỉ khi có `--confirm-full`. Báo cáo smoke bắt đầu bằng `f_smoke_`, báo cáo full không có tiền tố; không ghi đè báo cáo đã tồn tại. Muốn chạy lại phải lưu bản cũ và thay đường dẫn `--report-dir`; tuyệt đối không sửa `TRIALS` sau khi xem Test. Không huấn luyện ở app, không tải Test.

## Giao thức đánh giá

- **Train:** 34.304 snapshot, 2022–2023; **Validation:** 13.981 snapshot, 2024; cùng 58 feature B5 và target `dropout_next_2_terms`.
- **1 + 2 + 2 + 2 configs:** 1 Logistic Regression gốc; 2 Random Forest giới hạn số cây; 2 XGBoost; 2 LightGBM. Mọi config nằm sẵn trong source và báo cáo.
- **Xử lý categorical:** đúng 5 cột semantic; train-only `most_frequent` + OHE(ignore); numeric median; LR dùng StandardScaler, các mô hình cây không scale numeric.
- **Metric:** HK1 PR-AUC, HK1 Recall@Top10%, pooled PR-AUC, ROC-AUC, Brier, F1@0.5, PR-AUC lift vs prior, train time, predict time, ước lượng serialized size và phân tích HK1/HK2/HK3.
- **Top-k:** `ceil(N * budget)` trong từng nhóm, tie-break theo UID và học kỳ để kết quả ổn định.
- **Không lựa chọn threshold/calibration trong F.** F1@0.5 chỉ là phép so sánh theo ngưỡng cố định mang tính mô tả. G mới khóa threshold và calibration đúng quy trình.

## Diễn giải kết quả

Kết luận từ Validation **không phải** chứng cứ về hiệu quả trên sinh viên thật. Brier và xác suất chưa calibration cần được xử lý trong G. `F_recommendation` là shortlist để thảo luận Phase G, **không** phải quyết định cuối và không cho phép mở Test.

Tiêu chí cho shortlist được khai báo trước: xếp theo HK1 PR-AUC, sau đó HK1 Recall@Top10%, Brier pooled. Chỉ chuyển khỏi LR_B5 khi ứng viên cải thiện ít nhất 0.010 **absolute** ở HK1 PR-AUC, nếu không ưu tiên mô hình đơn giản. G có thể quyết định khác khi phân tích calibration, ổn định và chi phí, nhưng phải ghi lý do dựa trên Validation và khóa quyết định trước Test.

## Báo cáo

- `reports/ml_v3_2/model_comparison.csv`
- `reports/ml_v3_2/model_comparison_by_term.csv`
- `reports/ml_v3_2/model_comparison.json`
- `reports/ml_v3_2/model_comparison.md`

Không cần commit ngay ở bước này. Việc chọn model chính thức, calibration và threshold thuộc **Phase G**.
