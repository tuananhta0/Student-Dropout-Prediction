# Hiệu chỉnh dữ liệu và thực nghiệm dự báo thôi học

## Phạm vi và giả định

Dữ liệu nghiên cứu mới được tái sinh từ mã nguồn, seed, hồ sơ đầu vào và chương trình đào tạo của archive V3.2.1 đã đối chiếu SHA256. Bộ sinh sửa quy tắc điểm chữ (làm tròn half-up đến một chữ số), điều kiện thi bằng số giờ đã lưu, và bổ sung học phần học lại khi khối lượng đăng ký dưới mức tối thiểu nhưng vẫn còn môn nợ. Học kỳ cuối có số tín chỉ còn lại thấp hơn mức tối thiểu là ngoại lệ **giả định mô phỏng**, không phải quy định chính thức đã được chứng nhận. Không nâng điểm tuyển sinh giả: điểm thiếu giữ nguyên; điểm dưới ngưỡng đã đối chiếu được loại khỏi đặc trưng và có lý do riêng. Các ngưỡng tuyển sinh đã thu thập không bao phủ mọi ngành/phương thức.

Khi điểm thay đổi, kế hoạch học tập, tín chỉ, GPA, cảnh báo, quyết định mô phỏng, đặc trưng và nhãn phải được tái sinh đồng bộ. Không sử dụng nhãn cũ để đánh giá lịch sử mới. Các khóa sinh viên và chương trình đào tạo phải khớp dữ liệu gốc; số lượng lượt học, kết quả và sự kiện có thể thay đổi. Bộ gốc vẫn được giữ nguyên. Dữ liệu có bản ghi sau ngày thực tế vì là mô phỏng.

## Chạy lần lượt (PowerShell, tại thư mục gốc, Python 3.11)

```powershell
python -m pip install -r requirements.txt
python -B -X utf8 -m scripts.regenerate_dataset "$HOME\Downloads\CMC_Synthetic_Dataset_V3_2_1_CMC_IDS_RESEARCH_CANDIDATE.zip" --stress
$env:CMC_DATASET_ROOT = "$PWD\data\du_lieu_hieu_chinh"
python -B -X utf8 -m scripts.audit_dataset --source-only --output data/kiem_tra_truoc_huan_luyen
python -B -X utf8 -m scripts.review_policy
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -m scripts.train_model --output data/models/cmc-research-v1
```

Dừng nếu có lỗi. Bộ sinh không ghi đè thư mục đích; khi muốn chạy lại chọn một đích mới (`--destination`). Huấn luyện yêu cầu cả MAIN (20.000 sinh viên) và STRESS (5.000 sinh viên, tách riêng). Không trộn hai tập. Cần đủ dung lượng đĩa cho các bảng lịch sử và thư mục tạm, không xóa bộ gốc. Phải đợi quy trình trên máy kết thúc thành công trước khi bật mô hình mới. Các tệp sinh ra bên trong `data/` không commit.

## Thiết kế đánh giá

- Tạo đặc trưng cho toàn bộ học kỳ khả dụng từ HK1–HK12 theo ngày công bố. Học kỳ chưa có kết quả hoặc hồ sơ đã kết thúc không được tạo bằng cách điền feature giả. Nhãn thôi học trong hai kỳ tiếp theo chỉ dùng khi quan sát được; giữ nhãn thiếu do kiểm duyệt phải. Cửa sổ nhãn được tính lại từ sự kiện.
- Mốc đóng băng: 03/02/2027 (mốc demo). Train khóa 2022–2023 với nhãn có cửa sổ kết thúc không muộn hơn mốc. Validation khóa 2024 với cùng điều kiện nhãn. Test khóa 2025 tại hoặc sau mốc đóng băng, có nhãn quan sát được. Không trùng sinh viên. Train và validation có thể chồng lấn lịch nhưng mọi nhãn dùng để chọn mô hình đều đã quan sát được tại mốc đóng băng. Thực nghiệm rolling-origin riêng dùng nhãn train có horizon **trước** mỗi ngày bắt đầu đánh giá và tham số cố định, không chọn theo kết quả test.
- So sánh Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting và soft voting. Chọn tham số/trọng số và mô hình theo Average Precision trên validation. Không bắt buộc Ensemble thắng. Model chính bỏ feature đủ điều kiện xử lý hành chính; so sánh với phiên bản có feature này, và đánh giá riêng nhóm chưa đủ điều kiện xử lý. Ngưỡng chọn theo F2, giới hạn danh sách cảnh báo không quá 30% validation (giả định ngân sách can thiệp, không phải quy chế).
- Báo cáo Average Precision, ROC-AUC, Brier, precision, recall, F1, confusion matrix, prevalence, Recall@Top10%, phân nhóm học kỳ, reliability bins, fairness theo giới tính/khóa/ngành/bậc và STRESS. Fairness chỉ là kiểm tra sai khác trên dữ liệu mô phỏng, không chứng nhận công bằng. Nhóm ít mẫu/thiếu lớp nhãn không kết luận. Calibration là **đánh giá** độ tin cậy, chưa áp dụng phương pháp hiệu chuẩn xác suất riêng. Permutation importance trên validation, không phải SHAP hay nguyên nhân cá nhân.
- Phạm vi phục vụ: chỉ các học kỳ đủ mẫu và đủ hai lớp khi train. Không dùng dự báo được tạo hồi cứu trước ngày đóng băng của model, hết horizon hoặc cũ hơn hồ sơ học vụ để can thiệp. Xác suất của các kỳ ngoài miền hỗ trợ không được hiển thị như nguy cơ hiện tại. Báo cáo độ bao phủ thay vì coi người thiếu dự báo là nguy cơ thấp.

## Kiểm tra phiên bản mới

Mô hình, metadata, bảng chấm điểm và báo cáo được xuất đồng bộ vào thư mục phiên bản mới; dữ liệu và model phải có manifest tương ứng. Chương trình kiểm tra lại file đã lưu và audit trước khi xuất. Baseline `artifacts_model/` vẫn giữ nguyên. Xem `data/models/cmc-research-v1/bao_cao_thuc_nghiem.json`.

```powershell
$env:CMC_MODEL_DIR = "$PWD\data\models\cmc-research-v1"
foreach ($mode in @('DEMO','ASOF_2026_DEMO','FULL_SIMULATION')) {
    $env:CMC_DATA_MODE = $mode
    python -B -X utf8 -m tests.smoke_runtime
    if ($LASTEXITCODE -ne 0) { throw 'Lỗi kiểm tra mô hình' }
    python -B -X utf8 -m tests.smoke_academic_rules
    if ($LASTEXITCODE -ne 0) { throw 'Lỗi kiểm tra học vụ' }
}
$env:CMC_DATA_MODE = 'DEMO'
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Thử kịch bản nhiều học phần, môn mới và What-If thanh toán học phí (chỉ khi model mới có dự báo hợp lệ). Chuẩn hóa feature từ snapshot ML thật; giả định thanh toán trước khi chốt kỳ cập nhật đồng thời công nợ, tỷ lệ nợ và dữ liệu quá hạn liên quan. Không giả định sửa chuyên cần/GPA/điểm như tác động nhân quả; xác suất có thể tăng hoặc giảm. GPA là công cụ kịch bản toán học, không phải dự báo hồi quy. Không tự chọn kết quả tốt nhất để ghi vào README trước khi xem báo cáo tập đầy đủ.

Kiểm thử baseline chỉ dành cho đối chiếu lịch sử. Phải đặt `CMC_ALLOW_BASELINE=1` và chọn đúng `dataset_full` cùng `artifacts_model`; không tự lùi về baseline. Sau kiểm thử cần bỏ cờ và chọn lại cặp model chính thức.

Xem [bàn giao mô hình và dữ liệu](ban_giao.md) để tạo gói ZIP và xem quy ước đăng nhập demo; [tra cứu quy chế](tra_cuu_quy_che.md) để bật Gemini tùy chọn. Chỉ dùng localhost với dữ liệu mô phỏng.

## Đánh giá bổ sung không huấn luyện lại

`python -B -X utf8 -m scripts.review_experiment` kiểm tra cặp model/dataset, dựng lại đúng tập test đã khóa, đối chiếu AP và số mẫu, tổng hợp confusion matrix, ECE, fairness và kết quả theo học kỳ. Chỉ chọn snapshot quan sát đầu tiên mỗi sinh viên cho phép đo cấp sinh viên. Các khóa cũ là theo dõi sau mốc đóng băng, không phải tập test độc lập. Xem `data/kiem_tra_ml/bao_cao.json`.

Audit STRESS chạy trên thư mục `data/du_lieu_hieu_chinh/artifacts/runs/V3_2_STRESS` bằng `python -m scripts.audit_dataset --stress-run data/du_lieu_hieu_chinh/artifacts/runs/V3_2_STRESS --output data/kiem_tra_stress`. Không dùng STRESS cho huấn luyện hoặc runtime.
