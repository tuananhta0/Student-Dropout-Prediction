# 🎓 CMC Academic AI Portal & Student Dropout Prediction (V3.2.1)

Hệ thống **Trợ lý học vụ, tra cứu quy chế đào tạo và cảnh báo sớm nguy cơ thôi học** cho bài tập lớn Trí tuệ nhân tạo. Sử dụng dữ liệu mô phỏng 20.000 sinh viên, mô hình tổ hợp Random Forest + HistGradientBoosting và bộ quy tắc học vụ QĐ 22/2023.

## ✨ Tính năng chính

- **Sinh viên:** hồ sơ học kỳ, bảng điểm, tín chỉ, kịch bản GPA (kể cả học lại), cảnh báo quy chế và tra cứu điều kiện tốt nghiệp.
- **Cố vấn học tập:** thống kê lớp, hồ sơ sinh viên và danh sách nguy cơ còn hiệu lực.
- **Quản trị:** thống kê theo khoa, khóa, hệ đào tạo và tình trạng học tập.

Chatbot hiện dùng quy tắc và từ khóa; chưa phải RAG hoàn chỉnh. Xác thực phân quyền phía máy chủ chưa hoàn thiện: chỉ chạy localhost với dữ liệu mô phỏng. Xác suất không phải quyết định buộc thôi học hoặc bảo đảm kết quả tương lai. Model hiện có ROC-AUC 0,8309; Average Precision 0,5455 trên tập kiểm thử mô phỏng, chưa kiểm chứng với dữ liệu thực.

## 🚀 Cài đặt và chạy (PowerShell, Python 3.11)

```powershell
git clone -b Dung https://github.com/tuananhta0/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.install_dataset "$HOME\Downloads\CMC_Synthetic_Dataset_V3_2_1_CMC_IDS_RESEARCH_CANDIDATE.zip"
```

Nếu đã cài dataset, bổ sung bảng xét tốt nghiệp thay vì cài lại:

```powershell
python -m scripts.install_dataset "$HOME\Downloads\CMC_Synthetic_Dataset_V3_2_1_CMC_IDS_RESEARCH_CANDIDATE.zip" --supplement
```

```powershell
$env:CMC_DATA_MODE = 'DEMO'
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. Mặc định trình diễn tại mốc mô phỏng 03/02/2027 để có kết quả HK3 của khóa 2025. Ngày này không phải ngày hiện tại ngoài đời. Hai chế độ nghiên cứu cũ vẫn dùng được. Không cần huấn luyện lại để chạy demo.

## 🔑 Tài khoản demo

| Vai trò | Tài khoản | Mật khẩu |
|---|---|---|
| Sinh viên | `BIT250093` | `BIT250093` |
| Sinh viên | `BIT250006` | `BIT250006` |
| Sinh viên | `BIT250007` | `BIT250007` |
| Cố vấn | `GV001` | `GV001` |
| Quản trị | `ADMIN` | `ADMIN` |

## 📂 Cấu trúc thư mục

```text
Student-Dropout-Prediction/
├── core/                 # Quy tắc, chatbot, dữ liệu
│   ├── academic_rules.py
│   ├── chatbot_engine.py
│   └── data_pipeline.py
├── scripts/              # Cài dữ liệu, kiểm tra, huấn luyện, quét PDF
├── tests/                # Kiểm thử đơn vị và tích hợp
├── docs/                 # Tài liệu hướng dẫn tiếng Việt
├── templates/            # Giao diện web
├── regulations/          # Văn bản quy chế
├── artifacts_model/      # Model và dữ liệu chấm điểm
├── app.py                # Điểm khởi chạy FastAPI
├── requirements.txt
└── README.md
```

`dataset_full/`, `data/` và `.venv/` chỉ tồn tại trên máy, không commit vào Git.

## 🧪 Kiểm thử

```powershell
python -B -X utf8 -m scripts.audit_dataset
python -B -X utf8 -m scripts.review_policy
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -m tests.smoke_runtime
python -B -X utf8 -m tests.smoke_academic_rules
```

Xem [hướng dẫn dữ liệu](docs/huong_dan_du_lieu.md) và [quy chế, công thức tính toán](docs/quy_che_va_tinh_toan.md). Không chạy huấn luyện khi chưa giải quyết sai lệch làm tròn điểm và các dữ liệu phụ thuộc.
