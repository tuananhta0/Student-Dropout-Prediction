# 🎓 CMC Academic AI Portal & Student Dropout Prediction (V3.2.1)

Hệ thống **Trợ lý học vụ, tra cứu quy chế đào tạo và cảnh báo sớm nguy cơ thôi học** cho bài tập lớn Trí tuệ nhân tạo. Sử dụng dữ liệu mô phỏng 20.000 sinh viên, mô hình tổ hợp Random Forest + HistGradientBoosting và bộ quy tắc học vụ QĐ 22/2023.

## ✨ Tính năng chính

- **Sinh viên:** hồ sơ học kỳ, bảng điểm, tín chỉ, kịch bản GPA (kể cả học lại), cảnh báo quy chế và tra cứu điều kiện tốt nghiệp.
- **Cố vấn học tập:** thống kê lớp, hồ sơ sinh viên và danh sách nguy cơ còn hiệu lực.
- **Quản trị:** thống kê theo khoa, khóa, hệ đào tạo và tình trạng học tập.

Chatbot tra cứu quy chế theo Điều/Khoản/Trang PDF bằng BM25, hỗ trợ embedding đa ngôn ngữ và Gemini tùy chọn. Các phép tính học vụ và ML chạy trong Python, không gửi hồ sơ cá nhân đến LLM. Ứng dụng xác thực bằng phiên bearer và phân quyền phía máy chủ; chỉ chạy localhost với dữ liệu mô phỏng. Xác suất không phải quyết định buộc thôi học.

**Phiên bản nghiệm thu:** `cmc-research-v1`, dữ liệu MAIN đã hiệu chỉnh. Kết quả máy huấn luyện trên tập kiểm thử mô phỏng: ROC-AUC 0,8293; Average Precision 0,6291; Recall 57,74% ở ngưỡng 0,12. Nhóm cảnh báo sớm chưa đủ điều kiện xử lý hành chính: Recall 45,25%. Xác nhận từ metadata và báo cáo của gói được bàn giao, không sử dụng kết quả baseline cho model mới.

## 🚀 Cài đặt và chạy (PowerShell, Python 3.11)

```powershell
git clone -b Dung https://github.com/tuananhta0/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu đã huấn luyện model mới ở vị trí chuẩn, không cần cài lại. Nếu clone trên máy khác, cài gói model và dataset từ máy huấn luyện theo [hướng dẫn bàn giao](docs/ban_giao.md). Không tự động sử dụng model cũ nếu thiếu gói mới.

```powershell
python -m scripts.create_accounts
$env:CMC_DATA_MODE = 'DEMO'
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. Mặc định trình diễn mốc mô phỏng 03/02/2027; không phải ngày hiện tại ngoài đời. Đăng nhập bằng mật khẩu riêng do lệnh tạo tài khoản in ra một lần, không còn mật khẩu mặc định trùng tên đăng nhập.

## 🔑 Tài khoản demo

| Vai trò | Tài khoản | Phạm vi |
|---|---|---|
| Sinh viên | `BIT250093`, `BIT250006`, `BIT250007` | Hồ sơ bản thân |
| Cố vấn | `GV001` | Lớp ngành IT |
| Cố vấn | `GV002`, `GV003` | Lớp ngành BA, AI tương ứng |
| Quản trị | `ADMIN` | Toàn trường |

Phân công mô phỏng, không phải phân công thực tế của CMC. Xem [tra cứu quy chế và Gemini](docs/tra_cuu_quy_che.md), [bàn giao và bảo mật](docs/ban_giao.md).

## 📂 Cấu trúc thư mục

```text
Student-Dropout-Prediction/
├── core/                 # Quy tắc, chatbot, dữ liệu
│   ├── academic_rules.py
│   ├── chatbot_engine.py
│   ├── data_pipeline.py
│   ├── prediction.py
│   ├── analytics.py
│   ├── retrieval.py
│   ├── intent_router.py
│   ├── security.py
│   └── runtime_config.py
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
python -B -X utf8 -m scripts.benchmark_rag
python -B -X utf8 -m scripts.review_policy
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -m tests.smoke_runtime
python -B -X utf8 -m tests.smoke_academic_rules
python -B -X utf8 -m tests.smoke_security
```

Xem [hướng dẫn dữ liệu](docs/huong_dan_du_lieu.md) và [quy chế, công thức tính toán](docs/quy_che_va_tinh_toan.md).
## 🔬 Hiệu chỉnh và huấn luyện mô hình mới

Xem [hướng dẫn hiệu chỉnh, thực nghiệm và chọn phiên bản mô hình](docs/huan_luyen_va_thuc_nghiem.md). Quy trình mới tái sinh lịch sử học vụ đồng bộ, mở rộng snapshot qua các học kỳ và tách STRESS. Kết quả huấn luyện mới nằm trong `data/models/`, không ghi đè baseline. Không cần tái huấn luyện khi dùng gói phiên bản chính thức đã kiểm tra.
