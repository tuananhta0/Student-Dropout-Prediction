# 🎓 CMC Academic AI Portal & Student Dropout Prediction (V3.2.1)

Hệ thống **Trợ lý AI Học vụ, Tra cứu Quy chế Đào tạo (PDF) và Dự báo Sớm Rủi ro Thôi học** dành cho Trường Đại học CMC. Dự án kết hợp mô hình Học máy Tổ hợp (`Random Forest` + `HistGradientBoosting`) với hệ thống suy luận Quy chế QĐ 22/2023 và kiểm soát phân quyền 3 cấp (**Sinh viên – Cố vấn học tập – Quản trị viên**).

---

## ✨ Tính năng chính

1. **Dành cho Sinh viên (`STUDENT`)**:
   - Xem bảng điểm tổng hợp qua các kỳ, biểu đồ cột GPA (thang 4.0), tiến độ tích lũy tín chỉ và tỷ lệ chuyên cần.
   - Xem cảnh báo học tập tự động và giả lập kịch bản cải thiện nguy cơ thôi học (**What-If Simulation**).
   - Hỏi đáp trực tiếp Quy chế đào tạo (QĐ 22/2023) và giải các tình huống tính điểm liệt, học lại, cấm thi, bảo lưu, xét tốt nghiệp.
2. **Dành cho Cố vấn học tập / Quản lý Khoa (`ADVISOR`)**:
   - Quản lý danh sách lớp hành chính, xem thống kê sĩ số, GPA trung bình, số tín chỉ nợ và danh sách sinh viên có nguy cơ thôi học cao.
   - Phân tích so sánh lớp mạnh/yếu, phổ điểm và các môn học có tỷ lệ trượt cao.
3. **Dành cho Quản trị viên toàn trường (`ADMIN`)**:
   - Thống kê vĩ mô toàn bộ **19 Khoa/Ngành đào tạo**, **2.258 Lớp** và **20.000 Sinh viên** theo Khóa (2022–2025), Hệ đào tạo (`GM`/`SM`), Nơi ở và Phương thức tuyển sinh.
   - Theo dõi công nợ học phí toàn trường và các chỉ số đánh giá mô hình AI (`ROC-AUC = 0.8312`, `PR-AUC = 0.5460`).

---

## 🚀 Hướng dẫn Cài đặt & Chạy Web (Chạy ngay không cần train lại)

Kho lưu trữ đã đóng gói sẵn mô hình (`dropout_pipeline.joblib`) và dữ liệu chấm điểm rút gọn trong thư mục `artifacts_model/`, giúp bạn tải về và khởi chạy ngay lập tức.

### 1. Tải dự án về máy (Clone Repository)
```bash
git clone [https://github.com/tuananhta0/Student-Dropout-Prediction.git](https://github.com/tuananhta0/Student-Dropout-Prediction.git)
cd Student-Dropout-Prediction
```

### 2. Tạo môi trường ảo và cài đặt thư viện

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Trên macOS / Linux (Terminal):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Khởi chạy máy chủ Web
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Sau khi Terminal hiện thông báo `Application startup complete`, mở trình duyệt và truy cập:
👉 **http://127.0.0.1:8000**

---

## 🔑 Tài khoản Demo đăng nhập nhanh

Tại màn hình đăng nhập, bạn có thể bấm chọn trực tiếp 1 trong 3 nút tài khoản mẫu hoặc tự nhập:

| Phân quyền | Tài khoản (`Username`) | Mật khẩu (`Password`) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **Sinh viên (Có cảnh báo)** | `BIT250007` | `BIT250007` | SV ngành CNTT (Lớp `IT25-18`), có môn trượt & nợ tín chỉ |
| **Sinh viên (Học lực Giỏi)** | `BIT250006` | `BIT250006` | SV ngành CNTT (Lớp `IT25-16`), GPA cao, học tập an toàn |
| **Cố vấn học tập** | `GV001` | `GV001` | Xem báo cáo các Lớp học, Khoa & Hồ sơ từng Sinh viên |
| **Quản trị viên (Admin)** | `ADMIN` | `ADMIN` | Xem thống kê vĩ mô 19 Khoa/Ngành và toàn trường |

---

## 📂 Cấu trúc thư mục

```text
Student-Dropout-Prediction/
├── app.py                              # Máy chủ FastAPI & API phân quyền 3 cấp
├── chatbot_engine.py                   # Lõi NLP, giải bẫy quy chế, thống kê & mô phỏng What-If
├── train_model.py                      # Script huấn luyện mô hình Ensemble & xuất chỉ mục
├── regulations_scanner.py              # Công cụ quét văn bản từ PDF quy chế
├── requirements.txt                    # Danh sách thư viện Python
├── templates/
│   └── index.html                      # Giao diện Web (Login, Chatbot, Bảng điểm & Biểu đồ)
├── regulations/                        # Văn bản PDF Quy chế đào tạo & Thôi học ĐH CMC
└── artifacts_model/                    # Mô hình .joblib, metadata & dữ liệu đã chấm điểm
```