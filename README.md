# Student Dropout Prediction

Hệ thống phân tích và dự đoán nguy cơ sinh viên bỏ học dựa trên lịch sử học tập, điểm số và mức độ tương tác trên hệ thống LMS bằng các mô hình học máy.

---

## 1. Công nghệ sử dụng

- Ngôn ngữ: Python 3.10+
- Xử lý dữ liệu: Pandas, NumPy
- Trực quan hóa dữ liệu: Matplotlib, Seaborn
- Học máy: Scikit-learn, XGBoost, LightGBM
- Cân bằng dữ liệu: Imbalanced-learn
- Lưu trữ mô hình: Joblib
- Giao diện người dùng: Streamlit

---

## 2. Cấu trúc thư mục

```text
student-dropout-prediction/
├── data/
│   ├── raw/                  # Tệp dữ liệu CSV gốc
│   └── processed/            # Dữ liệu sau khi làm sạch và chuẩn hóa
├── notebooks/                # Notebook phục vụ phân tích và thử nghiệm
│   ├── 01_eda.ipynb          # Phân tích khám phá dữ liệu (EDA)
│   └── 02_model_training.ipynb # Thử nghiệm và đánh giá mô hình học máy
├── models/                   # Tệp mô hình sau huấn luyện (.joblib)
├── reports/                  # Báo cáo và biểu đồ đánh giá
│   ├── figures/              # Lưu trữ ma trận nhầm lẫn, ROC curve
│   └── cleaning_log.csv      # Nhật ký các bước làm sạch dữ liệu
├── src/
│   ├── __init__.py           # Tệp rỗng đánh dấu package
│   ├── preprocess.py         # Hàm tiền xử lý và mã hóa đặc trưng
│   ├── train.py              # Script huấn luyện và lưu mô hình
│   └── evaluate.py           # Tính toán độ đo đánh giá mô hình
├── app.py                    # Giao diện web dự báo nguy cơ sinh viên
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── .gitignore                # Danh sách tệp loại trừ khỏi Git
└── README.md                 # Tài liệu hướng dẫn kỹ thuật
```

---

## 3. Chi tiết nhiệm vụ theo tệp

- data/raw/: Lưu trữ tập dữ liệu gốc định dạng CSV.
- data/processed/: Lưu trữ dữ liệu sau khi làm sạch, mã hóa và chia tập dữ liệu.
- notebooks/01_eda.ipynb: Thống kê mô tả, phân tích tương quan và trực quan hóa phân phối dữ liệu.
- notebooks/02_model_training.ipynb: Thử nghiệm các thuật toán phân loại và so sánh hiệu năng.
- models/: Lưu trữ trọng số của mô hình tối ưu nhất phục vụ suy luận.
- reports/cleaning_log.csv: Ghi nhận các bước tiền xử lý, xử lý giá trị thiếu và biến đổi dữ liệu.
- src/preprocess.py: Đóng gói quy trình làm sạch dữ liệu thành hàm xử lý tự động.
- src/train.py: Tự động hóa quá trình huấn luyện và xuất tệp mô hình ra thư mục models/.
- src/evaluate.py: Tính toán các chỉ số Accuracy, Precision, Recall, F1-score và ROC-AUC.
- app.py: Giao diện tương tác cho phép nhập thông tin học tập và xuất cảnh báo mức độ rủi ro.
- requirements.txt: Khai báo danh sách thư viện phục vụ cài đặt môi trường.

---

## 4. Cài đặt môi trường

Khởi tạo môi trường ảo:

- Windows:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Cài đặt thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

---

## 5. Chuẩn bị dữ liệu

Đặt tệp dữ liệu CSV vào đường dẫn:
```text
data/raw/data.csv
```

---

## 6. Huấn luyện mô hình

Chạy script tiền xử lý và huấn luyện:
```bash
python src/train.py
```

---

## 7. Khởi chạy hệ thống

Khởi chạy ứng dụng Streamlit:
```bash
streamlit run app.py
```

Truy cập hệ thống tại địa chỉ: http://localhost:8501

---

## 8. Nhật ký làm sạch dữ liệu

Mọi bước xử lý dữ liệu bất thường (giá trị thiếu, trùng lặp, sai định dạng, chuẩn hóa thang điểm) được ghi lại tại `reports/cleaning_log.csv`, gồm: bước xử lý, lý do, số dòng trước/sau và số dòng bị ảnh hưởng.
