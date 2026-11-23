# 🎓 CMC Academic AI Portal & Student Dropout Prediction (V3.2.1)

Hệ thống **Trợ lý học vụ, tra cứu quy chế đào tạo và cảnh báo sớm nguy cơ thôi học** cho bài tập lớn Trí tuệ nhân tạo. Sử dụng dữ liệu mô phỏng 20.000 sinh viên, mô hình tổ hợp Random Forest + HistGradientBoosting và bộ quy tắc học vụ QĐ 22/2023.

## ✨ Tính năng chính

- **Sinh viên:** hồ sơ học kỳ, bảng điểm, tín chỉ, kịch bản GPA (kể cả học lại), cảnh báo quy chế và tra cứu điều kiện tốt nghiệp.
- **Cố vấn học tập:** thống kê lớp, hồ sơ sinh viên và danh sách nguy cơ còn hiệu lực.
- **Quản trị:** thống kê theo khoa, khóa, hệ đào tạo và tình trạng học tập.

Chatbot tra cứu quy chế theo Điều/Khoản/Trang PDF bằng BM25, hỗ trợ embedding đa ngôn ngữ và Gemini tùy chọn. Các phép tính học vụ và ML chạy trong Python, không gửi hồ sơ cá nhân đến LLM. Ứng dụng xác thực bằng phiên bearer và phân quyền phía máy chủ; chỉ chạy localhost với dữ liệu mô phỏng. Xác suất không phải quyết định buộc thôi học.

**Phiên bản nghiệm thu:** `cmc-research-v1`, dữ liệu MAIN đã hiệu chỉnh. Kết quả máy huấn luyện trên tập kiểm thử mô phỏng: ROC-AUC 0,8293; Average Precision 0,6291; Recall 57,74% ở ngưỡng 0,12. Nhóm cảnh báo sớm chưa đủ điều kiện xử lý hành chính: Recall 45,25%. Xác nhận từ metadata và báo cáo của gói được bàn giao, không sử dụng kết quả baseline cho model mới.

## 🚀 Cài đặt và chạy (Windows PowerShell, Python 3.11)

**Cài mới từ GitHub:** Clone nhánh `main` (nhánh công bố; `Dung` dành cho phát triển) và tải [gói runtime chính thức `cmc-research-v1.zip`](https://github.com/tuananhta0/Student-Dropout-Prediction/releases/tag/runtime-v1-7178718) trong GitHub Release. Gói gồm dataset MAIN mô phỏng và model đã huấn luyện; **không cần huấn luyện lại**. Thư mục `artifacts_model/` trong Git chỉ là baseline lịch sử, **không phải model nghiệm thu**.

### 1. Clone mã nguồn và cài dependencies

Yêu cầu: Windows PowerShell, Git, Python **3.11** có lệnh `py -3.11`.

```powershell
git clone --branch main --single-branch https://github.com/tuananhta0/Student-Dropout-Prediction.git
cd Student-Dropout-Prediction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

### 2. Tải gói runtime từ GitHub Release, xác minh SHA256 và cài đặt

Chạy tại **thư mục gốc repository** (đã kích hoạt `.venv`). SHA256 dưới đây là checksum cố định của asset đã được công bố trên GitHub, không phải mã hash tự lấy từ file vừa tải.

```powershell
$ErrorActionPreference = 'Stop'
$releaseUrl = 'https://github.com/tuananhta0/Student-Dropout-Prediction/releases/download/runtime-v1-7178718/cmc-research-v1.zip'
$expectedSha = '72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848'
$archive = 'data/cmc-research-v1.zip'

New-Item -ItemType Directory -Force -Path data | Out-Null
Invoke-WebRequest -Uri $releaseUrl -OutFile $archive
$actualSha = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha -ne $expectedSha) { throw 'SHA256 khong khop, dung cai dat!' }

python -B -X utf8 -m scripts.release_model install $archive --sha256 $expectedSha
if ($LASTEXITCODE -ne 0) { throw 'Khong the cai goi runtime' }
```

Gói sẽ được cài vào `data/releases/cmc-research-v1/` và kiểm tra từng file với manifest. Lệnh cài đặt **không ghi đè** nếu thư mục này đã tồn tại; máy đã cài đúng release có thể bỏ qua bước 2. Xem thêm [tài liệu bàn giao](docs/ban_giao.md).

### 3. Kiểm tra nhanh và khởi chạy ứng dụng

```powershell
Remove-Item Env:CMC_DATASET_ROOT, Env:CMC_MODEL_DIR, Env:CMC_ALLOW_BASELINE -ErrorAction SilentlyContinue
$env:CMC_DATA_MODE = 'DEMO'
$env:CMC_RAG_CLOUD = '0'

python -B -X utf8 -m tests.smoke_runtime
if ($LASTEXITCODE -ne 0) { throw 'Smoke runtime that bai' }
python -B -X utf8 -m tests.smoke_academic_rules
if ($LASTEXITCODE -ne 0) { throw 'Smoke hoc vu that bai' }
python -B -X utf8 -m tests.smoke_security
if ($LASTEXITCODE -ne 0) { throw 'Smoke phan quyen that bai' }

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Mở **http://127.0.0.1:8000**. Mặc định trình diễn mốc mô phỏng **03/02/2027**, không phải ngày hiện tại. Mật khẩu demo là mã tài khoản viết hoa (ví dụ `BIT250093` / `BIT250093`, `ADMIN` / `ADMIN`); không cần tạo file tài khoản. Chế độ offline vẫn tra cứu quy chế và tính học vụ, không cần Gemini API key. Muốn bật Gemini, xem [cấu hình RAG và cloud](docs/tra_cuu_quy_che.md); không đưa API key vào Git. Đây là hệ thống demo, chỉ chạy localhost với dữ liệu mô phỏng, không dùng để xác thực danh tính thực tế.

**Kiểm thử hồi quy đầy đủ (tùy chọn):** `python -B -X utf8 -m unittest discover -s tests -v` và `python -B -X utf8 -m scripts.benchmark_rag --all --e2e`. Node.js chỉ cần cho kiểm thử JavaScript, không cần để chạy giao diện đã biên dịch.

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
├── artifacts_model/      # Baseline lịch sử, KHÔNG phải model nghiệm thu
├── app.py                # Điểm khởi chạy FastAPI
├── requirements.txt
└── README.md
```

`dataset_full/`, `data/` và `.venv/` chỉ tồn tại trên máy, không commit vào Git. `artifacts_model/model_metadata.json` và pipeline trong Git là baseline lịch sử (ngưỡng 0,21); không dùng các chỉ số đó cho `cmc-research-v1` (ngưỡng 0,12). Hai thiết kế thực nghiệm khác nhau, không so sánh trực tiếp chỉ số giữa hai artifact. Nghiệm thu bằng release đã kiểm tra SHA256. Không tự bật `CMC_ALLOW_BASELINE` khi thiếu release.

## 🧪 Kiểm thử

Sau khi cài GitHub Release, `scripts.audit_dataset` và `scripts.review_policy` cần trỏ vào thư mục **dataset** bên trong release (khác đường dẫn mặc định `dataset_full/`). Chỉ đặt biến này khi chạy kiểm toán; không cần cho ứng dụng:

```powershell
if (-not (Test-Path 'data/releases/cmc-research-v1/dataset/artifacts/runs/V3_2_MAIN/manifests/generation_manifest.json')) { throw 'Chua cai runtime release' }
$env:CMC_DATASET_ROOT = (Resolve-Path 'data/releases/cmc-research-v1/dataset').Path
python -B -X utf8 -m scripts.audit_dataset
python -B -X utf8 -m scripts.review_policy
Remove-Item Env:CMC_DATASET_ROOT -ErrorAction SilentlyContinue
python -B -X utf8 -m scripts.benchmark_rag
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 -m tests.smoke_runtime
python -B -X utf8 -m tests.smoke_academic_rules
python -B -X utf8 -m tests.smoke_security
```

Xem [hướng dẫn dữ liệu](docs/huong_dan_du_lieu.md) và [quy chế, công thức tính toán](docs/quy_che_va_tinh_toan.md).
## 🔬 Hiệu chỉnh và huấn luyện mô hình mới

Xem [hướng dẫn hiệu chỉnh, thực nghiệm và chọn phiên bản mô hình](docs/huan_luyen_va_thuc_nghiem.md). Quy trình mới tái sinh lịch sử học vụ đồng bộ, mở rộng snapshot qua các học kỳ và tách STRESS. Kết quả huấn luyện mới nằm trong `data/models/`, không ghi đè baseline. Không cần tái huấn luyện khi dùng gói phiên bản chính thức đã kiểm tra.

Nghiệm thu cuối: xem [bàn giao](docs/ban_giao.md), [tra cứu và đánh giá RAG](docs/tra_cuu_quy_che.md) và [giới hạn thực nghiệm ML](docs/ket_qua_nghien_cuu.md). Giao diện chạy CSS/JS cục bộ, mật khẩu demo trùng mã tài khoản. Thống kê chỉ gồm hồ sơ đã có kết quả học kỳ; nguy cơ ML dự báo hai học kỳ tiếp theo từ snapshot.

Kiểm thử hồi quy bổ sung: `python -B -m scripts.benchmark_rag --all --e2e` (70 câu, fixture, không cần cloud) và `node --test tests/portal_async.test.cjs` (Node 22, chỉ cần cho kiểm thử giao diện, không cần khi chạy ứng dụng). Báo cáo mới ghi nguồn commit/dirty và checksum đầu vào; chạy lại sau commit sạch trước khi tạo gói nghiên cứu mới. Xem [bàn giao](docs/ban_giao.md).
