# Bàn giao mô hình và dữ liệu

Phiên bản chính thức dùng để nghiệm thu: **cmc-research-v1**, dataset MAIN đã hiệu chỉnh `cmc-policy-2026-01`. Gói phục vụ chạy ứng dụng gồm các bảng MAIN cần thiết, pipeline, metadata, snapshot đã chấm điểm, tóm tắt thống kê, manifest và báo cáo thực nghiệm. Giữ riêng dữ liệu STRESS, archive gốc và toàn bộ hồ sơ tái sinh để tái lập nghiên cứu; gói chạy không bao gồm STRESS. Không gửi gói sang dịch vụ bên ngoài hoặc commit vào Git nếu không có chủ đích. Joblib có thể thực thi mã khi nạp: chỉ cài gói do thành viên tin cậy tạo, đối chiếu SHA256 qua kênh riêng.

## Máy đã huấn luyện model

```powershell
python -B -X utf8 -m scripts.release_model pack
Get-FileHash data/cmc-research-v1.zip -Algorithm SHA256
```

Script kiểm tra toàn bộ bảng MAIN theo manifest, đồng thời kiểm tra model khớp manifest và không ghi đè gói đã tồn tại. Chia sẻ file ZIP và SHA256 riêng với nhóm. Không cần train lại.

## Máy thành viên khác

Clone nhánh Dung, cài dependencies, rồi tải gói và nhập SHA256 do thành viên tạo cung cấp:

```powershell
$sha = Read-Host 'Nhap SHA256 do nguoi tao goi cung cap'
python -B -X utf8 -m scripts.release_model install "$HOME\Downloads\cmc-research-v1.zip" --sha256 $sha
Remove-Item Env:CMC_DATASET_ROOT, Env:CMC_MODEL_DIR, Env:CMC_ALLOW_BASELINE -ErrorAction SilentlyContinue
python -B -X utf8 -m scripts.create_accounts
$env:CMC_DATA_MODE = 'DEMO'
python -B -X utf8 -m tests.smoke_runtime
python -B -X utf8 -m tests.smoke_academic_rules
python -B -X utf8 -m tests.smoke_security
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Ứng dụng tìm `data/releases/cmc-research-v1/` trước; nếu chưa cài release thì tìm cặp thư mục huấn luyện `data/du_lieu_hieu_chinh/` và `data/models/cmc-research-v1/`. Nếu thiếu hoặc không khớp manifest, chương trình dừng, không quay về baseline. Cũng có thể chỉ định đồng thời hai biến môi trường tương ứng để kiểm thử. Không dùng bản baseline làm kết quả mới. Cấu hình baseline chỉ dành cho kiểm thử lịch sử, bật rõ ràng `CMC_ALLOW_BASELINE=1` và chọn đúng dữ liệu/cặp artifact.

## Tài khoản và giới hạn bảo mật

`python -m scripts.create_accounts` tạo mật khẩu ngẫu nhiên riêng cho ADMIN, GV001–GV003, BIT250093, BIT250006, BIT250007. Mật khẩu mới chỉ hiện một lần; lưu riêng. File `data/accounts.json` chỉ chứa salt và hash PBKDF2-SHA256; không gửi kèm ZIP. Chạy lại không đổi mật khẩu cũ. Muốn thêm sinh viên: `python -m scripts.create_accounts --student BIT250008`. Đặt `CMC_ACCOUNTS_FILE` nếu cần file riêng. Không dùng mã sinh viên làm mật khẩu mặc định.

Phiên bearer có thời hạn 1 giờ, lưu bộ nhớ máy chủ và mất khi server khởi động lại; đăng xuất thu hồi token. Student chỉ truy cập bản thân; GV001 quản lý lớp ngành IT, GV002 ngành BA, GV003 ngành AI (phân công mô phỏng, có thể chỉnh cấu hình trong `core/security.py`); Admin toàn trường. API không chấp nhận vai trò do client tự khai báo. API chỉ trả trường cần thiết cho giao diện, HTML chatbot lọc loại script, link tùy ý, sự kiện và style không tin cậy. Dùng `textContent` cho nội dung nhập. Không gửi thông tin hồ sơ vào Gemini.

Giới hạn: chỉ localhost, một tiến trình Uvicorn; chưa có tài khoản thật, TLS, xác thực của trường, khóa phiên phân tán, kiểm toán vận hành, giới hạn tốc độ và kiểm thử thâm nhập độc lập. Không đưa lên Internet hoặc dùng dữ liệu thật. Giao diện vẫn dùng Tailwind CDN; không coi là môi trường sản xuất. Gói SHA256 xác minh toàn vẹn, không phải chữ ký xác thực nguồn. Thành viên khác cần tự nghiệm thu trên Python 3.11 và phiên bản trong requirements trước khi đánh dấu hoàn tất bàn giao.
