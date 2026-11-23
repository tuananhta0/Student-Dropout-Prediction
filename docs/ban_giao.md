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
$env:CMC_DATA_MODE = 'DEMO'
python -B -X utf8 -m tests.smoke_runtime
python -B -X utf8 -m tests.smoke_academic_rules
python -B -X utf8 -m tests.smoke_security
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Ứng dụng tìm `data/releases/cmc-research-v1/` trước; nếu chưa cài release thì tìm cặp thư mục huấn luyện `data/du_lieu_hieu_chinh/` và `data/models/cmc-research-v1/`. Nếu thiếu hoặc không khớp manifest, chương trình dừng, không quay về baseline. Cũng có thể chỉ định đồng thời hai biến môi trường tương ứng để kiểm thử. Không dùng bản baseline làm kết quả mới. Cấu hình baseline chỉ dành cho kiểm thử lịch sử, bật rõ ràng `CMC_ALLOW_BASELINE=1` và chọn đúng dữ liệu/cặp artifact.

## Tài khoản và giới hạn bảo mật

Mật khẩu demo là chính mã tài khoản viết hoa: `BIT250093` / `BIT250093`, `GV001` / `GV001`, `ADMIN` / `ADMIN`. Không cần tạo mật khẩu; file `data/accounts.json` cũ và biến `CMC_ACCOUNTS_FILE` không còn được đọc. Khởi động lại server để hủy phiên cũ. Mã tài khoản có thể đoán được, kể cả ADMIN, nên chỉ dùng localhost với dữ liệu mô phỏng, tuyệt đối không dùng dữ liệu thật hoặc triển khai ra mạng.

Phiên bearer có thời hạn 1 giờ, lưu bộ nhớ máy chủ và mất khi server khởi động lại; đăng xuất thu hồi token. Student chỉ truy cập bản thân; GV001 quản lý lớp ngành IT, GV002 ngành BA, GV003 ngành AI (phân công mô phỏng, có thể chỉnh cấu hình trong `core/security.py`); Admin toàn trường. API không chấp nhận vai trò do client tự khai báo. API chỉ trả trường cần thiết cho giao diện, HTML chatbot lọc loại script, link tùy ý, sự kiện và style không tin cậy. Dùng `textContent` cho nội dung nhập. Không gửi thông tin hồ sơ vào Gemini.

Giới hạn: chỉ localhost, một tiến trình Uvicorn; chưa có tài khoản thật, TLS, xác thực của trường, khóa phiên phân tán, kiểm toán vận hành, kiểm thử thâm nhập độc lập. Không đưa lên Internet hoặc dùng dữ liệu thật. Giao diện dùng CSS và JavaScript cục bộ; không coi là môi trường sản xuất. Gói SHA256 xác minh toàn vẹn, không phải chữ ký xác thực nguồn. Thành viên khác cần tự nghiệm thu trên Python 3.11 và phiên bản trong requirements trước khi đánh dấu hoàn tất bàn giao.

## Kiểm tra trước khi chia sẻ

- Đổi API key đã lộ tại Google AI Studio, nhập khóa mới qua `Read-Host -AsSecureString`; không chụp log có khóa. Cơ chế mã tài khoản làm mật khẩu chỉ dành cho dữ liệu mô phỏng; ai có quyền truy cập localhost đều có thể đăng nhập ADMIN.
- Giới hạn đăng nhập 30 yêu cầu/phút theo địa chỉ kết nối, chat 15 yêu cầu/phút theo tài khoản; một câu đang xử lý mỗi phiên. Chỉ chạy `--host 127.0.0.1 --no-proxy-headers`.
- Có thể đặt `CMC_ADVISOR_SCOPE_FILE` trỏ tới JSON tự quản lý, ví dụ `{"GV001":["IT25-09"],"GV002":["BA25-01"],"GV003":["AI25-01"]}`; khi bật, tài khoản không có danh sách được coi là không có lớp. Đây là ví dụ cấu hình, cần chọn mã lớp thực tế trong dữ liệu. Mặc định vẫn dùng phạm vi ngành mô phỏng.
- CSS được biên dịch cục bộ với Tailwind 4.1.10; không còn CDN. Chỉ khi sửa CSS cần Node và `npm install; npm run build:css`. CSP cấm inline script, chỉ cho JavaScript cục bộ; style động vẫn dùng inline. Cloud giới hạn 2 worker, 30 lời gọi SDK/phút và 500/24 giờ cho mỗi tiến trình; trả fallback sau 25 giây. SDK đang chạy có thể chưa dừng ngay; những lời gọi nội bộ SDK không phải bộ đếm hóa đơn.
- Dùng `python -m scripts.package_research <đường-dẫn-archive-nguồn>` để tạo gói `data/cmc-research-full.zip` chứa archive nguồn, bảng MAIN/STRESS hiệu chỉnh, model, manifest và các báo cáo sẵn có. Gói không dùng thay gói runtime `cmc-research-v1.zip`; chia sẻ riêng, không commit.
- Kiểm thử clone sạch bằng gói runtime trên máy thành viên khác; chạy unit test, smoke runtime, smoke academic, smoke security với cloud tắt. CI không cần dữ liệu nặng hoặc API key.

## Nghiệm thu cuối

Trên máy thành viên khác, clone vào thư mục mới bằng `git clone -b Dung --single-branch https://github.com/tuananhta0/Student-Dropout-Prediction.git`, tạo `.venv` bằng Python 3.11, `pip install -r requirements.txt`, cài release theo SHA256 đã nhận riêng. Không sao chép `.venv` hoặc API key. Tắt cloud, chạy unit test, ba chế độ smoke runtime/học vụ và smoke security, kiểm thử cả ba vai trò. Ghi commit, kết quả và máy thử (không ghi thông tin bí mật).

CI: `python -B -m scripts.benchmark_rag --all --e2e --output data/kiem_tra_rag/bao_cao_e2e.json` dùng HTTP TestClient và hồ sơ fixture; không kiểm chứng artifact thực. Báo cáo retrieval và benchmark e2e đều có quality gate.

Sau khi commit sạch, đóng gói nghiên cứu vào đường dẫn mới bằng `python -B -m scripts.package_research <archive-nguon> --output data/cmc-research-full-final.zip`. Manifest ghi commit, Git HEAD tree, trạng thái dirty, checksum các file; `report_origins` ghi nguồn riêng của từng báo cáo. Gói không ghi đè file cũ; chỉ đóng gói báo cáo đã có. Đối chiếu báo cáo đúng phiên bản trước khi chia sẻ. Dùng `Get-FileHash` đối chiếu SHA256 gói.

Chỉ bật Gemini khi cần, nhập khóa qua `Read-Host -AsSecureString`. Thu hồi khóa đã lộ tại Google AI Studio; không dùng khóa thật trong CI.

## Kiểm thử và chứng cứ phiên bản

Sau khi commit sạch, chạy lại benchmark gốc, mở rộng, e2e và cloud (nếu bật), `scripts.review_experiment --bootstrap 200`, audit MAIN/STRESS và `scripts.review_policy` để những báo cáo mới ghi đúng commit. Không cần train lại. Các báo cáo model gốc trong gói release là chứng cứ lịch sử của lần huấn luyện, không bị ghi đè. `report_origins` trong gói nghiên cứu phân biệt `current_clean`, `different_or_dirty`, `unknown` (báo cáo cũ chưa có metadata), `changed_after_recording` (CSV thay đổi sau khi xuất). Đây là thông tin provenance, không phải chữ ký số. Không sửa JSON cũ chỉ để đánh dấu khớp commit. Khi chạy từ ZIP không có Git, commit có thể là null.

Frontend: `node --test tests/portal_async.test.cjs` dùng Node 22, mô phỏng fetch và DOM để kiểm tra timeout kể cả đọc body JSON, đăng nhập lỗi, phản hồi từ phiên cũ và đổi lớp. CI chạy tự động; không cần Node để khởi động ứng dụng. Vẫn kiểm thử thủ công trên trình duyệt thật. Timeout giao diện 35 giây; logout/đổi phiên hủy yêu cầu đang chờ phía client, không đảm bảo server dừng ngay.

Có thể dùng chung API key của nhóm qua kênh riêng nếu chủ khóa chấp thuận; repository là public nên không hardcode, không lưu trong patch, không đưa vào gói dữ liệu hoặc Git. Khóa từng được chia sẻ nên thay trước khi sử dụng ngoài demo, giới hạn quota và theo dõi chi phí; đổi khóa do chủ sở hữu thực hiện. Kiểm thử offline không cần API key.
