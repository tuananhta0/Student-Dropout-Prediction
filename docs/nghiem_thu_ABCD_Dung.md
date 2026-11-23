# Nghiệm thu A–D sau patch `abb6991` trên nhánh `Dung`

**Base duy nhất:** `abb6991997f631a4140305e51f6872431931796f`. Patch **không sửa hoặc merge `main`**. Không đưa báo cáo, `.env`, key thật, ZIP mô hình hoặc dataset nguồn vào commit. Các kết quả sinh trong `data/` bị `.gitignore` loại trừ.

## A. Sửa RAG và regression

- [x] Nhận diện nghỉ học vì bệnh/đang điều trị, nghỉ một học kỳ theo Điều 21.
- [x] Không trích Điều không liên quan khi hỏi ngày/hạn đóng học phí; báo không có lịch trong quy chế.
- [x] Nhận diện chuyển đến, chuyển qua, chuyển sang, đổi trường theo Điều 22.
- [x] Thêm 40 câu gợi ý và regression HTTP cho biến thể, câu ghép và ca âm.
- [x] Phân biệt hỏi GPA lý thuyết với xem GPA cá nhân; hỏi giới hạn đăng ký học phần với xem bảng điểm.
- [ ] Chấm đúng nghĩa mỗi câu bằng mắt theo PDF nguồn bởi người độc lập.

Lệnh:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Python tests failed' }
& $py -B -X utf8 -m scripts.benchmark_rag --all --e2e --output data/kiem_tra_rag/hoi_quy_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'RAG regression failed' }
& $py -B -X utf8 -m scripts.benchmark_rag --e2e --cases tests/cases/bo_cau_ra_soat_40_de_xuat.json --output data/kiem_tra_rag/bo_40_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'RAG proposed questions failed' }
node --check static/portal.js
node --test tests/portal_async.test.cjs
if ($LASTEXITCODE -ne 0) { throw 'JS test failed' }
```

## B. RAG cloud và người chấm độc lập

- [x] Bảng hỏi *đề xuất* 40 câu; có khả năng tái chạy với `--cases`.
- [x] Có công cụ chấm 4 tiêu chí bằng tay trên cả BM25, Gemini và fallback: `scripts.review_rag_semantics`.
- [ ] Thành viên **không chỉnh retrieval** chuẩn bị thêm câu và giữ kín câu/đáp án trước khi chạy.
- [ ] Người chấm mở PDF gốc trên trình duyệt ở trang được dẫn, điền 0/1 cho `dung_ngu_nghia`, `day_du`, `trich_dan_dung`, `khong_bia_quy_dinh`, ghi lý do mọi câu 0. Đối chiếu Điều 14, 15, 18, 21, 22 và chủ đề ngoài phạm vi.
- [ ] Chạy Gemini thật và lưu báo cáo status; cần Internet, API key và quota hợp lệ. Không thể đánh dấu PASS chỉ từ test mock.

**Một key của nhóm:** Chủ key chạy một lần `& $py -B -m scripts.setup_shared_gemini`. Lệnh hỏi key kín, tạo `.env` được ignore; chuyển `.env` cho thành viên **qua kênh nội bộ**, không đưa vào GitHub Public. Các thành viên không cần tự tạo key. Chế độ đăng nhập demo `username=password` giữ nguyên.

```powershell
Remove-Item Env:CMC_RAG_CLOUD -ErrorAction SilentlyContinue
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --probe
if ($LASTEXITCODE -ne 0) { throw 'Cloud probe failed; inspect status (not secret)' }
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --cases tests/cases/bo_cau_ra_soat_40_de_xuat.json --output data/kiem_tra_rag/bo_40_cloud_ABCD.json
# Cloud benchmark có thể FAIL khi dưới 90% generation OK. Không tự xóa/hợp thức hóa.
```

**Lưu ý:** Cloud probe hiện chỉ kiểm một câu và không đo embeddings. Báo cáo đầy đủ có `so_cau_embedding` và `so_cau_gemini`. Khi Gemini lỗi/quota hạn chế, fallback trực tiếp không được tính là generation PASS. Cũng cần chấm nguồn trích dẫn thủ công.

Sau khi thành viên khác điền đủ bốn cột chấm điểm đã có sẵn trong CSV đối chiếu:

```powershell
& $py -B -X utf8 -m scripts.review_rag_semantics data/kiem_tra_rag/bo_40_cloud_ABCD_doi_chieu.csv --reviewer MA_THANH_VIEN
```

## C. ML, dataset và STRESS

- [x] Đã có script phân tầng theo HK, ngành, khóa, bậc, hệ; calibration/ECE và censoring label theo từng HK (`scripts.review_ml_extended`).
- [x] Script kiểm manifest model/MANIFEST MAIN và phát hiện trùng ID MAIN–STRESS nếu STRESS tồn tại.
- [ ] Cần runtime + dataset hiệu chỉnh thực tế trên máy để chạy; ZIP GitHub Release chỉ có MAIN, **không phải STRESS**.
- [ ] Audit STRESS hoàn chỉnh phải chạy `scripts.audit_dataset --stress-run`, khác với kiểm subset ở script phân tầng.
- [ ] Người báo cáo phải giải thích HK5/HK6 có thể chỉ có nhãn dương trong mẫu quan sát: AP/AUC theo HK không diễn giải được khi thiếu lớp âm. Không kết luận model chính xác 100%.
- [ ] Đọc và đưa vào báo cáo khoa học các bảng subgroup, ECE, bootstrap cùng giới hạn dữ liệu mô phỏng. Không suy ra quan hệ nhân quả hay hiệu suất trên sinh viên CMC thật.

```powershell
& $py -B -X utf8 -m scripts.review_experiment --bootstrap 200 --output data/kiem_tra_ml/xac_minh_model_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'Model report validation failed' }
& $py -B -X utf8 -m scripts.review_ml_extended --bootstrap 200 --output data/kiem_tra_ml/phan_tang_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'ML subgroup analysis failed' }
& $py -B -X utf8 -m scripts.audit_dataset --output data/kiem_tra_du_lieu/MAIN_ABCD
if ($LASTEXITCODE -ne 0) { throw 'MAIN audit failed' }
```

Để nghiệm thu STRESS, cần cài đủ tập hiệu chỉnh có `V3_2_STRESS` cùng cấp với `V3_2_MAIN` và đặt `CMC_DATASET_ROOT` trỏ đến thư mục gốc chứa `artifacts/runs/`:

```powershell
& $py -B -X utf8 -m scripts.review_ml_extended --bootstrap 200 --require-stress --output data/kiem_tra_ml/phan_tang_STRESS_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'STRESS missing or incompatible' }
& $py -B -X utf8 -m scripts.audit_dataset --stress-run data/du_lieu_hieu_chinh/artifacts/runs/V3_2_STRESS --output data/kiem_tra_du_lieu/STRESS_ABCD
if ($LASTEXITCODE -ne 0) { throw 'STRESS full audit failed' }
```

Hai kiểm tra MAIN/STRESS phải dùng cùng bản sinh hiệu chỉnh. Nếu STRESS chưa được cài, báo `NOT_INSTALLED`, không tính PASS.

## D. Tích hợp và bàn giao chỉ trên Dung

- [ ] Tải ZIP runtime thật từ GitHub Release, đối chiếu SHA256 đã công bố, kiểm manifest và mẫu bí mật bằng `scripts.review_public_runtime` (không nạp `.joblib`).
- [ ] Một thành viên **trên máy khác** clone `Dung`, cài Python 3.11 + dependencies và runtime; chạy unit test + 3 smoke test. Không thể hoàn tất thay người đó chỉ từ máy hiện tại.
- [ ] Chạy HTTP Uvicorn thực trên `127.0.0.1`, dùng `scripts.smoke_live_http` để kiểm login, phân quyền, RAG, hồ sơ, hai endpoint scenario và logout.
- [ ] Thao tác trực tiếp trong trình duyệt ở cả ba tài khoản, kiểm responsive và tính đúng nghĩa GPA, thanh toán, bảng điểm, biểu đồ, nguồn PDF. HTTP smoke **không** thay thế test UI.
- [ ] Chạy lại benchmark sau commit với `source_commit` đúng commit mới và `source_dirty=false`.
- [ ] Xác nhận GitHub Actions trên commit mới của `Dung` báo `success`.
- [ ] **Không merge hoặc sửa `main` trong patch này**. Các việc "merge Dung → main", "CI trên main" và "benchmark main sạch" **được hoãn** do yêu cầu không đụng main; chỉ thực hiện sau quyết định riêng của nhóm.

Runtime asset được công bố với tên `cmc-research-v1.zip`, SHA256:

`72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848`

Có thể tải qua trang GitHub Release rồi đặt tại `data/cmc-research-v1.zip`. **Không nhúng gói 66 MB vào patch.** Kiểm tra như sau:

```powershell
$expected = '72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848'
$archive = 'data/cmc-research-v1.zip'
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Runtime archive checksum failed' }
& $py -B -X utf8 -m scripts.review_public_runtime $archive --sha256 $expected --output data/kiem_tra_ban_giao/runtime_ABCD.json
if ($LASTEXITCODE -ne 0) { throw 'Runtime scan failed or needs review' }
```

Fresh-clone trên máy khác: `git clone --branch Dung --single-branch https://github.com/tuananhta0/Student-Dropout-Prediction.git`, `py -3.11 -m venv .venv`, dùng `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, `python -m scripts.release_model install <zip> --sha256 <hash>`, chạy `unittest`, ba smoke test, bật server rồi chạy live HTTP smoke dưới đây. Tuyệt đối không dùng `main` để nghiệm thu patch này.

Chạy server ở cửa sổ PowerShell thứ nhất:

```powershell
$env:CMC_DATA_MODE='DEMO'
$env:CMC_RAG_CLOUD='0'
& $py -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Chạy HTTP smoke ở cửa sổ PowerShell thứ hai (từ thư mục repository):

```powershell
& $py -B -X utf8 -m scripts.smoke_live_http --url http://127.0.0.1:8000
if ($LASTEXITCODE -ne 0) { throw 'Live HTTP tests failed' }
```

Checklist UI thủ công: (1) sinh viên `BIT250093` xem bảng điểm, điểm thiếu khác 0, nguồn trích PDF và GPA scenario; (2) mô phỏng thanh toán, kiểm số nợ trước/sau và ghi chú không suy diễn nhân quả; (3) cố vấn `GV001` chỉ xem đúng lớp được phân quyền; (4) `ADMIN` xem danh sách ngành; (5) chuyển vai trò và đăng xuất không lộ dữ liệu; (6) kích thước 375px, 768px, 1366px; (7) không có lỗi console; (8) ba câu RAG A; (9) kiểm tra `/api/data-view` ghi đúng mốc dữ liệu và demo synthetic.

Không ghi token, key, dữ liệu học sinh vào log nghiệm thu. Trường hợp `POST scenario` bị 422 do điều kiện hồ sơ được ghi riêng để người kiểm tra chọn kịch bản hợp lệ; không đánh dấu nghiệp vụ đã PASS chỉ vì HTTP phản hồi có chủ đích.

## E. Hồ sơ báo cáo, slide và vấn đáp

**Chưa thực hiện** theo yêu cầu. Các báo cáo kiểm thử mới chỉ là chứng cứ kỹ thuật, không tự động được coi là báo cáo học phần đã hoàn thiện.
