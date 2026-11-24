# Nghiệm thu A–D sau sửa lỗi RAG wave 2 trên `Dung`

Base duy nhất: `eb16d0def2122ce743a5e962d89e833ccf1960be`.

Tài liệu này phân biệt **mã đã sửa**, **test chạy được trong CI** và **các nghiệm thu đòi hỏi runtime/Gemini/người kiểm tra độc lập**. Không gộp những khái niệm này làm một.

> **Không merge/commit/push lên `main`.** Bản này phục vụ `Dung`. Dữ liệu là mô phỏng. Không đưa Gemini key, `.env` hoặc thông tin sinh viên thật lên GitHub. Nhóm vẫn dùng một Gemini key chung trong `.env` nội bộ; tài khoản demo giữ nguyên quy ước `username=password`.

## A. RAG: sửa và chống lỗi hồi quy

Đã sửa ở `core/retrieval.py` và `core/intent_router.py`:

1. “Một tín chỉ bao nhiêu giờ” → Điều 3 khoản 1, không nhầm đơn giá học phí. “Một tiết bao lâu” → Điều 5.
2. “Học phí có giảm cho sinh viên giỏi” → thông báo tài liệu đang dùng **không có chính sách miễn giảm cụ thể**, không tự dẫn Điều 20.
3. Chuyển ngành bằng hai mã viết hoa (`IT` → `BA`) và biến thể “đổi”, “chuyển”, không nhầm truy vấn hồ sơ.
4. Tạm dừng học vì tai nạn, phẫu thuật, điều trị dài ngày → Điều 21.
5. Quy đổi điểm `8,5`/`7,5` ra thang 4 → Điều 17; làm tròn điểm chữ → Điều 13.
6. Hỏi **quy định số lần cảnh báo** → Điều 18; hỏi **lịch sử cảnh báo của em** → hồ sơ cá nhân có phân quyền.
7. Câu hỏi ghép về bảo lưu/chuyển ngành giữ đủ Điều 21 và 22; câu ghép với miễn giảm học phí không bịa chính sách ngoài PDF.
8. HTTP Scenario: chỉ đánh dấu PASS khi mã trạng thái `200` và nội dung `gpa_before/gpa_after` hoặc `before/after/difference_pp` hợp lệ. `422` là lỗi/kịch bản chưa đủ điều kiện, **không được tính PASS**.

Test mới: `tests/test_rag_wave2.py` (regression HTTP và kiểm tra ca âm), `tests/cases/bo_cau_wave2_de_xuat.json` (30 câu *đề xuất do người sửa thuật toán soạn*, **không phải tập mù độc lập**).

Chạy trên nhánh Dung với Python 3.11 `.venv`:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE) { throw 'Python tests failed' }
& $py -B -X utf8 -m scripts.benchmark_rag --all --e2e --output data/kiem_tra_rag/wave2_baseline.json
if ($LASTEXITCODE) { throw 'RAG baseline failed' }
& $py -B -X utf8 -m scripts.benchmark_rag --e2e --cases tests/cases/bo_cau_wave2_de_xuat.json --output data/kiem_tra_rag/wave2_30.json
if ($LASTEXITCODE) { throw 'RAG wave2 failed' }
node --test tests/portal_async.test.cjs
if ($LASTEXITCODE) { throw 'JavaScript tests failed' }
```

## B. Gemini thật và kiểm tra ngữ nghĩa độc lập

**Không thể coi test fixture là kiểm chứng LLM.** Nhóm dùng một API key chung trong `.env` bỏ qua bởi Git. Khi có kết nối và quota hợp lệ:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --probe
if ($LASTEXITCODE) { throw 'Gemini probe failed' }
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --cases tests/cases/bo_cau_wave2_de_xuat.json --output data/kiem_tra_rag/wave2_cloud.json
if ($LASTEXITCODE) { throw 'Gemini benchmark gate failed' }
```

Đọc file `wave2_cloud.json`: `so_cau_gemini`, `so_cau_embedding`, `ty_le_gemini_thanh_cong`, `ty_le_fallback`, `trang_thai_generation`, `cloud_quality_gate`. Không coi fallback BM25 là Gemini generation thành công. Probe 1 câu không đủ để đánh giá cả bộ 30 câu.

**Bộ câu đánh giá mù:** Một thành viên **không tham gia sửa RAG** tự soạn 30–50 câu mới và giữ kín trước khi thử. Định dạng JSON: các cặp `["Câu hỏi", 18]` hoặc `["Câu ngoài phạm vi", null]`. Sau khi chạy `--cases`, thành viên đó phải mở trực tiếp PDF gốc và chấm đủ 4 cột của `_doi_chieu.csv`:

- `dung_ngu_nghia`: trả lời đúng điều khoản và ý nghĩa.
- `day_du`: đầy đủ điều kiện và ngoại lệ liên quan.
- `trich_dan_dung`: có Điều/Khoản và trang PDF phù hợp.
- `khong_bia_quy_dinh`: không bịa chi tiết tài liệu không có.

Chấm `0` phải ghi lý do. Sau đó:

```powershell
& $py -B -X utf8 -m scripts.review_rag_semantics data/kiem_tra_rag/wave2_cloud_doi_chieu.csv --reviewer MA_THANH_VIEN
```

Điểm 100% retrieval **không** phải điểm 100% ngữ nghĩa. Chỉ người kiểm tra độc lập mới được xác nhận độc lập.

## C. Machine Learning, censoring và MAIN/STRESS

Những script phân tầng đã có sẵn: `scripts.review_ml_extended`, `scripts.review_experiment`, `scripts.audit_dataset`.

- Model nghiên cứu `cmc-research-v1` cần dataset MAIN tương ứng; `scripts.review_ml_extended` kiểm manifest và tách Train/Val/Test. Không dùng `artifacts_model` baseline lịch sử để kết luận về model nghiên cứu.
- Phân tích theo học kỳ, chuyên ngành, khóa, bậc/hệ đào tạo, Calibration/ECE và Bootstrap theo sinh viên.
- Với HK5/HK6 chỉ có một lớp nhãn: báo số lượng, tỷ lệ censor, **không báo AUC/AP** cho nhóm một lớp.
- STRESS là bộ độc lập, phải được cài **cùng bản tái sinh hiệu chỉnh** với MAIN. Trạng thái `NOT_INSTALLED` không phải PASS.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.review_experiment --bootstrap 200 --output data/kiem_tra_ml/wave2_experiment.json
if ($LASTEXITCODE) { throw 'Experiment review failed' }
& $py -B -X utf8 -m scripts.review_ml_extended --bootstrap 200 --output data/kiem_tra_ml/wave2_subgroups.json
if ($LASTEXITCODE) { throw 'Extended ML review failed' }
& $py -B -X utf8 -m scripts.audit_dataset --output data/kiem_tra_du_lieu/wave2_MAIN
if ($LASTEXITCODE) { throw 'MAIN audit failed' }
```

Để **bắt buộc** xác nhận STRESS, đảm bảo `V3_2_STRESS` cùng cấp với `V3_2_MAIN`:

```powershell
& $py -B -X utf8 -m scripts.review_ml_extended --require-stress --bootstrap 200 --output data/kiem_tra_ml/wave2_STRESS.json
if ($LASTEXITCODE) { throw 'STRESS missing or invalid' }
& $py -B -X utf8 -m scripts.audit_dataset --stress-run data/du_lieu_hieu_chinh/artifacts/runs/V3_2_STRESS --output data/kiem_tra_du_lieu/wave2_STRESS
if ($LASTEXITCODE) { throw 'Full STRESS audit failed' }
```

Đối chiếu chéo `source_commit`, `source_dirty`, `dataset_manifest_sha256`, `model_manifest_sha256`, sample counts. Kết quả là từ **dữ liệu mô phỏng**, không chứng minh áp dụng trên hồ sơ thật hoặc quan hệ nhân quả. Thêm kết quả nghiệm thu đã chạy vào báo cáo nghiên cứu, không chế số liệu.

## D. Runtime Release, HTTP, trình duyệt và máy thành viên khác

Runtime được phát hành tại [release `runtime-v1-7178718`](https://github.com/tuananhta0/Student-Dropout-Prediction/releases/tag/runtime-v1-7178718). SHA256 đã công bố của `cmc-research-v1.zip`:

`72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848`

Tải asset từ GitHub Release về `data/cmc-research-v1.zip` trên máy nhóm và chạy:

```powershell
$zip = 'data/cmc-research-v1.zip'
$sha = '72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848'
if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw 'Release SHA256 mismatch' }
& $py -B -X utf8 -m scripts.review_public_runtime $zip --sha256 $sha
if ($LASTEXITCODE) { throw 'Runtime review failed' }
```

Cài runtime nếu chưa cài (không tự ghi đè thư mục):

```powershell
& $py -B -X utf8 -m scripts.release_model install $zip --sha256 $sha
```

Chạy Uvicorn tại máy cục bộ trong **PowerShell thứ nhất**:

```powershell
$env:CMC_DATA_MODE = 'DEMO'
& $py -B -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Trong **PowerShell thứ hai**, kiểm HTTP thật (chọn tài khoản sinh viên đang học, có dữ liệu học phần, có công nợ và dự báo VALID nếu tài khoản mẫu không đủ điều kiện):

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.smoke_live_http --url http://127.0.0.1:8000 --student BIT250093 --payment 1000 --output data/kiem_tra_ban_giao/wave2_http.json
if ($LASTEXITCODE) { throw 'HTTP business smoke failed; read output and choose eligible demo student if needed' }
```

**Bắt buộc nghiệm thu trực tiếp trong trình duyệt**: đăng nhập `ADMIN`, `GV001`, sinh viên demo; kiểm dashboard, biểu đồ, danh sách lớp, quyền truy cập chéo, bảng điểm, GPA Scenario, Payment Scenario, chatbot và liên kết PDF; kiểm cả cửa sổ thu hẹp 375 px và desktop. Không chấp nhận kiểm HTTP thay cho kiểm UI. Đối với Payment Scenario, yêu cầu JSON gồm xác suất trước/sau hợp lệ; `422` không được coi PASS.

**Fresh-clone:** Một thành viên khác clone `--branch Dung` trên máy riêng, dùng Python 3.11, cài `requirements.txt`, tải runtime ZIP, đối chiếu SHA256, chạy 3 smoke tests (`tests.smoke_runtime`, `tests.smoke_academic_rules`, `tests.smoke_security`), kiểm UI. Thành viên xác nhận kết quả thực tế; không tự đánh dấu bước này trên máy người tạo patch.

**Provenance và CI:** Chỉ sau khi commit + push mới chạy benchmark ở checkout sạch; xác minh `provenance.source_commit` đúng HEAD và `source_dirty=false`. Xem GitHub Actions workflow có `head_branch=Dung`, `head_sha` là commit mới, `conclusion=success`. **Không cần và không được thao tác `main` ở đợt này.**

## Checklist cuối

- [x] Sửa sáu vấn đề RAG và ngăn lỗi hồi quy bằng fixture.
- [x] Chạy lại Python unit tests, JS tests và RAG offline trên bộ câu do người sửa soạn.
- [x] Có công cụ HTTP strict và lệnh nghiệm thu runtime/Gemini/ML/STRESS trên máy nhóm.
- [ ] Gemini cloud thực tế và đo generation/fallback.
- [ ] Người khác soạn câu hỏi mù, chấm đúng ngữ nghĩa theo PDF.
- [ ] ML extended + MAIN/STRESS bằng release thật tại máy nhóm.
- [ ] Kiểm ZIP runtime thật, HTTP qua Uvicorn, UI cả ba vai trò.
- [ ] Fresh-clone trên máy thành viên khác.
- [ ] CI SUCCESS và benchmark có provenance sạch sau push lên `Dung`.
