# Nghiệm thu các việc còn lại trên nhánh `Dung`

> Baseline áp dụng: `862391b1b99944f88313a0e7ba68dcf57bb535ac`. **Không sửa, checkout, merge hay push `main`.** Mọi kết quả chỉ được chốt sau khi chạy lại trên commit mới, working tree sạch. Dữ liệu là mô phỏng, không suy luận chất lượng với hồ sơ sinh viên thật.

## Đã kiểm chứng ở baseline trước bản sửa này

- Python unit test: 148 PASS; kiểm định MAIN và STRESS: 0 lỗi chặn; ML extended `stress.status=PASS`.
- Phân tầng HK3: AP 0,6539, ROC-AUC 0,8570, 333/2960 dương tính. HK4: AP 0,5362, ROC-AUC 0,8190, 237/2763 dương tính.
- HK5 có 192/192 dương tính, HK6 có 97/97. Không báo AUC/AP riêng khi chỉ có một lớp; phải công bố giới hạn censoring.
- MAIN có 4.331 ngoại lệ đăng ký và 273 điểm tuyển sinh thiếu; STRESS có 1.045 và 77. Các con số **cần đối chiếu**, không phải 4.331 hoặc 1.045 vi phạm quy chế. Giữ nguyên `SYNTHETIC_SOURCE_MISSING` nếu có chứng cứ nguồn mô phỏng, không điền điểm tùy ý.
- Runtime Release `cmc-research-v1.zip` SHA256 `72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848` đã kiểm manifest. Model không cần train lại để sửa kiểm định.

## Bước 1: cài môi trường và kiểm tra source

Chỉ chạy sau khi áp dụng patch và push thành công lên `origin/Dung`:

```powershell
Set-Location C:\Code\AI\CMC_Dropout_Prediction
$py = '.\.venv\Scripts\python.exe'
if ((git branch --show-current).Trim() -ne 'Dung') { throw 'Wrong branch' }
& $py -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Python tests FAILED' }
node --test tests/portal_async.test.cjs
if ($LASTEXITCODE -ne 0) { throw 'JavaScript tests FAILED' }
```

## Bước 2: Gemini cloud và chấm câu trả lời độc lập

Giữ **một key Gemini chung** trong `.env` chỉ chia sẻ với nhóm; `core/retrieval.py` tự đọc `.env`, không nhập key vào script, file patch, ảnh chụp hoặc báo cáo. Khi cần bật cloud:

```powershell
$py = '.\.venv\Scripts\python.exe'
$env:CMC_RAG_CLOUD = '1'
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --probe
if ($LASTEXITCODE -ne 0) { throw 'Gemini probe FAILED' }
& $py -B -X utf8 -m scripts.benchmark_rag --cloud --cases tests/cases/bo_cau_wave2_de_xuat.json --output data/kiem_tra_rag/final_cloud.json
if ($LASTEXITCODE -ne 0) { throw 'Gemini 30-case benchmark FAILED' }
```

Báo cáo `final_cloud.json` phải có `cloud_requested=true`, `so_cau_gemini>0`, `ty_le_gemini_thanh_cong`, `ty_le_fallback` và `cloud_quality_gate=true`. BM25 fallback được tính riêng; **không** được coi là Gemini generation. Cloud có thể chậm hoặc bị rate/quota limit, nên có thể FAILED dù retrieval offline PASS. Không đăng log chứa API key.

**Chấm độc lập:** Thành viên KHÔNG sửa RAG tự tạo file JSON **ngoài Git** gồm 30–50 câu hỏi mới ở định dạng `[ ["Câu hỏi",18], ["Không có căn cứ",null] ]`, giữ bí mật trước khi chạy. Chạy cùng `--cases`, nhưng **không dùng các câu do người sửa lập trước để tự nhận là tập mù**. Người chấm mở PDF quy chế gốc, điền CSV `_doi_chieu.csv` theo `dung_ngu_nghia`, `day_du`, `trich_dan_dung`, `khong_bia_quy_dinh` (0/1) và ghi lý do nếu có 0. Sau đó:

```powershell
& $py -B -X utf8 -m scripts.review_rag_semantics data/kiem_tra_rag/final_cloud_doi_chieu.csv --reviewer MA_THANH_VIEN --output data/kiem_tra_rag/final_human_review.json
```

Phần mềm chỉ xác nhận bảng chấm hợp lệ. Tính **độc lập** phải được người chấm xác nhận, không thể tự chứng minh bằng script.

## Bước 3: ngoại lệ dữ liệu mô phỏng

Sau khi xác thực dataset MAIN/STRESS đã cài đúng phiên bản với release:

```powershell
$env:CMC_DATASET_ROOT = (Resolve-Path 'data/du_lieu_hieu_chinh').Path
$env:CMC_MODEL_DIR = (Resolve-Path 'data/releases/cmc-research-v1/model').Path
& $py -B -X utf8 -m scripts.review_policy
if ($LASTEXITCODE -ne 0) { throw 'Policy review FAILED' }
```

Đọc `data/doi_chieu_quy_che/review.json` và các bảng `registration.csv`, `admission_missing.csv`, `admission_evidence.csv`. Tách **ngoại lệ học phần cuối khóa** khỏi **thiếu khối lượng chưa được xác minh**. Đối với 273/77 giá trị thiếu điểm tuyển sinh, ghi nguyên nhân và tỷ lệ thiếu theo năm/phương thức; không biến `null` thành điểm 0. Cần xác minh căn cứ ngoại lệ từ quy chế hoặc giả định bộ sinh, không tự kết luận nhà trường đã phê duyệt. Không chỉnh sửa dataset nguồn sau khi phát hành model.

Nếu cần lập lại báo cáo ML/STRESS, chạy `scripts.review_ml_extended --require-stress` và `scripts.audit_dataset --stress-run` theo `docs/nghiem_thu_wave2_Dung.md`. Đừng gộp nhãn HK5/HK6 chỉ có một lớp vào nhận định ROC-AUC theo học kỳ.

## Bước 4: HTTP thật và trình duyệt Edge

Mở **PowerShell thứ nhất**, dùng runtime research model đã cài, chạy app trên localhost:

```powershell
$env:CMC_DATA_MODE = 'DEMO'
$env:CMC_RAG_CLOUD = '0'
.\.venv\Scripts\python.exe -B -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Mở **PowerShell thứ hai** cùng thư mục Git:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.smoke_live_http --url http://127.0.0.1:8000 --student BIT250093 --payment 1000 --output data/kiem_tra_ban_giao/final_http.json
if ($LASTEXITCODE -ne 0) { throw 'HTTP scenarios did not pass; choose an eligible student, do NOT accept 422' }
```

Chỉ công nhận GPA Scenario và Payment Scenario nếu API trả HTTP 200 cùng các giá trị kết quả hợp lệ. Nếu hồ sơ sinh viên mẫu không có công nợ hoặc prediction còn hiệu lực, chọn SV mô phỏng đủ điều kiện và ghi lại mã đã chọn, không sửa test để chấp nhận 422.

Để tự động chạy giao diện trên Microsoft Edge đã cài:

```powershell
& $py -m pip install playwright
& $py -B -X utf8 -m scripts.smoke_browser --url http://127.0.0.1:8000 --channel msedge --headed --output data/kiem_tra_ban_giao/final_browser.json
if ($LASTEXITCODE -ne 0) { throw 'Browser checks FAILED' }
```

Script mới chạy **sáu hành trình** (Student/Advisor/Admin × desktop 1280px và mobile 375px), kiểm tra đăng nhập, trang dữ liệu riêng vai trò, chat thật, lỗi JavaScript và overflow. Không ghi token/mật khẩu. `--screenshots data/kiem_tra_ban_giao/screenshots` là **tùy chọn**, chỉ dùng với hồ sơ tổng hợp demo.

Dù Playwright PASS, một thành viên vẫn cần xem trực tiếp tính dễ dùng, biểu đồ, nội dung bảng điểm, các scenario và thao tác logout ở cả sáu tổ hợp. Kiểm thử trên máy người viết code không thay thế fresh-clone trên máy người khác.

## Bước 5: báo cáo tình trạng, không nhận PASS giả

Sau khi các bước trên có báo cáo, chạy:

```powershell
& $py -B -X utf8 -m scripts.summarize_remaining
Get-Content data/kiem_tra_ban_giao/final_acceptance.md -Encoding UTF8
```

Trạng thái có thể là `PASS`, `NOT_RUN`, `NOT_PASSED`, `FAIL`, `STALE`, `HUMAN_REVIEWED`, `REVIEW_REQUIRED`, `MANUAL_PENDING`. Nếu báo cáo tạo trước commit mới hoặc trên checkout dirty thì có thể bị đánh dấu `STALE`. **Không tự nhận** đã chấm mù, đã kiểm tra trên máy thành viên khác, hay xác nhận UI bằng mắt chỉ vì file JSON tồn tại.

## Bước 6: fresh-clone trên máy thành viên khác

Có thể chuyển script `fresh_clone_wave2_Dung_ASCII.ps1` cho thành viên khác. Chạy với **thư mục chưa tồn tại**; script clone `--branch Dung`, cài venv Python 3.11, đối chiếu SHA256 Runtime Release, chạy unit/RAG/smoke. Sau khi xong, thành viên đó mở ứng dụng, kiểm trực tiếp ba vai trò và ghi kết quả, thời điểm, commit HEAD, OS, Python version trong biên bản riêng **không chứa token/secret**. Đừng đánh dấu `fresh_clone_other_machine` PASS nếu không có bằng chứng từ chính máy đó.

### Những việc chưa nằm trong patch

Không thể thực hiện hộ người khác việc **tạo bộ câu hỏi mù, chấm ngữ nghĩa độc lập, fresh-clone trên máy vật lý khác, hoặc kiểm thử GUI trực tiếp của chính thành viên**. Gemini trực tuyến cũng phải chạy với quota/kết nối hợp lệ trên máy nhóm. Patch này không thay thế bất kỳ bước nào bằng kết quả giả.
