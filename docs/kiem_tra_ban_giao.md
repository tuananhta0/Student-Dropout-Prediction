# Checklist nghiệm thu và công bố phiên bản

File này phân biệt **đã kiểm tra bằng mã nguồn** và **cần thao tác thật**. Không đánh dấu PASS chỉ vì có lệnh kiểm thử hoặc tài liệu hướng dẫn.

## Thay đổi có trong bản sửa trên `Dung`

- [x] Workflow khai báo `push` và `pull_request` cho cả `main`, `Dung`; hỗ trợ `workflow_dispatch`. Chỉ phát huy tác dụng trên `main` sau khi nhóm quyết định merge.
- [x] Chuẩn hóa câu hỏi "học lớp cải thiện" thành chủ đề Điều 15.
- [x] Với câu hỏi ghép học phí và nghỉ học/bảo lưu, giữ phần Điều 21 có căn cứ; không bịa mức học phí.
- [x] Thêm regression tests router, HTTP API; bổ sung hai câu vào benchmark E2E (tổng 70).
- [x] Hướng dẫn `CMC_DATASET_ROOT` cho kiểm toán khi chỉ cài runtime release.
- [x] README và hướng dẫn bàn giao lấy `main` làm nhánh người dùng cuối; giữ `Dung` để sửa mã.
- [x] Có bảng rà soát PDF nguồn Điều 15, 21, **không nhận là người chấm độc lập**.
- [x] Tài liệu hướng dẫn ghi provenance theo commit chạy báo cáo, không sửa chỉ số model lịch sử.

## Kiểm tra thực tế còn bắt buộc

- [ ] **CI trên `Dung`**: sau khi push, mở `Actions`, tìm workflow có `head_sha` đúng commit mới; yêu cầu `conclusion=success`. Nếu FAIL, không merge.
- [ ] **CI trên `main`**: chỉ sau khi nhóm review và merge, xác nhận workflow run có `head_branch=main`, `head_sha` đúng HEAD của `main`, `conclusion=success`. Việc chỉ chạy trên `Dung`/PR **không** hoàn thành mục này.
- [ ] **Máy thành viên khác**: trên Windows/Python 3.11, clone `main` sau khi nhóm merge, chạy đầy đủ các bước trong README: SHA256 download, `release_model install`, 3 smoke tests, mở giao diện, đăng nhập 3 vai trò, thử câu hỏi Điều 15/21. Lưu bằng chứng commit, phiên bản Python, SHA256 và kết quả (không ghi thông tin cá nhân/API key).
- [ ] **Gemini thật**: chủ khóa thay khóa cũ, bật cloud qua hướng dẫn `docs/tra_cuu_quy_che.md`, chạy `python -B -m scripts.benchmark_rag --cloud --probe` trước, rồi `python -B -m scripts.benchmark_rag --cloud --output data/kiem_tra_rag/bao_cao_cloud_<commit>.json`. Yêu cầu `so_cau_gemini` > 0 và `cloud_quality_gate=true`; không coi đây là kiểm định ngữ nghĩa.
- [ ] **Đánh giá ngữ nghĩa độc lập**: thành viên không sửa RAG tự chuẩn bị bộ câu mù, đối chiếu từng câu với PDF, điền CSV và chạy `scripts.review_rag`. Không dùng tập thử do chính người sửa thuật toán soạn để tự công nhận độc lập.
- [ ] **Khóa và an toàn công khai**: chủ khóa thu hồi khóa đã lộ, tạo khóa mới và giới hạn quota trên Google AI Studio. Kiểm tra ZIP runtime bằng chính file tải về, duyệt danh sách file, nội dung JSON/CSV/manifest, xác nhận không có dữ liệu thật và bí mật. Không mở Joblib ngoài nguồn tin cậy.
- [ ] **Provenance sau merge**: sau khi `main` đã merge, clone `main` sạch và chạy `python -B -m scripts.benchmark_rag --all --e2e --output data/kiem_tra_rag/bao_cao_e2e_main.json`, kiểm `provenance.source_commit == git rev-parse HEAD`, `source_dirty == false`. Không ghi đè metadata model trong Runtime Release.

## Kiểm tra nhanh bản runtime trên máy đã cài

```powershell
$sha = '72d0ebd7b27ffda8e8e80bd2efe1e72c3a86152cba8e80ac356f7c4e1f98e848'
$archive = 'data/cmc-research-v1.zip'
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw 'Runtime sai SHA256' }
python -B -X utf8 -m tests.smoke_runtime
if ($LASTEXITCODE -ne 0) { throw 'Runtime smoke FAIL' }
```

Để tái chạy audit quy chế từ bản phát hành: `$env:CMC_DATASET_ROOT = (Resolve-Path 'data/releases/cmc-research-v1/dataset').Path`, chạy `python -B -m scripts.review_policy`, rồi `Remove-Item Env:CMC_DATASET_ROOT -ErrorAction SilentlyContinue`.

**Giới hạn:** CI dùng fixture và bộ câu do người phát triển soạn; không thay thế kiểm tra runtime trên máy khác hoặc nghiệm thu Gemini/độ trung thực câu trả lời ngoài tập thử.
