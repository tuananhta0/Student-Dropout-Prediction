# Tra cứu quy chế đào tạo

Nguồn chính: QĐ 22/2023/QĐ-ĐHCMC-HĐT và Quy chế đính kèm, file `regulations/Quy-che-dao-tao-trinh-do-dai-hoc.pdf` (27 trang PDF). Trang 1 là quyết định, trang 2 bìa, trang 3–4 mục lục; nội dung 27 điều nằm ở trang PDF 5–27. `quy_che_chuan_hoa.tsv` có 50 đoạn tóm lược tiếng Việt, được đối chiếu trực tiếp với ảnh trang. Đây **không phải bản OCR nguyên văn**: những khoản dài có thể được gộp và lược bớt. Khi cần nội dung chính xác tuyệt đối, mở PDF từ trích dẫn. File PDF tóm tắt thôi học là tài liệu phụ, không ghi đè quy chế gốc. Phạm vi văn bản từ năm học 2023–2024; dùng như giả định đối với dữ liệu mô phỏng, không khẳng định quy chế hiện hành ngoài thực tế.

## Luồng xử lý

- `core/intent_router.py` phân luồng quy chế, hồ sơ, GPA, nguy cơ, thống kê; gọi công cụ Python sau kiểm tra quyền. Không đưa hồ sơ, token, điểm số, mã SV hay câu hỏi nguyên văn vào LLM.
- `core/retrieval.py`: chuẩn hóa truy vấn, Điều/Khoản ưu tiên lọc chính xác; các câu hỏi tự nhiên xếp hạng BM25; khi không đủ căn cứ trả lời từ chối. Có thể hỏi tiếp “Khoản 2 thì sao?” trong cùng phiên đăng nhập. Mỗi đoạn có ID, điều/khoản và số trang PDF thật. SHA256 khóa đúng bản nguồn. Khi PDF đổi phải đối chiếu lại nội dung và hash; thời điểm tra cứu không đồng nghĩa văn bản vẫn là quy định mới nhất.
- Chế độ mặc định không cần API, trả bằng chứng trực tiếp. Chế độ Gemini tùy chọn dùng embedding đa ngôn ngữ và cosine trên toàn corpus, hợp nhất với BM25; truy vấn được chiếu sang danh sách cụm chủ đề công khai cố định, không gửi câu hỏi nguyên văn. Cache theo hash corpus và model; thử lại một lần với lỗi 503/504/timeout. LLM diễn giải các đoạn bằng chứng, trả JSON với ID trích dẫn, kiểm tra ID hợp lệ; nếu lỗi quay về bằng chứng. ID hợp lệ không đảm bảo mọi câu sinh đúng ngữ nghĩa; cần kiểm tra nội dung trước nghiệm thu. Không gọi chế độ mặc định là RAG có LLM.

## Chạy thử

```powershell
python -B -X utf8 -m scripts.regulations_scanner
python -B -X utf8 -m scripts.benchmark_rag
```

Chỉ mục JSON xuất ở `data/kiem_tra_rag/`; API dùng TSV đã kiểm chứng, không phụ thuộc chỉ mục cũ trong artifact baseline. Báo cáo có 34 câu, trong đó 30 câu có căn cứ và 4 câu ngoài phạm vi. Hit rate, MRR, top1 và tỷ lệ từ chối được tính dựa trên tập nhỏ do người phát triển soạn. Không suy rộng thành chất lượng tổng quát.

## Bật Gemini

Cần khóa API Google AI Studio hợp lệ và mạng Internet. Chỉ bật nếu chấp nhận gửi nội dung quy chế công khai đến dịch vụ ngoài, có thể phát sinh phí. Khóa không đưa vào mã nguồn, không commit. Ví dụ PowerShell:

```powershell
$secret = Read-Host 'Nhap GEMINI API key' -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
$env:CMC_RAG_CLOUD = '1'
$env:CMC_EMBED_MODEL = 'gemini-embedding-001'
$env:CMC_LLM_MODEL = 'gemini-3.1-flash-lite'
python -B -X utf8 -m scripts.benchmark_rag --cloud
```

Nếu nhà cung cấp thay model hoặc quyền không hợp lệ, đổi ID theo tài khoản. Kiểm tra `so_cau_gemini` trong báo cáo: bằng 0 nghĩa là chưa kiểm thử generation thật. Kiểm tra thêm `so_cau_embedding` để xác nhận đường dẫn embedding thật. Đối chiếu số liệu, ngưỡng và trích dẫn trong từng câu trả lời với PDF gốc; không coi kiểm tra ID là chấm tự động độ trung thực. Không tự tạo chỉ số generation. Tắt cloud bằng `Remove-Item Env:CMC_RAG_CLOUD, Env:GEMINI_API_KEY -ErrorAction SilentlyContinue`.

### Chẩn đoán generation

Chạy `python -B -X utf8 -m scripts.benchmark_rag --cloud --probe` trước benchmark đầy đủ. Lệnh chỉ sinh một câu Điều/Khoản, không gọi embedding, không ghi đè báo cáo. Mã `OK` mới xác nhận generation thành công. `HTTP_429` là giới hạn API/quota; `HTTP_403` là lỗi quyền; `HTTP_404` là lỗi tài nguyên/model; `MAX_TOKENS`, `EMPTY_RESPONSE`, `INVALID_JSON`, `INVALID_CITATIONS` là lỗi đầu ra. `DISABLED` nghĩa là cloud chưa bật hoặc thiếu khóa. Nhật ký chỉ in mã lỗi an toàn, không in khóa, exception gốc hay nội dung phản hồi lỗi. Vẫn giữ bằng chứng trực tiếp khi generation thất bại.

## Bộ câu hỏi mở rộng

`python -m scripts.benchmark_rag --extended --output data/kiem_tra_rag/bao_cao_mo_rong.json` kiểm tra 18 câu diễn đạt khác, gồm 14 câu có nguồn và 4 câu ngoài phạm vi. Thêm `--cloud` để đo pipeline Gemini. CSV `*_doi_chieu.csv` có ô đánh giá thủ công để một thành viên khác đọc câu trả lời, so với PDF, chấm đúng ngữ nghĩa và độ đầy đủ; không tự đánh dấu đạt hoặc coi bộ hồi quy tự soạn là test độc lập. Luồng cloud chỉ truyền cụm chủ đề từ danh sách cố định và quy chế công khai; có thể mất sắc thái câu hỏi do chủ động hạn chế thông tin.

Để đánh giá độc lập, thành viên không sửa retrieval tự viết file JSON gồm các cặp `["câu hỏi", số_Điều]`, dùng `null` cho câu ngoài phạm vi. Không đưa bộ câu hỏi này vào tuning. Chạy `python -m scripts.benchmark_rag --cases data/cau_hoi_doc_lap.json --output data/kiem_tra_rag/bao_cao_doc_lap.json` (thêm `--cloud` khi cần).
