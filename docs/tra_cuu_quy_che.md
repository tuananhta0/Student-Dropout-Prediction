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

## Kiểm thử cuối và nguồn tham khảo

Kho chính vẫn là tóm lược 50 đoạn từ QĐ 22/2023, không phải toàn văn. Bản `Quy-dinh-thoi-hoc-DH-CMC.pdf` là tài liệu tổng hợp từ QĐ này; mở được qua liên kết tham khảo theo trang, được kiểm tra SHA256 và đối chiếu với đoạn chính. Truy vấn nhóm hình thức thôi học và quyền lợi sau thôi học tổng hợp các đoạn gốc liên quan. Không xem bản tổng hợp như quyết định mới.

Truy vấn nhiều Điều không tự cắt ở 12 đoạn; nếu quá nhiều thì trả bằng chứng trực tiếp. Câu hỏi đơn chủ đề cụ thể hạn chế nguồn ngoài chủ đề; khi Gemini sinh thành công chỉ hiển thị bằng chứng đã dẫn (trừ truy vấn Điều/Khoản trực tiếp, vốn cần toàn bộ đoạn). Kiểm tra ID trích dẫn không chứng minh độ đúng ngữ nghĩa. Semantic query gồm cụm chủ đề và khía cạnh đã cho phép, không gửi câu hỏi gốc; vẫn có thể mất chi tiết chưa có trong từ điển.

Offline: `python -B -m scripts.benchmark_rag --all --e2e --output data/kiem_tra_rag/bao_cao_e2e.json`. Quality gate: Hit@5, MRR@5 và top-1 tối thiểu 0,95, từ chối ngoài phạm vi 1,0, nguồn hợp lệ và tỷ lệ Điều mong đợi thực sự hiển thị bằng 1,0. Tập e2e dùng hồ sơ fixture.

Đánh giá độc lập: một thành viên không tham gia sửa retrieval chuẩn bị trước JSON `[[câu hỏi, số Điều hoặc null], ...]` (chỉ câu hỏi công khai, không chứa hồ sơ hay bí mật) từ PDF gốc. Không cho người chỉnh thuật toán biết đáp án trước khi chốt. Chạy `python -B -m scripts.benchmark_rag --cloud --cases <file-json> --output data/kiem_tra_rag/bao_cao_doc_lap.json`. Giữ CSV đối chiếu sinh ra; người chấm đối chiếu PDF và điền `dung_ngu_nghia`, `day_du` (1 đạt, 0 không đạt) cho tất cả câu Gemini, giải thích lỗi trong `ghi_chu_nguoi_cham`. Sau đó `python -B -m scripts.review_rag data/kiem_tra_rag/bao_cao_doc_lap_doi_chieu.csv --reviewer <ma-nguoi-cham>`. Chỉ đánh giá ngữ nghĩa của câu sinh thành công; báo riêng tỷ lệ fallback và retrieval. Không tự chấm hoặc tuyên bố kết quả độc lập khi chưa có người đánh giá thật.

SDK timeout 20 giây (20000 ms), deadline tổng thể 25 giây. `HTTP_400_DEADLINE` nhận diện deadline quá ngắn; `HTTP_400` là lỗi yêu cầu khác chưa phân loại. Không thử lại lỗi 400 và không in thông báo SDK gốc. Nếu probe thất bại, không chạy benchmark cloud tiếp. Với `--cloud`, `quality_gate` vẫn đo retrieval, nhưng exit code còn yêu cầu `cloud_quality_gate` (tối thiểu 90% câu có căn cứ sinh thành công, không tính fallback). Bộ câu chỉ ngoài phạm vi hay có chủ đích không yêu cầu LLM không phù hợp để nghiệm thu generation. Ngưỡng này đo độ khả dụng, không thay thế chấm ngữ nghĩa thủ công.

## Hồi quy tra cứu và phép tính

- Cú pháp `Điều 18 và 21`, `Điều 18 Khoản 1 và 99` được nhận diện; chỉ hiển thị phần có trong kho và báo từng số thiếu. Nếu có nhiều Điều kèm Khoản nhưng chưa rõ cách ghép, yêu cầu tra cứu từng Điều riêng. Không lưu context của truy vấn Điều/Khoản thiếu căn cứ.
- Với chủ đề có neo rõ, giới hạn BM25/embedding vào Điều tương ứng. Câu ghép như bảo lưu và phúc tra giữ nguồn của cả hai; Gemini thiếu nguồn của một chủ đề sẽ fallback. Câu ghép có phần nằm ngoài quy chế chỉ trả phần có căn cứ nếu nhận diện được. Lịch nghỉ lễ, hạn nộp học phí cụ thể, giờ thư viện thuộc thông báo vận hành, không tự suy diễn từ quy chế.
- Từ điển neo chủ đề và phép chiếu công khai có thể bỏ sót cách diễn đạt mới hoặc nhầm chủ đề; việc từ chối quá mức cũng cần kiểm thử. Đây không phải phân loại ngữ nghĩa tổng quát hoặc xác minh sự thật. Tập hồi quy 70 câu qua API kiểm tra thêm lỗi đã phát hiện; không thay cho đánh giá độc lập.
- Giả định GPA mục tiêu có số tín chỉ mới rõ ràng sử dụng đúng số nguyên dương, không vượt cap lưu trong hồ sơ, số tín chỉ còn lại và số học phần mới được giả định. Không tự đổi về 15 nếu người dùng chỉ định số khác. Điều kiện mở lớp và đăng ký kỳ tới vẫn cần xác minh.
- Giả định số lần cảnh báo dùng đúng ngưỡng Điều 18.2. Với nợ có chứng chỉ, chỉ trừ phần chứng chỉ có số lượng rõ hoặc yêu cầu tách ra; kết luận chỉ cho tiêu chí đang xét. Các câu thiếu điểm thành phần/tổng điểm vẫn kèm Điều 13.

Nguồn phiên bản báo cáo được ghi tại thời điểm chạy (commit, tree, dirty, dấu vân tay diff tracked, thời gian UTC, hash corpus và bộ câu hỏi; model cloud khi bật). CSV có sidecar `.provenance.json`; nếu người chấm sửa CSV, manifest ghi `changed_after_recording`, và báo cáo chấm lưu hash CSV đã chấm. Không tự nhận người chấm độc lập khi chưa xác nhận.

## Câu hỏi học vụ nhiều ý và GPA giả định

Nếu người dùng hỏi đơn giá học phí kết hợp với nghỉ học, chatbot chỉ trả nội dung nghỉ học có căn cứ Điều 21 và hướng dẫn tra thông báo học phí, không suy đoán con số. Hỏi nghỉ 3 hoặc 4 học kỳ do lý do cá nhân có phép tính giới hạn 3 học kỳ bằng Python theo Điều 21.1d; không áp ngưỡng này cho nghỉ do bệnh. Hỏi thi lại để cải thiện điểm môn đã đạt (C/B) trả quy định học lại cải thiện Điều 15 thay vì Điều 14 về thi lại thông thường.

Câu hỏi giả định `GPA học kỳ thứ hai 0,99 có bị cảnh báo?` được Rule Engine Python so với ngưỡng GPA Điều 18.1a, không lấy GPA từ hồ sơ sinh viên hoặc gọi Gemini. Kết luận chỉ cho một tiêu chí, không là quyết định cảnh báo chính thức. Những câu hỏi thiếu thông tin kỳ/GPA phải yêu cầu bổ sung, không tự đoán.
