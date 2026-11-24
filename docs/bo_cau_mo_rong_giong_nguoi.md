# Bộ câu hỏi mở rộng diễn đạt giống người (sinh offline có lọc chất lượng)

Tài liệu mô tả `scripts/generate_human_questions.py` và bộ xuất ra
`tests/cases/bo_cau_mo_rong_giong_nguoi.json`. Mục tiêu: sinh thêm câu hỏi tra cứu
học vụ với cách diễn đạt tự nhiên của sinh viên (xưng hô, mở đầu, từ kết, biến thể
từ đồng nghĩa) để mở rộng tập hồi quy RAG, không thay thế bộ kiểm thử độc lập.

## Phạm vi và giới hạn trung thực

- Đây là **quy trình sinh dữ liệu theo nguyên tắc few-shot + quy tắc, có vòng lặp
  lọc bằng bộ định tuyến offline**; không phải fine-tune trọng số mạng nơ-ron và
  không tự gọi LLM. Mọi câu đều do script kết hợp từ seed người thật viết và khuôn
  câu theo Điều, rồi chỉ giữ câu mà `RegulationSearch` trả đúng căn cứ.
- Bộ câu do người phát triển sinh ra, **KHÔNG phải tập kiểm thử độc lập/mù**. Không
  dùng để tuyên bố chất lượng ngữ nghĩa tổng quát; người chấm vẫn đối chiếu PDF khi
  đánh giá câu trả lời.
- Chỉ chứa câu hỏi công khai về quy chế; không chứa hồ sơ, mã sinh viên hay dữ
  liệu nhạy cảm. Câu ngoài phạm vi (`null`) là thông tin vận hành không có trong
  quy chế (học phí cụ thể, lịch đóng tiền, thư viện, gửi xe, học bổng...).

## Quy trình "train" dữ liệu (data-centric loop)

1. **Seed (câu người thật đã viết):** gộp `CASES`, `EXTENDED`, `REGRESSION` trong
   `scripts/benchmark_rag.py` và hai bộ `tests/cases/bo_cau_ra_soat_40_de_xuat.json`,
   `tests/cases/bo_cau_wave2_de_xuat.json` (148 cặp `[câu hỏi, Điều | null]`).
2. **Biến thể phong cách:** với câu seed đủ đầy đủ ngữ nghĩa, thêm mở đầu tự nhiên
   (`Cho em hỏi`, `Dạ cho em hỏi`, `Cho mình xin hỏi`, `Tôi muốn hỏi`, `Em thắc mắc
   là`...), đổi xưng hô, gắn từ kết lịch sự (`ạ`, `nhỉ`, `nha`) và thay cụm đồng
   nghĩa (`phúc tra` ↔ `chấm lại điểm`, `cảnh báo học tập` ↔ `nhắc nhở học tập`,
   `nghỉ học tạm thời` ↔ `bảo lưu học tập`...).
3. **Khuôn câu theo chủ đề:** viết sẵn câu theo từng Điều bám theo tiêu đề TSV quy
   chế (`regulations/quy_che_chuan_hoa.tsv`) để bổ sung các chủ đề ít seed.
4. **Lọc tự nhiên:** độ dài 14–170 ký tự, ít nhất 5 từ, có từ hỏi mạnh; loại từ
   khóa ngắn kiểu `quy mô lớp học`, loại trùng lặp sau chuẩn hóa dấu.
5. **Lọc chất lượng định tuyến:** chạy `RegulationSearch` offline (`CMC_RAG_CLOUD=0`):
   - câu có Điều mong đợi: chỉ giữ khi Điều đó xuất hiện trong bằng chứng hiển thị;
   - câu `null`: chỉ giữ khi không trả bằng chứng và intent thuộc `regulation`,
     `none`, `clarify`.
6. **Giới hạn và xuất:** tối đa 6 câu/Điều và 22 câu ngoài phạm vi; xuất JSON đúng
   schema `[[câu hỏi, số Điều | null], ...]` cho `scripts.benchmark_rag --cases`.

## Chạy

Tại thư mục gốc, Python 3.11/3.12, đã cài dependencies:

```powershell
$env:CMC_RAG_CLOUD = '0'
python -B -X utf8 -m scripts.generate_human_questions
python -B -X utf8 -m scripts.benchmark_rag --cases tests/cases/bo_cau_mo_rong_giong_nguoi.json --output data/kiem_tra_rag/mo_rong.json
```

Tham số hữu ích: `--max-per-article 6`, `--max-null 22`, `--dry-run` (chỉ sinh
ứng viên, không lọc retrieval). Báo cáo lọc chi tiết ghi ở
`data/kiem_tra_rag/cau_mo_rong_giong_nguoi_summary.json` (thư mục `data/` không
commit).

## Kết quả lần sinh trên máy hiện tại

- Seed: 148 cặp; ứng viên: 306; giữ: 156; loại ở bước lọc định tuyến: 2; tỷ lệ
  giữ: 0,510.
- 134 câu có căn cứ theo Điều, 22 câu ngoài phạm vi; phủ toàn bộ Điều 1–27.
- Chạy `benchmark_rag` offline: Hit@5 = 1,000; MRR@5 ≈ 0,996; Top-1 ≈ 0,993;
  từ chối ngoài phạm vi = 1,000; tỷ lệ Điều mong đợi hiển thị = 1,000; trích dẫn
  hợp lệ = đúng; `quality_gate = true`.
- Sinh ra ổn định giữa các lần chạy (chọn mẫu bằng md5, không dùng hash ngẫu nhiên).

Số liệu đo khả năng định tuyến/retrieval offline, không phải chấm đúng ngữ nghĩa
câu trả lời sinh ra bởi LLM.

## Phụ trợ: model phân loại chủ đề nhỏ (tùy chọn, minh hoạ "train")

`scripts/train_intent_classifier.py` huấn luyện TF-IDF (char n-gram 2–4) +
LogisticRegression trên 301 câu (seed + bộ mở rộng), nhãn là số Điều hoặc `none`,
đánh giá cross-validation; model lưu ở `data/models/` (thư mục ignored, không commit).

```powershell
python -B -X utf8 -m scripts.train_intent_classifier
```

Lần chạy mẫu: 301 câu, 28 lớp, 2-fold CV (một số lớp chỉ có 2 mẫu nên không dùng
5-fold), CV accuracy ≈ 0,861, macro F1 ≈ 0,893, weighted F1 ≈ 0,858. Script chỉ đo
khả năng tách chủ đề của tập câu; **không thay thế `core/intent_router.py`** (router
quy tắc + phân quyền vẫn là đường chạy chính của ứng dụng).

## Mở rộng thêm

- Bổ sung seed thật vào hai bộ JSON hoặc danh sách trong `benchmark_rag.py`, rồi
  chạy lại script; bộ mới tự loại trùng và tự lọc.
- Khi có `GEMINI_API_KEY`, có thể thay bước 2–3 bằng few-shot prompt cho Gemini
  (vài câu seed + yêu cầu diễn đạt lại tự nhiên), sau đó **bắt buộc** đưa toàn bộ
  câu sinh ra qua bước lọc 5 trước khi đưa vào bộ hồi quy; không commit key.
- Không gộp bộ này vào danh sách `CASES/EXTENDED/REGRESSION` làm quality gate chính
  nếu chưa có người chấm độc lập; chạy `--cases` riêng như báo cáo phụ trợ.

Xem quy ước đánh giá độc lập trong [tra_cuu_quy_che.md](tra_cuu_quy_che.md) và
[nghiem_thu_ABCD_Dung.md](nghiem_thu_ABCD_Dung.md).
