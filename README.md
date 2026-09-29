
```markdown
# Q-AChatBot

Hệ thống hỏi đáp tự động về quy chế đào tạo và sổ tay sinh viên áp dụng kiến trúc Retrieval-Augmented Generation (RAG).

---

## Cấu trúc thư mục

```text
student-handbook-rag/
├── data/
│   ├── raw/                  # Chứa tệp PDF quy chế gốc
│   └── processed/            # Chứa văn bản sau trích xuất (tùy chọn)
├── chroma_db/                # Dữ liệu vector do ChromaDB tự động tạo
├── src/
│   ├── __init__.py           # Tệp rỗng đánh dấu package
│   ├── ingest.py             # Script xử lý PDF và tạo cơ sở dữ liệu vector
│   ├── rag_engine.py         # Module định nghĩa chuỗi xử lý RAG và LLM
│   └── utils.py              # Các hàm phụ trợ tiền xử lý và metadata
├── app.py                    # Giao diện web người dùng Streamlit
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── .env.example              # Tệp cấu hình biến môi trường mẫu
├── .env                      # Tệp lưu khóa API bảo mật
├── .gitignore                # Danh sách tệp loại trừ khỏi Git
└── README.md                 # Tài liệu hướng dẫn kỹ thuật

```

---

## Chi tiết nhiệm vụ theo tệp

* `data/raw/`: Không viết mã nguồn. Sao chép trực tiếp các tệp PDF quy chế đào tạo, sổ tay sinh viên cần tra cứu vào thư mục này.
* `data/processed/`: Thư mục tùy chọn lưu dữ liệu trung gian sau khi làm sạch văn bản thô.
* `chroma_db/`: Thư mục do hệ thống tự sinh sau khi hoàn tất vector hóa, không chỉnh sửa thủ công.
* `src/__init__.py`: Giữ tệp rỗng để hỗ trợ cơ chế nhập module giữa các tệp trong thư mục `src`.
* `src/utils.py`: Lập trình các hàm bổ trợ như làm sạch ký tự thừa trong văn bản và định dạng thông tin trích dẫn nguồn.
* `src/ingest.py`: Lập trình quy trình đọc tài liệu PDF, phân tách thành các đoạn văn bản, gửi tới API embedding và lưu trữ vector vào ChromaDB.
* `src/rag_engine.py`: Lập trình kết nối với ChromaDB làm bộ truy xuất, thiết lập prompt mẫu hướng dẫn ngữ cảnh, kết nối mô hình Gemini 1.5 Flash và xuất hàm xử lý truy vấn hoàn chỉnh.
* `app.py`: Lập trình giao diện trò chuyện bằng Streamlit, quản lý lịch sử tương tác qua `session_state`, nhận câu hỏi từ người dùng, gọi hàm từ `rag_engine.py` và hiển thị câu trả lời kèm đoạn quy chế trích dẫn.
* `requirements.txt`: Khai báo đầy đủ tên và phiên bản của các thư viện phụ thuộc.
* `.env`: Khai báo giá trị khóa Google API phục vụ quá trình embedding và gọi mô hình ngôn ngữ.

---

## Cài đặt môi trường

Khởi tạo môi trường ảo:

* Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

* Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate

```

Cài đặt thư viện phụ thuộc:

```bash
pip install -r requirements.txt

```

---

## Cấu hình API

Sao chép tệp cấu hình mẫu và cập nhật khóa:

```bash
cp .env.example .env

```

Nội dung tệp `.env`:

```env
GOOGLE_API_KEY="dien_khoa_api_tai_day"

```

---

## Nạp dữ liệu

Sao chép các tệp PDF vào thư mục `data/raw/`, sau đó chạy script:

```bash
python src/ingest.py

```

---

## Cách chạy

Khởi chạy ứng dụng web:

```bash
streamlit run app.py

```

Truy cập hệ thống tại địa chỉ: `http://localhost:8501`.
