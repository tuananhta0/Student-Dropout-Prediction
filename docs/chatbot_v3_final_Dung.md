# CMC Academic AI V3 Final (Dung)

## Kiến trúc và giới hạn

V3 là bản tích lũy của các sửa đổi V2.1–V2.5.4 và một bộ điều phối hội thoại mới. Không thay dataset, model ML, tài liệu PDF hay thuật toán tính GPA, không huấn luyện lại. Các test cũ chạy chế độ offline để đảm bảo không mất chức năng đã có.

Khi `CMC_DATA_MODE=DEMO`, `CMC_RAG_CLOUD=1` và `CMC_CHAT_AGENT=1`, Gemini nhận câu hỏi bằng ngôn ngữ tự nhiên, chọn tối đa ba bước từ schema cố định. Chào hỏi/danh tính/những câu hỏi học vụ đơn giản đã có công cụ xác định và các phép tính rõ ràng vẫn đi theo đường nhanh tại Python. Các câu khác truy xuất ứng viên quy chế cục bộ trước, rồi dùng model điều phối; chỉ các đoạn liên quan được gửi, không gửi toàn kho cho hội thoại chung. Kiến thức chung không bị buộc phải có căn cứ PDF CMC. Hệ thống không có công cụ duyệt web, thông tin thời gian thực, quyền ghi cơ sở dữ liệu hoặc chạy code do model tạo.

- `general`: hội thoại, kiến thức chung, lập trình, viết và giải thích.
- `policy`: diễn giải từ các Khoản trong kho quy chế công khai. ID được kiểm tra và liên kết PDF được tạo tại máy chủ. Trường hợp thiếu căn cứ phải nói rõ.
- `identity`, `status`, `risk`, `personal_metric`: chỉ đọc hồ sơ được xác thực tại máy chủ. Không nhận số liệu từ Gemini.
- `analytics`: schema chỉ số và phạm vi cố định, kiểm tra vai trò, ngành, lớp và bộ lọc trước khi tính. Không coi tỷ lệ quan sát là tác động nhân quả.
- `local`: sử dụng công cụ tính toán/nghiệp vụ hiện có với câu hỏi gốc, không chạy SQL hoặc Python do LLM sinh.
- `clarify`, `unavailable`, `deny`: hỏi lại đúng trọng tâm, báo thiếu dữ liệu hoặc từ chối ngoài quyền.

Mọi trường hợp có mã sinh viên hoặc lớp rõ ràng đi qua kiểm tra quyền cũ trước khi có thể gọi cloud. Hệ thống không gửi token, mật khẩu, hàng dữ liệu cá nhân, kết quả ML hay kết quả công cụ vào prompt. **Khác với chế độ RAG trước đây, V3 gửi nội dung câu hỏi tự nhập tới Gemini khi được bật.** Chỉ dùng DEMO, không nhập tên thật, số điện thoại, mã hồ sơ thật hoặc bí mật. Bộ lọc chặn một số dạng định danh/bí mật, không bảo đảm ẩn danh mọi nội dung tự nhập. Ngữ cảnh hội thoại chung được giữ ngắn trong RAM theo phiên; dữ liệu công cụ học vụ không được lưu làm lịch sử gửi đi. Đăng xuất thu hồi phiên. V3 không kích hoạt trong các chế độ khác DEMO.

Gemini vẫn có thể hiểu sai, sinh thông tin sai hoặc lỗi dịch vụ/quota. Kiểm tra citation bảo đảm ID thuộc kho đã xác minh, không chứng minh mọi kết luận đều đúng ngữ nghĩa. Không tuyên bố trả lời đúng mọi câu hỏi hoặc dự báo trên sinh viên thật. Khi cloud lỗi, quay về truy vấn xác định offline hoặc thông báo rõ khả năng hội thoại chung đang không khả dụng. Giới hạn cloud 30 SDK calls/phút, 500/24 giờ theo tiến trình; hai worker và deadline 25 giây giữ nguyên. Không bỏ giới hạn để chạy benchmark đẹp.

## Chạy trực tiếp

1. Cài đúng Python 3.11, dependencies từ `requirements.txt`, runtime `cmc-research-v1` đã xác minh. `.env` riêng chứa `GEMINI_API_KEY` dùng chung của nhóm. Không commit file này.
2. Tại repo trên Windows: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start_v3_web_ASCII.ps1`.
3. Mở http://127.0.0.1:8000. Chế độ V3 được hiển thị trên giao diện. Student `BIT250093`, Advisor `GV001`, Admin `ADMIN`, mật khẩu demo là mã tài khoản. Đây là demo localhost, không triển khai với dữ liệu thật.
4. Thêm `-Offline` vào script nếu không dùng Gemini. Offline không thể cung cấp kiến thức chung linh hoạt như LLM.

## Kiểm thử và bàn giao

`python -B -X utf8 -m scripts.accept_chatbot_v3` chạy unit tests, Node, 77 câu hồi quy, runtime/academic/security smoke, HTTP và Edge Playwright ở chế độ offline, rồi khởi động Uvicorn với V3 và chạy bộ smoke hội thoại Gemini thật (bao gồm chuỗi câu đã phát hiện lỗi, kiến thức chung, hội thoại tiếp nối, GPA thực, quy chế và phân quyền cả ba vai trò). Báo cáo được lưu ở `data/kiem_tra_rag/v3_final`. Dừng server đang chạy trước khi nghiệm thu. Cổng mặc định HTTP offline 8000 và cloud 8001; nếu bị chiếm, dùng `--port` khác. Không tắt tiến trình không thuộc runner. Có thể dùng `--resume` khi đúng dấu vân tay mã nguồn để tiếp tục giai đoạn chưa hoàn thành.

Nếu có CSV 1.800 câu và công cụ audit cũ, có thể truyền `--bank` và `--audit` để lưu benchmark offline. Đây là bộ chẩn đoán của người phát triển, không phải benchmark V3 cloud hoặc tỷ lệ đúng ngữ nghĩa. V3 có thêm smoke cloud thực tế, không chạy tất cả 1.800 câu lên cloud gây vượt quota.

Không bắt buộc đánh giá độc lập theo yêu cầu của nhóm. Người phát triển có thể trực tiếp trò chuyện trên web. Báo cáo kết quả trung thực: smoke không chứng minh mọi câu trả lời đúng. Dữ liệu mô phỏng chưa xác thực dự báo trên người thật.

Patch phân phối là diff tích lũy đối với commit `284dac26cb11797cce14c634100d93efe6eece1e` của Dung. Công cụ bên ngoài chỉ áp dụng khi cây hiện tại trùng một trạng thái đã xác minh; không reset hoặc xóa thay đổi người dùng. Chỉ commit/push `Dung` nếu người dùng truyền `-Commit` và mọi kiểm thử bắt buộc đều PASS. Không checkout, merge hoặc push main. Fresh-clone kiểm tra SHA commit đã chốt, tải runtime chính thức có checksum cố định, cài môi trường mới, chạy lại acceptance và mở web.

Truy vấn đếm đơn giản theo một ngành, theo từng ngành hoặc toàn trường được thực hiện trực tiếp tại Python, trước bộ điều phối Gemini. Tên ngành phải khớp đầy đủ mã hoặc bí danh; các truy vấn có bộ lọc khác không bị rút gọn thành tổng số. Phạm vi ngành/lớp của Advisor được kiểm tra trước khi đếm. Smoke thực tế đối chiếu số đang học theo ngành IT với API lớp được phép và số toàn trường với API quản trị; yêu cầu ngành ngoài quyền phải nhận HTTP 403.

### Cloud resilience

Truy xuất ứng viên cục bộ, không gửi toàn bộ 50 đoạn quy chế. Câu trả lời policy phải dẫn ID đã cung cấp cho model; nếu thiếu nguồn thì quay về luồng cục bộ. Timeout SDK theo từng lần gọi (12 giây cho lần đầu nếu SDK hỗ trợ, lần sau tối đa 20 giây và chỉ khi còn ít nhất 10 giây sau backoff). Không retry 429, không đổi deadline 25 giây hoặc ngân sách 30/500. Lỗi tạm thời giữ chủ đề hội thoại chung và đánh dấu câu chưa được trả lời; không lưu hồ sơ hay kết quả công cụ. Smoke ghi thêm elapsed_ms và mã lỗi, giữ nguyên 21 kiểm tra. Không cam kết dịch vụ Gemini luôn khả dụng.
