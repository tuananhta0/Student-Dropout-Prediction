# Kiểm định định danh MAIN / STRESS trên nhánh Dung

Bộ sinh V3.2.1 sử dụng `record_uid` **cục bộ trong từng `scenario_id`**. Vì vậy
`V3_2_MAIN` và `V3_2_STRESS` đều có thể chứa `SIMV31_000001` nhưng là
hai sinh viên mô phỏng khác nhau.

Kiểm tra trực tiếp archive nguồn V3.2.1:

- MAIN: 20.000 sinh viên.
- STRESS: 5.000 sinh viên.
- `record_uid` trùng giữa hai kịch bản: 5.000; **hợp lệ theo thiết kế**.
- `student_id` trùng giữa hai kịch bản: **0**.
- `record_uid` và `student_id` trùng *trong mỗi kịch bản*: **0**.

Khóa định danh duy nhất khi xem đồng thời hai tập là
`(scenario_id, record_uid)`. **Không** merge hoặc split MAIN/STRESS chỉ dựa
vào `record_uid`. Huấn luyện dùng MAIN; STRESS được đánh giá tách riêng.
Bộ kiểm định vẫn từ chối `student_id` trùng giữa hai tập, ID trùng trong
cùng kịch bản hoặc `scenario_id` sai. Điều này chỉ xác nhận tách biệt
định danh, **không loại trừ mọi dạng data leakage khác**.

## Chạy nghiệm thu trên dữ liệu nghiên cứu đã cài

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Python tests failed' }
$env:CMC_DATASET_ROOT = (Resolve-Path '.\data\du_lieu_hieu_chinh').Path
$env:CMC_MODEL_DIR = (Resolve-Path '.\data\releases\cmc-research-v1\model').Path
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.review_ml_extended --require-stress --bootstrap 200 --output data/kiem_tra_ml/wave2_extended_STRESS.json
if ($LASTEXITCODE -ne 0) { throw 'STRESS extended failed' }
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.audit_dataset --stress-run '.\data\du_lieu_hieu_chinh\artifacts\runs\V3_2_STRESS' --output data/kiem_tra_du_lieu/wave2_STRESS
if ($LASTEXITCODE -ne 0) { throw 'STRESS full audit failed' }
```

Kiểm tra báo cáo `stress.student_id_overlap = 0`;
`stress.record_uid_overlap_local = 5000` có thể hoàn toàn hợp lệ.
Lưu ý: `full_stress_audit` trong báo cáo mở rộng được ghi `NOT_RUN`
cho đến khi lệnh `scripts.audit_dataset --stress-run` chạy thành công.

Nguồn code đối chiếu: `scripts/regenerate_dataset.py` sinh hai kịch bản;
`scripts/train_model.py` đã kiểm tra không trùng `student_id` khi đánh giá STRESS.
