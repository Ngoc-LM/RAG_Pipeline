# data/corpus/

Đặt các file văn bản QPPL `.md` / `.txt` vào đây, mỗi file mở đầu bằng khối
frontmatter phẳng. Xem mục "Chuẩn bị corpus" trong README gốc để biết các key
bắt buộc (`doc_id`, `title`) và quy ước trục ký tự mà `gold_spans` tham chiếu tới.

Kiểm tra sau khi thêm file:

```bash
python run.py ingest        # báo lỗi ngay nếu frontmatter thiếu key hoặc doc_id trùng
```

Thư mục này đang trống. Fixture giả lập để chạy thử nằm ở `tests/fixtures/corpus/`.
