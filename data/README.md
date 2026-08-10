# data/corpus/

Đặt các file văn bản QPPL **`.txt`** vào `data/corpus/`. `load_corpus` chỉ đọc
phần mở rộng `.txt`; file `.md` sẽ bị bỏ qua.

Mỗi file mở đầu bằng khối frontmatter phẳng với sáu key bắt buộc — `doc_id`,
`title`, `doc_type`, `issued_date`, `effective_from`, `status` — cộng
`effective_to` để `null` nếu văn bản còn hiệu lực. `status` chỉ nhận `active`
hoặc `expired`. Xem mục "Chuẩn bị corpus" trong README gốc để có ví dụ đầy đủ và
quy ước trục ký tự mà `gold_spans` tham chiếu tới.

Kiểm tra sau khi thêm file:

```bash
python -m src.chunk --corpus data/corpus
```

Lệnh này parse toàn bộ corpus và sẽ báo lỗi ngay nếu frontmatter thiếu key,
`status` không hợp lệ, `doc_id` trùng, offset của khoản không khớp body, hoặc có
phần body không thuộc chunk nào. Chạy được tức là corpus đã sạch về cấu trúc.

Thư mục `data/corpus/` đang trống. Fixture giả lập để chạy thử nằm ở
`tests/fixtures/corpus/`.
