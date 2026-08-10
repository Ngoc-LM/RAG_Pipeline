# eval/

`questions.json` — gold set, mảng các object theo schema mô tả ở mục "Gold set"
trong README gốc.

`char_start` / `char_end` trỏ vào body đã chuẩn hoá, không phải file gốc. Lấy body
bằng `python run.py ingest` (ghi ra `outputs/bodies/<doc_id>.txt`) và đối chiếu
bằng `python run.py show-span --doc-id ... --start ... --end ...`.

Trước khi tin số liệu: `python run.py check-leakage`.

Ví dụ đầy đủ 5 câu (đủ các `type`) nằm ở `tests/fixtures/questions.json`.
