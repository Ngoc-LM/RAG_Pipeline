# eval/

`questions.json` — gold set, mảng các object theo schema mô tả ở mục "Gold set"
trong README gốc.

`char_start` / `char_end` trỏ vào `Document.body` — phần sau frontmatter, đã chuẩn
hoá CRLF→LF và NFC, **không strip**. Không phải offset trong file gốc, và không
phải offset sau khi cắt bỏ dòng trống đầu file.

Đừng đếm tay. Dùng `tools/annotate.py`:

```bash
python -m tools.annotate --corpus data/corpus --grep "thời hạn phản hồi"
python -m tools.annotate --corpus data/corpus --article luat_91_2025:9
python -m tools.annotate --corpus data/corpus --grep "phản hồi" --emit q07
python -m tools.annotate --corpus data/corpus --validate eval/questions.json
```

`--grep` trả về offset của khoản chứa chuỗi tìm được; `--article` in nguyên một
Điều kèm offset từng khoản; `--emit` sinh sẵn một question object để dán vào mảng;
`--validate` soi lại toàn bộ file và trả exit code khác 0 nếu có span sai phạm vi
hoặc không nằm trọn trong một Điều.

Ví dụ đầy đủ các `type` sẽ được thêm cùng tầng evaluate. Hiện chưa có file mẫu
trong repo.
