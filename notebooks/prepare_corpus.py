# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chuẩn bị corpus cho RAG_Pipeline
#
# Notebook này biến 4 file nguồn (`.pdf` / `.doc`) thành `data/corpus/*.txt` đúng
# định dạng mà `src/ingest.py` đọc được.
#
# **Toàn bộ logic nằm trong `tools/normalize_raw.py`** — notebook chỉ upload file,
# gọi hàm và in báo cáo. Mã chôn trong cell notebook thì không test được, không
# diff được và không tái hiện được.
#
# File này lưu ở định dạng jupytext percent. Để mở dưới dạng notebook:
#
# ```bash
# jupytext --to notebook notebooks/prepare_corpus.py
# ```
#
# Chạy tuần tự từ Cell 1. Hai cell cần **điền tay**: Cell 4 (`SOURCES`),
# Cell 7 (`KEEP`) và Cell 8 (`FRONTMATTER`).

# %% [markdown]
# ## Cell 1 — Cài đặt
#
# `libreoffice` mất khoảng **2 phút** và chỉ cần chạy **một lần mỗi session**.
# Nếu runtime bị ngắt kết nối thì phải chạy lại.
#
# - `poppler-utils` → `pdftotext` (đọc PDF)
# - `libreoffice` → `soffice` (chuyển `.doc` nhị phân cũ sang `.docx`)
# - `python-docx` → đọc `.docx`

# %%
# !apt-get -qq install poppler-utils libreoffice > /dev/null
# !pip -q install python-docx jupytext

# !soffice --version
# !pdftotext -v

# %% [markdown]
# ## Cell 2 — Clone repo
#
# Bước này chưa cần API key: chỉ dùng `src.ingest`, `src.chunk`, `src.intervals`.

# %%
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Ngoc-LM/RAG_Pipeline"
REPO = Path("/content/RAG_Pipeline")

if not REPO.exists():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "-q"], check=True)

# numpy và rank_bm25 là đủ cho ingest/chunk; google-genai và groq chỉ cần ở bước
# index nên bỏ qua để khỏi chờ.
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "numpy", "rank_bm25"], check=True)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "raw"
CORPUS = REPO / "data" / "corpus"
RAW.mkdir(exist_ok=True)
CORPUS.mkdir(parents=True, exist_ok=True)

from tools.normalize_raw import (  # noqa: E402
    ExtractionError,
    corpus_audit,
    detect_format,
    extract_text,
    normalize,
    outline,
    select_articles,
    word_count,
    write_corpus_file,
)

print(f"repo   : {REPO}")
print(f"raw    : {RAW}")
print(f"corpus : {CORPUS}")

# %% [markdown]
# ## Cell 3 — Upload file nguồn
#
# Chọn cả 4 file cùng lúc. Cột **định dạng** đọc từ magic bytes chứ không từ đuôi
# file: `.doc` nhị phân cũ (OLE2) hay bị đặt tên `.docx` và ngược lại, mà
# `python-docx` chỉ đọc được ZIP.

# %%
from google.colab import files  # noqa: E402

uploaded = files.upload()
for name in uploaded:
    Path(name).replace(RAW / name)

print(f"\n{'tên file':<44} {'KB':>9}  định dạng")
print("-" * 68)
for path in sorted(RAW.iterdir()):
    if path.suffix == ".txt":
        continue
    print(f"{path.name:<44} {path.stat().st_size / 1024:>9.1f}  {detect_format(path)}")

# %% [markdown]
# ## Cell 4 — Trích text thô
#
# **Điền `SOURCES`** theo đúng tên file in ra ở Cell 3.
#
# Text thô được ghi ra `raw/<doc_id>.raw.txt` để mở xem khi cần lần lỗi.
# PDF trích ra quá ngắn hoặc mất dấu sẽ raise ngay tại đây — corpus rỗng mà không
# báo lỗi thì phải tới tận bước eval mới phát hiện.

# %%
SOURCES: dict[str, str] = {
    "luat_91_2025": "luat-bao-ve-du-lieu-ca-nhan.pdf",
    "nd_356_2025": "nghi-dinh-356-2025.doc",
    "nd_13_2023": "nghi-dinh-13-2023.doc",
    "luat_attt_86_2015": "luat-an-toan-thong-tin-mang.doc",
}

raw_text: dict[str, str] = {}
for doc_id, filename in SOURCES.items():
    source = RAW / filename
    try:
        text = extract_text(source)
    except ExtractionError as error:
        print(f"✗ {doc_id:<20} {error}")
        continue
    raw_text[doc_id] = text
    (RAW / f"{doc_id}.raw.txt").write_text(text, encoding="utf-8")
    print(f"✓ {doc_id:<20} {len(text):>8,} ký tự  ({detect_format(source)})")

# %% [markdown]
# ## Cell 5 — Chuẩn hoá
#
# Sáu bước theo thứ tự: dọn ký tự → bỏ số trang và header chạy trang → **nối dòng
# bị wrap** → gộp khoảng trắng → cắt phần trước `Chương`/`Điều 1.` → cắt từ
# `Nơi nhận`/`PHỤ LỤC` trở đi.
#
# Bước nối dòng là bước quan trọng nhất: PDF ngắt dòng cứng giữa câu, không nối
# lại thì mỗi dòng vật lý thành một đoạn và khoản vỡ vụn.

# %%
clean_text: dict[str, str] = {}
for doc_id, text in raw_text.items():
    normalized, report = normalize(text)
    clean_text[doc_id] = normalized
    print(f"── {doc_id}")
    print(f"   {report.summary()}")
    print()

# %% [markdown]
# ## Cell 6 — Soi cấu trúc TRƯỚC khi cắt phạm vi
#
# Đọc cây này rồi mới quyết định giữ Điều nào ở Cell 7.
#
# Ba cảnh báo cần để ý:
# - **0 khoản** — Điều chỉ có một đoạn văn (hợp lệ), hoặc khoản không được nhận diện
# - **khoản nhảy cóc** — dòng `N.` bị wrap nên mất một khoản
# - **dài > 2000 token** — gần như chắc chắn dòng `Điều N.` kế tiếp bị wrap nên hai
#   Điều dính vào nhau

# %%
for doc_id, text in clean_text.items():
    print(f"═══ {doc_id} " + "═" * (60 - len(doc_id)))
    print(outline(text))
    print()

# %% [markdown]
# ## Cell 7 — Cắt phạm vi
#
# **Điền `KEEP`** theo cây ở Cell 6. Dùng `range(1, 30)` hoặc list cụ thể.
#
# `select_articles` giữ lại các Điều được chọn **và** dòng tiêu đề Chương chứa
# chúng. Không đánh số lại: số Điều gốc là thứ người đọc dùng để tra ngược văn
# bản, đánh số lại làm mọi trích dẫn sai.

# %%
KEEP: dict[str, dict[str, object]] = {
    "luat_91_2025": {"articles": range(1, 30)},
    "nd_356_2025": {"articles": range(1, 20)},
    "nd_13_2023": {"articles": range(1, 25)},
    "luat_attt_86_2015": {"articles": range(1, 20)},
}

WORD_BUDGET = 25_000

selected: dict[str, str] = {}
total_words = 0
for doc_id, text in clean_text.items():
    trimmed = select_articles(text, KEEP.get(doc_id, {"articles": []}))
    selected[doc_id] = trimmed
    words = word_count(trimmed)
    total_words += words
    print(f"{doc_id:<22} {words:>7,} từ")

print("-" * 32)
print(f"{'tổng':<22} {total_words:>7,} từ")
if total_words > WORD_BUDGET:
    print(
        f"\n⚠ Vượt ngân sách {WORD_BUDGET:,} từ. Corpus càng to thì gold set càng "
        "khó phủ và quota embedding free tier càng căng — cân nhắc bớt Điều."
    )

# %% [markdown]
# ## Cell 8 — Gắn frontmatter và ghi ra `data/corpus/`
#
# **Điền `FRONTMATTER`**. `render_frontmatter` kiểm `REQUIRED_KEYS` và
# `VALID_STATUS` nhập thẳng từ `src/ingest.py`, nên định dạng không thể trôi khỏi
# nhau mà phải chạy cả pipeline mới phát hiện.

# %%
FRONTMATTER: dict[str, dict[str, object]] = {
    "luat_91_2025": {
        "doc_id": "luat_91_2025",
        "title": "Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15",
        "doc_type": "luat",
        "issued_date": "2025-06-26",
        "effective_from": "2026-01-01",
        "effective_to": None,
        "status": "active",
    },
    "nd_356_2025": {
        "doc_id": "nd_356_2025",
        "title": "Nghị định 356/2025/NĐ-CP quy định chi tiết Luật Bảo vệ dữ liệu cá nhân",
        "doc_type": "nghi_dinh",
        "issued_date": "2025-12-31",
        "effective_from": "2026-01-01",
        "effective_to": None,
        "status": "active",
    },
    "nd_13_2023": {
        "doc_id": "nd_13_2023",
        "title": "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
        "doc_type": "nghi_dinh",
        "issued_date": "2023-04-17",
        "effective_from": "2023-07-01",
        "effective_to": "2026-01-01",
        "status": "expired",
    },
    "luat_attt_86_2015": {
        "doc_id": "luat_attt_86_2015",
        "title": "Luật An toàn thông tin mạng số 86/2015/QH13",
        "doc_type": "luat",
        "issued_date": "2015-11-19",
        "effective_from": "2016-07-01",
        "effective_to": None,
        "status": "active",
    },
}

for doc_id, text in selected.items():
    path = write_corpus_file(CORPUS, FRONTMATTER[doc_id], text)
    print(f"✓ {path.relative_to(REPO)}  ({path.stat().st_size / 1024:.1f} KB)")

# %% [markdown]
# ## Cell 9 — Kiểm tra khép vòng
#
# Nạp lại chính những file vừa ghi bằng `src/ingest.py` và `src/chunk.py`. Đây là
# bài kiểm thật: nếu cell này chạy sạch thì corpus dùng được.
#
# `chunk_document` gọi `verify_coverage` bên trong, nên lỗ thủng coverage sẽ raise
# ngay chứ không âm thầm trôi xuống bước eval.

# %%
import traceback  # noqa: E402

from src.chunk import _report, chunk_corpus  # noqa: E402
from src.ingest import load_corpus  # noqa: E402

try:
    documents = load_corpus(CORPUS)
    chunks = chunk_corpus(documents)
except Exception:
    print("✗ Nạp corpus thất bại — sửa text rồi chạy lại từ Cell 7:\n")
    traceback.print_exc()
else:
    # Cùng hàm mà `python -m src.chunk` dùng, nên số liệu khớp nhau.
    _report(documents, chunks)
    print()
    print(corpus_audit(documents, chunks, KEEP))

# %% [markdown]
# ## Cell 10 — Tải về
#
# Giải nén vào `data/corpus/` của repo local rồi commit.
#
# **Chỉ commit `data/corpus/*.txt`.** Thư mục `raw/` và mọi `*.doc *.docx *.pdf`
# đã nằm trong `.gitignore` — file nguồn nặng, là nhị phân, và không phải thứ
# repo cần lưu.

# %%
import shutil  # noqa: E402

archive = shutil.make_archive("/content/corpus", "zip", root_dir=CORPUS)
print(f"{archive}  ({Path(archive).stat().st_size / 1024:.1f} KB)")
files.download(archive)
