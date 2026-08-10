"""Đọc data/corpus/ thành Document đã chuẩn hoá, giữ trục ký tự ổn định.

Quy ước trục ký tự (gold_spans trong eval/questions.json phải theo đúng quy ước
này): body = phần sau khối frontmatter, đã NFC-normalize, đã đổi CRLF/CR thành
LF, đã strip hai đầu. `run.py dump-bodies` ghi đúng chuỗi đó ra
outputs/bodies/<doc_id>.txt để soạn gold span mà không phải đoán.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from src.schema import Document

FRONTMATTER_FENCE = "---"
REQUIRED_KEYS = ("doc_id", "title")
SUPPORTED_SUFFIXES = (".md", ".txt")


class FrontmatterError(ValueError):
    """Frontmatter thiếu key bắt buộc hoặc sai định dạng."""


def normalize_text(raw: str) -> str:
    """Chuẩn hoá về dạng duy nhất mà mọi char offset trong repo tham chiếu tới."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def parse_frontmatter(raw: str, source: str) -> tuple[dict[str, str], str]:
    """Tách khối `--- key: value ---` ở đầu file khỏi phần thân.

    Cố ý chỉ hỗ trợ `key: value` phẳng thay vì kéo thêm PyYAML — corpus không
    cần cấu trúc lồng nhau, và giữ dependency đúng như ràng buộc đề bài.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    if not text.startswith(FRONTMATTER_FENCE):
        raise FrontmatterError(f"{source}: thiếu khối frontmatter mở đầu bằng '---'")
    end = text.find(f"\n{FRONTMATTER_FENCE}", len(FRONTMATTER_FENCE))
    if end == -1:
        raise FrontmatterError(f"{source}: frontmatter không có '---' đóng")

    block = text[len(FRONTMATTER_FENCE) : end]
    body_start = text.find("\n", end + 1 + len(FRONTMATTER_FENCE))
    body = "" if body_start == -1 else text[body_start + 1 :]

    meta: dict[str, str] = {}
    for lineno, line in enumerate(block.split("\n"), start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise FrontmatterError(f"{source}:{lineno}: dòng không có dạng 'key: value'")
        key, _, value = stripped.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")

    missing = [k for k in REQUIRED_KEYS if not meta.get(k)]
    if missing:
        raise FrontmatterError(f"{source}: frontmatter thiếu key bắt buộc {missing}")
    return meta, body


def load_document(path: Path) -> Document:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"), path.name)
    normalized = normalize_text(body)
    if not normalized:
        raise FrontmatterError(f"{path.name}: phần thân rỗng")
    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        body=normalized,
        meta={k: v for k, v in meta.items() if k not in ("doc_id", "title")},
        source_path=path.name,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Đọc mọi .md/.txt trong thư mục, sắp theo doc_id cho ổn định."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục corpus: {corpus_dir}")
    paths = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"Không có file {SUPPORTED_SUFFIXES} nào trong {corpus_dir}")

    docs = [load_document(p) for p in paths]
    seen: dict[str, str] = {}
    for doc in docs:
        if doc.doc_id in seen:
            raise FrontmatterError(
                f"doc_id trùng '{doc.doc_id}' giữa {seen[doc.doc_id]} và {doc.source_path}"
            )
        seen[doc.doc_id] = doc.source_path
    return sorted(docs, key=lambda d: d.doc_id)


def write_documents(docs: list[Document], out_path: Path, body_dir: Path) -> None:
    """Ghi documents.json và bản body thuần để đối chiếu gold span."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([d.to_json() for d in docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for doc in docs:
        (body_dir / f"{doc.doc_id}.txt").write_text(doc.body, encoding="utf-8")
