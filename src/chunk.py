"""Chia chunk theo cấu trúc văn bản QPPL, kích thước chỉ là ràng buộc phụ.

Cắt ở ranh giới Điều trước, chỉ khi một Điều vượt CHUNK_MAX_CHARS mới cắt tiếp
ở ranh giới Khoản. Cắt cứng giữa câu là phương án cuối. Mọi chunk mang
char_start/char_end tuyệt đối vào Document.body nên coverage đo được bất kể
tham số chunk thay đổi thế nào.
"""

from __future__ import annotations

import re

from src.schema import Chunk, Document

ARTICLE_RE = re.compile(r"^#{0,6}\s*(Điều\s+\d+[a-zA-ZđĐ]?)\s*[.:．]", re.MULTILINE)
CHAPTER_RE = re.compile(r"^#{0,6}\s*(Chương\s+(?:[IVXLCDM]+|\d+))\b", re.MULTILINE)
SECTION_RE = re.compile(r"^#{0,6}\s*(Mục\s+\d+)\b", re.MULTILINE)
CLAUSE_RE = re.compile(r"^\d+\.\s", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _trim(body: str, start: int, end: int) -> tuple[int, int]:
    """Co hai đầu khoảng về phần không phải khoảng trắng, giữ offset chính xác."""
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def _headers(body: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    return [(m.start(), m.group(1).strip()) for m in pattern.finditer(body)]


def _label_before(headers: list[tuple[int, str]], position: int) -> str:
    label = ""
    for offset, text in headers:
        if offset <= position:
            label = text
        else:
            break
    return label


def _segments(body: str) -> list[tuple[int, int]]:
    """Ranh giới cấp Điều; nếu văn bản không có Điều thì lùi về cấp đoạn."""
    starts = [m.start() for m in ARTICLE_RE.finditer(body)]
    if not starts:
        bounds = [0] + [m.end() for m in PARAGRAPH_RE.finditer(body)] + [len(body)]
        return [(s, e) for s, e in zip(bounds, bounds[1:]) if s < e]
    if starts[0] > 0:
        starts.insert(0, 0)
    edges = starts + [len(body)]
    return [(s, e) for s, e in zip(edges, edges[1:]) if s < e]


def _hard_split(start: int, end: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Cắt cứng theo cửa sổ trượt có chồng lấn, cho Khoản dài bất thường."""
    stride = max(1, max_chars - overlap)
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        windows.append((cursor, min(cursor + max_chars, end)))
        if cursor + max_chars >= end:
            break
        cursor += stride
    return windows


def _split_segment(
    body: str, start: int, end: int, target_chars: int, max_chars: int, overlap: int
) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]

    clause_starts = [start] + [
        m.start() + start for m in CLAUSE_RE.finditer(body[start:end]) if m.start() > 0
    ]
    units = [(s, e) for s, e in zip(clause_starts, clause_starts[1:] + [end]) if s < e]

    pieces: list[tuple[int, int]] = []
    group_start, group_end = units[0]
    for unit_start, unit_end in units[1:]:
        if unit_end - group_start <= target_chars:
            group_end = unit_end
            continue
        pieces.append((group_start, group_end))
        group_start, group_end = unit_start, unit_end
    pieces.append((group_start, group_end))

    out: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        if piece_end - piece_start > max_chars:
            out.extend(_hard_split(piece_start, piece_end, max_chars, overlap))
        else:
            out.append((piece_start, piece_end))
    return out


def chunk_document(
    doc: Document, target_chars: int, max_chars: int, overlap: int
) -> list[Chunk]:
    chapters = _headers(doc.body, CHAPTER_RE)
    sections = _headers(doc.body, SECTION_RE)
    articles = _headers(doc.body, ARTICLE_RE)

    chunks: list[Chunk] = []
    for seg_start, seg_end in _segments(doc.body):
        for raw_start, raw_end in _split_segment(
            doc.body, seg_start, seg_end, target_chars, max_chars, overlap
        ):
            start, end = _trim(doc.body, raw_start, raw_end)
            if start >= end:
                continue
            crumbs = [doc.title]
            for label in (
                _label_before(chapters, seg_start),
                _label_before(sections, seg_start),
                _label_before(articles, seg_start),
            ):
                if label:
                    crumbs.append(label)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    breadcrumb=" > ".join(crumbs),
                    text=doc.body[start:end],
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks


def chunk_corpus(
    docs: list[Document], target_chars: int, max_chars: int, overlap: int
) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, target_chars, max_chars, overlap)]
