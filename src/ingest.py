"""Đọc văn bản QPPL thành cấu trúc Chương / Điều / Khoản.

Ranh giới cấu trúc chỉ được nhận ở ĐẦU DÒNG. Trong văn bản QPPL, các từ "Điều",
"Chương", "khoản" xuất hiện dày đặc bên trong nội dung dưới dạng tham chiếu chéo
("quy định tại khoản 2 Điều 9"), nên bất kỳ cách nhận dạng nào không neo vào đầu
dòng đều sẽ cắt nhầm giữa câu.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)
KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\s*$")
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
CLAUSE_RE = re.compile(r"^(\d+)\.\s")
POINT_RE = re.compile(r"^[a-zđ]\)\s")

REQUIRED_KEYS = ("doc_id", "title", "doc_type", "issued_date", "effective_from", "status")
VALID_STATUS = ("active", "expired")


class ParseError(ValueError):
    """Frontmatter sai định dạng, hoặc offset của clause không khớp body."""


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    doc_type: str
    issued_date: str
    effective_from: str
    effective_to: str | None
    status: str


@dataclass(frozen=True)
class Clause:
    """Một khoản. clause_no = 0 nghĩa là Điều không chia khoản."""

    clause_no: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Article:
    chapter: str | None
    article_no: int
    article_title: str
    clauses: list[Clause] = field(default_factory=list)


@dataclass(frozen=True)
class Document:
    meta: DocMeta
    body: str
    articles: list[Article] = field(default_factory=list)


def _parse_scalar(raw: str) -> str | None:
    """null / rỗng -> None; bóc một lớp nháy nếu có."""
    value = raw.strip()
    if value in ("null", "~", ""):
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_frontmatter(raw: str, source: str) -> tuple[DocMeta, str]:
    """Tách khối `--- key: value ---` khỏi thân văn bản.

    Regex + xử lý tay thay vì pyyaml: frontmatter ở đây chỉ có scalar phẳng,
    kéo cả một parser YAML về chỉ để đọc bảy dòng là không đáng.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ParseError(f"{source}: không tìm thấy khối frontmatter '---' hợp lệ ở đầu file")

    fields: dict[str, str | None] = {}
    for offset, line in enumerate(match.group(1).split("\n"), start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kv = KV_RE.match(stripped)
        if kv is None:
            raise ParseError(f"{source}:{offset}: không đúng dạng 'key: value' -> {stripped!r}")
        fields[kv.group(1)] = _parse_scalar(kv.group(2))

    missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
    if missing:
        raise ParseError(f"{source}: frontmatter thiếu key bắt buộc {missing}")
    if fields["status"] not in VALID_STATUS:
        raise ParseError(
            f"{source}: status={fields['status']!r} không hợp lệ, phải thuộc {VALID_STATUS}"
        )

    meta = DocMeta(
        doc_id=str(fields["doc_id"]),
        title=str(fields["title"]),
        doc_type=str(fields["doc_type"]),
        issued_date=str(fields["issued_date"]),
        effective_from=str(fields["effective_from"]),
        effective_to=fields.get("effective_to"),
        status=str(fields["status"]),
    )
    body = unicodedata.normalize("NFC", text[match.end() :])
    return meta, body


@dataclass(frozen=True)
class _Line:
    start: int
    stripped: str
    end: int
    """Offset ngay sau ký tự không-khoảng-trắng cuối cùng của dòng."""


def _scan_lines(body: str) -> list[_Line]:
    lines: list[_Line] = []
    position = 0
    for raw in body.splitlines(keepends=True):
        content = raw.rstrip()
        lines.append(_Line(start=position, stripped=content.strip(), end=position + len(content)))
        position += len(raw)
    return lines


def _chapter_label(lines: list[_Line], index: int) -> str:
    """"Chương I" kèm tiêu đề chương ở dòng kế tiếp nếu có.

    Tiêu đề chương là tuỳ chọn, nên chỉ nhận dòng không-rỗng kế tiếp khi nó
    không phải một ranh giới cấu trúc khác.
    """
    label = lines[index].stripped
    for following in lines[index + 1 :]:
        if not following.stripped:
            continue
        if (
            ARTICLE_RE.match(following.stripped)
            or CHAPTER_RE.match(following.stripped)
            or CLAUSE_RE.match(following.stripped)
            or POINT_RE.match(following.stripped)
        ):
            return label
        return f"{label}. {following.stripped}"
    return label


def _build_clauses(
    body: str, lines: list[_Line], first: int, stop: int, doc_id: str, article_no: int
) -> list[Clause]:
    """Cắt vùng nội dung của một Điều thành các khoản.

    Số khoản phải tăng liên tiếp từ 1. Dòng "3." xuất hiện khi đang chờ khoản 2
    gần như luôn là đánh số trong nội dung chứ không phải ranh giới khoản, nên
    được coi là văn bản thường và ghi cảnh báo.
    """
    groups: list[tuple[int, list[_Line]]] = []
    expected = 1

    for line in lines[first:stop]:
        match = CLAUSE_RE.match(line.stripped)
        if match is not None:
            number = int(match.group(1))
            if number == expected:
                groups.append((number, [line]))
                expected += 1
                continue
            logger.warning(
                "%s Điều %d: bỏ qua ranh giới khoản '%s' (đang chờ khoản %d), "
                "coi là nội dung thường",
                doc_id,
                article_no,
                line.stripped[:40],
                expected,
            )
        if not line.stripped:
            if groups:
                groups[-1][1].append(line)
            continue
        if groups:
            groups[-1][1].append(line)
        else:
            groups.append((0, [line]))

    clauses: list[Clause] = []
    for number, member_lines in groups:
        filled = [line for line in member_lines if line.stripped]
        if not filled:
            continue
        start, end = filled[0].start, filled[-1].end
        clauses.append(
            Clause(clause_no=number, text=body[start:end], char_start=start, char_end=end)
        )
    return clauses


def parse_body(body: str, doc_id: str) -> list[Article]:
    lines = _scan_lines(body)
    chapters: dict[int, str] = {}
    headers: list[tuple[int, int, str]] = []
    boundaries: list[int] = []

    current_chapter: str | None = None
    for index, line in enumerate(lines):
        if CHAPTER_RE.match(line.stripped):
            current_chapter = _chapter_label(lines, index)
            boundaries.append(index)
            continue
        article = ARTICLE_RE.match(line.stripped)
        if article is not None:
            headers.append((index, int(article.group(1)), article.group(2).strip()))
            boundaries.append(index)
            if current_chapter is not None:
                chapters[index] = current_chapter

    articles: list[Article] = []
    for index, article_no, title in headers:
        later = [b for b in boundaries if b > index]
        stop = min(later) if later else len(lines)
        articles.append(
            Article(
                chapter=chapters.get(index),
                article_no=article_no,
                article_title=title,
                clauses=_build_clauses(body, lines, index + 1, stop, doc_id, article_no),
            )
        )
    return articles


def _verify_offsets(document: Document) -> None:
    """Offset phải cắt lại đúng text. Dùng raise thay vì assert vì `python -O`
    sẽ bỏ assert, mà mất bất biến này thì mọi gold_span đều vô nghĩa."""
    for article in document.articles:
        for clause in article.clauses:
            actual = document.body[clause.char_start : clause.char_end]
            if actual != clause.text:
                raise ParseError(
                    f"lệch offset: doc_id={document.meta.doc_id} "
                    f"article_no={article.article_no} clause_no={clause.clause_no} "
                    f"[{clause.char_start}, {clause.char_end}) "
                    f"body={actual[:60]!r} != text={clause.text[:60]!r}"
                )


def load_document(path: Path) -> Document:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"), path.name)
    document = Document(meta=meta, body=body, articles=parse_body(body, meta.doc_id))
    _verify_offsets(document)
    return document


def load_corpus(directory: Path) -> list[Document]:
    """Đọc mọi .txt trong thư mục, sắp theo doc_id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục corpus: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".txt")
    if not paths:
        raise FileNotFoundError(f"Không có file .txt nào trong {directory}")

    documents = [load_document(path) for path in paths]
    seen: dict[str, str] = {}
    for document in documents:
        doc_id = document.meta.doc_id
        if doc_id in seen:
            raise ParseError(f"doc_id trùng '{doc_id}' giữa {seen[doc_id]} và {document.meta.title}")
        seen[doc_id] = document.meta.title
    return sorted(documents, key=lambda d: d.meta.doc_id)
