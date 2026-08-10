"""CLI hỗ trợ gán nhãn gold_span bằng tay.

Gold span là offset ký tự thô, gõ tay thì sai lúc nào không biết. Công cụ này
làm ba việc: tìm chuỗi và trả về offset của KHOẢN chứa nó, in nguyên một Điều
kèm offset từng khoản, và soi lại file gold set đã soạn để bắt span gán nhầm.

    python -m tools.annotate --corpus data/corpus --grep "thời hạn phản hồi"
    python -m tools.annotate --corpus data/corpus --article luat_91_2025:9
    python -m tools.annotate --corpus data/corpus --grep "phản hồi" --emit q07
    python -m tools.annotate --corpus data/corpus --validate eval/questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

from src import config
from src.chunk import citation_label, estimate_tokens
from src.ingest import Article, Clause, Document, load_corpus

EXCERPT_WIDTH = 300
VALIDATE_PREVIEW = 120


def _markers() -> tuple[str, str]:
    """Đảo màu khi ra terminal, dấu ngoặc khi bị pipe hoặc chuyển tiếp."""
    if sys.stdout.isatty():
        return "\x1b[7m", "\x1b[0m"
    return "»", "«"


def fold(text: str, no_accent: bool) -> tuple[str, list[int]]:
    """Chuẩn hoá để so khớp, kèm ánh xạ ngược về offset gốc.

    Bỏ dấu làm đổi độ dài chuỗi, nên không thể tìm trên chuỗi đã bỏ dấu rồi
    dùng thẳng chỉ số đó. `positions[i]` là offset trong `text` của ký tự sinh
    ra ký tự thứ i của chuỗi đã chuẩn hoá.
    """
    folded: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        piece = character.lower()
        if no_accent:
            decomposed = unicodedata.normalize("NFD", piece)
            piece = "".join(c for c in decomposed if not unicodedata.combining(c))
            piece = piece.replace("đ", "d")
        for produced in piece:
            folded.append(produced)
            positions.append(index)
    return "".join(folded), positions


def find_matches(document: Document, needle: str, no_accent: bool) -> list[tuple[int, int]]:
    """Span tuyệt đối của mọi lần xuất hiện trong body."""
    haystack, positions = fold(document.body, no_accent)
    pattern, _ = fold(needle, no_accent)
    if not pattern:
        return []

    spans: list[tuple[int, int]] = []
    cursor = haystack.find(pattern)
    while cursor != -1:
        start = positions[cursor]
        end = positions[cursor + len(pattern) - 1] + 1
        spans.append((start, end))
        cursor = haystack.find(pattern, cursor + 1)
    return spans


def locate(document: Document, position: int) -> tuple[Article, Clause] | None:
    """Khoản chứa một offset. None nếu offset rơi vào tiêu đề Điều hoặc khe trống."""
    for article in document.articles:
        for clause in article.clauses:
            if clause.char_start <= position < clause.char_end:
                return article, clause
    return None


def article_span(article: Article) -> tuple[int, int] | None:
    if not article.clauses:
        return None
    return article.clauses[0].char_start, article.clauses[-1].char_end


def _has_numbered(article: Article) -> bool:
    return any(c.clause_no > 0 for c in article.clauses)


def short_label(article: Article, clause: Clause) -> str:
    if clause.clause_no == 0:
        return f"Điều {article.article_no} (đoạn mở đầu)" if _has_numbered(article) else (
            f"Điều {article.article_no}"
        )
    return f"Điều {article.article_no} Khoản {clause.clause_no}"


def full_label(document: Document, article: Article, clause: Clause) -> str:
    return citation_label(
        document.meta.title,
        article.article_no,
        str(clause.clause_no),
        clause.clause_no == 0 and _has_numbered(article),
    )


def excerpt(text: str, rel_start: int, rel_end: int, width: int = EXCERPT_WIDTH) -> str:
    """Cắt còn `width` ký tự quanh phần khớp, phần khớp được highlight."""
    on, off = _markers()
    if len(text) <= width:
        start, end = 0, len(text)
    else:
        margin = max(0, (width - (rel_end - rel_start)) // 2)
        end = min(len(text), max(rel_start - margin, 0) + width)
        start = max(0, end - width)

    visible_start = max(start, rel_start)
    visible_end = min(end, rel_end)
    if visible_start < visible_end:
        rendered = (
            text[start:visible_start]
            + on
            + text[visible_start:visible_end]
            + off
            + text[visible_end:end]
        )
    else:
        rendered = text[start:end]

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{rendered}{suffix}".replace("\n", " ")


def cmd_grep(
    documents: Sequence[Document], needle: str, no_accent: bool, emit: str | None
) -> int:
    hits: list[tuple[Document, Article, Clause]] = []

    for document in documents:
        for start, _ in find_matches(document, needle, no_accent):
            found = locate(document, start)
            if found is None:
                print(
                    f"{document.meta.doc_id} | (ngoài phạm vi khoản, offset {start}) | "
                    "khớp nằm ở tiêu đề Điều hoặc khe giữa các khoản"
                )
                continue
            article, clause = found
            hits.append((document, article, clause))

            relative = start - clause.char_start
            print(
                f"{document.meta.doc_id} | {short_label(article, clause)} | "
                f"{clause.char_start}-{clause.char_end}"
            )
            print(f"  {full_label(document, article, clause)}")
            print(f"  {excerpt(clause.text, relative, relative + (len(needle)))}")
            print()

    if not hits:
        print(f"Không tìm thấy {needle!r} trong {len(documents)} văn bản.")
        return 1

    if emit is not None:
        print(_emit_fragment(emit, hits))
    return 0


def _emit_fragment(qid: str, hits: Sequence[tuple[Document, Article, Clause]]) -> str:
    """JSON dán thẳng làm một phần tử của mảng trong eval/questions.json."""
    spans: list[dict[str, object]] = []
    for document, _, clause in hits:
        span = {
            "doc_id": document.meta.doc_id,
            "char_start": clause.char_start,
            "char_end": clause.char_end,
        }
        if span not in spans:
            spans.append(span)

    skeleton = {
        "qid": qid,
        "question": "TODO",
        "type": "factoid_1hop",
        "answerable": True,
        "gold_spans": spans,
        "gold_answer": "TODO",
    }
    return "--- dán vào eval/questions.json ---\n" + json.dumps(
        skeleton, ensure_ascii=False, indent=2
    )


def cmd_article(documents: Sequence[Document], target: str) -> int:
    if ":" not in target:
        print(f"--article cần dạng doc_id:số_điều, nhận được {target!r}", file=sys.stderr)
        return 1
    doc_id, _, raw_no = target.rpartition(":")
    if not raw_no.isdigit():
        print(f"số điều không hợp lệ: {raw_no!r}", file=sys.stderr)
        return 1

    document = next((d for d in documents if d.meta.doc_id == doc_id), None)
    if document is None:
        available = ", ".join(sorted(d.meta.doc_id for d in documents))
        print(f"Không có doc_id {doc_id!r}. Có: {available}", file=sys.stderr)
        return 1

    article = next((a for a in document.articles if a.article_no == int(raw_no)), None)
    if article is None:
        numbers = ", ".join(str(a.article_no) for a in document.articles)
        print(f"{doc_id} không có Điều {raw_no}. Có: {numbers}", file=sys.stderr)
        return 1

    span = article_span(article)
    print(f"{doc_id} | Điều {article.article_no}. {article.article_title}")
    print(f"  chương : {article.chapter or '(không có)'}")
    print(f"  span   : {span[0]}-{span[1]}" if span else "  span   : (không có khoản)")
    print()
    for clause in article.clauses:
        print(
            f"  {short_label(article, clause)} | {clause.char_start}-{clause.char_end} "
            f"| ~{estimate_tokens(clause.text)} token"
        )
        for line in clause.text.split("\n"):
            print(f"      {line}")
        print()
    return 0


def cmd_validate(documents: Sequence[Document], path: Path) -> int:
    if not path.is_file():
        print(f"Không thấy file gold set: {path}", file=sys.stderr)
        return 1

    by_id = {d.meta.doc_id: d for d in documents}
    questions = json.loads(path.read_text(encoding="utf-8"))
    errors = 0
    checked = 0

    for question in questions:
        qid = question.get("qid", "?")
        for index, span in enumerate(question.get("gold_spans", [])):
            checked += 1
            doc_id = span.get("doc_id")
            start, end = span.get("char_start"), span.get("char_end")
            document = by_id.get(doc_id)

            if document is None:
                print(f"LỖI {qid} span {index}: doc_id {doc_id!r} không có trong corpus")
                errors += 1
                continue

            length = len(document.body)
            if not (isinstance(start, int) and isinstance(end, int)):
                print(f"LỖI {qid} span {index}: char_start/char_end phải là số nguyên")
                errors += 1
                continue
            if not 0 <= start < end <= length:
                print(
                    f"LỖI {qid} span {index}: [{start}, {end}) ngoài phạm vi "
                    f"— body của {doc_id} dài {length}"
                )
                errors += 1
                continue

            container = next(
                (
                    a
                    for a in document.articles
                    if (s := article_span(a)) is not None and s[0] <= start and end <= s[1]
                ),
                None,
            )
            if container is None:
                print(
                    f"LỖI {qid} span {index}: [{start}, {end}) không nằm trọn trong một Điều "
                    "— nhiều khả năng gán nhầm ranh giới"
                )
                errors += 1
                continue

            found = locate(document, start)
            where = short_label(container, found[1]) if found else f"Điều {container.article_no}"
            preview = document.body[start : start + VALIDATE_PREVIEW].replace("\n", " ")
            tail = "…" if end - start > VALIDATE_PREVIEW else ""
            print(f"{qid} span {index} | {doc_id} [{start}, {end}) | {where}")
            print(f'    "{preview}{tail}"')

    print(f"\n{checked} span, {errors} lỗi.")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.annotate", description="Hỗ trợ gán nhãn gold_span bằng tay"
    )
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--grep", help="Chuỗi cần tìm, không phân biệt hoa thường")
    parser.add_argument(
        "--no-accent", action="store_true", help="Bỏ dấu hai phía trước khi so khớp"
    )
    parser.add_argument("--article", help="In nguyên một Điều, dạng doc_id:số_điều")
    parser.add_argument("--emit", metavar="QID", help="In JSON fragment cho các kết quả --grep")
    parser.add_argument("--validate", metavar="PATH", help="Soi lại gold set đã soạn")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.grep, args.article, args.validate)):
        build_parser().print_help()
        return 1

    documents = load_corpus(Path(args.corpus))
    status = 0
    if args.grep:
        status |= cmd_grep(documents, args.grep, args.no_accent, args.emit)
    if args.article:
        status |= cmd_article(documents, args.article)
    if args.validate:
        status |= cmd_validate(documents, Path(args.validate))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
