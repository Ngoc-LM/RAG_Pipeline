"""Chia chunk lấy KHOẢN làm đơn vị cơ bản, không dùng cửa sổ trượt cố định.

Ranh giới ngữ nghĩa của văn bản QPPL là khoản: một khoản là một quy phạm trọn
vẹn, đọc tách ra vẫn đúng nghĩa. Cửa sổ trượt cố định cắt ngang giữa khoản làm
hỏng cả hai đầu — retrieval mất tín hiệu vì điều kiện và hệ quả nằm ở hai chunk
khác nhau, còn trích dẫn thì mất tính hợp lệ vì không còn trỏ được tới một đơn
vị mà người đọc tra ngược lại được trong văn bản gốc.

Cắt câu chỉ là phương án cho khoản dài quá trần, và gộp chỉ xảy ra giữa các
khoản liền kề trong cùng một Điều.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src import config
from src.ingest import Article, Clause, DocMeta, Document, load_corpus
from src.intervals import interval_union

SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
ABBREVIATIONS = frozenset(
    {
        "tp.", "tt.", "ts.", "ths.", "gs.", "pgs.", "nđ.", "qđ.", "tw.",
        "đ.", "tr.", "vd.", "v.v.", "stt.", "kt.", "tl.", "kg.", "khcn.",
    }
)


def estimate_tokens(text: str) -> int:
    """Ước lượng token bằng len(text) / CHARS_PER_TOKEN.

    Corpus cỡ ~200 chunk thì sai số vài phần trăm của phép ước lượng này không
    đổi được quyết định nào: nó chỉ dùng để so với trần cắt và sàn gộp. Kéo một
    tokenizer thật về đổi lấy thêm dependency, thêm thời gian nạp và thêm một
    thứ phải giữ đồng bộ với model — không đáng.
    """
    return max(1, round(len(text) / config.CHARS_PER_TOKEN))


def citation_label(
    doc_title: str, article_no: int, clause_range: str, is_lead_in: bool
) -> str:
    """Nhãn trích dẫn hiển thị cho người đọc.

    Đoạn mở đầu của một Điều có chia khoản phải được gọi tên riêng: nếu ghi
    "Điều 9 <title>" thì trông như đang trích dẫn cả Điều, trong khi thực tế chỉ
    trích phần chapeau. "Điều 9 <title>" chỉ dùng khi Điều thật sự không chia khoản.
    """
    if is_lead_in:
        return f"Điều {article_no} (đoạn mở đầu) {doc_title}"
    if clause_range == "0":
        return f"Điều {article_no} {doc_title}"
    return f"Điều {article_no} Khoản {clause_range} {doc_title}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    effective_to: str | None
    chapter: str | None
    article_no: int
    article_title: str
    clause_range: str
    text: str
    char_start: int
    char_end: int
    n_tokens: int
    is_lead_in: bool = False

    @property
    def citation_label(self) -> str:
        """Chuỗi hiển thị trong câu trả lời cuối cùng, không phải chunk_id."""
        return citation_label(
            self.doc_title, self.article_no, self.clause_range, self.is_lead_in
        )

    @property
    def indexed_text(self) -> str:
        """Text đem đi đánh chỉ mục.

        Gắn cả nhãn trích dẫn lẫn tiêu đề Điều vào đầu. Tiêu đề Điều ("Thời hạn
        lưu trữ và huỷ dữ liệu") là tín hiệu truy xuất mạnh nhất của văn bản
        QPPL, nhưng nó nằm ngoài mọi khoản nên chỉ chunk đầu của Điều mới có nó
        trong `text`; các chunk còn lại sẽ mất tín hiệu đó nếu không gắn ở đây.
        """
        header = self.citation_label
        if self.article_title:
            header = f"{header} — {self.article_title}"
        return f"{header}\n{self.text}"


def _is_false_boundary(text: str, position: int) -> bool:
    """Dấu chấm ở `position` có phải kết câu thật không."""
    preceding = text[:position]
    token = re.search(r"\S+$", preceding)
    if token is None:
        return True
    word = token.group(0).lower() + "."
    if word in ABBREVIATIONS:
        return True
    if token.group(0).isdigit():
        return True
    return len(token.group(0)) == 1 and token.group(0).isupper()


def split_sentences(text: str, base: int) -> list[tuple[int, int]]:
    """Span câu tuyệt đối, phủ liền mạch toàn bộ `text`.

    Khoảng trắng giữa hai câu thuộc về câu đứng trước, nhờ vậy hợp các span
    luôn bằng đúng span gốc — điều kiện để metric coverage không bị thủng lỗ.
    """
    cuts: list[int] = []
    for match in SENTENCE_END_RE.finditer(text):
        if not _is_false_boundary(text, match.start()):
            cuts.append(match.end())

    spans: list[tuple[int, int]] = []
    previous = 0
    for cut in cuts:
        if cut > previous:
            spans.append((base + previous, base + cut))
            previous = cut
    if previous < len(text):
        spans.append((base + previous, base + len(text)))
    return spans or [(base, base + len(text))]


def _pack_with_overlap(body: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Gom câu thành phần <= CHUNK_MAX_TOKENS, phần liền kề chồng lấn nhau."""
    target_overlap = config.CHUNK_OVERLAP_RATIO * config.CHUNK_MAX_TOKENS
    parts: list[tuple[int, int]] = []
    start_index = 0

    while start_index < len(spans):
        end_index = start_index
        tokens = 0
        while end_index < len(spans):
            span_tokens = estimate_tokens(body[spans[end_index][0] : spans[end_index][1]])
            if end_index > start_index and tokens + span_tokens > config.CHUNK_MAX_TOKENS:
                break
            tokens += span_tokens
            end_index += 1

        parts.append((spans[start_index][0], spans[end_index - 1][1]))
        if end_index >= len(spans):
            break

        overlap_count, accumulated = 0, 0.0
        while end_index - overlap_count - 1 > start_index and accumulated < target_overlap:
            span = spans[end_index - overlap_count - 1]
            accumulated += estimate_tokens(body[span[0] : span[1]])
            overlap_count += 1
        start_index = max(start_index + 1, end_index - max(1, overlap_count))

    return parts


@dataclass(frozen=True)
class _Group:
    clauses: list[Clause]
    must_split: bool


def _span_tokens(body: str, group: Sequence[Clause]) -> int:
    return estimate_tokens(body[group[0].char_start : group[-1].char_end])


def _group_clauses(body: str, clauses: Sequence[Clause]) -> list[_Group]:
    """Tách hai pha để quy tắc cắt và quy tắc gộp không thể hợp thành nhau.

    Pha 1 đánh dấu mọi khoản vượt trần: chúng sẽ bị cắt thành part và không bao
    giờ tham gia gộp. Pha 2 chỉ gộp các khoản chưa bị cắt, chỉ khi tổng vẫn nằm
    dưới trần, và không bao giờ gộp xuyên qua một khoản đã bị cắt.

    Nhờ vậy mỗi chunk hoặc là một part của đúng một khoản, hoặc là hợp của các
    khoản nguyên vẹn — không bao giờ vừa gộp vừa cắt. Đánh đổi: khoản ngắn nằm
    cạnh một khoản dài sẽ đứng riêng dưới sàn gộp.
    """
    marked = [(c, estimate_tokens(c.text) > config.CHUNK_MAX_TOKENS) for c in clauses]
    groups: list[_Group] = []
    current: list[Clause] = []

    for clause, must_split in marked:
        # Đoạn mở đầu (clause_no 0) đứng riêng: gộp nó vào khoản 1 sẽ tạo ra
        # clause_range "0-1", một nhãn trích dẫn vô nghĩa.
        if must_split or clause.clause_no == 0:
            if current:
                groups.append(_Group(current, False))
                current = []
            groups.append(_Group([clause], must_split))
            continue

        if current and _span_tokens(body, current + [clause]) > config.CHUNK_MAX_TOKENS:
            groups.append(_Group(current, False))
            current = [clause]
        else:
            current = current + [clause]

        if _span_tokens(body, current) >= config.CHUNK_MIN_TOKENS:
            groups.append(_Group(current, False))
            current = []

    if current:
        previous = groups[-1] if groups else None
        mergeable = (
            previous is not None
            and not previous.must_split
            and previous.clauses[0].clause_no != 0
            and _span_tokens(body, previous.clauses + current) <= config.CHUNK_MAX_TOKENS
        )
        if mergeable and previous is not None:
            groups[-1] = _Group(previous.clauses + current, False)
        else:
            groups.append(_Group(current, False))
    return groups


def _clause_range(group: Sequence[Clause]) -> str:
    if len(group) == 1 or group[0].clause_no == group[-1].clause_no:
        return str(group[0].clause_no)
    return f"{group[0].clause_no}-{group[-1].clause_no}"


def _header_prefix_start(article: Article) -> int:
    """Điểm lùi về của chunk đầu tiên trong một Điều.

    Lùi tới đầu dòng "Điều N."; nếu Điều đó mở đầu một Chương thì lùi tiếp tới
    đầu khối "Chương X" để tiêu đề chương cũng nằm trong chunk.
    """
    if article.chapter_header_start is not None:
        return article.chapter_header_start
    return article.header_start


def chunk_article(document: Document, article: Article) -> list[Chunk]:
    body = document.body
    meta = document.meta
    chunks: list[Chunk] = []
    prefix_start = _header_prefix_start(article)

    has_numbered = any(c.clause_no > 0 for c in article.clauses)
    groups = _group_clauses(body, article.clauses)

    if not groups:
        line_end = body.find("\n", article.header_start)
        return [
            _make_chunk(
                meta, article, body, prefix_start,
                len(body) if line_end == -1 else line_end, "0", 0, False,
            )
        ]

    for group_index, group in enumerate(groups):
        members = group.clauses
        start, end = members[0].char_start, members[-1].char_end
        clause_range = _clause_range(members)
        is_lead_in = members[0].clause_no == 0 and has_numbered
        if group.must_split:
            parts = _pack_with_overlap(body, split_sentences(body[start:end], start))
        else:
            parts = [(start, end)]

        for part_index, (part_start, part_end) in enumerate(parts):
            if group_index == 0 and part_index == 0:
                part_start = prefix_start
            chunks.append(
                _make_chunk(
                    meta, article, body, part_start, part_end,
                    clause_range, part_index, is_lead_in,
                )
            )
    return chunks


def _make_chunk(
    meta: DocMeta,
    article: Article,
    body: str,
    char_start: int,
    char_end: int,
    clause_range: str,
    part_index: int,
    is_lead_in: bool,
) -> Chunk:
    text = body[char_start:char_end]
    return Chunk(
        chunk_id=f"{meta.doc_id}#a{article.article_no}#c{clause_range}#p{part_index}",
        doc_id=meta.doc_id,
        doc_title=meta.title,
        doc_type=meta.doc_type,
        status=meta.status,
        effective_from=meta.effective_from,
        effective_to=meta.effective_to,
        chapter=article.chapter,
        article_no=article.article_no,
        article_title=article.article_title,
        clause_range=clause_range,
        text=text,
        char_start=char_start,
        char_end=char_end,
        n_tokens=estimate_tokens(text),
        is_lead_in=is_lead_in,
    )


def verify_coverage(document: Document, chunks: Sequence[Chunk]) -> None:
    """Hợp các chunk phải phủ toàn bộ body, phần chừa ra chỉ được là khoảng trắng.

    Dùng raise chứ không assert vì `python -O` bỏ assert, và một lỗ thủng ở đây
    hỏng âm thầm: mọi gold_span chạm vùng không được phủ sẽ có Cov@k tụt dưới
    ngưỡng vĩnh viễn, cho Recall@k = 0 ở mọi arm mà không có dấu hiệu báo lỗi.
    """
    covered = interval_union(
        [(c.char_start, c.char_end) for c in chunks if c.doc_id == document.meta.doc_id]
    )
    holes: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        if cursor < start:
            holes.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(document.body):
        holes.append((cursor, len(document.body)))

    for start, end in holes:
        fragment = document.body[start:end]
        if fragment.strip():
            raise ValueError(
                f"lỗ thủng coverage: doc_id={document.meta.doc_id} [{start}, {end}) "
                f"không thuộc chunk nào: {fragment.strip()[:80]!r}"
            )


def chunk_document(document: Document) -> list[Chunk]:
    chunks = [c for article in document.articles for c in chunk_article(document, article)]
    verify_coverage(document, chunks)
    return chunks


def chunk_corpus(documents: Sequence[Document]) -> list[Chunk]:
    return [c for document in documents for c in chunk_document(document)]


def _percentile(values: Sequence[int], fraction: float) -> float:
    """Nội suy tuyến tính, đủ cho một bảng thống kê mô tả."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _report(documents: Sequence[Document], chunks: Sequence[Chunk]) -> None:
    articles = sum(len(d.articles) for d in documents)
    tokens = [c.n_tokens for c in chunks]
    median = _percentile(tokens, 0.5)
    p75 = _percentile(tokens, 0.75)

    print(f"documents : {len(documents)}")
    print(f"articles  : {articles}")
    print(f"chunks    : {len(chunks)}")
    print(
        "n_tokens  : min={} p25={:.0f} median={:.0f} p75={:.0f} max={}".format(
            min(tokens, default=0), _percentile(tokens, 0.25), median, p75, max(tokens, default=0)
        )
    )

    for status in ("active", "expired"):
        count = sum(1 for c in chunks if c.status == status)
        share = 100.0 * count / len(chunks) if chunks else 0.0
        print(f"status {status:<8}: {count} chunk ({share:.1f}%)")

    if median > 0 and p75 / median > 3:
        print(
            f"\nCẢNH BÁO: p75/median = {p75 / median:.1f} > 3. "
            "Phân phối lệch bất thường — nhiều khả năng logic gộp hoặc cắt đang sai."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Thống kê chunk trên một corpus")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    args = parser.parse_args()

    documents = load_corpus(Path(args.corpus))
    _report(documents, chunk_corpus(documents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
