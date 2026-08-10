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
from src.ingest import Article, Clause, Document, load_corpus

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

    @property
    def citation_label(self) -> str:
        """Chuỗi hiển thị trong câu trả lời cuối cùng, không phải chunk_id."""
        if self.clause_range == "0":
            return f"Điều {self.article_no} {self.doc_title}"
        return f"Điều {self.article_no} Khoản {self.clause_range} {self.doc_title}"

    @property
    def indexed_text(self) -> str:
        """Text đem đi đánh chỉ mục: gắn nhãn trích dẫn để chunk mang ngữ cảnh
        Điều/Khoản, nếu không thì một khoản tách rời gần như vô nghĩa."""
        return f"{self.citation_label}\n{self.text}"


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


def _group_clauses(body: str, clauses: Sequence[Clause]) -> list[list[Clause]]:
    """Gộp khoản ngắn với khoản liền kề, chỉ trong phạm vi một Điều."""
    groups: list[list[Clause]] = []
    current: list[Clause] = []

    for clause in clauses:
        current.append(clause)
        span_tokens = estimate_tokens(body[current[0].char_start : current[-1].char_end])
        if span_tokens >= config.CHUNK_MIN_TOKENS:
            groups.append(current)
            current = []

    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _clause_range(group: Sequence[Clause]) -> str:
    if len(group) == 1 or group[0].clause_no == group[-1].clause_no:
        return str(group[0].clause_no)
    return f"{group[0].clause_no}-{group[-1].clause_no}"


def chunk_article(document: Document, article: Article) -> list[Chunk]:
    body = document.body
    meta = document.meta
    chunks: list[Chunk] = []

    for group in _group_clauses(body, article.clauses):
        start, end = group[0].char_start, group[-1].char_end
        clause_range = _clause_range(group)
        if estimate_tokens(body[start:end]) <= config.CHUNK_MAX_TOKENS:
            parts = [(start, end)]
        else:
            parts = _pack_with_overlap(body, split_sentences(body[start:end], start))

        for part_index, (part_start, part_end) in enumerate(parts):
            text = body[part_start:part_end]
            chunks.append(
                Chunk(
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
                    char_start=part_start,
                    char_end=part_end,
                    n_tokens=estimate_tokens(text),
                )
            )
    return chunks


def chunk_document(document: Document) -> list[Chunk]:
    return [c for article in document.articles for c in chunk_article(document, article)]


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
