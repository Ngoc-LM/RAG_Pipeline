"""Phát hiện câu hỏi gold copy từ vựng của chính đoạn văn nguồn.

Câu hỏi trùng nhiều từ với gold_span sẽ làm BM25 thắng một cách giả tạo và thổi
phồng Recall@k của mọi arm. Đây là bước QC cho eval/questions.json, chạy trước
khi tin bất kỳ con số nào trong bảng ablation.
"""

from __future__ import annotations

from typing import Any, Sequence

import config
from src.schema import Document, Question
from src.tokenize_vi import syllables


def jaccard(left: str, right: str) -> float:
    """Jaccard trên tập unigram âm tiết; dùng unigram để không phạt hai lần."""
    a, b = set(syllables(left)), set(syllables(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def span_text(docs: dict[str, Document], doc_id: str, start: int, end: int) -> str:
    doc = docs.get(doc_id)
    if doc is None:
        raise KeyError(f"gold_span trỏ tới doc_id không tồn tại: {doc_id}")
    if not 0 <= start < end <= len(doc.body):
        raise ValueError(
            f"gold_span ngoài phạm vi cho {doc_id}: [{start}, {end}) "
            f"nhưng body dài {len(doc.body)}"
        )
    return doc.body[start:end]


def check_leakage(
    questions: Sequence[Question],
    documents: Sequence[Document],
    threshold: float = config.LEAKAGE_JACCARD_MAX,
) -> dict[str, Any]:
    by_id = {d.doc_id: d for d in documents}
    rows: list[dict[str, Any]] = []
    for question in questions:
        for index, span in enumerate(question.gold_spans):
            text = span_text(by_id, span.doc_id, span.char_start, span.char_end)
            score = jaccard(question.question, text)
            rows.append(
                {
                    "qid": question.qid,
                    "span_index": index,
                    "doc_id": span.doc_id,
                    "jaccard": round(score, 4),
                    "flagged": score > threshold,
                }
            )
    flagged = [r for r in rows if r["flagged"]]
    return {
        "threshold": threshold,
        "n_spans": len(rows),
        "n_flagged": len(flagged),
        "max_jaccard": max((r["jaccard"] for r in rows), default=0.0),
        "flagged": flagged,
        "rows": rows,
    }
