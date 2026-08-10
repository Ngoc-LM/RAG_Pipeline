"""Truy xuất: dense, BM25, hợp nhất RRF, rerank listwise bằng LLM.

Bốn arm dùng chung một đường đi để bảng ablation so sánh được công bằng:
    bm25          chỉ từ khoá
    dense         chỉ ngữ nghĩa
    hybrid        RRF hợp nhất hai danh sách trên
    hybrid_rerank hybrid rồi cho LLM chấm lại — arm mặc định của pipeline
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np

import config
from src.index import Index
from src.llm import gemini_generate
from src.schema import Chunk
from src.tokenize_vi import tokenize

ARMS: Final[tuple[str, ...]] = ("bm25", "dense", "hybrid", "hybrid_rerank")
RERANK_SCALE: Final[int] = 10

RERANK_PROMPT = """Bạn là bộ chấm điểm mức liên quan cho hệ thống tra cứu văn bản pháp luật Việt Nam.

Câu hỏi:
{question}

Dưới đây là {n} đoạn văn bản ứng viên, đánh số từ 1 đến {n}:
{candidates}

Chấm mỗi đoạn một điểm nguyên từ 0 đến {scale} theo mức hữu ích để TRẢ LỜI câu hỏi trên:
- {scale}: chứa trực tiếp thông tin trả lời được câu hỏi
- 5: cùng chủ đề và có ích một phần, nhưng không đủ để trả lời
- 0: không liên quan, hoặc chỉ trùng từ ngữ bề mặt

Chấm độc lập từng đoạn, không so sánh tương đối, không giả định đoạn nào cũng phải có điểm cao.
Trả về đúng JSON dạng: {{"scores": [{{"index": 1, "score": 0}}, ...]}} đủ cả {n} đoạn."""


@dataclass(frozen=True)
class Retrieval:
    arm: str
    chunks: tuple[Chunk, ...]
    scores: tuple[float, ...]
    rerank_top_score: float | None
    """Điểm rerank cao nhất, chuẩn hoá về [0,1]. None ở arm không có rerank."""


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    """k chỉ số điểm cao nhất, phá hoà bằng chỉ số nhỏ hơn cho ổn định."""
    k = min(k, scores.size)
    if k <= 0:
        return []
    partial = np.argpartition(-scores, k - 1)[:k]
    return sorted(partial.tolist(), key=lambda i: (-scores[i], i))


def dense_search(index: Index, query_vector: np.ndarray, k: int) -> list[int]:
    """Cosine brute-force; embedding đã chuẩn hoá L2 nên tích vô hướng là cosine."""
    if index.embeddings is None:
        raise ValueError("Index được dựng với with_dense=False, không chạy được arm dense")
    return _top_indices(index.embeddings @ query_vector, k)


def bm25_search(index: Index, question: str, k: int) -> list[int]:
    return _top_indices(np.asarray(index.bm25.get_scores(tokenize(question))), k)


def rrf_merge(rankings: Sequence[Sequence[int]], k: int) -> list[int]:
    """Reciprocal Rank Fusion: cộng 1/(k + hạng) qua các danh sách.

    Hợp nhất theo hạng chứ không theo điểm vì điểm BM25 và cosine không cùng
    thang đo, chuẩn hoá chúng về một thang luôn là lựa chọn tuỳ tiện.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: (-fused[i], i))


def rerank(
    question: str, candidates: Sequence[Chunk], *, offline: bool
) -> list[tuple[int, float]]:
    """Chấm lại toàn bộ ứng viên trong ĐÚNG MỘT lời gọi LLM.

    Pointwise mỗi ứng viên một call sẽ tốn n lần quota; free tier giới hạn RPM
    thấp nên listwise là lựa chọn bắt buộc chứ không chỉ là tối ưu.
    """
    if not candidates:
        return []

    numbered = "\n\n".join(
        f"[{i}] {c.indexed_text}" for i, c in enumerate(candidates, start=1)
    )
    prompt = RERANK_PROMPT.format(
        question=question, n=len(candidates), candidates=numbered, scale=RERANK_SCALE
    )
    raw = gemini_generate(
        task="rerank",
        model=config.RERANK_MODEL,
        input_obj={
            "question": question,
            "candidates": [c.indexed_text for c in candidates],
            "scale": RERANK_SCALE,
        },
        prompt=prompt,
        temperature=config.RERANK_TEMPERATURE,
        max_tokens=config.RERANK_MAX_TOKENS,
        thinking_budget=config.RERANK_THINKING_BUDGET,
        offline=offline,
    )

    scores = [0.0] * len(candidates)
    for item in _parse_scores(raw):
        position = int(item["index"]) - 1
        if 0 <= position < len(candidates):
            scores[position] = max(0.0, min(1.0, float(item["score"]) / RERANK_SCALE))

    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order]


def _parse_scores(raw: str) -> list[dict[str, Any]]:
    """Đọc JSON của reranker, chấp nhận cả object bọc lẫn array trần."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("scores", [])
    return [x for x in parsed if isinstance(x, dict) and "index" in x and "score" in x]


def retrieve(
    index: Index,
    question: str,
    *,
    arm: str,
    offline: bool,
    query_vector: np.ndarray | None = None,
    top_k: int | None = None,
) -> Retrieval:
    if arm not in ARMS:
        raise ValueError(f"arm không hợp lệ: {arm} (chọn trong {ARMS})")
    limit = top_k or config.TOP_K_CONTEXT

    if arm == "bm25":
        order = bm25_search(index, question, config.TOP_K_BM25)
    elif arm == "dense":
        if query_vector is None:
            raise ValueError("arm dense cần query_vector")
        order = dense_search(index, query_vector, config.TOP_K_DENSE)
    else:
        if query_vector is None:
            raise ValueError(f"arm {arm} cần query_vector")
        order = rrf_merge(
            [
                dense_search(index, query_vector, config.TOP_K_DENSE),
                bm25_search(index, question, config.TOP_K_BM25),
            ],
            k=config.RRF_K,
        )

    if arm != "hybrid_rerank":
        chunks = [index.chunks[i] for i in order[:limit]]
        return Retrieval(
            arm=arm,
            chunks=tuple(chunks),
            scores=tuple(1.0 / (rank + 1) for rank in range(len(chunks))),
            rerank_top_score=None,
        )

    candidates = [index.chunks[i] for i in order[: config.TOP_K_RERANK_CANDIDATES]]
    ranked = rerank(question, candidates, offline=offline)
    chunks = [candidates[i] for i, _ in ranked[:limit]]
    scores = [score for _, score in ranked[:limit]]
    return Retrieval(
        arm=arm,
        chunks=tuple(chunks),
        scores=tuple(scores),
        rerank_top_score=max(scores) if scores else 0.0,
    )
