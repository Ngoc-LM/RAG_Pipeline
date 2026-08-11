"""Bốn arm truy xuất: bm25, dense, hybrid (RRF), hybrid_rerank (listwise).

Bốn arm dùng chung một chỉ mục và một hàm vào, khác nhau đúng ở cách xếp hạng —
nhờ vậy bảng ablation so sánh được cơ chế xếp hạng chứ không so nhầm hai đường
dữ liệu khác nhau.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src import config
from src.chunk import Chunk
from src.index import Index, build_index
from src.ingest import load_corpus
from src.llm import CacheMiss, gemini_generate


@dataclass(frozen=True)
class Retrieved:
    """Một chunk ở một hạng, kèm mọi điểm thành phần dẫn tới hạng đó."""

    chunk: Chunk
    rank: int
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "citation_label": self.chunk.citation_label,
            "status": self.chunk.status,
            "char_start": self.chunk.char_start,
            "char_end": self.chunk.char_end,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "fusion_score": None if self.fusion_score is None else round(self.fusion_score, 6),
            "rerank_score": self.rerank_score,
        }


def rank_by_score(scores: np.ndarray, depth: int) -> list[int]:
    """Chỉ số chunk xếp giảm dần theo điểm, hoà thì chỉ số nhỏ đứng trước.

    Phá hoà bằng chỉ số chứ không để thứ tự tuỳ ý: BM25 trả 0.0 cho rất nhiều
    chunk cùng lúc, mà một thứ tự không tất định ở đó sẽ làm mọi con số đánh giá
    nhảy giữa hai lần chạy trên cùng dữ liệu.
    """
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:depth]]


def rrf_fuse(rank_lists: Sequence[Sequence[int]], k: int = config.RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank(d)), rank tính từ 1.

    Chunk vắng mặt trong một danh sách thì đơn giản là không cộng gì từ danh
    sách đó — không cần điểm phạt, vì độ sâu cắt đã là hình phạt rồi.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranked in rank_lists:
        for position, index in enumerate(ranked, start=1):
            fused[index] += 1.0 / (k + position)
    return dict(fused)


def _order_fused(fused: dict[int, float], depth: int) -> list[int]:
    return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))][:depth]


# --- Rerank listwise ------------------------------------------------------
RERANK_INSTRUCTION = """Bạn đang chấm mức liên quan của các trích đoạn văn bản quy phạm pháp luật đối với một câu hỏi.

Câu hỏi: {question}

Chấm MỖI trích đoạn theo thang số nguyên 0-{max_score}:
{max_score} = chứa trực tiếp căn cứ để trả lời câu hỏi
2 = không tự trả lời được nhưng là căn cứ bổ sung cần thiết (định nghĩa, điều kiện, ngoại lệ được dẫn chiếu)
1 = cùng chủ đề nhưng không góp phần trả lời
0 = không liên quan

Quy tắc hiệu lực: trích đoạn thuộc văn bản có trạng thái `expired` KHÔNG còn là căn cứ hợp lệ. Nếu một trích đoạn `active` trong danh sách đã bàn cùng nội dung thì chấm trích đoạn `expired` đó 0 điểm.

Trả về JSON đúng dạng, chấm đủ cả {n} trích đoạn, không thêm trường nào khác:
{{"scores": [{{"id": 1, "score": 2}}, ...]}}

Các trích đoạn:
{candidates}"""


def _render_candidates(chunks: Sequence[Chunk]) -> str:
    blocks: list[str] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        hieu_luc = chunk.status
        if chunk.effective_to:
            hieu_luc = f"{chunk.status}, hết hiệu lực {chunk.effective_to}"
        blocks.append(
            f"[{ordinal}] {chunk.citation_label} — {chunk.article_title}\n"
            f"    văn bản: {chunk.doc_id} · hiệu lực: {hieu_luc}\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def _parse_rerank(raw: str, n: int) -> dict[int, int]:
    """Đọc JSON reranker thành {ordinal: điểm}, bỏ ordinal ngoài phạm vi.

    Model bịa ra id không có trong danh sách thì bỏ, chứ không ánh xạ đại sang
    một chunk nào đó. Ứng viên không được chấm sẽ nhận 0 ở tầng trên: im lặng
    không phải bằng chứng ủng hộ, và cách tính này không thưởng cho reranker trả
    thiếu để rút ngắn output.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Reranker trả về JSON hỏng: {raw[:200]!r}") from exc

    items = parsed.get("scores") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError(f"Reranker trả về cấu trúc lạ: {raw[:200]!r}")

    scores: dict[int, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            ordinal = int(item["id"])
            value = int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= ordinal <= n:
            scores[ordinal] = max(0, min(config.RERANK_MAX_SCORE, value))
    if not scores:
        raise ValueError(f"Reranker không chấm được ứng viên nào: {raw[:200]!r}")
    return scores


def rerank(question: str, chunks: Sequence[Chunk], *, offline: bool) -> list[float]:
    """Chấm cả danh sách trong ĐÚNG MỘT lời gọi LLM, trả điểm đã chuẩn hoá [0, 1].

    Pointwise mỗi ứng viên một lời gọi sẽ tốn 30 lần quota cho mỗi câu hỏi; free
    tier giới hạn RPM thấp nên listwise là bắt buộc chứ không chỉ là tối ưu.
    Đánh đổi đã biết: model thấy cả danh sách nên điểm không độc lập hoàn toàn
    giữa các ứng viên.
    """
    prompt = RERANK_INSTRUCTION.format(
        question=question,
        max_score=config.RERANK_MAX_SCORE,
        n=len(chunks),
        candidates=_render_candidates(chunks),
    )
    digest = hashlib.sha256(
        "\n".join(c.indexed_text for c in chunks).encode("utf-8")
    ).hexdigest()

    raw = gemini_generate(
        task="rerank",
        model=config.RERANK_MODEL,
        # Danh tính lời gọi = câu hỏi + danh sách ứng viên ĐÚNG THỨ TỰ trình bày
        # (listwise nên thứ tự ảnh hưởng output), kèm digest nội dung để đổi tham
        # số chunk mà chunk_id không đổi vẫn sinh key mới.
        input_obj={
            "question": question,
            "candidates": [c.chunk_id for c in chunks],
            "candidates_digest": digest,
        },
        prompt=prompt,
        temperature=config.RERANK_TEMPERATURE,
        max_tokens=config.RERANK_MAX_TOKENS,
        thinking_budget=config.RERANK_THINKING_BUDGET,
        offline=offline,
    )
    scored = _parse_rerank(raw, len(chunks))
    return [scored.get(o, 0) / config.RERANK_MAX_SCORE for o in range(1, len(chunks) + 1)]


# --- Arm ------------------------------------------------------------------
def retrieve(
    index: Index,
    question: str,
    *,
    arm: str = "hybrid_rerank",
    top_k: int | None = None,
    offline: bool = False,
) -> list[Retrieved]:
    """Trả top_k chunk theo arm chỉ định, hạng 1 là tốt nhất."""
    if arm not in config.ARMS:
        raise ValueError(f"arm không hợp lệ: {arm!r} (chọn trong {config.ARMS})")
    if not question.strip():
        raise ValueError("Câu hỏi rỗng")

    limit = top_k or config.RETRIEVE_TOP_K
    total = len(index)

    if arm == "bm25":
        scores = index.bm25_scores(question)
        ordered = rank_by_score(scores, limit)
        return [
            Retrieved(index.chunks[i], rank=r, score=float(scores[i]), bm25_rank=r)
            for r, i in enumerate(ordered, start=1)
        ]

    if arm == "dense":
        scores = index.dense_scores(question, offline=offline)
        ordered = rank_by_score(scores, limit)
        return [
            Retrieved(index.chunks[i], rank=r, score=float(scores[i]), dense_rank=r)
            for r, i in enumerate(ordered, start=1)
        ]

    depth = min(total, max(limit, config.RERANK_CANDIDATES))
    bm25_ranked = rank_by_score(index.bm25_scores(question), depth)
    dense_ranked = rank_by_score(index.dense_scores(question, offline=offline), depth)
    bm25_rank = {i: r for r, i in enumerate(bm25_ranked, start=1)}
    dense_rank = {i: r for r, i in enumerate(dense_ranked, start=1)}
    fused = rrf_fuse([bm25_ranked, dense_ranked])
    fused_order = _order_fused(fused, depth)

    if arm == "hybrid":
        return [
            Retrieved(
                index.chunks[i],
                rank=r,
                score=fused[i],
                bm25_rank=bm25_rank.get(i),
                dense_rank=dense_rank.get(i),
                fusion_score=fused[i],
            )
            for r, i in enumerate(fused_order[:limit], start=1)
        ]

    candidates = [index.chunks[i] for i in fused_order]
    rerank_scores = rerank(question, candidates, offline=offline)
    # Hoà điểm rerank thì giữ thứ tự RRF: thang 0-3 hoà rất nhiều, và rơi về một
    # thứ tự tuỳ ý ở đó sẽ vứt bỏ toàn bộ tín hiệu của tầng hợp nhất bên dưới.
    reranked = sorted(
        range(len(fused_order)), key=lambda p: (-rerank_scores[p], p)
    )[:limit]
    return [
        Retrieved(
            index.chunks[fused_order[p]],
            rank=r,
            score=rerank_scores[p],
            bm25_rank=bm25_rank.get(fused_order[p]),
            dense_rank=dense_rank.get(fused_order[p]),
            fusion_score=fused[fused_order[p]],
            rerank_score=rerank_scores[p],
        )
        for r, p in enumerate(reranked, start=1)
    ]


def _load_question(qid: str) -> str:
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    for item in questions:
        if item["qid"] == qid:
            return str(item["question"])
    raise SystemExit(f"Không có qid {qid} trong {config.QUESTIONS_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy thử một câu hỏi qua một arm")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--arm", default="hybrid_rerank", choices=config.ARMS)
    parser.add_argument("--question")
    parser.add_argument("--qid", help="lấy câu hỏi từ eval/questions.json")
    parser.add_argument("--top-k", type=int, default=config.RETRIEVE_TOP_K)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if not args.question and not args.qid:
        parser.error("cần --question hoặc --qid")
    question = args.question or _load_question(args.qid)

    index = build_index(load_corpus(Path(args.corpus)))
    try:
        results = retrieve(
            index, question, arm=args.arm, top_k=args.top_k, offline=args.offline
        )
    except CacheMiss as exc:
        print(exc)
        return 2

    print(f"arm      : {args.arm}")
    print(f"câu hỏi  : {question}\n")
    for item in results:
        parts = [f"{item.rank:>2}. {item.score:.4f}  {item.chunk.citation_label}"]
        detail = [
            f"{name}={value}"
            for name, value in (
                ("bm25", item.bm25_rank),
                ("dense", item.dense_rank),
                ("rrf", None if item.fusion_score is None else f"{item.fusion_score:.4f}"),
            )
            if value is not None
        ]
        parts.append(f"      [{item.chunk.status}] {' '.join(detail)}")
        parts.append(f"      {item.chunk.text.strip()[:120]}…")
        print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
