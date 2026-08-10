"""Metric truy xuất theo coverage mức tập kết quả.

Không có khái niệm "chunk này là gold". Một gold_span được coi là đã truy xuất
được khi HỢP các chunk trong top-k phủ ít nhất THETA_COVERAGE độ dài span:

    Cov@k(span) = |span ∩ union(top-k chunks cùng doc)| / |span|
    hit@k(span) <=> Cov@k(span) >= THETA
    r*(span)    = min{k : Cov@k(span) >= THETA},  MRR = 1/r*  (không tồn tại -> 0)

Định nghĩa này bất biến với chunk_size theo cấu tạo: đổi tham số chunk không
làm đổi tử số lẫn mẫu số một cách giả tạo, nên các arm ablation so được với nhau.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence

from src.intervals import Interval, coverage, interval_union
from src.schema import Chunk, Question, Span


def coverage_curve(span: Span, ranked: Sequence[Chunk]) -> list[float]:
    """Cov@k của một span với k chạy từ 1 tới len(ranked); đơn điệu không giảm."""
    target: Interval = (span.char_start, span.char_end)
    accumulated: list[Interval] = []
    curve: list[float] = []
    for chunk in ranked:
        if chunk.doc_id == span.doc_id:
            accumulated.append((chunk.char_start, chunk.char_end))
        curve.append(coverage(interval_union(accumulated), target))
    return curve


def first_rank_reaching(curve: Sequence[float], theta: float) -> int | None:
    """Hạng nhỏ nhất (1-based) mà coverage đạt ngưỡng."""
    for index, value in enumerate(curve, start=1):
        if value >= theta:
            return index
    return None


def question_metrics(
    question: Question,
    ranked: Sequence[Chunk],
    k_values: Iterable[int],
    theta: float,
) -> dict[str, Any]:
    """Metric của một câu hỏi có gold_span. Câu unanswerable không đi qua đây."""
    curves = [coverage_curve(span, ranked) for span in question.gold_spans]
    ranks = [first_rank_reaching(curve, theta) for curve in curves]

    strict_rank = None if any(r is None for r in ranks) else max(ranks)  # type: ignore[type-var]
    hit_ranks = [r for r in ranks if r is not None]
    any_rank = min(hit_ranks) if hit_ranks else None

    def cov_at(curve: Sequence[float], k: int) -> float:
        if not curve:
            return 0.0
        return curve[min(k, len(curve)) - 1]

    per_k: dict[str, Any] = {}
    for k in k_values:
        covs = [cov_at(curve, k) for curve in curves]
        per_k[str(k)] = {
            "recall_strict": float(all(c >= theta for c in covs)),
            "recall_any": float(any(c >= theta for c in covs)),
            "mean_cov": mean(covs) if covs else 0.0,
        }

    return {
        "qid": question.qid,
        "type": question.type,
        "per_k": per_k,
        "rr_strict": 1.0 / strict_rank if strict_rank else 0.0,
        "rr_any": 1.0 / any_rank if any_rank else 0.0,
        "first_rank_strict": strict_rank,
        "first_rank_any": any_rank,
    }


def aggregate(rows: Sequence[dict[str, Any]], k_values: Iterable[int]) -> dict[str, Any]:
    """Trung bình trên toàn bộ câu hỏi, kèm tách theo type."""
    if not rows:
        return {"n": 0}

    def summarize(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(subset)}
        for k in k_values:
            key = str(k)
            out[f"recall_strict@{k}"] = mean(r["per_k"][key]["recall_strict"] for r in subset)
            out[f"recall_any@{k}"] = mean(r["per_k"][key]["recall_any"] for r in subset)
            out[f"mean_cov@{k}"] = mean(r["per_k"][key]["mean_cov"] for r in subset)
        out["mrr_strict"] = mean(r["rr_strict"] for r in subset)
        out["mrr_any"] = mean(r["rr_any"] for r in subset)
        return out

    result = summarize(rows)
    result["by_type"] = {
        qtype: summarize([r for r in rows if r["type"] == qtype])
        for qtype in sorted({r["type"] for r in rows})
    }
    return result


def abstain_scores(
    decisions: Sequence[tuple[Question, bool]]
) -> dict[str, float]:
    """P/R/F1 của quyết định abstain, coi 'nên abstain' (unanswerable) là lớp dương."""
    tp = sum(1 for q, abstained in decisions if abstained and not q.answerable)
    fp = sum(1 for q, abstained in decisions if abstained and q.answerable)
    fn = sum(1 for q, abstained in decisions if not abstained and not q.answerable)
    tn = sum(1 for q, abstained in decisions if not abstained and q.answerable)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
