"""Hiệu chỉnh hai ngưỡng abstain trên gold set.

TAU_RETRIEVE gác trước generate (điểm rerank cao nhất), TAU_GROUND gác sau
generate (support_ratio). Tách đôi vì chúng bắt hai loại lỗi khác nhau: câu hỏi
ngoài phạm vi corpus chết ở gate đầu, còn câu hỏi có chunk trông hợp lý nhưng
model bịa thêm thì chỉ gate sau mới bắt được.

Tín hiệu được thu ĐÚNG MỘT LẦN với cả hai ngưỡng bằng 0, rồi mọi điểm trong
lưới được mô phỏng lại trên tín hiệu đó — nếu không thì mỗi ô của lưới lại tốn
một lượt generate và judge.
"""

from __future__ import annotations

from typing import Any, Sequence

import config
from src.evaluate import embed_queries
from src.index import Index
from src.metrics import abstain_scores
from src.retrieve import retrieve
from src.schema import Question
from src.verify import answer_with_verification


def collect_signals(
    index: Index, questions: Sequence[Question], *, offline: bool, arm: str
) -> list[dict[str, Any]]:
    """Chạy pipeline với ngưỡng bằng 0 để lấy điểm rerank và support_ratio lượt 1."""
    vectors = embed_queries(questions, offline=offline)
    signals: list[dict[str, Any]] = []

    for question in questions:
        result = retrieve(
            index,
            question.question,
            arm=arm,
            offline=offline,
            query_vector=vectors[question.qid],
            top_k=config.TOP_K_CONTEXT,
        )
        _, record = answer_with_verification(
            question.qid,
            question.question,
            result.chunks,
            offline=offline,
            tau_ground=0.0,
        )
        attempt = record["attempt_1"]
        signals.append(
            {
                "qid": question.qid,
                "answerable": question.answerable,
                "type": question.type,
                "rerank_top_score": result.rerank_top_score or 0.0,
                "support_ratio": (attempt or {}).get("support_ratio", 0.0),
                "model_abstained": bool((attempt or {}).get("abstained", True)),
            }
        )
    return signals


def decide(signal: dict[str, Any], tau_retrieve: float, tau_ground: float) -> bool:
    """True nghĩa là abstain."""
    if signal["model_abstained"]:
        return True
    if signal["rerank_top_score"] < tau_retrieve:
        return True
    return signal["support_ratio"] < tau_ground


def sweep(
    signals: Sequence[dict[str, Any]],
    questions: Sequence[Question],
    *,
    tau_retrieve_values: Sequence[float] = config.TAU_RETRIEVE_SWEEP,
    tau_ground_values: Sequence[float] = config.TAU_GROUND_SWEEP,
) -> dict[str, Any]:
    """Quét lưới hai ngưỡng, tối ưu F1 của quyết định abstain."""
    by_qid = {q.qid: q for q in questions}
    grid: list[dict[str, Any]] = []

    for tau_retrieve in tau_retrieve_values:
        for tau_ground in tau_ground_values:
            decisions = [
                (by_qid[s["qid"]], decide(s, tau_retrieve, tau_ground)) for s in signals
            ]
            scores = abstain_scores(decisions)
            grid.append(
                {
                    "tau_retrieve": tau_retrieve,
                    "tau_ground": tau_ground,
                    **{k: round(v, 4) for k, v in scores.items()},
                }
            )

    best = max(grid, key=lambda row: (row["abstain_f1"], -row["fp"], -row["fn"]))
    return {
        "note": (
            "Tín hiệu thu ở tau=0 nên nhánh sinh lại không được mô phỏng; "
            "F1 ở đây là cận dưới của cấu hình có retry."
        ),
        "n_questions": len(signals),
        "best": best,
        "current_config": {
            "tau_retrieve": config.TAU_RETRIEVE,
            "tau_ground": config.TAU_GROUND,
        },
        "grid": grid,
        "signals": list(signals),
    }
