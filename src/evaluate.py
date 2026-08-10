"""Đánh giá retrieval và answer, xuất bảng ablation.

Hai chế độ tách biệt vì chi phí khác nhau một bậc:
  retrieval  chạy mọi arm trên toàn bộ gold set; có embed và rerank (đều cache),
             KHÔNG sinh câu trả lời và KHÔNG gọi judge.
  full       chỉ chạy trên các arm được chỉ định; có generate + verify + judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

import config
from src.index import Index, build_index
from src.llm import embed_texts, groq_generate
from src.metrics import abstain_scores, aggregate, question_metrics
from src.retrieve import ARMS, retrieve
from src.schema import Chunk, Document, Question
from src.verify import answer_with_verification

CORRECTNESS_PROMPT = """Bạn chấm xem câu trả lời của hệ thống có khớp với đáp án chuẩn hay không.

Câu hỏi: {question}
Đáp án chuẩn: {gold}
Câu trả lời của hệ thống: {answer}

Câu trả lời được coi là ĐÚNG nếu nó truyền đạt cùng nội dung thực chất với đáp án chuẩn.
Khác cách diễn đạt, dài hơn, hoặc có thêm trích dẫn thì vẫn tính là đúng.
Sai số liệu, sai điều kiện, thiếu ý chính, hoặc mâu thuẫn với đáp án chuẩn thì tính là sai.

Trả về đúng JSON dạng: {{"correct": true, "reason": "ngắn gọn"}}"""


def load_questions(path: Path) -> list[Question]:
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy gold set: {path}")
    return [Question.from_json(o) for o in json.loads(path.read_text(encoding="utf-8"))]


def embed_queries(questions: Sequence[Question], *, offline: bool) -> dict[str, np.ndarray]:
    """Embed mọi câu hỏi trong một lượt batch, tra theo qid."""
    matrix = embed_texts(
        [q.question for q in questions],
        task_type=config.EMBED_TASK_QUERY,
        offline=offline,
    )
    return {q.qid: matrix[i] for i, q in enumerate(questions)}


def _maybe_embed_queries(
    questions: Sequence[Question], arms: Sequence[str], *, offline: bool
) -> dict[str, np.ndarray]:
    """Chỉ embed khi có arm cần vector; chạy riêng arm bm25 thì không tốn API nào."""
    if all(arm == "bm25" for arm in arms):
        return {}
    return embed_queries(questions, offline=offline)


def retrieval_eval(
    index: Index,
    questions: Sequence[Question],
    *,
    arms: Sequence[str],
    offline: bool,
    top_k: int,
) -> dict[str, Any]:
    """Metric truy xuất cho từng arm. Bỏ qua câu unanswerable (không có gold span)."""
    scored = [q for q in questions if q.gold_spans]
    vectors = _maybe_embed_queries(questions, arms, offline=offline)

    results: dict[str, Any] = {}
    for arm in arms:
        rows = []
        for question in scored:
            result = retrieve(
                index,
                question.question,
                arm=arm,
                offline=offline,
                query_vector=vectors.get(question.qid),
                top_k=top_k,
            )
            rows.append(
                question_metrics(
                    question, result.chunks, config.EVAL_K_VALUES, config.THETA_COVERAGE
                )
            )
        results[arm] = {"summary": aggregate(rows, config.EVAL_K_VALUES), "rows": rows}
    return results


def judge_correctness(
    question: str, gold: str, answer: str, *, offline: bool
) -> tuple[bool, str]:
    raw = groq_generate(
        task="judge",
        input_obj={"mode": "correctness", "question": question, "gold": gold, "answer": answer},
        prompt=CORRECTNESS_PROMPT.format(question=question, gold=gold, answer=answer),
        offline=offline,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, "judge trả JSON hỏng"
    return bool(parsed.get("correct", False)), str(parsed.get("reason", ""))


def full_eval(
    index: Index,
    questions: Sequence[Question],
    *,
    arms: Sequence[str],
    offline: bool,
    tau_retrieve: float = config.TAU_RETRIEVE,
    tau_ground: float = config.TAU_GROUND,
) -> dict[str, Any]:
    """Sinh câu trả lời có kiểm chứng và chấm chất lượng answer cho từng arm."""
    vectors = _maybe_embed_queries(questions, arms, offline=offline)
    results: dict[str, Any] = {}

    for arm in arms:
        records: list[dict[str, Any]] = []
        decisions: list[tuple[Question, bool]] = []

        for question in questions:
            result = retrieve(
                index,
                question.question,
                arm=arm,
                offline=offline,
                query_vector=vectors.get(question.qid),
                top_k=config.TOP_K_CONTEXT,
            )
            gate_score = result.rerank_top_score

            if gate_score is not None and gate_score < tau_retrieve:
                record = {
                    "qid": question.qid,
                    "question": question.question,
                    "attempt_1": None,
                    "attempt_2": None,
                    "final_action": "abstain",
                    "final_answer": config.ABSTAIN_MESSAGE,
                    "gate": "retrieve",
                }
                answer_text = config.ABSTAIN_MESSAGE
                abstained = True
            else:
                answer, record = answer_with_verification(
                    question.qid,
                    question.question,
                    result.chunks,
                    offline=offline,
                    tau_ground=tau_ground,
                )
                record["gate"] = None
                answer_text = answer.text
                abstained = answer.abstained

            record["arm"] = arm
            record["rerank_top_score"] = gate_score
            record["context_chunk_ids"] = [c.chunk_id for c in result.chunks]

            if question.answerable and not abstained:
                correct, reason = judge_correctness(
                    question.question, question.gold_answer, answer_text, offline=offline
                )
                record["correct"] = correct
                record["correct_reason"] = reason

            decisions.append((question, abstained))
            records.append(record)

        results[arm] = {"summary": _answer_summary(records, decisions), "records": records}
    return results


def _answer_summary(
    records: Sequence[dict[str, Any]], decisions: Sequence[tuple[Question, bool]]
) -> dict[str, Any]:
    """Gộp số liệu answer, tách faithfulness trước và sau tầng verify."""
    pre = [
        r["attempt_1"]["support_ratio"]
        for r in records
        if r.get("attempt_1") and not r["attempt_1"]["abstained"]
    ]
    accepted = [r for r in records if r["final_action"] in ("accept", "retry_accept")]
    post = [
        (r["attempt_2"] or r["attempt_1"])["support_ratio"]
        for r in accepted
        if r.get("attempt_1")
    ]
    graded = [r for r in records if "correct" in r]

    return {
        "n": len(records),
        "faithfulness_pre_verify": mean(pre) if pre else 0.0,
        "faithfulness_post_verify": mean(post) if post else 0.0,
        "n_accept": sum(1 for r in records if r["final_action"] == "accept"),
        "n_retry_accept": sum(1 for r in records if r["final_action"] == "retry_accept"),
        "n_abstain": sum(1 for r in records if r["final_action"] == "abstain"),
        "n_gate_retrieve": sum(1 for r in records if r.get("gate") == "retrieve"),
        "answer_accuracy": mean(float(r["correct"]) for r in graded) if graded else 0.0,
        "n_graded": len(graded),
        **abstain_scores(list(decisions)),
    }


def chunk_size_sweep(
    documents: Sequence[Document],
    questions: Sequence[Question],
    sizes: Sequence[int],
    *,
    arm: str,
    offline: bool,
) -> dict[str, Any]:
    """Quét CHUNK_TARGET_CHARS. Metric coverage bất biến nên các dòng so được."""
    from src.chunk import chunk_corpus

    out: dict[str, Any] = {}
    for size in sizes:
        chunks: list[Chunk] = chunk_corpus(
            list(documents), size, int(size * 1.5), config.CHUNK_HARD_SPLIT_OVERLAP
        )
        index = build_index(chunks, offline=offline)
        result = retrieval_eval(
            index, questions, arms=[arm], offline=offline, top_k=max(config.EVAL_K_VALUES)
        )
        out[str(size)] = {
            "n_chunks": len(chunks),
            "mean_chunk_chars": mean(len(c.text) for c in chunks),
            "summary": result[arm]["summary"],
        }
    return out


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def retrieval_table(results: dict[str, Any]) -> str:
    """Bảng ablation retrieval dạng markdown."""
    ks = config.EVAL_K_VALUES
    header = (
        ["arm"]
        + [f"R@{k} strict" for k in ks]
        + [f"R@{k} any" for k in ks]
        + [f"Cov@{max(ks)}", "MRR strict", "MRR any"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in ARMS:
        if arm not in results:
            continue
        s = results[arm]["summary"]
        row = (
            [arm]
            + [_fmt(s[f"recall_strict@{k}"]) for k in ks]
            + [_fmt(s[f"recall_any@{k}"]) for k in ks]
            + [_fmt(s[f"mean_cov@{max(ks)}"]), _fmt(s["mrr_strict"]), _fmt(s["mrr_any"])]
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def by_type_table(results: dict[str, Any], k: int) -> str:
    """Recall strict@k tách theo loại câu hỏi — cột distractor là chỗ rerank phải thắng."""
    types = sorted(
        {t for arm in results.values() for t in arm["summary"].get("by_type", {})}
    )
    header = ["arm"] + types
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in ARMS:
        if arm not in results:
            continue
        by_type = results[arm]["summary"].get("by_type", {})
        row = [arm] + [
            _fmt(by_type[t][f"recall_strict@{k}"]) if t in by_type else "-" for t in types
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def answer_table(results: dict[str, Any]) -> str:
    header = [
        "arm",
        "faithful pre",
        "faithful post",
        "accuracy",
        "accept",
        "retry_accept",
        "abstain",
        "abstain F1",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm, payload in results.items():
        s = payload["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _fmt(s["faithfulness_pre_verify"]),
                    _fmt(s["faithfulness_post_verify"]),
                    _fmt(s["answer_accuracy"]),
                    str(s["n_accept"]),
                    str(s["n_retry_accept"]),
                    str(s["n_abstain"]),
                    _fmt(s["abstain_f1"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def chunk_table(sweep: dict[str, Any]) -> str:
    ks = config.EVAL_K_VALUES
    header = ["chunk_target", "n_chunks", "mean_chars"] + [f"R@{k} strict" for k in ks]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for size, payload in sorted(sweep.items(), key=lambda kv: int(kv[0])):
        s = payload["summary"]
        lines.append(
            "| "
            + " | ".join(
                [size, str(payload["n_chunks"]), f"{payload['mean_chunk_chars']:.0f}"]
                + [_fmt(s[f"recall_strict@{k}"]) for k in ks]
            )
            + " |"
        )
    return "\n".join(lines)
