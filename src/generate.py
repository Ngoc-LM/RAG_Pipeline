"""Sinh câu trả lời có trích dẫn, và điều phối vòng sinh lại theo kết quả kiểm chứng.

Bộ sinh trả về TỪNG MỆNH ĐỀ kèm đúng tập trích dẫn của riêng mệnh đề đó, chứ
không trả một khối text rồi tách claim ở bước sau. Tách câu tiếng Việt bằng
heuristic sẽ vỡ ở "Điều 1.", "khoản 2.", "0,5%" — mà claim tách sai thì mọi con
số groundedness đều sai theo, một cách âm thầm.

Hướng phụ thuộc: generate -> verify. `src.verify` chỉ nhận kiểu nguyên thuỷ nên
không cần biết gì về module này.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src import config
from src.chunk import Chunk
from src.index import build_index
from src.ingest import load_corpus
from src.llm import CacheMiss, gemini_generate
from src.retrieve import Retrieved, retrieve
from src.verify import (
    CheckA,
    CheckB,
    check_citations,
    check_grounding,
    grounded,
    render_context,
    retry_feedback,
)

GENERATE_INSTRUCTION = """Bạn trả lời câu hỏi pháp lý CHỈ dựa trên các trích đoạn được cung cấp dưới đây.

Câu hỏi: {question}

Quy tắc:
- Tách câu trả lời thành các mệnh đề ngắn, mỗi mệnh đề là một ý kiểm chứng được độc lập.
- Mỗi mệnh đề kèm đúng những trích đoạn làm căn cứ cho CHÍNH mệnh đề đó, ghi bằng số hiệu.
- Chỉ được dùng số hiệu trong khoảng [1]..[{n}]. Không viện dẫn bất kỳ nguồn nào khác.
- Không suy đoán, không bổ sung kiến thức ngoài trích đoạn, không nêu con số không có trong trích đoạn.
- Trích đoạn thuộc văn bản `expired` không phải căn cứ hợp lệ khi đã có trích đoạn `active` cùng nội dung.
- Nếu các trích đoạn không đủ để trả lời, trả về abstain thay vì trả lời một phần.

Các trích đoạn:
{context}
{feedback}
Trả về JSON đúng một trong hai dạng:
{{"abstain": false, "claims": [{{"text": "...", "citations": [1, 3]}}]}}
{{"abstain": true, "reason": "..."}}"""


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": list(self.citations)}


@dataclass(frozen=True)
class Draft:
    """Một lượt sinh, chưa qua kiểm chứng."""

    claims: tuple[Claim, ...]
    abstain: bool
    reason: str

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.claims)

    @property
    def citations(self) -> tuple[tuple[int, ...], ...]:
        return tuple(c.citations for c in self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain": self.abstain,
            "reason": self.reason,
            "claims": [c.to_dict() for c in self.claims],
        }


def _parse_draft(raw: str) -> Draft:
    """Đọc JSON bộ sinh.

    Cố ý KHÔNG lọc trước mệnh đề thiếu trích dẫn hay trích dẫn ngoài phạm vi:
    đó đúng là việc của Check A. Lọc ở đây sẽ giấu mất lỗi mà tầng kiểm chứng
    sinh ra để bắt, và làm mọi con số về Check A đẹp một cách giả tạo.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bộ sinh trả về JSON hỏng: {raw[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Bộ sinh trả về cấu trúc lạ: {raw[:200]!r}")

    if parsed.get("abstain"):
        return Draft(claims=(), abstain=True, reason=str(parsed.get("reason", "")))

    claims: list[Claim] = []
    for item in parsed.get("claims") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw_citations = item.get("citations") or []
        citations: list[int] = []
        for value in raw_citations if isinstance(raw_citations, list) else []:
            try:
                citations.append(int(value))
            except (TypeError, ValueError):
                continue
        claims.append(Claim(text=text, citations=tuple(citations)))
    return Draft(claims=tuple(claims), abstain=False, reason="")


def draft_answer(
    question: str,
    chunks: Sequence[Chunk],
    *,
    attempt: int,
    feedback: str,
    offline: bool,
) -> Draft:
    """Một lượt sinh. `feedback` rỗng ở lượt đầu, có nội dung ở lượt sinh lại."""
    prompt = GENERATE_INSTRUCTION.format(
        question=question,
        n=len(chunks),
        context=render_context(chunks),
        feedback=f"\n{feedback}\n" if feedback else "\n",
    )
    raw = gemini_generate(
        task="generate",
        model=config.GEN_MODEL,
        # `attempt` và `feedback` PHẢI nằm trong cache key: lượt hai dùng prompt
        # khác lượt một, nên nó cũng phải là một entry cache khác. Thiếu hai
        # trường này thì lượt sinh lại chỉ đọc lại đúng câu trả lời vừa bị bác.
        input_obj={
            "question": question,
            "context": [c.chunk_id for c in chunks],
            "attempt": attempt,
            "feedback": feedback,
        },
        prompt=prompt,
        temperature=config.GEN_TEMPERATURE,
        max_tokens=config.GEN_MAX_TOKENS,
        thinking_budget=config.GEN_THINKING_BUDGET,
        offline=offline,
    )
    return _parse_draft(raw)


@dataclass(frozen=True)
class Attempt:
    index: int
    draft: Draft
    check_a: CheckA
    check_b: CheckB | None

    @property
    def support_ratio(self) -> float | None:
        return None if self.check_b is None else self.check_b.support_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.index,
            "draft": self.draft.to_dict(),
            "check_a": self.check_a.to_dict(),
            "check_b": None if self.check_b is None else self.check_b.to_dict(),
        }


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    citations: tuple[str, ...]
    abstained: bool
    abstain_stage: str | None
    attempts: tuple[Attempt, ...]
    retrieval_score: float | None

    @property
    def support_ratio_first(self) -> float | None:
        """Faithfulness TRƯỚC tầng verify — tức là của lượt sinh đầu tiên."""
        return self.attempts[0].support_ratio if self.attempts else None

    @property
    def support_ratio_final(self) -> float | None:
        """Faithfulness SAU tầng verify. So với cột trên để đọc được tầng này
        thực sự thêm bao nhiêu giá trị chứ không chỉ thêm bao nhiêu chi phí."""
        for attempt in reversed(self.attempts):
            if attempt.support_ratio is not None:
                return attempt.support_ratio
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "abstained": self.abstained,
            "abstain_stage": self.abstain_stage,
            "retrieval_score": self.retrieval_score,
            "support_ratio_first": self.support_ratio_first,
            "support_ratio_final": self.support_ratio_final,
            "attempts": [a.to_dict() for a in self.attempts],
        }


def render_answer(draft: Draft, chunks: Sequence[Chunk]) -> tuple[str, tuple[str, ...]]:
    """Ghép mệnh đề thành câu trả lời, giữ số hiệu trích dẫn ngay sau từng mệnh đề."""
    sentences = [
        f"{claim.text} {''.join(f'[{c}]' for c in claim.citations)}".strip()
        for claim in draft.claims
    ]
    used = sorted({c for claim in draft.claims for c in claim.citations})
    labels = tuple(
        f"[{ordinal}] {chunks[ordinal - 1].citation_label}"
        for ordinal in used
        if 1 <= ordinal <= len(chunks)
    )
    return " ".join(sentences), labels


def _abstain(question: str, stage: str, attempts, score) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=config.ABSTAIN_TEXT,
        citations=(),
        abstained=True,
        abstain_stage=stage,
        attempts=tuple(attempts),
        retrieval_score=score,
    )


def answer_question(
    question: str,
    results: Sequence[Retrieved],
    *,
    offline: bool,
) -> AnswerResult:
    """Điều phối: gác ngưỡng truy xuất -> sinh -> Check A -> Check B -> sinh lại.

    Ngưỡng gác trước generate chỉ áp dụng khi có điểm rerank. Điểm BM25 không
    chặn trên còn cosine nằm trong [-1, 1], nên một ngưỡng chung cho cả bốn arm
    sẽ mang ý nghĩa khác nhau ở mỗi arm — thà không gác còn hơn gác bằng một con
    số không so sánh được.
    """
    if not results:
        return _abstain(question, "retrieve", [], None)

    top_score = results[0].rerank_score
    if top_score is not None and top_score < config.TAU_RETRIEVE:
        return _abstain(question, "retrieve", [], top_score)

    chunks = [r.chunk for r in results]
    attempts: list[Attempt] = []
    feedback = ""

    for index in range(1, config.MAX_GENERATE_ATTEMPTS + 1):
        draft = draft_answer(
            question, chunks, attempt=index, feedback=feedback, offline=offline
        )
        if draft.abstain:
            attempts.append(Attempt(index, draft, CheckA(True, ()), None))
            return _abstain(question, "model", attempts, top_score)

        check_a = check_citations(draft.citations, len(chunks))
        check_b = (
            check_grounding(
                question, draft.texts, draft.citations, chunks, offline=offline
            )
            if check_a.ok
            else None
        )
        attempts.append(Attempt(index, draft, check_a, check_b))

        if check_a.ok and check_b is not None and grounded(check_b):
            text, labels = render_answer(draft, chunks)
            return AnswerResult(
                question=question,
                answer=text,
                citations=labels,
                abstained=False,
                abstain_stage=None,
                attempts=tuple(attempts),
                retrieval_score=top_score,
            )

        feedback = retry_feedback(check_a, check_b, draft.texts)

    stage = "check_a" if not attempts[-1].check_a.ok else "check_b"
    return _abstain(question, stage, attempts, top_score)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trả lời một câu hỏi có trích dẫn")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--arm", default="hybrid_rerank", choices=config.ARMS)
    parser.add_argument("--question")
    parser.add_argument("--qid")
    parser.add_argument("--top-k", type=int, default=config.RETRIEVE_TOP_K)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if not args.question and not args.qid:
        parser.error("cần --question hoặc --qid")
    question = args.question
    if args.qid:
        from src.evaluate import load_questions

        found = [q for q in load_questions(config.QUESTIONS_PATH) if q.qid == args.qid]
        if not found:
            parser.error(f"không có qid {args.qid}")
        question = found[0].question

    index = build_index(load_corpus(Path(args.corpus)))
    try:
        results = retrieve(
            index, question, arm=args.arm, top_k=args.top_k, offline=args.offline
        )
        answer = answer_question(question, results, offline=args.offline)
    except CacheMiss as exc:
        print(exc)
        return 2

    print(f"câu hỏi : {question}\n")
    print(answer.answer)
    if answer.citations:
        print("\nCăn cứ:")
        for label in answer.citations:
            print(f"  {label}")
    print(
        f"\nabstain={answer.abstained} ({answer.abstain_stage})"
        f"  support_ratio: đầu={answer.support_ratio_first} "
        f"cuối={answer.support_ratio_final}  lượt sinh={len(answer.attempts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
