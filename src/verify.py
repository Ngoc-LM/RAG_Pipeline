"""Tầng kiểm chứng hai tầng giữa generate và câu trả lời cuối cùng.

Check A — cú pháp, miễn phí: mọi citation phải trỏ tới chunk thực sự có trong
prompt, và mọi claim phải có ít nhất một citation. Fail thì sinh lại NGAY,
không tốn một lời gọi judge nào.

Check B — ngữ nghĩa, đúng một lời gọi Groq: gửi toàn bộ claim trong một call,
judge trả về từng claim có được đoạn trích mà nó dẫn hỗ trợ hay không.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import config
from src.generate import abstain, generate_answer
from src.llm import groq_generate
from src.schema import Answer, Chunk, FinalAction

JUDGE_PROMPT = """Bạn là giám khảo kiểm tra tính có căn cứ (groundedness) của câu trả lời về pháp luật Việt Nam.

Câu hỏi:
{question}

Dưới đây là các khẳng định, mỗi khẳng định kèm đúng những đoạn trích mà nó viện dẫn:
{claims}

Với TỪNG khẳng định, xác định nó có được suy ra trực tiếp từ các đoạn trích kèm theo nó hay không.
- supported = true: nội dung khẳng định nằm trong đoạn trích, không cần thêm giả định nào.
- supported = false: đoạn trích không nói điều đó, chỉ nói gần giống, nói ngược lại, hoặc phải suy diễn thêm mới ra.

Chỉ xét đoạn trích kèm theo mỗi khẳng định. Không dùng kiến thức pháp luật của bạn để bào chữa cho khẳng định.
Trả về đúng JSON dạng: {{"results": [{{"claim_id": 1, "supported": true, "reason": "ngắn gọn"}}]}} đủ mọi khẳng định."""


@dataclass(frozen=True)
class Attempt:
    answer: Answer
    support_ratio: float
    failed_claims: tuple[str, ...]
    invalid_citations: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "answer": self.answer.text,
            "claims": [c.to_json() for c in self.answer.claims],
            "support_ratio": self.support_ratio,
            "failed_claims": list(self.failed_claims),
            "invalid_citations": list(self.invalid_citations),
            "abstained": self.answer.abstained,
        }


def check_citations(answer: Answer, chunks: Sequence[Chunk]) -> list[str]:
    """Check A. Trả về danh sách id sai; claim không citation tính là lỗi."""
    allowed = {c.chunk_id for c in chunks}
    invalid: list[str] = []
    for claim in answer.claims:
        if not claim.citations:
            invalid.append(f"claim {claim.claim_id}: không có citation")
        invalid.extend(c for c in claim.citations if c not in allowed)
    return invalid


def judge_claims(
    question: str, answer: Answer, chunks: Sequence[Chunk], *, offline: bool
) -> list[dict[str, Any]]:
    """Check B. Toàn bộ claim đi trong một lời gọi Groq duy nhất."""
    if not answer.claims:
        return []
    by_id = {c.chunk_id: c for c in chunks}

    blocks: list[str] = []
    payload: list[dict[str, Any]] = []
    for claim in answer.claims:
        cited = [by_id[cid].indexed_text for cid in claim.citations if cid in by_id]
        quoted = "\n".join(f"  - {text}" for text in cited) or "  (không có đoạn trích)"
        blocks.append(f"Khẳng định {claim.claim_id}: {claim.text}\nĐoạn trích:\n{quoted}")
        payload.append({"claim_id": claim.claim_id, "text": claim.text, "cited": cited})

    raw = groq_generate(
        task="judge",
        input_obj={"question": question, "claims": payload},
        prompt=JUDGE_PROMPT.format(question=question, claims="\n\n".join(blocks)),
        offline=offline,
    )

    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return []
    results = parsed.get("results", []) if isinstance(parsed, dict) else parsed
    return [r for r in results if isinstance(r, dict) and "claim_id" in r]


def support_ratio(answer: Answer, results: Sequence[dict[str, Any]]) -> tuple[float, list[str]]:
    """Tỉ lệ claim được hỗ trợ, cùng nội dung các claim bị bác.

    Claim mà judge không nhắc tới bị coi là KHÔNG được hỗ trợ: im lặng không
    phải là bằng chứng ủng hộ, và cách tính này không thưởng cho judge trả thiếu.
    """
    if not answer.claims:
        return 0.0, []
    supported = {
        int(r["claim_id"]) for r in results if bool(r.get("supported", False))
    }
    failed = [c.text for c in answer.claims if c.claim_id not in supported]
    return (len(answer.claims) - len(failed)) / len(answer.claims), failed


def answer_with_verification(
    qid: str,
    question: str,
    chunks: Sequence[Chunk],
    *,
    offline: bool,
    tau_ground: float = config.TAU_GROUND,
) -> tuple[Answer, dict[str, Any]]:
    """Sinh, kiểm chứng, sinh lại một lần nếu cần, rồi chốt hành động cuối."""
    attempts: list[Attempt] = []
    feedback: tuple[str, ...] = ()
    final_answer = abstain(qid, question, config.ABSTAIN_MESSAGE)
    final_action: FinalAction = "abstain"

    for attempt_index in range(1 + config.MAX_REGENERATE_ATTEMPTS):
        is_last = attempt_index == config.MAX_REGENERATE_ATTEMPTS
        answer = generate_answer(
            qid, question, chunks, offline=offline, failed_claims=feedback
        )

        if answer.abstained:
            attempts.append(Attempt(answer, 0.0, (), ()))
            break

        invalid = check_citations(answer, chunks)
        if invalid:
            attempts.append(Attempt(answer, 0.0, (), tuple(invalid)))
            if is_last:
                break
            feedback = tuple(
                f"Citation không hợp lệ: {i}. Chỉ được dẫn id có trong danh sách."
                for i in invalid
            )
            continue

        ratio, failed = support_ratio(
            answer, judge_claims(question, answer, chunks, offline=offline)
        )
        attempts.append(Attempt(answer, ratio, tuple(failed), ()))

        if ratio >= tau_ground:
            final_answer = answer
            final_action = "accept" if attempt_index == 0 else "retry_accept"
            break
        if is_last:
            break
        feedback = tuple(failed)

    record = {
        "qid": qid,
        "question": question,
        "attempt_1": attempts[0].to_json() if attempts else None,
        "attempt_2": attempts[1].to_json() if len(attempts) > 1 else None,
        "final_action": final_action,
        "final_answer": final_answer.text,
    }
    return final_answer, record
