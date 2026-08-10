"""Sinh câu trả lời có trích dẫn, xuất ra JSON schema cố định.

Model sinh THEO CÂU, mỗi câu kèm đúng tập citation của riêng nó, thay vì sinh
một khối text rồi tách claim ở bước sau. Tách câu tiếng Việt bằng heuristic sẽ
vỡ ở "Điều 1.", "khoản 2.", "0,5%" — mà claim tách sai thì mọi con số
groundedness phía sau đều sai theo.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import config
from src.llm import gemini_generate
from src.schema import Answer, Chunk, Claim

GENERATE_PROMPT = """Bạn trả lời câu hỏi về văn bản pháp luật Việt Nam, CHỈ dựa vào các đoạn trích được cung cấp.

Câu hỏi:
{question}

Các đoạn trích (mỗi đoạn có một id trong ngoặc vuông):
{context}

Quy tắc bắt buộc:
- Chỉ dùng thông tin có trong các đoạn trích trên. Tuyệt đối không dùng kiến thức bên ngoài.
- Tách câu trả lời thành từng câu ngắn. Mỗi câu là một khẳng định độc lập, kiểm chứng được.
- Mỗi câu phải ghi id của những đoạn trích trực tiếp chứng minh câu đó. Chỉ dùng id có trong danh sách trên.
- Nếu các đoạn trích không đủ căn cứ để trả lời, đặt "answerable": false và để "claims" rỗng.
- Không suy diễn, không khái quát hoá, không thêm điều kiện mà đoạn trích không nói.
{feedback}
Trả về đúng JSON dạng:
{{"answerable": true, "claims": [{{"text": "một câu khẳng định", "citations": ["id1"]}}]}}"""

FEEDBACK_BLOCK = """
LƯU Ý — lần trả lời trước bị bác bỏ. Các khẳng định sau KHÔNG được đoạn trích hỗ trợ:
{failed}
Hãy sửa hoặc loại bỏ chúng. Nếu phần nào không thể trả lời chỉ dựa vào đoạn trích, hãy nói rõ điều đó thay vì đoán.
"""


def format_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.indexed_text}" for c in chunks)


def abstain(qid: str, question: str, reason: str) -> Answer:
    return Answer(
        qid=qid, question=question, claims=(), abstained=True, abstain_reason=reason
    )


def generate_answer(
    qid: str,
    question: str,
    chunks: Sequence[Chunk],
    *,
    offline: bool,
    failed_claims: Sequence[str] = (),
) -> Answer:
    """Một lượt sinh. `failed_claims` khác rỗng nghĩa là đây là lượt sinh lại.

    failed_claims đi vào cả prompt lẫn cache key: ở temperature=0, sinh lại với
    prompt y hệt sẽ cho ra output y hệt, nên lượt retry bắt buộc phải khác input.
    """
    if not chunks:
        return abstain(qid, question, config.ABSTAIN_MESSAGE)

    feedback = (
        FEEDBACK_BLOCK.format(failed="\n".join(f"- {c}" for c in failed_claims))
        if failed_claims
        else ""
    )
    prompt = GENERATE_PROMPT.format(
        question=question, context=format_context(chunks), feedback=feedback
    )
    raw = gemini_generate(
        task="generate",
        model=config.GEN_MODEL,
        input_obj={
            "question": question,
            "context": [
                {"id": c.chunk_id, "text": c.indexed_text} for c in chunks
            ],
            "failed_claims": list(failed_claims),
        },
        prompt=prompt,
        temperature=config.GEN_TEMPERATURE,
        max_tokens=config.GEN_MAX_TOKENS,
        thinking_budget=config.GEN_THINKING_BUDGET,
        offline=offline,
    )
    return parse_answer(qid, question, raw)


def parse_answer(qid: str, question: str, raw: str) -> Answer:
    """Đọc JSON của model; JSON hỏng hoặc rỗng đều quy về abstain."""
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return abstain(qid, question, config.ABSTAIN_MESSAGE)
    if not isinstance(parsed, dict) or not parsed.get("answerable", False):
        return abstain(qid, question, config.ABSTAIN_MESSAGE)

    claims: list[Claim] = []
    for index, item in enumerate(parsed.get("claims", []), start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        citations = tuple(
            str(c) for c in item.get("citations", []) if isinstance(c, (str, int))
        )
        claims.append(Claim(claim_id=index, text=text, citations=citations))

    if not claims:
        return abstain(qid, question, config.ABSTAIN_MESSAGE)
    return Answer(qid=qid, question=question, claims=tuple(claims), abstained=False)
