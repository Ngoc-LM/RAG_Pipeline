"""Hai tầng kiểm chứng, chi phí khác hẳn nhau.

- Check A — cú pháp, MIỄN PHÍ. Mọi trích dẫn phải trỏ tới một trích đoạn thực sự
  có trong prompt, và mọi mệnh đề phải có ít nhất một trích dẫn. Fail thì sinh
  lại ngay, không tốn một lời gọi judge nào.
- Check B — ngữ nghĩa, ĐÚNG MỘT lời gọi LLM judge cho toàn bộ mệnh đề.

Module này cố ý chỉ nhận kiểu dữ liệu nguyên thuỷ (chuỗi, số, Chunk) chứ không
import gì từ `src.generate`. Nhờ vậy hướng phụ thuộc là generate -> verify, không
có vòng, và hai tầng kiểm chứng test được độc lập với bộ sinh.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from src import config
from src.chunk import Chunk
from src.llm import judge_generate


@dataclass(frozen=True)
class CheckA:
    """Kết quả kiểm cú pháp trích dẫn."""

    ok: bool
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": list(self.problems)}


def check_citations(citations_per_claim: Sequence[Sequence[int]], n_context: int) -> CheckA:
    """Mọi mệnh đề phải có trích dẫn, và mọi trích dẫn phải nằm trong [1, n].

    Đây là tầng bắt lỗi rẻ nhất: một trích dẫn trỏ ra ngoài danh sách là bằng
    chứng chắc chắn model đang bịa nguồn, không cần hỏi judge mới biết.
    """
    problems: list[str] = []
    if not citations_per_claim:
        problems.append("không có mệnh đề nào")

    for position, citations in enumerate(citations_per_claim, start=1):
        if not citations:
            problems.append(f"mệnh đề {position} không có trích dẫn")
            continue
        for ordinal in citations:
            if not 1 <= ordinal <= n_context:
                problems.append(
                    f"mệnh đề {position} trích dẫn [{ordinal}] ngoài danh sách "
                    f"(chỉ có [1]..[{n_context}])"
                )
    return CheckA(ok=not problems, problems=tuple(problems))


@dataclass(frozen=True)
class ClaimVerdict:
    claim_index: int
    supported: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CheckB:
    verdicts: tuple[ClaimVerdict, ...]
    support_ratio: float

    @property
    def rejected(self) -> tuple[ClaimVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_ratio": round(self.support_ratio, 4),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


JUDGE_INSTRUCTION = """Bạn kiểm chứng một câu trả lời pháp lý. Với MỖI mệnh đề, xác định các trích đoạn được viện dẫn cho chính mệnh đề đó có ĐỦ để kết luận mệnh đề đó hay không.

Chấm `supported: false` nếu mệnh đề nói nhiều hơn những gì trích đoạn được viện dẫn nêu ra — kể cả khi mệnh đề đúng trên thực tế. Bạn đang đo tính có căn cứ, không đo tính đúng.

Câu hỏi gốc: {question}

Các trích đoạn:
{context}

Các mệnh đề cần kiểm:
{claims}

Trả về JSON đúng dạng, chấm đủ cả {n} mệnh đề:
{{"verdicts": [{{"claim_id": 1, "supported": true, "reason": "ngắn gọn"}}]}}"""


def render_context(chunks: Sequence[Chunk]) -> str:
    """Danh sách trích đoạn đánh số [1]..[n], dùng chung cho prompt sinh và judge."""
    blocks: list[str] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        hieu_luc = chunk.status
        if chunk.effective_to:
            hieu_luc = f"{chunk.status}, hết hiệu lực {chunk.effective_to}"
        blocks.append(
            f"[{ordinal}] {chunk.citation_label} — {chunk.article_title}\n"
            f"    hiệu lực: {hieu_luc}\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


def _render_claims(texts: Sequence[str], citations: Sequence[Sequence[int]]) -> str:
    return "\n".join(
        f"{position}. {text}  (viện dẫn: {', '.join(f'[{c}]' for c in cited)})"
        for position, (text, cited) in enumerate(zip(texts, citations), start=1)
    )


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Bóc khối JSON ra khỏi văn bản tự do.

    Judge chạy ở chế độ JSON mode thì trả JSON trần và hàm này không đụng gì. Gemma
    không có JSON mode nên gói kết quả trong ```json … ```; bóc fence ở tầng parse
    giữ cho tầng gọi API không phải biết model nào cần xử lý đặc biệt.
    """
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    fenced = FENCE_RE.search(raw)
    if fenced:
        return fenced.group(1).strip()
    brace = re.search(r"(\{.*\})", raw, re.DOTALL)
    return brace.group(1) if brace else stripped


def _parse_verdicts(raw: str, n_claims: int) -> dict[int, ClaimVerdict]:
    try:
        parsed = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge trả về JSON hỏng: {raw[:200]!r}") from exc

    items = parsed.get("verdicts") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError(f"Judge trả về cấu trúc lạ: {raw[:200]!r}")

    found: dict[int, ClaimVerdict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            claim_id = int(item["claim_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= claim_id <= n_claims:
            found[claim_id] = ClaimVerdict(
                claim_index=claim_id,
                supported=bool(item.get("supported")),
                reason=str(item.get("reason", "")),
            )
    return found


def check_grounding(
    question: str,
    claim_texts: Sequence[str],
    citations_per_claim: Sequence[Sequence[int]],
    chunks: Sequence[Chunk],
    *,
    offline: bool,
) -> CheckB:
    """Chấm toàn bộ mệnh đề trong ĐÚNG MỘT lời gọi judge.

    Mệnh đề mà judge không nhắc tới bị tính là KHÔNG được hỗ trợ. Im lặng không
    phải bằng chứng ủng hộ, và cách tính này không thưởng cho judge trả thiếu để
    rút ngắn output.
    """
    if not claim_texts:
        return CheckB(verdicts=(), support_ratio=0.0)

    prompt = JUDGE_INSTRUCTION.format(
        question=question,
        context=render_context(chunks),
        claims=_render_claims(claim_texts, citations_per_claim),
        n=len(claim_texts),
    )
    raw = judge_generate(
        task="judge",
        input_obj={
            "question": question,
            "claims": list(claim_texts),
            "citations": [list(c) for c in citations_per_claim],
            "context": [c.chunk_id for c in chunks],
        },
        prompt=prompt,
        offline=offline,
    )
    found = _parse_verdicts(raw, len(claim_texts))

    verdicts = tuple(
        found.get(
            position,
            ClaimVerdict(position, supported=False, reason="judge không chấm mệnh đề này"),
        )
        for position in range(1, len(claim_texts) + 1)
    )
    supported = sum(1 for v in verdicts if v.supported)
    return CheckB(verdicts=verdicts, support_ratio=supported / len(verdicts))


def grounded(check: CheckB, tau: float | None = None) -> bool:
    """`tau` để None thì lấy config; truyền tay khi quét lưới ở calibrate."""
    return check.support_ratio >= (config.TAU_GROUND if tau is None else tau)


def retry_feedback(check_a: CheckA, check_b: CheckB | None, claim_texts: Sequence[str]) -> str:
    """Khối văn bản chèn vào prompt lượt hai.

    Bắt buộc phải có nội dung KHÁC lượt một: ở temperature = 0, một prompt y hệt
    sẽ cho ra output y hệt, nên "sinh lại" mà không đổi prompt là một vòng lặp
    tốn quota để nhận lại đúng câu trả lời vừa bị bác.
    """
    lines: list[str] = []
    if not check_a.ok:
        lines.append("Lượt trước sai về trích dẫn:")
        lines.extend(f"- {problem}" for problem in check_a.problems)
    if check_b is not None and check_b.rejected:
        lines.append("Lượt trước có mệnh đề không được trích đoạn hỗ trợ:")
        for verdict in check_b.rejected:
            text = claim_texts[verdict.claim_index - 1]
            lines.append(f'- "{text}" — {verdict.reason}')
    lines.append(
        "Sửa hoặc BỎ HẲN những mệnh đề trên. Không thêm mệnh đề mới nếu không có "
        "căn cứ trong danh sách trích đoạn. Nếu sau khi bỏ thì không còn gì để "
        "trả lời, hãy trả về abstain."
    )
    return "\n".join(lines)
