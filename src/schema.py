"""Kiểu dữ liệu dùng chung và (de)serialize JSON cho outputs/."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

QuestionType = Literal[
    "factoid_1hop",
    "multihop",
    "distractor",
    "unanswerable_oos",
    "unanswerable_nearmiss",
]

FinalAction = Literal["accept", "retry_accept", "abstain"]


@dataclass(frozen=True)
class Document:
    """Một văn bản QPPL đã chuẩn hoá.

    `body` là trục ký tự chuẩn: mọi char_start/char_end trong repo — của chunk
    lẫn của gold_span — đều là chỉ số vào đúng chuỗi này.
    """

    doc_id: str
    title: str
    body: str
    meta: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    breadcrumb: str
    text: str
    char_start: int
    char_end: int

    @property
    def indexed_text(self) -> str:
        """Text đem đi embed và đánh chỉ mục BM25.

        Gắn breadcrumb vào đầu để chunk mang theo ngữ cảnh Chương/Điều, nếu
        không thì một Khoản tách rời gần như vô nghĩa với retriever.
        """
        return f"{self.breadcrumb}\n{self.text}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Chunk":
        return Chunk(
            chunk_id=obj["chunk_id"],
            doc_id=obj["doc_id"],
            breadcrumb=obj["breadcrumb"],
            text=obj["text"],
            char_start=int(obj["char_start"]),
            char_end=int(obj["char_end"]),
        )


@dataclass(frozen=True)
class Span:
    """Đoạn văn bản làm căn cứ đúng, nửa mở [char_start, char_end)."""

    doc_id: str
    char_start: int
    char_end: int

    @property
    def length(self) -> int:
        return max(0, self.char_end - self.char_start)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Span":
        return Span(
            doc_id=obj["doc_id"],
            char_start=int(obj["char_start"]),
            char_end=int(obj["char_end"]),
        )


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    type: QuestionType
    answerable: bool
    gold_spans: tuple[Span, ...]
    gold_answer: str

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Question":
        return Question(
            qid=obj["qid"],
            question=obj["question"],
            type=obj["type"],
            answerable=bool(obj["answerable"]),
            gold_spans=tuple(Span.from_json(s) for s in obj.get("gold_spans", [])),
            gold_answer=obj.get("gold_answer", ""),
        )


@dataclass(frozen=True)
class Claim:
    """Một câu của câu trả lời cùng đúng tập citation của riêng nó.

    Sinh theo câu thay vì sinh một khối text rồi tách lại: tách câu tiếng Việt
    bằng heuristic sẽ vỡ ở "Điều 1.", "khoản 2.", số thập phân — mà claim tách
    sai thì mọi con số groundedness đều sai theo.
    """

    claim_id: int
    text: str
    citations: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    qid: str
    question: str
    claims: tuple[Claim, ...]
    abstained: bool
    abstain_reason: str | None = None

    @property
    def text(self) -> str:
        """Câu trả lời hiển thị, mỗi câu kèm marker citation ngay sau nó."""
        if self.abstained:
            return self.abstain_reason or ""
        parts = []
        for claim in self.claims:
            marks = "".join(f"[{cid}]" for cid in claim.citations)
            parts.append(f"{claim.text} {marks}".strip())
        return " ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.text,
            "claims": [c.to_json() for c in self.claims],
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
        }
