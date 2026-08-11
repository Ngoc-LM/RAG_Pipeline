"""Đo truy xuất bằng coverage trên trục ký tự, không bằng nhãn "chunk gold".

Gán nhãn ở mức chunk buộc phải chọn một ngưỡng overlap, mà mọi ngưỡng như vậy
đều thiên lệch theo kích thước chunk. Ở đây đo ở mức TẬP KẾT QUẢ:

    Cov@k(span) = |span ∩ union(top-k chunk cùng doc_id)| / |span|
    hit@k(span) ⟺ Cov@k(span) >= THETA_COVERAGE
    r*(span)    = min{k : Cov@k(span) >= THETA}
    MRR         = 1/r*        (không đạt ở mọi k -> 0)

Nhờ tính trên hợp các khoảng [char_start, char_end), metric bất biến với
chunk_size theo đúng định nghĩa: chia nhỏ một chunk thành hai chunk kề nhau
không đổi hợp, nên không đổi coverage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from src import config
from src.chunk import Chunk
from src.index import Index, build_index
from src.ingest import Document, load_corpus
from src.intervals import Interval, coverage, interval_union
from src.llm import CacheMiss
from src.retrieve import Retrieved, retrieve

RETRIEVAL_TYPES = ("factoid_1hop", "multihop", "distractor")


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    char_start: int
    char_end: int

    @property
    def interval(self) -> Interval:
        return (self.char_start, self.char_end)


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    type: str
    answerable: bool
    gold_spans: tuple[GoldSpan, ...]


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            qid=item["qid"],
            question=item["question"],
            type=item["type"],
            answerable=bool(item["answerable"]),
            gold_spans=tuple(
                GoldSpan(s["doc_id"], int(s["char_start"]), int(s["char_end"]))
                for s in item["gold_spans"]
            ),
        )
        for item in raw
    ]


def check_questions(questions: Sequence[Question], documents: Sequence[Document]) -> None:
    """Chặn sớm gold_span trỏ ra ngoài corpus.

    Không có bước này thì một doc_id gõ sai chỉ làm Cov = 0 lặng lẽ, và bảng kết
    quả trông như retriever kém chứ không như dữ liệu hỏng.
    """
    lengths = {d.meta.doc_id: len(d.body) for d in documents}
    for question in questions:
        for span in question.gold_spans:
            if span.doc_id not in lengths:
                raise ValueError(f"{question.qid}: doc_id {span.doc_id!r} không có trong corpus")
            if not 0 <= span.char_start < span.char_end <= lengths[span.doc_id]:
                raise ValueError(
                    f"{question.qid}: span [{span.char_start}, {span.char_end}) "
                    f"ngoài phạm vi {span.doc_id} (len={lengths[span.doc_id]})"
                )
        if question.answerable and not question.gold_spans:
            raise ValueError(f"{question.qid}: answerable nhưng không có gold_span")
        if not question.answerable and question.gold_spans:
            raise ValueError(f"{question.qid}: unanswerable nhưng vẫn có gold_span")


# --- Coverage -------------------------------------------------------------
def coverage_curve(span: GoldSpan, ranked: Sequence[Chunk]) -> list[float]:
    """Cov@k với k = 1..len(ranked), đơn điệu không giảm.

    Chỉ gom chunk CÙNG doc_id với span: trục ký tự là trục riêng của từng
    document, nên trộn chunk khác văn bản vào sẽ cho những phần giao hoàn toàn
    ảo giữa hai khoảng số học tình cờ chồng nhau.
    """
    intervals: list[Interval] = []
    curve: list[float] = []
    for chunk in ranked:
        if chunk.doc_id == span.doc_id:
            intervals.append((chunk.char_start, chunk.char_end))
        curve.append(coverage(interval_union(intervals), span.interval))
    return curve


def first_reaching(curve: Sequence[float], theta: float) -> int | None:
    """r* = hạng nhỏ nhất đạt ngưỡng, tính từ 1. None nếu không bao giờ đạt."""
    for position, value in enumerate(curve, start=1):
        if value >= theta:
            return position
    return None


@dataclass(frozen=True)
class SpanResult:
    span: GoldSpan
    curve: tuple[float, ...]
    r_star: int | None

    def cov_at(self, k: int) -> float:
        if not self.curve:
            return 0.0
        return self.curve[min(k, len(self.curve)) - 1]


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    type: str
    spans: tuple[SpanResult, ...]
    ranked_chunk_ids: tuple[str, ...]

    def strict_hit(self, k: int) -> bool:
        """Mọi gold_span đều đạt ngưỡng. Cột chính cho câu multi-hop."""
        return bool(self.spans) and all(
            s.cov_at(k) >= config.THETA_COVERAGE for s in self.spans
        )

    def any_hit(self, k: int) -> bool:
        return any(s.cov_at(k) >= config.THETA_COVERAGE for s in self.spans)

    def mean_cov(self, k: int) -> float:
        """Trung bình coverage TRONG một câu trước, để câu 2 span không đếm đôi."""
        return fmean([s.cov_at(k) for s in self.spans]) if self.spans else 0.0

    @property
    def r_star_strict(self) -> int | None:
        """Hạng mà tại đó gom đủ căn cứ cho TOÀN BỘ span của câu."""
        if not self.spans or any(s.r_star is None for s in self.spans):
            return None
        return max(s.r_star for s in self.spans if s.r_star is not None)

    @property
    def mrr_strict(self) -> float:
        rank = self.r_star_strict
        return 0.0 if rank is None else 1.0 / rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "type": self.type,
            "r_star_strict": self.r_star_strict,
            "mrr_strict": round(self.mrr_strict, 6),
            "spans": [
                {
                    "doc_id": s.span.doc_id,
                    "char_start": s.span.char_start,
                    "char_end": s.span.char_end,
                    "r_star": s.r_star,
                    "cov": {str(k): round(s.cov_at(k), 4) for k in config.EVAL_K_VALUES},
                }
                for s in self.spans
            ],
            "ranked_chunk_ids": list(self.ranked_chunk_ids),
        }


def evaluate_question(question: Question, results: Sequence[Retrieved]) -> QuestionResult:
    ranked = [r.chunk for r in results]
    spans = []
    for span in question.gold_spans:
        curve = coverage_curve(span, ranked)
        spans.append(
            SpanResult(span, tuple(curve), first_reaching(curve, config.THETA_COVERAGE))
        )
    return QuestionResult(
        qid=question.qid,
        type=question.type,
        spans=tuple(spans),
        ranked_chunk_ids=tuple(c.chunk_id for c in ranked),
    )


# --- Tổng hợp -------------------------------------------------------------
def aggregate(results: Iterable[QuestionResult]) -> dict[str, Any]:
    """Trung bình theo CÂU, không theo span, để câu 2 span không có trọng số đôi."""
    items = list(results)
    if not items:
        return {"n": 0}
    return {
        "n": len(items),
        "recall_strict": {
            str(k): round(fmean([float(r.strict_hit(k)) for r in items]), 4)
            for k in config.EVAL_K_VALUES
        },
        "recall_any": {
            str(k): round(fmean([float(r.any_hit(k)) for r in items]), 4)
            for k in config.EVAL_K_VALUES
        },
        "mean_cov": {
            str(k): round(fmean([r.mean_cov(k) for r in items]), 4)
            for k in config.EVAL_K_VALUES
        },
        "mrr_strict": round(fmean([r.mrr_strict for r in items]), 4),
    }


def evaluate_arm(
    index: Index,
    questions: Sequence[Question],
    *,
    arm: str,
    offline: bool,
) -> dict[str, Any]:
    depth = max(config.EVAL_K_VALUES)
    answerable = [q for q in questions if q.answerable]
    per_question = [
        evaluate_question(q, retrieve(index, q.question, arm=arm, top_k=depth, offline=offline))
        for q in answerable
    ]
    by_type = {
        kind: aggregate([r for r in per_question if r.type == kind])
        for kind in RETRIEVAL_TYPES
    }
    return {
        "arm": arm,
        "theta_coverage": config.THETA_COVERAGE,
        "depth": depth,
        "overall": aggregate(per_question),
        "by_type": {k: v for k, v in by_type.items() if v.get("n")},
        "questions": [r.to_dict() for r in per_question],
    }


def run(
    index: Index,
    questions: Sequence[Question],
    *,
    arms: Sequence[str],
    offline: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Chạy từng arm, bỏ qua arm thiếu cache thay vì làm hỏng cả lượt chạy.

    Với cache rỗng thì chỉ arm bm25 chạy được; trả về một bảng bm25 kèm ghi chú
    vẫn hữu ích hơn là không trả gì. Arm bị bỏ được đánh dấu rõ và làm exit code
    khác 0, để không ai đọc nhầm bảng thiếu cột thành bảng đầy đủ.
    """
    report: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for arm in arms:
        try:
            report[arm] = evaluate_arm(index, questions, arm=arm, offline=offline)
        except CacheMiss as exc:
            skipped[arm] = str(exc).splitlines()[0]
    return report, skipped


def write_report(
    report: dict[str, Any],
    questions: Sequence[Question],
    skipped: dict[str, str],
    directory: Path | None = None,
) -> Path:
    target = (directory or config.EVAL_DIR) / "retrieval.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "theta_coverage": config.THETA_COVERAGE,
        "k_values": list(config.EVAL_K_VALUES),
        "n_questions": len(questions),
        "n_answerable": sum(1 for q in questions if q.answerable),
        "n_unanswerable": sum(1 for q in questions if not q.answerable),
        "skipped_arms": skipped,
        "arms": report,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


# --- Bảng in ra ------------------------------------------------------------
REPORT_K = (1, 5, 8, 30)


def _row(name: str, stats: dict[str, Any]) -> str:
    cells = [f"{name:<16}", f"{stats['n']:>4}"]
    cells += [f"{stats['recall_strict'][str(k)]:>10.3f}" for k in REPORT_K]
    cells += [f"{stats['recall_any'][str(config.RETRIEVE_TOP_K)]:>9.3f}"]
    cells += [f"{stats['mean_cov'][str(config.RETRIEVE_TOP_K)]:>11.3f}"]
    cells += [f"{stats['mrr_strict']:>7.3f}"]
    return " ".join(cells)


def _header() -> str:
    top = config.RETRIEVE_TOP_K
    parts = [f"{'':<16}", f"{'n':>4}"]
    parts += [f"{'Rs@' + str(k):>10}" for k in REPORT_K]
    parts += [f"{'Ra@' + str(top):>9}", f"{'cov@' + str(top):>11}", f"{'MRR':>7}"]
    return " ".join(parts)


def print_report(report: dict[str, Any], skipped: dict[str, str], questions) -> None:
    print(f"θ = {config.THETA_COVERAGE}   Rs = recall strict, Ra = recall any\n")
    print(_header())
    print("-" * len(_header()))
    for arm, data in report.items():
        print(_row(arm, data["overall"]))
    print()

    for arm, data in report.items():
        if not data["by_type"]:
            continue
        print(f"[{arm}] tách theo type")
        print(_header())
        print("-" * len(_header()))
        for kind, stats in data["by_type"].items():
            print(_row(kind, stats))
        print()

    unanswerable = sum(1 for q in questions if not q.answerable)
    print(
        f"{unanswerable} câu unanswerable không tham gia metric truy xuất "
        "(chúng đo chất lượng abstain ở tầng generate)."
    )
    for arm, reason in skipped.items():
        print(f"BỎ QUA arm {arm}: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Đo truy xuất bằng coverage")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--questions", default=str(config.QUESTIONS_PATH))
    parser.add_argument("--out", default=str(config.EVAL_DIR))
    parser.add_argument(
        "--arms", default="all", help="danh sách arm ngăn bằng dấu phẩy, hoặc 'all'"
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    arms = list(config.ARMS) if args.arms == "all" else args.arms.split(",")
    unknown = [a for a in arms if a not in config.ARMS]
    if unknown:
        parser.error(f"arm không hợp lệ: {unknown} (chọn trong {list(config.ARMS)})")

    documents = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))
    check_questions(questions, documents)

    report, skipped = run(
        build_index(documents), questions, arms=arms, offline=args.offline
    )
    path = write_report(report, questions, skipped, Path(args.out))
    print_report(report, skipped, questions)
    print(f"\nChi tiết từng câu: {path}")
    return 2 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
