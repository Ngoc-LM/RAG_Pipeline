"""Quét lưới hai ngưỡng abstain trên gold set.

Lưới chạy PIPELINE THẬT ở từng điểm chứ không mô phỏng bằng số học trên điểm đã
thu được. Mô phỏng sẽ trượt khỏi hành vi thật ở đúng chỗ khó nhất: vòng sinh lại
phụ thuộc TAU_GROUND, nên "câu trả lời ở ngưỡng 0.5" không suy ra được từ "câu
trả lời ở ngưỡng 0.8".

Chạy thật mà vẫn rẻ là nhờ cache: kết quả truy xuất không phụ thuộc ngưỡng nên
tính một lần; còn prompt lượt hai dựng từ danh sách mệnh đề bị judge bác, mà danh
sách đó cũng không phụ thuộc ngưỡng. Tổng lại mỗi câu chỉ tốn tối đa 2 lời gọi
generate và 2 lời gọi judge cho TOÀN BỘ lưới, không phải 2 lời gọi mỗi điểm lưới.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src import config
from src.evaluate import (
    Question,
    abstain_stats,
    answer_all,
    check_questions,
    faithfulness_stats,
    load_questions,
    retrieve_all,
)
from src.index import build_index
from src.ingest import load_corpus
from src.llm import CacheMiss
from src.retrieve import Retrieved


@dataclass(frozen=True)
class GridPoint:
    tau_retrieve: float
    tau_ground: float
    abstain: dict[str, Any]
    faithfulness: dict[str, Any]

    @property
    def f1(self) -> float:
        return float(self.abstain["f1"])

    @property
    def missed_abstain(self) -> int:
        """Trả lời một câu không có căn cứ — lỗi nguy hiểm hơn từ chối nhầm."""
        return int(self.abstain["missed_abstain"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_retrieve": self.tau_retrieve,
            "tau_ground": self.tau_ground,
            "abstain": self.abstain,
            "faithfulness": self.faithfulness,
        }


def sweep(
    questions: Sequence[Question],
    retrievals: Sequence[Sequence[Retrieved]],
    *,
    offline: bool,
    tau_retrieve_grid: Sequence[float] = config.CALIBRATE_TAU_RETRIEVE_GRID,
    tau_ground_grid: Sequence[float] = config.CALIBRATE_TAU_GROUND_GRID,
) -> list[GridPoint]:
    points: list[GridPoint] = []
    for tau_retrieve in tau_retrieve_grid:
        for tau_ground in tau_ground_grid:
            answers = answer_all(
                questions, retrievals, offline=offline,
                tau_retrieve=tau_retrieve, tau_ground=tau_ground,
            )
            points.append(
                GridPoint(
                    tau_retrieve=tau_retrieve,
                    tau_ground=tau_ground,
                    abstain=abstain_stats(questions, answers),
                    faithfulness=faithfulness_stats(answers),
                )
            )
    return points


def best_point(points: Sequence[GridPoint]) -> GridPoint:
    """F1 cao nhất; hoà thì ưu tiên ít `missed_abstain` hơn, rồi ngưỡng thấp hơn.

    Hai loại lỗi không đối xứng: trả lời một câu không có căn cứ trong corpus là
    đúng dạng lỗi mà cả pipeline này sinh ra để chặn, còn từ chối nhầm chỉ gây
    phiền. Nên khi F1 bằng nhau, phá hoà về phía an toàn. Mức phá hoà cuối cùng
    là ngưỡng thấp hơn, để không siết chặt hơn mức dữ liệu biện minh được.
    """
    if not points:
        raise ValueError("Lưới rỗng")
    return min(
        points,
        key=lambda p: (-p.f1, p.missed_abstain, p.tau_retrieve, p.tau_ground),
    )


def plateau(points: Sequence[GridPoint], best: GridPoint) -> list[GridPoint]:
    """Mọi điểm đạt đúng F1 cao nhất.

    Với gold set chỉ có 4 câu unanswerable, argmax gần như luôn nằm trên một vùng
    bằng phẳng rộng. Hình dạng vùng đó mới là thứ biện minh cho lựa chọn ngưỡng,
    chứ không phải bản thân điểm argmax.
    """
    return [p for p in points if p.f1 == best.f1]


def write_report(points: Sequence[GridPoint], best: GridPoint, directory: Path | None = None) -> Path:
    target = (directory or config.EVAL_DIR) / "calibration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau_retrieve_grid": list(config.CALIBRATE_TAU_RETRIEVE_GRID),
        "tau_ground_grid": list(config.CALIBRATE_TAU_GROUND_GRID),
        "best": best.to_dict(),
        "plateau_size": len(plateau(points, best)),
        "grid": [p.to_dict() for p in points],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


HEADER = (
    f"{'tau_ret':>8} {'tau_gnd':>8} {'F1':>7} {'P':>7} {'R':>7} "
    f"{'từ chối nhầm':>13} {'bỏ sót':>8} {'trả lời':>8}"
)


def _row(point: GridPoint, mark: str) -> str:
    stats = point.abstain
    return (
        f"{point.tau_retrieve:>8.2f} {point.tau_ground:>8.2f} "
        f"{stats['f1']:>7.3f} {stats['precision']:>7.3f} {stats['recall']:>7.3f} "
        f"{stats['wrong_abstain']:>13} {stats['missed_abstain']:>8} "
        f"{stats['true_answer']:>8} {mark}"
    )


def print_report(points: Sequence[GridPoint], best: GridPoint, n_unanswerable: int) -> None:
    print(HEADER)
    print("-" * len(HEADER))
    on_plateau = set(id(p) for p in plateau(points, best))
    for point in points:
        mark = " <<<" if point is best else (" ~" if id(point) in on_plateau else "")
        print(_row(point, mark))

    print(
        f"\nChọn: TAU_RETRIEVE = {best.tau_retrieve}, TAU_GROUND = {best.tau_ground} "
        f"(F1 = {best.f1:.3f}, vùng bằng phẳng {len(plateau(points, best))} điểm)"
    )
    print(
        f"CẢNH BÁO: chỉ có {n_unanswerable} câu unanswerable, nên recall của lớp "
        f"abstain nhảy theo bước {1 / n_unanswerable:.2f} nếu n > 0. Con số F1 ở "
        "đây có khoảng tin cậy rất rộng — đọc hình dạng vùng bằng phẳng, đừng đọc "
        "riêng điểm argmax."
        if n_unanswerable
        else "CẢNH BÁO: không có câu unanswerable nào, F1 không có ý nghĩa."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quét lưới hai ngưỡng abstain")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--questions", default=str(config.QUESTIONS_PATH))
    parser.add_argument("--out", default=str(config.EVAL_DIR))
    parser.add_argument("--arm", default="hybrid_rerank", choices=config.ARMS)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    documents = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))
    check_questions(questions, documents)

    try:
        retrievals = retrieve_all(
            build_index(documents), questions, arm=args.arm, offline=args.offline
        )
        points = sweep(questions, retrievals, offline=args.offline)
    except CacheMiss as exc:
        print(exc)
        return 2

    best = best_point(points)
    path = write_report(points, best, Path(args.out))
    print_report(points, best, sum(1 for q in questions if not q.answerable))
    print(f"\nLưới đầy đủ: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
