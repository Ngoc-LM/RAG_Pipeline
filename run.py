"""CLI đầu-cuối. Mỗi lệnh con là một tầng, `all` chạy tuần tự cả bốn.

Module này cố ý chỉ làm việc nối dây: mọi logic nằm trong `src/`, để không có
đường chạy nào chỉ tồn tại khi gọi qua CLI mà chưa từng được test gọi tới.

Mọi lệnh nhận `--offline`: đọc cache, cache miss thì báo rõ key còn thiếu và trả
exit code 2 thay vì gọi API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import calibrate as calibrate_mod
from src import config
from src.evaluate import (
    answer_all,
    check_questions,
    evaluate_answers,
    load_questions,
    print_answer_report,
    print_report,
    retrieve_all,
    run as run_retrieval_eval,
    write_answer_report,
    write_report,
)
from src.generate import answer_question
from src.index import build_index, write_manifest
from src.ingest import load_corpus
from src.llm import CacheMiss, MissingAPIKey
from src.retrieve import retrieve


def _index(args) -> int:
    index = build_index(load_corpus(Path(args.corpus)))
    path = write_manifest(index, Path(args.out) / "index")
    print(f"chunks   : {len(index)}")
    print(f"manifest : {path}")
    if args.embed:
        print(f"embedding: {index.embeddings(offline=args.offline).shape} (đã chuẩn hoá L2)")
    return 0


def _ask(args) -> int:
    index = build_index(load_corpus(Path(args.corpus)))
    results = retrieve(index, args.question, arm=args.arm, offline=args.offline)
    answer = answer_question(args.question, results, offline=args.offline)

    print(f"câu hỏi : {args.question}\n")
    print(answer.answer)
    if answer.citations:
        print("\nCăn cứ:")
        for label in answer.citations:
            print(f"  {label}")
    print(
        f"\nabstain={answer.abstained} ({answer.abstain_stage})  "
        f"support_ratio: đầu={answer.support_ratio_first} "
        f"cuối={answer.support_ratio_final}  lượt sinh={len(answer.attempts)}"
    )
    return 0


def _load(args):
    documents = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))
    check_questions(questions, documents)
    return documents, questions


def _eval_retrieval(args) -> int:
    documents, questions = _load(args)
    arms = list(config.ARMS) if args.arm == "all" else [args.arm]
    report, skipped = run_retrieval_eval(
        build_index(documents), questions, arms=arms, offline=args.offline
    )
    path = write_report(report, questions, skipped, Path(args.out) / "eval")
    print_report(report, skipped, questions)
    print(f"\nChi tiết từng câu: {path}")
    return 2 if skipped else 0


def _answer(args) -> int:
    documents, questions = _load(args)
    arm = "hybrid_rerank" if args.arm == "all" else args.arm
    retrievals = retrieve_all(
        build_index(documents), questions, arm=arm, offline=args.offline
    )
    answers = answer_all(questions, retrievals, offline=args.offline)
    report = evaluate_answers(questions, answers, arm=arm)
    path = write_answer_report(report, Path(args.out) / "eval")
    print_answer_report(report)
    print(f"\nChi tiết từng câu: {path}")
    return 0


def _calibrate(args) -> int:
    documents, questions = _load(args)
    arm = "hybrid_rerank" if args.arm == "all" else args.arm
    retrievals = retrieve_all(
        build_index(documents), questions, arm=arm, offline=args.offline
    )
    points = calibrate_mod.sweep(questions, retrievals, offline=args.offline)
    best = calibrate_mod.best_point(points)
    path = calibrate_mod.write_report(points, best, Path(args.out) / "eval")
    calibrate_mod.print_report(
        points, best, sum(1 for q in questions if not q.answerable)
    )
    print(f"\nLưới đầy đủ: {path}")
    return 0


def _all(args) -> int:
    """Chạy cả bốn tầng, không dừng ở tầng đầu tiên thiếu cache.

    Với cache rỗng thì `answer` và `calibrate` chắc chắn hỏng, nhưng `index` và
    `eval --arm bm25` vẫn ra kết quả. Dừng hẳn ở tầng hỏng đầu tiên sẽ giấu mất
    phần chạy được, nên ở đây gom lỗi lại và tóm tắt ở cuối.
    """
    worst = 0
    failed: list[str] = []
    for name, step in (
        ("index", _index), ("eval", _eval_retrieval),
        ("answer", _answer), ("calibrate", _calibrate),
    ):
        print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}")
        try:
            worst = max(worst, step(args))
        except (CacheMiss, MissingAPIKey) as exc:
            print(exc)
            failed.append(name)
            worst = max(worst, 2 if isinstance(exc, CacheMiss) else 3)

    if failed:
        print(f"\nCác tầng chưa chạy được (thiếu cache hoặc API key): {', '.join(failed)}")
    return worst


COMMANDS = {
    "index": _index,
    "ask": _ask,
    "eval": _eval_retrieval,
    "answer": _answer,
    "calibrate": _calibrate,
    "all": _all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Pipeline RAG cho văn bản quy phạm pháp luật"
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("question", nargs="?", help="chỉ dùng cho lệnh `ask`")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--questions", default=str(config.QUESTIONS_PATH))
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR))
    parser.add_argument(
        "--arm", default="hybrid_rerank", choices=(*config.ARMS, "all"),
        help="`all` chỉ có nghĩa với lệnh `eval`; các lệnh khác lấy hybrid_rerank",
    )
    parser.add_argument("--embed", action="store_true", help="chỉ dùng cho lệnh `index`")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ask" and not args.question:
        parser.error("lệnh `ask` cần một câu hỏi")

    try:
        return COMMANDS[args.command](args)
    except CacheMiss as exc:
        print(exc)
        return 2
    except MissingAPIKey as exc:
        print(f"{exc}\nHoặc chạy lại với --offline nếu cache đã có sẵn.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
