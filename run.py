"""CLI của pipeline. Mọi kết quả trung gian ghi ra outputs/ dưới dạng JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import config
from src.calibrate import collect_signals, sweep
from src.check_leakage import check_leakage
from src.chunk import chunk_corpus
from src.evaluate import (
    answer_table,
    by_type_table,
    chunk_size_sweep,
    chunk_table,
    full_eval,
    load_questions,
    retrieval_eval,
    retrieval_table,
)
from src.index import Index, build_index, save_index
from src.ingest import load_corpus, write_documents
from src.llm import CacheMiss, MissingAPIKey, load_dotenv
from src.retrieve import ARMS, retrieve
from src.schema import Document
from src.verify import answer_with_verification

DEFAULT_FULL_ARMS = ("bm25", "hybrid_rerank")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path.relative_to(config.ROOT)}")


def load_documents(args: argparse.Namespace) -> list[Document]:
    docs = load_corpus(Path(args.corpus_dir))
    write_documents(docs, config.OUTPUT_DIR / "documents.json", config.BODY_DIR)
    return docs


def load_index(
    args: argparse.Namespace, docs: Sequence[Document], *, with_dense: bool = True
) -> Index:
    chunks = chunk_corpus(
        list(docs), args.chunk_size, args.chunk_max, config.CHUNK_HARD_SPLIT_OVERLAP
    )
    print(f"{len(docs)} document -> {len(chunks)} chunk")
    index = build_index(chunks, offline=args.offline, with_dense=with_dense)
    save_index(index, config.OUTPUT_DIR / "index")
    return index


def needs_dense(arms: Sequence[str]) -> bool:
    return any(arm != "bm25" for arm in arms)


def cmd_ingest(args: argparse.Namespace) -> int:
    docs = load_documents(args)
    for doc in docs:
        print(f"  {doc.doc_id}: {len(doc.body)} ký tự ({doc.source_path})")
    return 0


def cmd_show_span(args: argparse.Namespace) -> int:
    """In đúng đoạn mà một gold_span trỏ tới, để soạn eval/questions.json."""
    docs = {d.doc_id: d for d in load_corpus(Path(args.corpus_dir))}
    doc = docs.get(args.doc_id)
    if doc is None:
        print(f"Không có doc_id '{args.doc_id}'. Có: {sorted(docs)}", file=sys.stderr)
        return 1
    if not 0 <= args.start < args.end <= len(doc.body):
        print(
            f"Khoảng ngoài phạm vi: body của {args.doc_id} dài {len(doc.body)}",
            file=sys.stderr,
        )
        return 1
    print(f"--- {args.doc_id}[{args.start}:{args.end}] ---")
    print(doc.body[args.start : args.end])
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    load_index(args, load_documents(args))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    dense = needs_dense([args.arm])
    index = load_index(args, load_documents(args), with_dense=dense)
    result = retrieve(
        index,
        args.question,
        arm=args.arm,
        offline=args.offline,
        query_vector=_query_vector(args) if dense else None,
        top_k=config.TOP_K_CONTEXT,
    )
    score = result.rerank_top_score
    if score is not None and score < config.TAU_RETRIEVE:
        print(f"{config.ABSTAIN_MESSAGE} (điểm rerank cao nhất {score:.2f} < "
              f"TAU_RETRIEVE {config.TAU_RETRIEVE})")
        return 0

    answer, record = answer_with_verification(
        "ask", args.question, result.chunks, offline=args.offline
    )
    print(f"\n{answer.text}\n")
    print("Nguồn:")
    for chunk in result.chunks:
        print(f"  [{chunk.chunk_id}] {chunk.breadcrumb} "
              f"(ký tự {chunk.char_start}-{chunk.char_end})")
    write_json(config.OUTPUT_DIR / "ask.json", record)
    return 0


def _query_vector(args: argparse.Namespace):
    from src.llm import embed_texts

    return embed_texts(
        [args.question], task_type=config.EMBED_TASK_QUERY, offline=args.offline
    )[0]


def cmd_answer(args: argparse.Namespace) -> int:
    """Chạy toàn bộ gold set qua pipeline có kiểm chứng."""
    index = load_index(args, load_documents(args), with_dense=needs_dense([args.arm]))
    questions = load_questions(Path(args.questions))
    results = full_eval(index, questions, arms=[args.arm], offline=args.offline)
    records = results[args.arm]["records"]
    write_json(config.OUTPUT_DIR / "verify.json", records)
    write_json(
        config.OUTPUT_DIR / "answers.json",
        [
            {"qid": r["qid"], "question": r["question"], "answer": r["final_answer"],
             "final_action": r["final_action"], "citations": r["context_chunk_ids"]}
            for r in records
        ],
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    docs = load_documents(args)
    default_arms = list(DEFAULT_FULL_ARMS) if args.full else list(ARMS)
    arms = args.arms.split(",") if args.arms else default_arms
    index = load_index(args, docs, with_dense=needs_dense(arms))
    questions = load_questions(Path(args.questions))
    sections: dict[str, str] = {}

    if not args.full:
        results = retrieval_eval(
            index,
            questions,
            arms=arms,
            offline=args.offline,
            top_k=max(config.EVAL_K_VALUES),
        )
        write_json(config.OUTPUT_DIR / "eval_retrieval.json", results)
        sections["Ablation truy xuất"] = retrieval_table(results)
        sections[f"Recall strict@{config.TOP_K_CONTEXT} theo loại câu hỏi"] = (
            by_type_table(results, config.TOP_K_CONTEXT)
        )
        if args.chunk_sweep:
            swept = chunk_size_sweep(
                docs, questions, config.CHUNK_SIZE_SWEEP,
                arm="hybrid", offline=args.offline,
            )
            write_json(config.OUTPUT_DIR / "eval_chunk_sweep.json", swept)
            sections["Quét chunk_size (arm hybrid)"] = chunk_table(swept)
    else:
        results = full_eval(index, questions, arms=arms, offline=args.offline)
        write_json(config.OUTPUT_DIR / "eval_answer.json", results)
        for arm in arms:
            write_json(
                config.OUTPUT_DIR / f"verify_{arm}.json", results[arm]["records"]
            )
        sections["Chất lượng câu trả lời"] = answer_table(results)

    print("\n" + "\n\n".join(f"## {t}\n\n{c}" for t, c in sections.items()))
    _write_results(sections)
    return 0


def _write_results(sections: dict[str, str]) -> None:
    """Cập nhật results.md theo từng mục.

    Giữ các mục ở lần chạy trước (ví dụ bảng retrieval khi lần này chỉ chạy
    --full) nhưng ghi đè mục trùng tên, để file không phình ra vì chạy lại.
    """
    store = config.OUTPUT_DIR / "report_sections.json"
    merged: dict[str, str] = {}
    if store.is_file():
        merged.update(json.loads(store.read_text(encoding="utf-8")))
    merged.update(sections)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    body = "\n\n".join(f"## {title}\n\n{content}" for title, content in merged.items())
    path = config.ROOT / "results.md"
    path.write_text(
        "# Kết quả\n\nSinh bằng `run.py evaluate`. "
        "Tái hiện không cần API key: `python run.py evaluate --offline`.\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    print(f"-> {path.name}")


def cmd_calibrate(args: argparse.Namespace) -> int:
    index = load_index(args, load_documents(args))
    questions = load_questions(Path(args.questions))
    signals = collect_signals(index, questions, offline=args.offline, arm="hybrid_rerank")
    report = sweep(signals, questions)
    write_json(config.OUTPUT_DIR / "calibration.json", report)
    best = report["best"]
    print(
        f"\nTối ưu: TAU_RETRIEVE={best['tau_retrieve']} TAU_GROUND={best['tau_ground']} "
        f"-> abstain F1={best['abstain_f1']} (fp={best['fp']:.0f} fn={best['fn']:.0f})"
    )
    print(f"config.py hiện tại: {report['current_config']}")
    return 0


def cmd_check_leakage(args: argparse.Namespace) -> int:
    docs = load_corpus(Path(args.corpus_dir))
    questions = load_questions(Path(args.questions))
    report = check_leakage(questions, docs)
    write_json(config.OUTPUT_DIR / "leakage.json", report)
    print(
        f"\n{report['n_flagged']}/{report['n_spans']} span vượt ngưỡng "
        f"{report['threshold']} (max Jaccard {report['max_jaccard']})"
    )
    for row in report["flagged"]:
        print(f"  ! {row['qid']} span {row['span_index']}: {row['jaccard']}")
    return 0 if report["n_flagged"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline RAG cho văn bản QPPL tiếng Việt")
    parser.add_argument("--offline", action="store_true",
                        help="Chỉ đọc cache; cache miss thì báo lỗi thay vì gọi API")
    parser.add_argument("--corpus-dir", default=str(config.CORPUS_DIR))
    parser.add_argument("--questions", default=str(config.QUESTIONS_PATH))
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_TARGET_CHARS)
    parser.add_argument("--chunk-max", type=int, default=config.CHUNK_MAX_CHARS)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Đọc corpus, ghi documents.json và bodies/")
    subparsers.add_parser("index", help="Chunk và xây chỉ mục")
    subparsers.add_parser("calibrate", help="Quét lưới hai ngưỡng abstain")
    subparsers.add_parser("check-leakage", help="QC gold set: Jaccard câu hỏi vs gold_span")

    span = subparsers.add_parser("show-span", help="In text của một khoảng ký tự")
    span.add_argument("--doc-id", required=True)
    span.add_argument("--start", type=int, required=True)
    span.add_argument("--end", type=int, required=True)

    ask = subparsers.add_parser("ask", help="Hỏi một câu")
    ask.add_argument("question")
    ask.add_argument("--arm", default="hybrid_rerank", choices=ARMS)

    answer = subparsers.add_parser("answer", help="Chạy toàn bộ gold set")
    answer.add_argument("--arm", default="hybrid_rerank", choices=ARMS)

    evaluate = subparsers.add_parser("evaluate", help="Metric và bảng ablation")
    group = evaluate.add_mutually_exclusive_group()
    group.add_argument("--retrieval-only", dest="full", action="store_false",
                       help="Mọi arm, không sinh câu trả lời, không gọi judge (mặc định)")
    group.add_argument("--full", dest="full", action="store_true",
                       help="Sinh câu trả lời + verify + judge, chỉ trên --arms")
    evaluate.set_defaults(full=False)
    evaluate.add_argument(
        "--arms",
        default=None,
        help="Danh sách arm ngăn cách bằng dấu phẩy; mặc định mọi arm với "
             "--retrieval-only và bm25,hybrid_rerank với --full",
    )
    evaluate.add_argument("--chunk-sweep", action="store_true")
    return parser


COMMANDS = {
    "ingest": cmd_ingest,
    "index": cmd_index,
    "show-span": cmd_show_span,
    "ask": cmd_ask,
    "answer": cmd_answer,
    "evaluate": cmd_evaluate,
    "calibrate": cmd_calibrate,
    "check-leakage": cmd_check_leakage,
}


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        return COMMANDS[args.command](args)
    except (CacheMiss, MissingAPIKey) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
