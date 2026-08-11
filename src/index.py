"""Chỉ mục lai trên cùng một danh sách chunk: BM25 âm tiết + embedding dày.

Hai chỉ mục dùng chung đúng một danh sách chunk và đúng một chuỗi đầu vào
(`Chunk.indexed_text`), nên chỉ số hàng là danh tính chung — RRF ở tầng trên chỉ
việc hợp nhất hai danh sách chỉ số, không cần ánh xạ id qua lại.

Phía embedding nạp TRỄ. Arm `bm25` không cần vector nào, nên bắt nó chờ 160 lời
gọi embedding chỉ để chạy được là sai; lười hoá cũng khiến `--offline` với cache
rỗng vẫn chạy được arm thuần từ khoá thay vì chết ngay ở bước dựng chỉ mục.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.chunk import Chunk, chunk_corpus
from src.ingest import Document, load_corpus
from src.llm import CacheMiss, embed_texts
from src.tokenize_vi import tokenize

MANIFEST_NAME = "chunks.json"


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25Okapi
    _embeddings: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def bm25_scores(self, question: str) -> np.ndarray:
        """Điểm BM25 của mọi chunk. Không chặn trên, không so được với cosine."""
        return np.asarray(self.bm25.get_scores(tokenize(question)), dtype=np.float64)

    def embeddings(self, *, offline: bool) -> np.ndarray:
        """Ma trận (n, EMBED_DIM) đã chuẩn hoá L2, nhúng một lần rồi nhớ lại."""
        if self._embeddings is None:
            self._embeddings = embed_texts(
                [c.indexed_text for c in self.chunks],
                task_type=config.EMBED_TASK_DOCUMENT,
                offline=offline,
            )
        return self._embeddings

    def dense_scores(self, question: str, *, offline: bool) -> np.ndarray:
        """Cosine giữa câu hỏi và mọi chunk.

        Cả hai phía đã chuẩn hoá L2 nên cosine chính là tích vô hướng: một phép
        nhân ma trận numpy trên (160, 768) là đủ, không cần chỉ mục xấp xỉ.

        Câu hỏi nhúng bằng `RETRIEVAL_QUERY` còn chunk bằng `RETRIEVAL_DOCUMENT`
        — embedding bất đối xứng, nên cùng một chuỗi ở hai vai trò cho hai vector
        khác nhau và hai cache key khác nhau.
        """
        query = embed_texts(
            [question], task_type=config.EMBED_TASK_QUERY, offline=offline
        )
        return self.embeddings(offline=offline) @ query[0]


def build_index(documents: Sequence[Document]) -> Index:
    """Dựng chỉ mục từ corpus đã parse. Không chạm mạng."""
    chunks = chunk_corpus(documents)
    if not chunks:
        raise ValueError("Corpus rỗng: không dựng được chỉ mục")

    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ValueError(f"chunk_id trùng: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)

    return Index(chunks=chunks, bm25=BM25Okapi([tokenize(c.indexed_text) for c in chunks]))


def chunk_record(chunk: Chunk) -> dict[str, Any]:
    """Bản ghi JSON của một chunk, đủ để dựng lại mọi con số offline."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "doc_type": chunk.doc_type,
        "status": chunk.status,
        "effective_from": chunk.effective_from,
        "effective_to": chunk.effective_to,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "article_no": chunk.article_no,
        "article_title": chunk.article_title,
        "clause_range": chunk.clause_range,
        "citation_label": chunk.citation_label,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "n_tokens": chunk.n_tokens,
        "is_lead_in": chunk.is_lead_in,
        "text": chunk.text,
    }


def write_manifest(index: Index, directory: Path | None = None) -> Path:
    """Ghi ảnh chụp danh sách chunk ra `outputs/index/chunks.json`.

    Vector KHÔNG nằm ở đây: chúng đã có trong `outputs/cache/embed/` theo từng
    text. Chép sang nơi thứ hai chỉ tạo ra hai nguồn sự thật lệch pha nhau khi
    đổi tham số chunk.
    """
    target = (directory or config.INDEX_DIR) / MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
        "chunk_min_tokens": config.CHUNK_MIN_TOKENS,
        "chunk_overlap_ratio": config.CHUNK_OVERLAP_RATIO,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "n_chunks": len(index),
        "chunks": [chunk_record(c) for c in index.chunks],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng chỉ mục và ghi manifest")
    parser.add_argument("--corpus", default=str(config.CORPUS_DIR))
    parser.add_argument("--out", default=str(config.INDEX_DIR))
    parser.add_argument(
        "--embed",
        action="store_true",
        help="nhúng luôn toàn bộ chunk (cần GEMINI_API_KEY hoặc cache đã có)",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    index = build_index(load_corpus(Path(args.corpus)))
    path = write_manifest(index, Path(args.out))
    print(f"chunks   : {len(index)}")
    print(f"manifest : {path}")

    if not args.embed:
        print("embedding: bỏ qua (thêm --embed để nhúng)")
        return 0

    try:
        matrix = index.embeddings(offline=args.offline)
    except CacheMiss as exc:
        print(exc)
        return 2
    print(f"embedding: {matrix.shape} (đã chuẩn hoá L2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
