"""Xây chỉ mục: embedding dày lưu .npy và chỉ mục BM25 trong bộ nhớ.

Corpus cỡ vài nghìn chunk nên cosine brute-force bằng một phép nhân ma trận
numpy đã đủ nhanh, và giải thích được từng bước — không cần FAISS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

import config
from src.llm import embed_texts
from src.schema import Chunk
from src.tokenize_vi import tokenize


@dataclass(frozen=True)
class Index:
    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray | None
    bm25: BM25Okapi

    def __post_init__(self) -> None:
        if self.embeddings is not None and len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Lệch số lượng: {len(self.chunks)} chunk vs "
                f"{self.embeddings.shape[0]} vector"
            )


def build_index(chunks: Sequence[Chunk], *, offline: bool, with_dense: bool = True) -> Index:
    """with_dense=False bỏ hẳn bước embed, để chạy arm bm25 mà không cần API key."""
    embeddings = (
        embed_texts(
            [c.indexed_text for c in chunks],
            task_type=config.EMBED_TASK_DOCUMENT,
            offline=offline,
        )
        if with_dense
        else None
    )
    bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])
    return Index(chunks=tuple(chunks), embeddings=embeddings, bm25=bm25)


def save_index(index: Index, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if index.embeddings is not None:
        np.save(directory / "embeddings.npy", index.embeddings)
    (directory / "chunks.json").write_text(
        json.dumps([c.to_json() for c in index.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(directory: Path) -> list[Chunk]:
    path = directory / "chunks.json"
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có {path}; chạy `run.py index` trước")
    return [Chunk.from_json(o) for o in json.loads(path.read_text(encoding="utf-8"))]
