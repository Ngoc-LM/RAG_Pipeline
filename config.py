"""Hằng số của các module TẦNG TRÊN chưa được dựng lại trên nền móng mới.

Nền móng hiện tại (`src/ingest.py`, `src/chunk.py`) đọc hằng số từ
`src/config.py`. File này chỉ còn giữ những hằng số của index / retrieve /
generate / verify / evaluate / calibrate / run — các module viết trước khi đổi
sang cấu trúc Chương-Điều-Khoản và còn phải viết lại. Những hằng số dùng chung
được re-export từ `src/config.py` để không có hai nguồn sự thật.
"""

from __future__ import annotations

from typing import Final

from src.config import (  # noqa: F401  (re-export cho module tầng trên)
    CACHE_DIR,
    CORPUS_DIR,
    OUTPUTS_DIR,
    PROMPT_VERSION,
    QUESTIONS_PATH,
    ROOT,
    THETA_COVERAGE,
)

OUTPUT_DIR: Final = OUTPUTS_DIR
BODY_DIR: Final = OUTPUTS_DIR / "bodies"

# --- Model ---------------------------------------------------------------
EMBED_MODEL: Final[str] = "gemini-embedding-001"
GEN_MODEL: Final[str] = "gemini-2.5-flash"
RERANK_MODEL: Final[str] = "gemini-2.5-flash-lite"
JUDGE_MODEL: Final[str] = "llama-3.3-70b-versatile"

EMBED_DIM: Final[int] = 768
EMBED_TASK_DOCUMENT: Final[str] = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY: Final[str] = "RETRIEVAL_QUERY"
EMBED_BATCH_SIZE: Final[int] = 32

GEN_TEMPERATURE: Final[float] = 0.0
GEN_MAX_TOKENS: Final[int] = 2048
RERANK_TEMPERATURE: Final[float] = 0.0
RERANK_MAX_TOKENS: Final[int] = 2048
JUDGE_TEMPERATURE: Final[float] = 0.0
JUDGE_MAX_TOKENS: Final[int] = 2048
GEN_THINKING_BUDGET: Final[int] = 0
RERANK_THINKING_BUDGET: Final[int] = 0

# --- Retrieval -----------------------------------------------------------
TOP_K_DENSE: Final[int] = 20
TOP_K_BM25: Final[int] = 20
RRF_K: Final[int] = 60
TOP_K_RERANK_CANDIDATES: Final[int] = 20
TOP_K_CONTEXT: Final[int] = 5
EVAL_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10)

# --- Ngưỡng abstain ------------------------------------------------------
TAU_RETRIEVE: Final[float] = 0.45
TAU_GROUND: Final[float] = 0.8
TAU_RETRIEVE_SWEEP: Final[tuple[float, ...]] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
TAU_GROUND_SWEEP: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MAX_REGENERATE_ATTEMPTS: Final[int] = 1
ABSTAIN_MESSAGE: Final[str] = "Không đủ căn cứ trong tài liệu."

# --- QC ------------------------------------------------------------------
LEAKAGE_JACCARD_MAX: Final[float] = 0.3

# --- Backoff -------------------------------------------------------------
RETRY_MAX_ATTEMPTS: Final[int] = 6
RETRY_BASE_SECONDS: Final[float] = 2.0
RETRY_MAX_SLEEP_SECONDS: Final[float] = 60.0
RETRY_JITTER: Final[float] = 0.25
