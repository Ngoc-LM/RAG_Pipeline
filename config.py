"""Hằng số điều khiển toàn pipeline. Không hardcode tham số ở module khác."""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent
CORPUS_DIR: Final[Path] = ROOT / "data" / "corpus"
QUESTIONS_PATH: Final[Path] = ROOT / "eval" / "questions.json"
OUTPUT_DIR: Final[Path] = ROOT / "outputs"
CACHE_DIR: Final[Path] = OUTPUT_DIR / "cache"
BODY_DIR: Final[Path] = OUTPUT_DIR / "bodies"

# --- Model ---------------------------------------------------------------
EMBED_MODEL: Final[str] = "gemini-embedding-001"
GEN_MODEL: Final[str] = "gemini-2.5-flash"
RERANK_MODEL: Final[str] = "gemini-2.5-flash-lite"
JUDGE_MODEL: Final[str] = "llama-3.3-70b-versatile"

EMBED_DIM: Final[int] = 768
"""Matryoshka truncation của gemini-embedding-001 (gốc 3072).

768 giảm dung lượng cache commit vào repo ~4 lần. Bắt buộc re-normalize L2 sau
khi cắt chiều vì chỉ vector 3072 chiều mới được chuẩn hoá sẵn.
"""

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
"""Tắt thinking của họ 2.5.

Cả sinh câu trả lời lẫn rerank ở đây đều là tác vụ trích rút bám sát context,
không phải suy luận nhiều bước. Tắt thinking giữ output ổn định hơn ở
temperature=0 và không đốt quota free tier vào token ẩn. Đây là hằng số nên
bật lại để so sánh chỉ là sửa một dòng.
"""

PROMPT_VERSION: Final[int] = 1
"""Tăng tay mỗi khi sửa bất kỳ template prompt nào.

Cache key băm nội dung *tham số* của lời gọi (task, input, params) chứ không
băm chuỗi prompt đã render. Nhờ vậy đổi cách trình bày prompt không làm hỏng
toàn bộ cache, nhưng cũng có nghĩa là cache KHÔNG tự phát hiện được khi template
đổi. prompt_version là cái chốt thủ công đó: tăng số này để vô hiệu hoá các
entry cũ thay vì phải xoá thư mục cache.
"""

# --- Chunk ---------------------------------------------------------------
CHUNK_TARGET_CHARS: Final[int] = 1200
CHUNK_MAX_CHARS: Final[int] = 1800
CHUNK_HARD_SPLIT_OVERLAP: Final[int] = 150
"""Overlap chỉ dùng khi buộc phải cắt cứng một Khoản dài quá CHUNK_MAX_CHARS."""

CHUNK_SIZE_SWEEP: Final[tuple[int, ...]] = (600, 1200, 2400)
"""Các giá trị CHUNK_TARGET_CHARS quét trong bảng ablation."""

# --- Retrieval -----------------------------------------------------------
TOP_K_DENSE: Final[int] = 20
TOP_K_BM25: Final[int] = 20
RRF_K: Final[int] = 60
TOP_K_RERANK_CANDIDATES: Final[int] = 20
TOP_K_CONTEXT: Final[int] = 5
"""Số chunk thực sự đưa vào prompt sinh câu trả lời."""

EVAL_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10)

# --- Ngưỡng đánh giá và abstain ------------------------------------------
THETA_COVERAGE: Final[float] = 0.8
"""Ngưỡng coverage để coi một gold_span là đã được truy xuất đủ căn cứ."""

TAU_RETRIEVE: Final[float] = 0.45
"""Gate TRƯỚC generate, so với điểm rerank cao nhất đã chuẩn hoá về [0,1]."""

TAU_GROUND: Final[float] = 0.8
"""Gate SAU generate, so với support_ratio = #claim được hỗ trợ / #claim."""

TAU_RETRIEVE_SWEEP: Final[tuple[float, ...]] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
TAU_GROUND_SWEEP: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

MAX_REGENERATE_ATTEMPTS: Final[int] = 1
"""Số lần sinh lại sau khi verify fail. Tổng số lượt = 1 + giá trị này."""

ABSTAIN_MESSAGE: Final[str] = "Không đủ căn cứ trong tài liệu."

# --- QC ------------------------------------------------------------------
LEAKAGE_JACCARD_MAX: Final[float] = 0.3
"""Jaccard unigram giữa câu hỏi và text gold_span vượt ngưỡng này -> cảnh báo."""

# --- Backoff -------------------------------------------------------------
RETRY_MAX_ATTEMPTS: Final[int] = 6
RETRY_BASE_SECONDS: Final[float] = 2.0
RETRY_MAX_SLEEP_SECONDS: Final[float] = 60.0
RETRY_JITTER: Final[float] = 0.25
