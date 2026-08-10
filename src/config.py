"""Mọi hằng số điều khiển pipeline. Không hardcode tham số ở module khác."""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CORPUS_DIR: Final[Path] = ROOT / "data" / "corpus"
OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
CACHE_DIR: Final[Path] = OUTPUTS_DIR / "cache"
QUESTIONS_PATH: Final[Path] = ROOT / "eval" / "questions.json"

# --- Chunk ---------------------------------------------------------------
CHUNK_MAX_TOKENS: Final[int] = 600
"""Trần cho một khoản. Vượt thì cắt tiếp tại ranh giới câu.

Chọn 600 vì phần lớn khoản trong văn bản QPPL nằm dưới ngưỡng này, nên đường
cắt câu là ngoại lệ chứ không phải đường đi thường xuyên — giữ được ranh giới
ngữ nghĩa cho đa số chunk.
"""

CHUNK_MIN_TOKENS: Final[int] = 120
"""Dưới ngưỡng này thì gộp với khoản liền kề trong cùng một Điều.

Khoản một dòng kiểu "Chính phủ quy định chi tiết Điều này" gần như không mang
tín hiệu truy xuất khi đứng một mình, và làm loãng chỉ mục.
"""

CHUNK_OVERLAP_RATIO: Final[float] = 0.15
"""Tỉ lệ chồng lấn giữa hai phần liền kề khi buộc phải cắt một khoản dài.

Đủ để câu bị cắt ngang vẫn còn ngữ cảnh ở phần sau, chưa đủ nhiều để làm phình
chỉ mục.
"""

CHARS_PER_TOKEN: Final[float] = 3.5
"""Ước lượng số ký tự mỗi token cho tiếng Việt có dấu."""

# --- Đánh giá ------------------------------------------------------------
THETA_COVERAGE: Final[float] = 0.8
"""Ngưỡng coverage để coi một gold_span là đã truy xuất đủ căn cứ. Dùng ở
bước evaluate, khai báo sẵn ở đây cho tập trung."""

# --- LLM -----------------------------------------------------------------
EMBED_MODEL: Final[str] = "gemini-embedding-001"
GEN_MODEL: Final[str] = "gemini-2.5-flash"
RERANK_MODEL: Final[str] = "gemini-2.5-flash-lite"
JUDGE_MODEL: Final[str] = "llama-3.3-70b-versatile"
"""Judge khác họ model với bộ sinh: dùng Gemini chấm output của Gemini thì có
thiên lệch tự ưu ái."""

EMBED_DIM: Final[int] = 768
"""Cắt Matryoshka từ 3072 để cache commit vào repo nhỏ đi ~4 lần. Bắt buộc
chuẩn hoá L2 lại sau khi cắt vì chỉ vector 3072 chiều mới được chuẩn hoá sẵn."""

EMBED_BATCH_SIZE: Final[int] = 32

EMBED_TASK_DOCUMENT: Final[str] = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY: Final[str] = "RETRIEVAL_QUERY"
"""Embedding bất đối xứng: chunk và câu hỏi được nhúng bằng hai task_type khác
nhau, nên cùng một chuỗi ở hai vai trò cho hai vector — và hai cache key khác nhau."""
JUDGE_TEMPERATURE: Final[float] = 0.0
JUDGE_MAX_TOKENS: Final[int] = 2048

PROMPT_VERSION: Final[int] = 1
"""Tăng tay mỗi khi sửa bất kỳ template prompt nào.

Cache key băm payload có cấu trúc chứ không băm chuỗi prompt đã render, nên
template đổi mà payload không đổi sẽ không tự bị phát hiện. Đây là cái chốt
thủ công để vô hiệu hoá entry cũ thay vì phải xoá thư mục cache.
"""

# --- Backoff -------------------------------------------------------------
RETRY_MAX_ATTEMPTS: Final[int] = 6
RETRY_BASE_SECONDS: Final[float] = 2.0
RETRY_MAX_SLEEP_SECONDS: Final[float] = 60.0
RETRY_JITTER: Final[float] = 0.25
