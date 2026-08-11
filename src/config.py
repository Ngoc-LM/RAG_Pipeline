"""Mọi hằng số điều khiển pipeline. Không hardcode tham số ở module khác."""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CORPUS_DIR: Final[Path] = ROOT / "data" / "corpus"
OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
CACHE_DIR: Final[Path] = OUTPUTS_DIR / "cache"
INDEX_DIR: Final[Path] = OUTPUTS_DIR / "index"
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

# --- Truy xuất -----------------------------------------------------------
ARMS: Final[tuple[str, ...]] = ("bm25", "dense", "hybrid", "hybrid_rerank")
"""Bốn arm ablation. `hybrid_rerank` là arm mặc định của pipeline."""

RETRIEVE_TOP_K: Final[int] = 8
"""Số chunk cuối cùng đưa vào prompt sinh câu trả lời."""

RERANK_CANDIDATES: Final[int] = 30
"""Độ sâu ứng viên lấy từ RRF đưa vào reranker.

Đủ sâu để câu multi-hop có cơ hội gom cả hai văn bản vào cùng một danh sách,
đủ nông để cả danh sách nằm gọn trong một lời gọi listwise.
"""

RRF_K: Final[int] = 60
"""Hằng số làm mượt của Reciprocal Rank Fusion: score = Σ 1/(RRF_K + rank).

RRF hợp nhất theo HẠNG chứ không theo điểm, vì điểm BM25 (không chặn trên) và
cosine (trong [-1, 1]) không cùng thang đo — mọi cách chuẩn hoá về một thang đều
tuỳ tiện. 60 là giá trị gốc của bài báo, giữ nguyên vì corpus này quá nhỏ để
tinh chỉnh nó một cách có ý nghĩa thống kê.
"""

RERANK_MAX_SCORE: Final[int] = 3
"""Thang điểm NGUYÊN của reranker, chuẩn hoá về [0, 1] khi trả ra.

Thang nguyên hẹp ổn định hơn hẳn thang thực: hỏi LLM một số thực trong [0, 1]
thì cùng một ứng viên nhận 0.85 hay 0.9 tuỳ lượt, còn ranh giới "2 hay 3" thì
model giữ nhất quán hơn nhiều.
"""

RERANK_TEMPERATURE: Final[float] = 0.0
RERANK_MAX_TOKENS: Final[int] = 2048
RERANK_THINKING_BUDGET: Final[int] = 0

# --- Đánh giá ------------------------------------------------------------
LEAKAGE_JACCARD_MAX: Final[float] = 0.3
"""Ngưỡng Jaccard unigram âm tiết giữa câu hỏi và text gold_span.

Câu hỏi copy lại từ vựng của chính khoản nguồn làm BM25 thắng một cách giả tạo
và thổi phồng Recall của MỌI arm cùng lúc — bảng ablation mất luôn khả năng
phân biệt arm nào thực sự tốt hơn. Đo trên unigram chứ không phải bigram: bigram
sẽ phạt hai lần cùng một chỗ trùng (vừa tính "bảo", "hiểm" vừa tính "bảo_hiểm")
nên ngưỡng mất ý nghĩa.
"""

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
