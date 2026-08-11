"""Mọi hằng số điều khiển pipeline. Không hardcode tham số ở module khác."""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CORPUS_DIR: Final[Path] = ROOT / "data" / "corpus"
OUTPUTS_DIR: Final[Path] = ROOT / "outputs"
CACHE_DIR: Final[Path] = OUTPUTS_DIR / "cache"
INDEX_DIR: Final[Path] = OUTPUTS_DIR / "index"
EVAL_DIR: Final[Path] = OUTPUTS_DIR / "eval"
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
"""Ngưỡng coverage để coi một gold_span là đã truy xuất đủ căn cứ.

0.8 chứ không phải 1.0: chunk cuối cùng chạm vào span thường cắt ngang vài chữ ở
mép, và đòi phủ tuyệt đối sẽ phạt một retriever thực chất đã lấy đủ căn cứ để
trả lời. Cũng không hạ xuống 0.5, vì phủ nửa khoản trong văn bản QPPL rất dễ mất
đúng vế điều kiện hoặc vế ngoại lệ — tức là mất phần quyết định nghĩa của quy phạm.
"""

EVAL_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 8, 10, 20, 30)
"""Các mốc k báo cáo. Trần 30 = RERANK_CANDIDATES.

Báo cáo @k vượt quá RERANK_CANDIDATES là vô nghĩa với arm hybrid_rerank (nó chỉ
nhìn thấy 30 ứng viên), nên cả bốn arm cùng cắt ở 30 để so được với nhau.
"""

# --- LLM -----------------------------------------------------------------
EMBED_MODEL: Final[str] = "gemini-embedding-001"

GEN_MODEL: Final[str] = "gemini-3.1-flash-lite"
RERANK_MODEL: Final[str] = "gemini-3.1-flash-lite"
"""Thiết kế ban đầu: `gemini-2.5-flash` sinh câu trả lời, `gemini-2.5-flash-lite`
rerank — tầng mạnh cho việc khó, tầng rẻ cho việc gọi nhiều.

Hai lần buộc phải đổi, cả hai đều do ràng buộc bên ngoài chứ không do thiết kế:

1. Cả hai model 2.5 trả 404 "no longer available to new users" với API key tạo
   mới, nên chuyển sang thế hệ 3.
2. Free tier của `gemini-3.5-flash` chỉ cho 20 request/NGÀY — không đủ cho 20 câu
   hỏi nhân số lượt sinh lại. Nên bộ sinh cũng chạy `flash-lite`.

Hệ quả: phân tầng chi phí hiện xẹp xuống, hai vai dùng chung một model. Đây là
thứ ĐẦU TIÊN nên đảo lại nếu có tài khoản trả phí — chỉ cần sửa GEN_MODEL.

Không dùng gemini-3.6-flash vì nó từ chối thinking_budget = 0, tức không tắt được
phần suy nghĩ — mất cả tính tất định lẫn quyền kiểm soát quota. Không dùng các
bí danh `-latest` vì chúng trôi theo thời gian, mà cache key băm tên model: bí
danh đổi ngầm sẽ làm cache trỏ sai model mà không có dấu hiệu nào.
"""

JUDGE_PROVIDER: Final[str] = "google"
JUDGE_MODEL: Final[str] = "gemma-4-31b-it"
"""Judge phải KHÁC HỌ MODEL với bộ sinh: dùng Gemini chấm output của Gemini thì
có thiên lệch tự ưu ái.

Lựa chọn đầu tiên là `llama-3.3-70b-versatile` trên Groq — khác cả nhà cung cấp
lẫn họ model. Môi trường phát triển chặn `api.groq.com` ở tầng mạng, nên mặc định
chuyển sang Gemma: khác họ model và khác công thức huấn luyện so với Gemini, dù
cùng nhà cung cấp. Yếu hơn phương án Groq đúng ở điểm "cùng vendor", và đây là
đánh đổi có ý thức chứ không phải nhầm lẫn.

Quay lại Groq là sửa hai dòng:
    JUDGE_PROVIDER = "groq"
    JUDGE_MODEL = "llama-3.3-70b-versatile"

Gemma không hỗ trợ response_mime_type = "application/json" (trả rỗng), nên nó
được gọi ở chế độ text thường và `_parse_verdicts` bóc khối JSON ra khỏi fence.

Cố ý CHỈ có một hằng JUDGE_MODEL: giữ thêm một hằng "tên model Groq" nằm chờ sẽ
tạo ra hai nguồn sự thật, mà cái không được dùng thì không ai phát hiện khi nó sai.
"""

EMBED_DIM: Final[int] = 768
"""Cắt Matryoshka từ 3072 để cache commit vào repo nhỏ đi ~4 lần. Bắt buộc
chuẩn hoá L2 lại sau khi cắt vì chỉ vector 3072 chiều mới được chuẩn hoá sẵn."""

EMBED_BATCH_SIZE: Final[int] = 32

EMBED_TASK_DOCUMENT: Final[str] = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY: Final[str] = "RETRIEVAL_QUERY"
"""Embedding bất đối xứng: chunk và câu hỏi được nhúng bằng hai task_type khác
nhau, nên cùng một chuỗi ở hai vai trò cho hai vector — và hai cache key khác nhau."""
JUDGE_TEMPERATURE: Final[float] = 0.0

JUDGE_MAX_TOKENS: Final[int] = 8192
"""Rộng gấp bốn các trần khác vì Gemma KHÔNG cho tắt thinking.

`thinking_budget = 0` bị Gemma từ chối bằng 400, và token suy nghĩ tiêu chung
ngân sách với token trả lời. Ở 2048, câu hỏi khó (judge phải cân nhắc mệnh đề có
bỏ sót ngoại lệ hay không) đốt sạch ngân sách vào suy luận rồi trả về RỖNG với
`finish_reason = MAX_TOKENS`. Ở 8192 thì dừng bình thường.
"""

GEN_TEMPERATURE: Final[float] = 0.0
GEN_MAX_TOKENS: Final[int] = 4096
GEN_THINKING_BUDGET: Final[int] = 0
"""Tắt thinking cho bộ sinh: free tier tính quota theo cả token suy nghĩ, mà
nhiệm vụ ở đây là trích xuất có ràng buộc chứ không phải suy luận nhiều bước.
Bật lên là một tham số đáng thử nếu quota cho phép."""

# --- Sinh và kiểm chứng --------------------------------------------------
MAX_GENERATE_ATTEMPTS: Final[int] = 2
"""Sinh lại tối đa một lần. Lượt hai bắt buộc dùng prompt KHÁC lượt một."""

ABSTAIN_TEXT: Final[str] = "Không đủ căn cứ trong tài liệu."

TAU_RETRIEVE: Final[float] = 0.0
"""Ngưỡng gác TRƯỚC generate, so với điểm rerank cao nhất (đã chuẩn hoá [0, 1]).

Ý định: bắt câu hỏi nằm ngoài phạm vi corpus và chặn trước khi tốn lượt sinh nào.
Chỉ áp dụng được cho arm có rerank — điểm BM25 và cosine không cùng thang nên
không có ngưỡng chung nào đúng cho cả bốn arm.

GIÁ TRỊ 0.0 LÀ MỘT KẾT QUẢ RỖNG, KHÔNG PHẢI MỘT PHÁT HIỆN. Quét lưới trên gold
set cho ra bốn hàng giống hệt nhau: mọi giá trị TAU_RETRIEVE đều cho cùng F1,
cùng faithfulness, cùng số câu bị từ chối. Lý do là cả 4 câu unanswerable bị chặn
HAI LẦN độc lập — điểm rerank của chúng lần lượt là 0.0, 0.0, 0.33, 0.0, mà bản
thân bộ sinh cũng tự abstain đúng cả 4 câu. Cửa gác chưa bao giờ là mắt xích
quyết định, nên dữ liệu không nói được gì về nó.

Không có bằng chứng cho một giá trị khác 0 thì lấy 0. Hai điều cần biết trước khi
tin con số này: (a) nó dựa trên đúng 4 câu unanswerable; (b) cửa gác còn một lý do
mà metric này không đo được — nó tiết kiệm một lượt sinh cho mỗi câu truy xuất
hỏng, và trên free tier có hạn ngạch theo ngày thì đó không phải lợi ích nhỏ.
Hiệu chuẩn lại khi gold set có nhiều câu unanswerable hơn.
"""

TAU_GROUND: Final[float] = 1.0
"""Ngưỡng gác SAU generate, so với support_ratio từ LLM judge.

Tách khỏi TAU_RETRIEVE vì hai ngưỡng bắt hai loại lỗi khác nhau: ngưỡng trước
bắt "corpus không có câu trả lời", ngưỡng sau bắt "chunk trông hợp lý nhưng model
bịa thêm chi tiết". Gộp một ngưỡng thì không thể chỉnh riêng từng loại lỗi.

Khác TAU_RETRIEVE, ngưỡng này CÓ bằng chứng: tác dụng đơn điệu trên toàn lưới,
faithfulness sau verify đi 0.9597 -> 0.9806 -> 1.0000 khi siết 0.5 -> 0.75 -> 1.0,
và không gây một lần từ chối nhầm nào.

Giá phải trả cho 1.0 là 3 lượt sinh lại thay vì 0. Nếu quota là ràng buộc chặt
hơn groundedness thì 0.75 cho 0.9806 với chỉ 1 lượt sinh lại — cả hai con số nằm
trong outputs/eval/calibration.json để chọn lại có căn cứ.
"""

CALIBRATE_TAU_RETRIEVE_GRID: Final[tuple[float, ...]] = (0.0, 0.34, 0.67, 1.0)
"""Điểm rerank bị LƯỢNG TỬ HOÁ, nên lưới mịn hơn là vô nghĩa.

Thang nguyên 0-RERANK_MAX_SCORE chuẩn hoá về [0, 1] chỉ cho đúng 4 giá trị khả dĩ
(0, 1/3, 2/3, 1). Bốn ngưỡng ở đây rơi vào bốn khe giữa các giá trị đó, tức là bốn
hành vi phân biệt được — thêm điểm lưới nữa chỉ tạo ra các dòng trùng nhau và làm
bảng trông như đã dò kỹ hơn thực tế.
"""

CALIBRATE_TAU_GROUND_GRID: Final[tuple[float, ...]] = (
    0.0, 0.25, 0.34, 0.5, 0.67, 0.75, 1.0
)
"""support_ratio cũng lượng tử hoá, nhưng theo SỐ MỆNH ĐỀ của từng câu (1/n, 2/n…)
nên mẫu số đổi theo câu. Lưới này phủ các phân số thường gặp với 1-4 mệnh đề."""

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
