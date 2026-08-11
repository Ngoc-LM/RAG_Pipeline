# Pipeline RAG cho văn bản quy phạm pháp luật tiếng Việt

Trả lời câu hỏi pháp lý bằng tiếng Việt, **mỗi mệnh đề kèm trích dẫn tới đúng
Điều — Khoản** của văn bản gốc, và từ chối trả lời khi không đủ căn cứ.

Truy xuất lai (BM25 + embedding) → hợp nhất RRF → rerank listwise bằng LLM → sinh
câu trả lời có trích dẫn → hai tầng kiểm chứng quyết định chấp nhận / sinh lại /
từ chối.

Python thuần + numpy + rank_bm25 + google-genai + groq. Không LangChain, không
LlamaIndex, không FAISS — corpus cỡ này thì cosine brute-force bằng một phép nhân
ma trận numpy vừa đủ nhanh vừa giải thích được từng bước.

## Chạy thử trong 30 giây (không cần API key)

Toàn bộ cache lời gọi API đã được commit trong `outputs/cache/`, nên bảng kết quả
dưới đây dựng lại được offline:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py eval --arm all --offline    # bảng truy xuất 4 arm
.venv/bin/python -m pytest                          # 246 test, không chạm mạng
```

## Kết quả

Đo trên 16 câu answerable của gold set 20 câu (`eval/questions.json`).
`Rs` = recall **strict** (mọi `gold_span` của câu đều đạt ngưỡng coverage 0.8),
`Ra` = recall **any** (ít nhất một span đạt).

| arm | Rs@1 | Rs@5 | Rs@8 | Rs@30 | Ra@8 | cov@8 | MRR |
|---|---|---|---|---|---|---|---|
| `bm25` | 0.125 | 0.500 | 0.562 | 0.938 | 0.812 | 0.688 | 0.281 |
| `dense` | 0.500 | 0.812 | 0.938 | 1.000 | 0.938 | 0.938 | 0.646 |
| `hybrid` | 0.438 | 0.875 | 0.875 | 1.000 | 1.000 | 0.938 | 0.601 |
| **`hybrid_rerank`** | **0.625** | **0.875** | **1.000** | **1.000** | **1.000** | **1.000** | **0.751** |

Tách theo loại câu ở `Rs@1` — nơi đọc được arm nào giải quyết việc gì:

| loại | `bm25` | `dense` | `hybrid` | `hybrid_rerank` |
|---|---|---|---|---|
| factoid 1 bước (8 câu) | 0.250 | 0.750 | 0.625 | **0.875** |
| multi-hop (5 câu) | 0.000 | 0.000 | 0.000 | 0.000 |
| distractor (3 câu) | 0.000 | 0.667 | 0.667 | **1.000** |

Ba điều đáng nói, kể cả điều bất lợi:

- **Rerank thắng trọn bộ bẫy văn bản hết hiệu lực.** Ba câu `distractor` cố tình
  đặt một Nghị định đã hết hiệu lực cạnh văn bản thay thế nó. BM25 đưa văn bản sai
  lên hạng 1 cả ba lần; `hybrid_rerank` đưa văn bản đúng lên hạng 1 cả ba lần.
- **Multi-hop `Rs@8` đi từ 0.200 lên 1.000**, nhưng `Rs@1` vẫn là 0.000 ở **mọi**
  arm — và đó là điều hiển nhiên chứ không phải thất bại: câu multi-hop cần hai
  span ở hai văn bản khác nhau, một chunk duy nhất không thể phủ đủ.
- **`hybrid` KÉM HƠN `dense` đơn thuần ở `Rs@1`** (0.438 so với 0.500). RRF kéo
  theo cả lựa chọn hạng-1 tồi của BM25; chỉ tầng rerank mới gỡ lại được. Hợp nhất
  lai không miễn phí, và bảng này cho thấy chỗ nó phải trả giá.

Chi tiết `r*` từng span từng câu nằm trong `outputs/eval/retrieval.json`.

### Chất lượng câu trả lời (cả 20 câu, arm `hybrid_rerank`)

```
Quyết định abstain (lớp dương = đáng lẽ phải abstain)
  abstain đúng   : 4        từ chối nhầm : 0
  bỏ sót abstain : 0        trả lời đúng : 16
  P = 1.000   R = 1.000   F1 = 1.000

Groundedness
  support_ratio  : 0.9597 trước verify  ->  1.0000 sau verify
  sinh lại       : 3        Check A hỏng : 0
```

Cột "trước / sau verify" là chỗ đọc được tầng kiểm chứng đáng giá bao nhiêu. Ba
câu bị judge bác một phần, và cả ba đều được lượt sinh lại cứu:

| qid | loại | support_ratio lượt 1 → lượt 2 |
|---|---|---|
| `q07` | factoid | 0.667 → **1.000** |
| `q09` | multi-hop | 0.889 → **1.000** |
| `q16` | distractor | 0.800 → **1.000** |

Không câu nào phải abstain sau khi sinh lại, tức cơ chế "chèn danh sách mệnh đề
bị bác vào prompt lượt hai" hoạt động đúng ý đồ chứ không chỉ đốt thêm quota.

Năm câu trả lời đầy đủ kèm trích dẫn: **[`outputs/demo_answers.md`](outputs/demo_answers.md)**.

## Các lệnh khác

```bash
python run.py index --embed         # dựng chỉ mục, nhúng toàn bộ chunk
python run.py ask "Khách đòi xoá dữ liệu, bên tôi có bao lâu?"
python run.py answer                # cả gold set qua generate + verify
python run.py calibrate             # quét lưới hai ngưỡng abstain
python run.py all                   # bốn bước, tuần tự
```

Mọi lệnh nhận `--corpus`, `--questions`, `--out`, `--arm`, `--offline`. `run.py`
chỉ nối dây; toàn bộ logic nằm trong `src/` để không có đường chạy nào chỉ tồn tại
khi gọi qua CLI mà chưa từng được test gọi tới. Exit code: `2` = cache miss lúc
`--offline`, `3` = thiếu API key.

Các module cũng chạy trực tiếp được khi cần soi một tầng:

```bash
python -m src.chunk  --corpus data/corpus                    # thống kê chunk
python -m src.retrieve --corpus data/corpus --arm bm25 --qid q16
python -m tools.annotate --help
```

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Embedding | `gemini-embedding-001` | 768 chiều, cắt Matryoshka + chuẩn hoá L2 |
| Sinh câu trả lời | `gemini-3.1-flash-lite` | JSON có `response_schema` |
| Rerank | `gemini-3.1-flash-lite` | listwise, một lời gọi cho cả danh sách |
| LLM judge | `gemma-4-31b-it` | khác họ model với bộ sinh |

Judge cố ý **khác họ model** với bộ sinh: dùng chính Gemini để chấm câu trả lời
của Gemini thì sẽ có thiên lệch tự ưu ái.

### Ba lần buộc phải đổi model, và vì sao

Thiết kế ban đầu là `gemini-2.5-flash` sinh, `gemini-2.5-flash-lite` rerank,
`llama-3.3-70b-versatile` trên Groq làm judge. Cả ba đều phải đổi vì ràng buộc
bên ngoài, không phải vì thiết kế sai:

1. **Cả hai model 2.5 trả `404 no longer available to new users`** với API key tạo
   mới → chuyển sang thế hệ 3.
2. **`api.groq.com` bị network policy của môi trường phát triển chặn** → judge
   chuyển sang Gemma. Gemma khác họ model và khác công thức huấn luyện so với
   Gemini, nhưng **cùng nhà cung cấp** — yếu hơn phương án Groq đúng ở điểm đó.
   Quay lại Groq chỉ cần sửa hai dòng trong `src/config.py`.
3. **`gemini-3.5-flash` free tier chỉ cho 20 request/NGÀY**, không đủ cho 20 câu
   hỏi nhân số lượt sinh lại → bộ sinh cũng dùng `flash-lite`. Phân tầng chi phí
   (tầng mạnh cho việc khó, tầng rẻ cho việc gọi nhiều) vì vậy hiện đang xẹp
   xuống; đây là thứ đầu tiên nên đảo lại nếu có tài khoản trả phí.

Không dùng bí danh `-latest` vì chúng trôi theo thời gian mà cache key băm tên
model — bí danh đổi ngầm sẽ làm cache trỏ sai model mà không có dấu hiệu nào.
Không dùng `gemini-3.6-flash` vì nó từ chối `thinking_budget = 0`.

### Bốn lỗi chỉ lộ ra khi chạy thật

Không test nào bắt được những lỗi này, vì chúng nằm ở hành vi của nhà cung cấp:

- **`response_mime_type="application/json"` KHÔNG đủ để có JSON hợp lệ.** Model
  trả JSON thiếu dấu `}` đóng trong khi `finish_reason` vẫn báo `STOP`, tất định
  ở `temperature = 0`. Phải khai báo `response_schema` mới ràng buộc được cấu
  trúc ở tầng giải mã.
- **Gemma không cho tắt thinking** (`thinking_budget = 0` → 400), và token suy
  nghĩ tiêu chung ngân sách với token trả lời. Ở `max_output_tokens = 2048` nó
  đốt sạch ngân sách vào suy luận rồi trả **rỗng** với `finish_reason = MAX_TOKENS`.
  Vì vậy `JUDGE_MAX_TOKENS = 8192`.
- **Hạn ngạch theo NGÀY thì backoff không cứu được.** `RETRY_MAX_ATTEMPTS = 6` xử
  lý tốt 429 do vượt RPM, nhưng vô dụng với `GenerateRequestsPerDayPerProjectPerModel`.
- **404 không được thử lại.** `RETRYABLE_STATUS` không chứa 404, nên khi gặp model
  đã bị khoá thì pipeline dừng ngay thay vì lùi sáu lần vô ích — phân loại lỗi
  theo status code thay vì theo chuỗi trong message đã trả công đúng chỗ này.

`check_finish()` trong `src/llm.py` tách riêng "chạm trần token" khỏi "rỗng vì lý
do khác": gộp hai ca này vào một thông điệp là tự làm khó mình, vì cách sửa hoàn
toàn khác nhau (nới trần so với đổi prompt).

## Cấu trúc

| Thành phần | File |
|---|---|
| Hằng số tập trung | `src/config.py` |
| Parse frontmatter + Chương/Mục/Điều/Khoản | `src/ingest.py` |
| Chia chunk theo khoản | `src/chunk.py` |
| Số học khoảng cho coverage | `src/intervals.py` |
| Tokenizer tiếng Việt cho BM25 | `src/tokenize_vi.py` |
| Cache + backoff cho API | `src/llm.py` |
| Chỉ mục BM25 + embedding | `src/index.py` |
| Bốn arm truy xuất | `src/retrieve.py` |
| Sinh câu trả lời có trích dẫn | `src/generate.py` |
| Kiểm chứng hai tầng | `src/verify.py` |
| Đo truy xuất + chất lượng câu trả lời | `src/evaluate.py` |
| Hiệu chuẩn hai ngưỡng abstain | `src/calibrate.py` |
| CLI đầu-cuối | `run.py` |
| CLI gán nhãn gold_span | `tools/annotate.py` |
| Chuẩn bị corpus từ .doc/.docx/.pdf | `tools/normalize_raw.py` + `notebooks/` |

## Cài đặt

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # điền GEMINI_API_KEY và GROQ_API_KEY
```

`.env` nằm trong `.gitignore`. Không commit key.

**Không cần key** cho: toàn bộ `pytest`, `src.chunk`, `tools.annotate`, và mọi lệnh
chạy với `--offline` — cache đã commit phủ đủ cả bốn arm truy xuất. Chỉ cần key khi
muốn hỏi một câu **mới** chưa có trong cache.

## Chuẩn bị corpus

Mỗi văn bản là một file **`.txt`** trong `data/corpus/` (`load_corpus` chỉ đọc
`.txt`), mở đầu bằng khối frontmatter phẳng dạng `key: value`:

```
---
doc_id: luat_91_2025
title: "Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15"
doc_type: luat
issued_date: "2025-06-26"
effective_from: "2026-01-01"
effective_to: null
status: active
---

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
...
```

Sáu key **bắt buộc**: `doc_id`, `title`, `doc_type`, `issued_date`,
`effective_from`, `status`. `status` bị kiểm và chỉ nhận `active` hoặc `expired`.
`effective_to` để `null` nghĩa là còn hiệu lực. `doc_id` phải duy nhất trong corpus.

Chỉ hỗ trợ `key: value` phẳng, không lồng nhau — để không phải kéo thêm PyYAML
chỉ vì bảy dòng metadata.

Bộ parse nhận diện `Chương`, `Điều`, khoản đánh số và điểm `a)` `b)` **chỉ khi
chúng ở đầu dòng**. Đây là điểm mấu chốt: trong văn bản QPPL, "Điều", "Chương",
"khoản" xuất hiện dày đặc bên trong nội dung dưới dạng tham chiếu chéo ("quy định
tại khoản 2 Điều 9"), nên bất kỳ cách nhận dạng nào không neo vào đầu dòng đều cắt
nhầm giữa câu. Thêm hai chốt nữa: số khoản phải tăng liên tiếp từ 1 (nhảy cóc thì
coi là nội dung thường và ghi cảnh báo), và điểm `a)` `b)` thuộc về khoản đang mở
chứ không tách thành đơn vị riêng.

## Trục ký tự — đọc trước khi soạn gold set

`gold_spans` trỏ tới **offset ký tự trong `Document.body`**, không phải trong file
gốc. `body` bắt đầu **ngay sau** dòng `---` đóng, đã đổi CRLF/CR thành LF và chuẩn
hoá NFC, và **không bị strip**. Trong fixture `qc_99_2099`, `body[0]` là một ký tự
xuống dòng, khối `Chương I` bắt đầu ở offset 1, dòng `Điều 1.` ở offset 26, và
đoạn văn của Điều 1 ở offset 53.

Đừng đếm tay. Dùng `tools/annotate.py`:

```bash
python -m tools.annotate --corpus data/corpus --grep "thời hạn phản hồi"
python -m tools.annotate --corpus data/corpus --article luat_91_2025:9
python -m tools.annotate --corpus data/corpus --grep "phản hồi" --emit q07
python -m tools.annotate --corpus data/corpus --validate eval/questions.json
python -m tools.annotate --corpus data/corpus --leakage eval/questions.json
```

- `--grep` tìm chuỗi (không phân biệt hoa thường, thêm `--no-accent` để bỏ dấu ở
  cả hai phía) và trả về offset của **khoản** chứa nó, kèm nhãn trích dẫn đầy đủ
  và một đoạn 300 ký tự có highlight. Khớp rơi vào tiêu đề Điều hoặc khe giữa các
  khoản cũng được báo — đó là ca dễ gán nhầm nhất.
- `--article doc_id:9` in nguyên Điều 9 kèm offset từng khoản.
- `--emit qid` in một question object dán thẳng vào mảng trong `eval/questions.json`.
- `--validate` kiểm `0 <= char_start < char_end <= len(body)`, in 120 ký tự đầu của
  mỗi span để xác nhận bằng mắt, và báo lỗi nếu span không nằm trọn trong một Điều.
  Trả exit code khác 0 khi có lỗi, cắm vào CI được.
- `--leakage` đo Jaccard trên unigram âm tiết giữa câu hỏi và text của `gold_span`,
  gắn cờ câu vượt `LEAKAGE_JACCARD_MAX` và trả exit code khác 0. Câu multi-span đo
  trên hợp text của các span; câu `unanswerable_*` không tính.

## Gold set

`eval/questions.json` là một mảng:

```json
{
  "qid": "q01",
  "question": "...",
  "type": "factoid_1hop | multihop | distractor | unanswerable_oos | unanswerable_nearmiss",
  "answerable": true,
  "gold_spans": [{ "doc_id": "luat_91_2025", "char_start": 1240, "char_end": 1533 }],
  "gold_answer": "câu trả lời ngắn 1-2 câu"
}
```

Câu `unanswerable_*` có `answerable: false` và `gold_spans: []`; chúng không tham
gia metric truy xuất, chỉ dùng để đo chất lượng abstain.

20 câu, 21 gold span. Mọi offset đều lấy từ `tools/annotate.py` chạy thật trên
`data/corpus/`, không đếm tay.

| loại | số câu | dùng để đo |
|---|---|---|
| `factoid_1hop` | 8 | truy xuất một khoản duy nhất |
| `multihop` | 5 | cột **strict**: cả hai span phải đạt ngưỡng coverage |
| `distractor` | 3 | rerank có đọc `status` không |
| `unanswerable_oos` | 2 | abstain khi chủ đề nằm ngoài corpus |
| `unanswerable_nearmiss` | 2 | abstain khi corpus bàn đúng chủ đề nhưng không có con số |

Bốn văn bản đóng bốn vai khác nhau, và tỉ lệ gold span bám theo tỉ lệ độ dài để
Recall tổng không bị một văn bản chi phối:

| văn bản | số từ | tỉ lệ corpus | gold span | tỉ lệ span |
|---|---|---|---|---|
| `luat_91_2025` | 9.307 | 52% | 11 | 52% |
| `nd_356_2025` | 6.084 | 34% | 7 | 33% |
| `luat_attt_86_2015` | 2.512 | 14% | 3 | 14% |
| `nd_13_2023` (`expired`) | 6.985 | — | **0** | chỉ đóng vai mồi nhử |

### Ba cặp mồi nhử

`nd_13_2023` đã hết hiệu lực từ 01/01/2026, bị thay bằng `nd_356_2025`, và cố ý
được giữ trong corpus vì nó bàn **đúng** những chủ đề mà hai văn bản còn hiệu lực
bàn — chỉ khác đáp án. Ba câu `distractor` khai thác đúng chỗ đó:

| qid | span đúng | mồi nhử | vì sao khó |
|---|---|---|---|
| `q14` | `luat_91` Điều 23 Khoản 1 | `nd_13` Điều 23 Khoản 1 | chỉ văn bản hết hiệu lực nêu đích danh "Bộ Công an (Cục A05)" và "Mẫu số 03" — trả lời sai thì lộ ngay ở phần trích dẫn |
| `q15` | `nd_356` Điều 5 Khoản 4 | `nd_13` Điều 16 Khoản 5 | xoá dữ liệu trong **72 giờ** (cũ) so với **02 ngày làm việc + 20/30 ngày** (mới). Không có khe hở từ vựng nào giữa hai bên, nên câu này cô lập đúng một câu hỏi: retriever có phân biệt hiệu lực không |
| `q16` | `luat_91` Điều 28 Khoản 1 | `nd_13` Điều 21 Khoản 1 | từ "tiếp thị" **chỉ** xuất hiện trong văn bản hết hiệu lực nên BM25 bị kéo về mồi nhử, và hai văn bản cho đáp án **ngược nhau** |

Cặp mồi nhử không nằm trong `eval/questions.json`: schema đánh giá chỉ biết tới
span đúng, còn "chunk nào là mồi nhử" là thông tin phân tích lỗi, không phải nhãn.

### Chống rò rỉ từ vựng

Câu hỏi chép lại từ vựng của chính khoản nguồn làm BM25 thắng một cách giả tạo và
thổi phồng Recall của **mọi** arm cùng lúc — bảng ablation mất luôn khả năng phân
biệt arm nào thực sự tốt hơn. Vì vậy câu hỏi được viết theo lời người dùng thật
("bên tôi", "khách gửi yêu cầu", "đứa trẻ 10 tuổi") thay vì thuật ngữ của văn bản.

`--leakage` trên bộ hiện tại: **0/16 câu vượt ngưỡng**, cao nhất 0.211. Ngưỡng đo
trên unigram âm tiết chứ không phải bigram — bigram phạt hai lần cùng một chỗ trùng
(vừa tính `bảo`, `hiểm` vừa tính `bảo_hiểm`) nên ngưỡng mất ý nghĩa.

Một hệ quả đo được: khoản càng ngắn thì mẫu số Jaccard càng nhỏ và điểm càng dễ
vọt. `q08` trỏ tới một khoản chỉ 136 ký tự và bản nháp đầu đạt 0.303; diễn đạt lại
bằng từ đời thường ("hạ tầng phân giải địa chỉ web" thay cho "hệ thống máy chủ tên
miền") kéo xuống 0.122 mà không đổi đáp án.

## Chia chunk: đơn vị là KHOẢN

Ranh giới ngữ nghĩa của văn bản QPPL là khoản: một khoản là một quy phạm trọn vẹn,
đọc tách ra vẫn đúng nghĩa. Cửa sổ trượt cố định cắt ngang giữa khoản làm hỏng cả
hai đầu — retrieval mất tín hiệu vì điều kiện và hệ quả nằm ở hai chunk khác nhau,
còn trích dẫn thì mất tính hợp lệ vì không còn trỏ tới một đơn vị mà người đọc tra
ngược lại được trong văn bản gốc.

Hai pha tách rời, để quy tắc cắt và quy tắc gộp không thể hợp thành nhau:

1. **Cắt** — khoản có `n_tokens > CHUNK_MAX_TOKENS` (600) bị cắt tại ranh giới câu
   với chồng lấn `CHUNK_OVERLAP_RATIO` (0.15), và bị đánh dấu **không tham gia gộp**.
2. **Gộp** — chỉ gộp các khoản chưa bị cắt, chỉ khi tổng vẫn dưới trần, và không
   bao giờ gộp xuyên qua một khoản đã bị cắt. Khoản dưới `CHUNK_MIN_TOKENS` (120)
   nằm cạnh một khoản dài sẽ đứng riêng — đánh đổi đã chấp nhận.

Bất biến: mỗi chunk **hoặc** là một part của đúng một khoản, **hoặc** là hợp của
các khoản nguyên vẹn. Không bao giờ vừa gộp vừa cắt.

Tách câu không cắt tại dấu chấm nằm trong viết tắt (`TP.`, `NĐ-CP`), số thứ tự,
hay chữ cái đầu viết hoa đơn lẻ.

Ước lượng token bằng `len(text) / CHARS_PER_TOKEN` (3.5). Với corpus vài trăm chunk,
sai số vài phần trăm của phép ước lượng này không đổi được quyết định nào — nó chỉ
dùng để so với trần cắt và sàn gộp. Kéo một tokenizer thật về đổi lấy thêm
dependency, thêm thời gian nạp và thêm một thứ phải giữ đồng bộ với model.

### Tiêu đề Điều và Chương nằm trong chunk đầu tiên

Tiêu đề `Điều N. ...` và khối `Chương X` không thuộc khoản nào. Nếu để nguyên như
vậy, chúng không thuộc chunk nào, và hợp các chunk chỉ phủ 85-95% body. Hậu quả cụ
thể: một `gold_span` gồm tiêu đề Điều cộng khoản 1 chỉ đạt `Cov ≈ 0.69` **kể cả khi
lấy toàn bộ chunk của document** — tức `Recall@k = 0` vĩnh viễn ở mọi arm, mà không
có gì báo lỗi.

Nên chunk **đầu tiên** của mỗi Điều có `char_start` lùi về đầu dòng tiêu đề Điều;
nếu Điều đó mở đầu một Chương thì lùi tiếp về đầu khối tiêu đề Chương.
`verify_coverage()` chạy trong `chunk_document()` và **raise** nếu phần body không
được phủ có chứa ký tự khác khoảng trắng.

Hệ quả phụ: chunk đầu của một Điều có thể vượt `CHUNK_MAX_TOKENS` đúng bằng độ dài
khối tiêu đề. Trần áp cho nội dung khoản, không áp cho phần tiêu đề cõng thêm.

`indexed_text` (chuỗi đem đi đánh chỉ mục) gắn cả nhãn trích dẫn lẫn tiêu đề Điều
vào đầu mỗi chunk, vì từ chunk thứ hai của một Điều trở đi thì tiêu đề không còn
nằm trong `text` — mà tiêu đề Điều là tín hiệu truy xuất mạnh nhất của văn bản QPPL.

### Nhãn trích dẫn

`citation_label` là chuỗi hiển thị cho người đọc, tách hẳn khỏi `chunk_id`:

| Trường hợp | Nhãn |
|---|---|
| Khoản đơn | `Điều 4 Khoản 1 <title>` |
| Khoản gộp | `Điều 5 Khoản 1-4 <title>` |
| Điều không chia khoản | `Điều 1 <title>` |
| Đoạn mở đầu của Điều có chia khoản | `Điều 3 (đoạn mở đầu) <title>` |

Đoạn mở đầu phải được gọi tên riêng: ghi `Điều 3 <title>` cho nó thì trông như
đang trích dẫn cả Điều, trong khi thực tế chỉ trích phần chapeau. Vì lý do đó đoạn
mở đầu cũng không bao giờ gộp với khoản 1 — gộp sẽ tạo ra `clause_range` `"0-1"`.

## Đo truy xuất bằng coverage, không bằng "chunk gold"

Repo này **không** có khái niệm "chunk nào là chunk gold". Gán nhãn ở mức chunk
buộc phải chọn một ngưỡng overlap, mà mọi ngưỡng như vậy đều thiên lệch theo kích
thước chunk: lấy `overlap >= 0.5 · |span|` thì chunk nhỏ không bao giờ đạt; đổi sang
`0.5 · min(|span|, |chunk|)` thì chunk 50 ký tự chỉ cần phủ 25 ký tự của một span
1000 ký tự cũng thành gold. Cả hai đều làm hỏng bảng ablation theo kích thước chunk.

Thay vào đó, đo ở mức **tập kết quả**:

```
Cov@k(span) = |span ∩ union(các chunk trong top-k)| / |span|
hit@k(span) ⟺ Cov@k(span) >= THETA_COVERAGE          (mặc định 0.8)
r*(span)    = min{k : Cov@k(span) >= THETA}
MRR         = 1/r*                                    (không tồn tại -> 0)
```

MRR ở đây đọc là "đến hạng thứ mấy thì retriever gom đủ căn cứ để trả lời".

Phần giao tính trên **hợp các khoảng `[char_start, char_end)`** rồi mới cắt với
span — không cộng độ dài text của từng chunk, vì chunk chồng lấn sẽ bị đếm hai lần
và coverage vọt quá 1. Có test cho đúng ca đó: hai chunk `[0,70)` và `[30,100)`
phủ span `[0,100)` cho `cov = 1.0`, trong khi cộng độ dài text sẽ ra 1.4.

Hợp khoảng chỉ gom chunk **cùng `doc_id`** với span. Trục ký tự là trục riêng của
từng văn bản, nên chunk `[0, 100)` của văn bản khác chồng lên span về mặt số học
mà không phủ một chữ nào của nó.

Tính chất then chốt — **bất biến với `chunk_size` theo đúng định nghĩa**: chia một
chunk thành hai chunk kề nhau không đổi hợp, nên không đổi coverage. Đó là thứ mà
nhãn "chunk gold" không thể có, và `test_evaluate.py::test_coverage_bat_bien_voi_chunk_size`
canh đúng tính chất đó.

Câu multi-hop báo cáo hai cột: **strict** (mọi `gold_span` đều đạt ngưỡng) và
**any** (ít nhất một span đạt). Strict là cột chính, và `r*_strict` là hạng của
span **chậm nhất** — hạng mà tại đó retriever đã gom đủ căn cứ cho cả câu.

Không báo cáo nDCG: với relevance dạng coverage thì IDCG phải dựng từ một lời giải
phủ tối ưu, và con số đó nói về bộ giải set-cover nhiều hơn là về retriever. Thay
bằng `mean_cov@k`, vốn đã là metric có thứ bậc và không cần chuẩn hoá tuỳ tiện.
`mean_cov` lấy trung bình **trong một câu trước** rồi mới trung bình qua các câu,
để câu 2 span không mang trọng số gấp đôi câu 1 span.

Bốn câu `unanswerable` không tham gia metric truy xuất — chúng đo chất lượng
abstain, mà abstain thì thuộc tầng generate.

### Cột strict đã bắt được đúng thứ nó sinh ra để bắt

Con số thuyết phục nhất cho việc tách hai cột nằm ở nhóm multi-hop của arm `bm25`:

```
multihop (5 câu)   Ra@8 = 1.000     Rs@8 = 0.200
```

BM25 gần như **luôn** tìm được *một* trong hai span, và gần như **không bao giờ**
tìm đủ *cả hai*. Một bảng chỉ có cột "any" sẽ báo 100% và giấu sạch vấn đề. Sau
khi thêm dense + rerank, `Rs@8` của nhóm này lên 1.000 — tức tầng rerank giải
quyết đúng chỗ mà cột strict chỉ ra.

`Rs@1` của nhóm multi-hop bằng **0.000 ở cả bốn arm**, và đó là giới hạn cấu trúc
chứ không phải thất bại của retriever: câu multi-hop cần hai span ở hai văn bản
khác nhau, mà top-1 chỉ có một chunk. Chỉ số này không có cách nào khác 0.

`outputs/eval/retrieval.json` giữ `r*` của từng span từng câu, nên truy được ngay
câu nào hỏng ở đâu. Ví dụ `q10` với arm `bm25` có `r*` từng span là `[None, 2]`:
span thứ hai lên hạng 2, span thứ nhất không xuất hiện trong 30 ứng viên đầu. Cùng
câu đó với `hybrid_rerank` là `[8, 1]` — cả hai span đều vào được top-8.

## Bốn arm ablation

| arm | mô tả | cần gọi API lần đầu |
|---|---|---|
| `bm25` | chỉ từ khoá | không bao giờ |
| `dense` | chỉ embedding | embedding |
| `hybrid` | RRF hợp nhất hai danh sách trên | embedding |
| `hybrid_rerank` | hybrid rồi rerank listwise — arm mặc định của pipeline | embedding + rerank |

Cache của cả bốn arm đã được commit, nên `--offline` chạy lại được toàn bộ bảng
mà không cần key nào.

Bốn arm dùng chung **một** chỉ mục và **một** hàm vào (`retrieve.retrieve`), khác
nhau đúng ở cách xếp hạng. Nếu mỗi arm có đường dữ liệu riêng thì bảng ablation
so nhầm hai đường dữ liệu chứ không so hai cơ chế xếp hạng.

RRF hợp nhất theo **hạng** chứ không theo điểm, vì điểm BM25 không chặn trên còn
cosine nằm trong `[-1, 1]`; mọi cách chuẩn hoá hai thang đó về một đều tuỳ tiện.
`score(d) = Σ 1/(RRF_K + rank(d))`, hạng tính từ 1, chunk vắng mặt trong một danh
sách thì không cộng gì từ danh sách đó — không cần điểm phạt, vì độ sâu cắt đã là
hình phạt rồi.

Phá hoà điểm bằng chỉ số chunk ở mọi nơi. BM25 trả 0.0 cho rất nhiều chunk cùng
lúc, mà một thứ tự không tất định ở đó làm mọi con số đánh giá nhảy giữa hai lần
chạy trên cùng dữ liệu.

Bảng tách theo `type` là chỗ đáng nhìn nhất: câu `distractor` (nhiều văn bản cùng
bàn một chủ đề, chỉ một chứa đáp án) là nơi rerank phải chứng minh nó đáng tiền.
Ví dụ đo được, `q16` với arm `bm25` trên corpus thật:

```
1. 63.66  Điều 21 Khoản 1-3 Nghị định 13/2023/NĐ-CP   [expired]   ← mồi nhử
2. 38.87  Điều 28 Khoản 5-7 Luật 91/2025/QH15         [active]
3. 32.29  Điều 28 Khoản 1-2 Luật 91/2025/QH15         [active]    ← span đúng
```

Từ khoá "tiếp thị" chỉ có trong văn bản đã hết hiệu lực, nên BM25 đẩy nó lên hạng
1 với điểm gần gấp đôi. Đây chính là khoảng cách mà rerank phải bù.

### Rerank listwise

Chấm cả danh sách trong **đúng một** lời gọi LLM. Pointwise mỗi ứng viên một call
sẽ tốn `RERANK_CANDIDATES` lần quota cho mỗi câu hỏi; free tier giới hạn RPM thấp
nên listwise là bắt buộc chứ không chỉ là tối ưu. Đánh đổi đã biết: model nhìn
thấy cả danh sách nên điểm không độc lập hoàn toàn giữa các ứng viên.

Thang điểm là **số nguyên 0-3**, chuẩn hoá về `[0, 1]` khi trả ra. Hỏi LLM một số
thực trong `[0, 1]` thì cùng một ứng viên nhận 0.85 hay 0.9 tuỳ lượt; ranh giới
"2 hay 3" thì model giữ nhất quán hơn nhiều. Điểm đã chuẩn hoá này cũng chính là
đại lượng mà ngưỡng abstain trước generate sẽ so.

Ứng viên được đánh số thứ tự `[1]`..`[n]` trong prompt chứ không dùng `chunk_id`:
id ngắn thì output ngắn hơn và model ít bịa hơn, còn ánh xạ ngược về `chunk_id`
làm ở phía Python nên không mất mát gì.

Ba quy tắc phòng thủ khi đọc output, tất cả đều có test:

- Ordinal ngoài `[1, n]` bị **bỏ**, không ánh xạ đại sang một chunk nào đó.
- Ứng viên không được chấm nhận **0**. Im lặng không phải bằng chứng ủng hộ, và
  cách tính này không thưởng cho reranker trả thiếu để rút ngắn output.
- Không chấm được ứng viên nào, hoặc JSON hỏng, thì **raise**. Âm thầm rơi về
  thứ tự RRF sẽ biến một tầng chết thành một tầng trông như đang chạy.

Hoà điểm rerank thì **giữ thứ tự RRF**. Thang 0-3 hoà rất nhiều, và rơi về một
thứ tự tuỳ ý ở đó là vứt bỏ toàn bộ tín hiệu của tầng hợp nhất bên dưới.

Prompt rerank có nêu trạng thái hiệu lực của từng ứng viên kèm quy tắc "văn bản
`expired` không còn là căn cứ hợp lệ khi đã có ứng viên `active` cùng nội dung".
Nói thẳng: điều đó khiến ba câu `distractor` đo **cả** hiệu lực của câu lệnh này
chứ không chỉ đo năng lực nội tại của model. Đó là một lựa chọn thiết kế có ý
thức — lọc cứng theo `status` ở tầng chỉ mục thì rẻ hơn nhưng sai, vì văn bản hết
hiệu lực vẫn cần truy xuất được khi câu hỏi hỏi về giai đoạn trước.

### Chỉ mục và phép nhúng nạp trễ

BM25 và embedding dùng chung đúng một danh sách chunk và đúng một chuỗi đầu vào
(`Chunk.indexed_text`), nên chỉ số hàng là danh tính chung: RRF chỉ việc hợp nhất
hai danh sách chỉ số, không cần ánh xạ id qua lại.

Phía embedding **nạp trễ**: `build_index()` không chạm mạng, vector chỉ được nhúng
ở lần gọi `dense_scores()` đầu tiên. Arm `bm25` không cần vector nào, nên bắt nó
chờ 160 lời gọi embedding chỉ để chạy được là sai; lười hoá cũng khiến `--offline`
với cache rỗng vẫn chạy được arm thuần từ khoá thay vì chết ngay ở bước dựng chỉ
mục.

Cosine tính bằng một phép nhân ma trận `(160, 768) × (768,)`. Cả hai phía đã chuẩn
hoá L2 nên tích vô hướng **chính là** cosine — không cần chỉ mục xấp xỉ, và không
có cấu trúc dữ liệu nào phải giải thích thêm.

`outputs/index/chunks.json` là ảnh chụp danh sách chunk kèm tham số chunk đã dùng.
Vector **không** nằm ở đó: chúng đã có trong `outputs/cache/embed/` theo từng text.
Chép sang nơi thứ hai chỉ tạo ra hai nguồn sự thật lệch pha nhau khi đổi tham số.

### Tokenizer tiếng Việt cho BM25

Tiếng Việt tách âm tiết bằng khoảng trắng, nên tách theo whitespace chỉ cho ra âm
tiết rời — "bảo", "hiểm", "xã", "hội" khớp lung tung với mọi văn bản có chữ "xã".
Bù bằng **bigram âm tiết liền kề** (`bảo_hiểm`, `xã_hội`) thay vì kéo `pyvi` hay
`underthesea` về (chúng kéo theo scikit-learn). Chữ số được giữ và tách riêng vì
"Điều 60", "12 tháng" là tín hiệu mạnh; bigram `điều_60` còn mạnh hơn.

## Sinh câu trả lời và kiểm chứng

Bộ sinh trả về **từng mệnh đề kèm đúng tập citation của riêng mệnh đề đó**, không
trả một khối text rồi tách claim ở bước sau — tách câu tiếng Việt bằng heuristic
sẽ vỡ ở "Điều 1.", "khoản 2.", "0,5%", mà claim tách sai thì mọi con số
groundedness đều sai theo, một cách âm thầm.

Trích dẫn dùng **số hiệu `[1]`..`[n]`** trỏ vào danh sách trích đoạn trong prompt,
không dùng `chunk_id`. Output ngắn hơn, model ít bịa hơn, và Check A trở thành một
phép so sánh khoảng số chính xác thay vì một phép khớp chuỗi.

Hai tầng kiểm chứng, chi phí khác hẳn nhau:

- **Check A — cú pháp, miễn phí.** Mọi citation phải trỏ tới trích đoạn thực sự có
  trong prompt, và mọi claim phải có ít nhất một citation. Fail thì sinh lại ngay,
  **không tốn một lời gọi judge nào** — `check_b` để `None` chứ không phải 0.0, để
  "chưa chấm" không bị đọc nhầm thành "chấm và trượt".
- **Check B — ngữ nghĩa, đúng một lời gọi Groq.** Toàn bộ claim đi trong một
  request; judge trả `[{claim_id, supported, reason}]`. `support_ratio` = số claim
  được hỗ trợ / tổng số claim. Claim mà judge không nhắc tới bị tính là **không**
  được hỗ trợ — im lặng không phải bằng chứng ủng hộ, và cách tính này không
  thưởng cho judge trả thiếu.

Judge được dặn rõ là đang đo **tính có căn cứ, không đo tính đúng**: một mệnh đề
đúng trên thực tế nhưng nói nhiều hơn trích đoạn được viện dẫn vẫn phải bị bác.
Không tách bạch chỗ này thì judge sẽ chấm bằng kiến thức nền của chính nó và
`support_ratio` mất hết ý nghĩa.

Bộ parse của generate **cố ý không lọc trước** mệnh đề thiếu citation hay citation
ngoài phạm vi. Đó đúng là việc của Check A; lọc sớm sẽ giấu mất lỗi mà tầng kiểm
chứng sinh ra để bắt và làm mọi con số về Check A đẹp một cách giả tạo.

### Sinh lại: prompt và cache key đều phải khác

Sinh lại tối đa một lần, và lượt hai **bắt buộc khác** lượt một ở hai chỗ:

1. **Prompt** — chèn danh sách claim đã bị bác kèm lý do và yêu cầu sửa hoặc bỏ
   hẳn. Ở `temperature = 0`, prompt y hệt cho output y hệt.
2. **Cache key** — `attempt` và `feedback` nằm trong `input_obj`. Thiếu hai trường
   này thì lượt sinh lại chỉ đọc lại đúng câu trả lời vừa bị bác, và vòng lặp
   "sinh lại" trở thành một vòng lặp tốn quota để nhận lại y nguyên.

`test_generate_verify.py::test_luot_sinh_lai_dung_cache_key_khac` canh điểm 2 bằng
cách gieo cache **chỉ cho lượt 1**: nếu lượt 2 dùng chung key thì nó sẽ đọc lại
được và không raise, nên `CacheMiss` ở đó chính là bằng chứng hai lượt là hai lời
gọi khác nhau.

Vẫn fail sau hai lượt thì trả `"Không đủ căn cứ trong tài liệu."`, kèm
`abstain_stage` ghi rõ chết ở đâu: `retrieve`, `model`, `check_a` hay `check_b`.

`support_ratio_first` và `support_ratio_final` được báo cáo tách nhau, để đọc được
tầng verify thực sự thêm bao nhiêu giá trị chứ không chỉ thêm bao nhiêu chi phí.

### Hai ngưỡng abstain

Tách đôi vì chúng bắt hai loại lỗi khác nhau: `TAU_RETRIEVE` gác **trước** generate
(so với điểm rerank cao nhất) bắt câu hỏi ngoài phạm vi corpus và chặn trước khi
tốn lượt sinh nào; `TAU_GROUND` gác **sau** generate (so với `support_ratio`) bắt
trường hợp trích đoạn trông hợp lý nhưng model bịa thêm chi tiết. Gộp một ngưỡng
thì không thể chỉnh riêng từng loại lỗi.

Ngưỡng trước **chỉ áp dụng cho arm có rerank**. Điểm BM25 không chặn trên còn
cosine nằm trong `[-1, 1]`, nên cùng một con số ngưỡng mang ý nghĩa khác nhau ở mỗi
arm — thà không gác còn hơn gác bằng một đại lượng không so sánh được. Ba arm còn
lại đi thẳng vào generate và chỉ chịu ngưỡng sau.

Giá trị hiện tại (`TAU_RETRIEVE = 0.0`, `TAU_GROUND = 1.0`) đến từ `run.py
calibrate` chạy thật, không phải đặt tay — xem phần quét lưới ngay dưới đây, kể
cả phần kết quả rỗng của `TAU_RETRIEVE`.

### Quét lưới ngưỡng

Lớp dương là **"đáng lẽ phải abstain"**, tức 4 câu `unanswerable`. Hai loại lỗi
được tách riêng chứ không gộp vào một con số:

| | ý nghĩa |
|---|---|
| `wrong_abstain` | từ chối một câu trả lời được — phiền, nhưng an toàn |
| `missed_abstain` | trả lời một câu không có căn cứ — đúng dạng lỗi pipeline này sinh ra để chặn |

Hệ thống không bao giờ abstain nhận precision = 0 theo quy ước ở đây, tức F1 = 0.
Đó là hành vi mong muốn: nó không được thưởng vì né bài toán.

Lưới chạy **pipeline thật** ở từng điểm chứ không mô phỏng bằng số học trên điểm
đã thu được. Mô phỏng sẽ trượt khỏi hành vi thật ở đúng chỗ khó nhất: vòng sinh
lại phụ thuộc `TAU_GROUND`, nên "câu trả lời ở ngưỡng 0.5" không suy ra được từ
"câu trả lời ở ngưỡng 0.8".

Chạy thật mà vẫn rẻ là nhờ cache, và nhờ hai quan sát:

- Kết quả truy xuất **không** phụ thuộc ngưỡng nào, nên tính đúng một lần
  (`retrieve_all`) rồi dùng lại cho mọi điểm lưới.
- Prompt lượt hai dựng từ danh sách mệnh đề bị judge bác, mà danh sách đó cũng
  không phụ thuộc ngưỡng — chỉ *quyết định có sinh lại hay không* mới phụ thuộc.

Tổng lại, mỗi câu tốn tối đa 2 lời gọi generate và 2 lời gọi judge cho **toàn bộ**
lưới, không phải 2 lời gọi mỗi điểm lưới.

**Lưới `TAU_RETRIEVE` chỉ có 4 điểm, và đó là con số đúng.** Điểm rerank là thang
nguyên `0..RERANK_MAX_SCORE` chuẩn hoá về `[0, 1]`, nên nó chỉ nhận đúng 4 giá trị
(`0`, `1/3`, `2/3`, `1`). Bốn ngưỡng `(0.0, 0.34, 0.67, 1.0)` rơi vào bốn khe giữa
các giá trị đó — thêm điểm lưới nữa chỉ tạo ra các dòng trùng nhau và làm bảng
trông như đã dò kỹ hơn thực tế.

Chọn điểm theo bốn tiêu chí, đúng thứ tự đó: F1 cao nhất → ít `missed_abstain`
hơn (phá hoà về phía an toàn) → **faithfulness cao hơn** → ngưỡng thấp hơn.

### Lưới phẳng: kết quả thật, và nó buộc phải sửa tiêu chí chọn

Chạy thật trên gold set cho một kết quả không lường trước: **cả 28 điểm lưới đều
F1 = 1.000**. Vùng bằng phẳng phủ kín lưới.

Đó chính là lúc việc in cả vùng bằng phẳng trả công. Nếu chỉ in argmax thì bảng
sẽ trông như "đã tìm được ngưỡng tối ưu", trong khi sự thật là **mục tiêu chính
không phân biệt được điểm nào với điểm nào**. Tách hai ngưỡng ra thì thấy chúng
khác hẳn nhau:

| | kết luận từ lưới |
|---|---|
| `TAU_RETRIEVE` | **kết quả rỗng.** Bốn hàng giống hệt nhau. Cả 4 câu unanswerable bị chặn hai lần độc lập — điểm rerank của chúng là 0.0/0.0/0.33/0.0, *và* bộ sinh cũng tự abstain đúng cả 4. Cửa gác chưa bao giờ là mắt xích quyết định. |
| `TAU_GROUND` | **có tác dụng đo được.** Faithfulness đi 0.9597 → 0.9806 → 1.0000 khi siết 0.5 → 0.75 → 1.0, đơn điệu trên toàn lưới, và không gây một lần từ chối nhầm nào. Giá phải trả: 0 → 1 → 3 lượt sinh lại. |

Tiêu chí chọn ban đầu chỉ tối ưu F1 abstain. Vì F1 hoà khắp lưới, nó rơi thẳng
xuống tiêu chí cuối là "ngưỡng thấp nhất" — và chọn đúng ô có faithfulness **tệ
nhất** (0.9597 thay vì 1.0000). Đó là một lỗi thật của tiêu chí, chỉ lộ ra khi có
dữ liệu thật, nên `best_point` được bổ sung faithfulness làm tiêu chí phá hoà thứ
ba. Bài học: **lưới phẳng ở mục tiêu chính không có nghĩa mọi điểm tương đương.**

Kết quả: `TAU_RETRIEVE = 0.0`, `TAU_GROUND = 1.0`. Giá trị `TAU_RETRIEVE = 0.0`
được ghi rõ trong `src/config.py` là *kết quả rỗng, không phải phát hiện* — không
có bằng chứng cho giá trị nào khác 0, và cửa gác vẫn còn một lý do mà metric này
không đo được (nó tiết kiệm một lượt sinh cho mỗi câu truy xuất hỏng, đáng kể khi
free tier có hạn ngạch theo ngày).

**Cảnh báo cỡ mẫu, in ra ngay trong báo cáo.** Gold set chỉ có 4 câu
`unanswerable`, nên recall của lớp abstain nhảy theo bước 0.25 và F1 có khoảng tin
cậy rất rộng. Một lưới phẳng hoàn toàn với n = 4 nói về cỡ mẫu nhiều hơn là về
ngưỡng.

## Cache và chế độ offline

Cache **luôn bật**, kể cả khi chạy online. `offline` không phải một nhánh code
riêng — nó chỉ đổi hành vi lúc cache miss: online thì gọi API rồi ghi cache, offline
thì raise `CacheMiss` kèm key, provider, model, task và 200 ký tự đầu của input.
Nhờ vậy đường đi dữ liệu ở hai chế độ là một, không có nhánh nào chỉ chạy lúc demo
mà chưa từng chạy lúc thật.

Cache key = SHA-256 của canonical JSON:

```json
{"provider": "...", "model": "...", "task": "...", "input": ..., "params": {...}, "prompt_version": 1}
```

`params` chứa mọi thứ ảnh hưởng tới output: `temperature`, `max_output_tokens`,
`output_dimensionality`, `task_type`, `thinking_budget`. Băm **payload có cấu
trúc** chứ không băm chuỗi prompt đã render, để đổi cách trình bày prompt không
vứt sạch cache; bù lại template đổi mà payload không đổi sẽ không tự phát hiện
được — đó là việc của `PROMPT_VERSION` trong `src/config.py`, **tăng tay** mỗi khi
sửa template.

Embedding cache theo **từng text** (không theo batch), nên đổi thứ tự hay đổi kích
thước batch không làm hỏng cache; batching chỉ tồn tại ở tầng gọi API. Các text
trùng nhau được gom theo digest trước khi chia batch, nên chỉ tốn một lời gọi.

Layout: `outputs/cache/{embed,generate,rerank,judge}/<sha256>.json`, mỗi file lưu
`{"key_payload": {...}, "response": ..., "created_at": ...}`. Lưu cả `key_payload`
để khi hai lời gọi tưởng giống nhau mà ra key khác thì diff được ngay.

Dùng 768 chiều thay vì 3072 để cache commit vào repo nhỏ đi khoảng bốn lần.
`gemini-embedding-001` chỉ chuẩn hoá sẵn ở 3072 chiều, nên sau khi cắt chiều bắt
buộc chuẩn hoá L2 lại — có kiểm tra `‖v‖₂ ≈ 1.0` raise ngay tại chỗ.

Mọi lời gọi API đều qua `with_backoff`: luỹ thừa cơ số 2 + jitter, tối đa 6 lần.
Phân loại lỗi đáng thử lại dựa vào **status code** đọc từ exception của SDK và các
lỗi mạng thuần (`ConnectionError`, `TimeoutError`) — không dựa vào substring trong
message, vì chuỗi `"max_output_tokens 500"` sẽ khớp marker `"500"` và biến một lỗi
tham số vĩnh viễn thành sáu lần thử lại vô ích.

## Test

```bash
python -m pytest
```

Không test nào chạm mạng. Đáng chú ý:

- `test_ingest_chunk.py::test_char_offsets_exact` — `body[char_start:char_end] == text`
  ở mọi kích thước chunk. Test này fail thì mọi `gold_span` đều vô nghĩa.
- `test_ingest_chunk.py::test_hop_chunk_phu_toan_bo_body` — phần body không được
  chunk nào phủ chỉ được chứa khoảng trắng.
- `test_ingest_chunk.py::test_khong_bao_gio_vua_gop_vua_cat` — bất biến hai pha.
- `test_ingest_chunk.py::test_cross_reference_not_parsed` — tham chiếu chéo giữa
  đoạn không tạo ra Điều/Khoản mới.
- `test_intervals.py` — hợp khoảng: chồng lấn, lồng nhau, kề nhau, trùng lặp, rỗng.
- `test_cache.py` — cái gì làm đổi cache key, `CacheMiss` lúc offline, phân loại
  lỗi retryable, và dedup embedding.
- `test_annotate.py` — ánh xạ offset khi bỏ dấu, và các ca `--validate` bắt lỗi.
- `test_index_retrieve.py::test_arm_bm25_chay_duoc_offline_voi_cache_rong` — arm
  từ khoá không tạo ra một file cache nào; đây là bằng chứng cho phép nạp trễ.
- `test_index_retrieve.py::test_rrf_khong_phu_thuoc_thang_diem` — nhân điểm lên
  1000 lần mà giữ nguyên thứ hạng thì kết quả hợp nhất không đổi.
- `test_index_retrieve.py::test_arm_rerank_hoa_diem_thi_giu_thu_tu_rrf` — reranker
  chấm mọi ứng viên bằng nhau thì thứ tự ra đúng bằng thứ tự RRF.

- `test_evaluate.py::test_coverage_bat_bien_voi_chunk_size` — chia một chunk thành
  bốn chunk kề nhau không đổi coverage. Đây là lý do tồn tại của cả metric.
- `test_evaluate.py::test_coverage_chi_tinh_chunk_cung_document` — chunk của văn
  bản khác không được tính vào phần giao dù khoảng số học có chồng nhau.
- `test_evaluate.py::test_coverage_chunk_chong_lan_khong_vuot_qua_1` — cộng độ dài
  text sẽ ra 1.4, hợp khoảng ra đúng 1.0.

- `test_generate_verify.py::test_luot_sinh_lai_dung_cache_key_khac` — gieo cache
  chỉ cho lượt 1; `CacheMiss` ở lượt 2 là bằng chứng hai lượt là hai lời gọi khác.
- `test_generate_verify.py::test_check_a_fail_thi_khong_goi_judge` — tầng rẻ chặn
  trước tầng đắt.
- `test_generate_verify.py::test_diem_rerank_duoi_nguong_thi_abstain_truoc_khi_sinh`
  — cache rỗng + offline mà không raise, tức là không tốn lượt sinh nào.

- `test_calibrate_run.py::test_tau_retrieve_tach_duoc_cau_ngoai_pham_vi` — lưới
  chạy qua `answer_question` thật, nên bao logic hai ngưỡng chứ không chỉ bao
  phép số học tổng hợp.
- `test_calibrate_run.py::test_khong_bao_gio_abstain_thi_f1_bang_0` — hệ thống né
  bài toán không được thưởng.

Phía dense, rerank, generate và judge được kiểm bằng cách **gieo sẵn cache** rồi
chạy đúng đường dữ liệu thật, chứ không monkeypatch hàm gọi API. Nhờ vậy test bao
luôn cả hình dạng cache key — thứ mà một stub sẽ bỏ lọt.

`tests/fixtures/corpus/` chứa hai văn bản **giả lập** (Quy chế 99/2099 và Thông tư
88/2088 hư cấu, một `active` một `expired`) để chạy được toàn bộ pipeline mà không
đụng corpus thật. Chúng không phải văn bản pháp luật có thật và không được dùng làm
nguồn tra cứu.

## Giới hạn đã biết

- Reranker chấm trong một lời gọi listwise: model thấy cả danh sách nên điểm không
  hoàn toàn độc lập giữa các ứng viên.
- Quy tắc hiệu lực nằm trong prompt rerank, nên ba câu `distractor` đo cả câu lệnh
  đó chứ không chỉ đo model. Số liệu phải đọc kèm ghi chú này.
- `RERANK_CANDIDATES = 30` là trần cứng: gold_span nằm ngoài top-30 của RRF thì
  rerank không có cơ hội cứu. Recall@30 của arm `hybrid` vì vậy là trần trên của
  `hybrid_rerank`, và cần được báo cáo cùng nhau.
- Hai ngưỡng abstain được hiệu chuẩn trên **4** câu unanswerable. Con số F1 có
  khoảng tin cậy rất rộng; đó là lý do `calibrate` in cả vùng bằng phẳng.
- Ngưỡng được chọn và đánh giá trên cùng một gold set, không có tập held-out.
  Với 20 câu thì chia tập sẽ làm cả hai nửa vô nghĩa, nhưng con số vì thế là
  **in-sample** và phải đọc như vậy.
- Bigram âm tiết chỉ xấp xỉ ranh giới từ ghép, không thay được phân từ thật; từ
  ghép ba âm tiết trở lên chỉ được bắt một phần.
- Nếu một câu đơn lẻ dài hơn trần cắt thì hai part liền kề kề nhau chứ không chồng
  lấn — không thể vừa chồng lấn vừa tiến. Hợp các span vẫn đúng bằng span gốc.
- `verify_coverage` sẽ raise nếu văn bản có phần mở đầu (lời nói đầu, căn cứ ban
  hành) nằm trước `Chương`/`Điều` đầu tiên, vì phần đó chưa thuộc chunk nào. Gặp
  trên corpus thật thì cần thêm một chunk cấp document cho phần mở đầu.
- Gold set do chính tác giả pipeline soạn. Có `--leakage` chặn việc chép từ vựng,
  nhưng không loại được thiên lệch vô thức khi người viết câu hỏi cũng là người
  biết hệ thống truy xuất thế nào.

## Giả định

Những điều dưới đây được **giả định**, không được kiểm chứng, và là chỗ nên chất
vấn đầu tiên nếu kết quả trông đẹp quá:

1. **Văn bản QPPL tiếng Việt có cấu trúc Chương / Mục / Điều / Khoản nhất quán và
   các tiêu đề luôn nằm ở đầu dòng.** Toàn bộ `src/ingest.py` đứng trên giả định
   này. Corpus thật sau khi chuẩn hoá thoả mãn nó; một văn bản chép tay hoặc OCR
   lệch dòng sẽ phá vỡ bộ parse. Có chốt an toàn: số khoản phải tăng liên tiếp từ
   1, và `verify_coverage()` raise nếu có phần body không thuộc chunk nào.
2. **Một khoản là một quy phạm trọn vẹn, đọc tách ra vẫn đúng nghĩa.** Đây là cơ
   sở để lấy khoản làm đơn vị chunk. Không đúng với khoản chỉ có ý nghĩa khi đọc
   cùng đoạn mở đầu của Điều — bù một phần bằng việc gắn tiêu đề Điều vào
   `indexed_text` của mọi chunk.
3. **Ước lượng token bằng `len(text) / 3.5` là đủ.** Nó chỉ dùng để so với trần
   cắt và sàn gộp, không dùng để tính chi phí, nên sai số vài phần trăm không đổi
   quyết định nào.
4. **`gold_span` do con người gán là chân lý.** Mọi offset đều lấy từ
   `tools/annotate.py` chạy thật, có `--validate` kiểm biên và kiểm "span nằm trọn
   trong một Điều". Nhưng việc *chọn* khoản nào là căn cứ đúng vẫn là phán đoán
   của người gán.
5. **`support_ratio` từ một LLM judge xấp xỉ được groundedness.** Judge khác họ
   model với bộ sinh để tránh thiên lệch tự ưu ái, nhưng không có ai chấm lại
   judge. Đây là mắt xích chưa được kiểm chứng độc lập.
6. **Ngưỡng coverage 0.8 tách được "đủ căn cứ" khỏi "chưa đủ".** Chọn theo lập
   luận (mép chunk hay cắt vài chữ; phủ nửa khoản dễ mất vế điều kiện), không
   chọn bằng cách quét trên dữ liệu.
7. **Corpus đóng.** Pipeline giả định câu trả lời hoặc nằm trong 4 văn bản này,
   hoặc không tồn tại. Nó không biết gì về văn bản sửa đổi bổ sung ban hành sau.

## Nếu có thêm thời gian

Theo thứ tự giá trị trên chi phí:

1. **Mở rộng nhóm `unanswerable` từ 4 lên ~20 câu.** Đây là ràng buộc chặt nhất
   hiện tại: hiệu chuẩn hai ngưỡng trên 4 điểm dương thì recall nhảy theo bước
   0.25 và F1 gần như không có ý nghĩa thống kê. Rẻ, vì câu unanswerable không
   cần gán `gold_span`.
2. **Tách tập hiệu chuẩn khỏi tập đánh giá.** Cần khoảng 60 câu để chia 40/20 mà
   cả hai nửa còn nói được điều gì. Chừng nào chưa có thì mọi số abstain là
   in-sample.
3. **Cho reranker nhìn sâu hơn 30 ứng viên, hoặc lặp truy xuất cho câu multi-hop.**
   `Rs@30 = 0.938` của `bm25` cho thấy trần vẫn còn dư địa, và câu multi-hop cần
   hai span thì một lượt truy xuất đơn không phải hình dạng đúng — truy xuất lượt
   hai dùng chính span đã tìm được làm truy vấn sẽ hợp lý hơn.
4. **Đo lại với `chunk_size` khác nhau.** Metric coverage được thiết kế để bất
   biến với `chunk_size`, và đã có test cho tính chất đó, nhưng chưa có bảng thực
   nghiệm quét `CHUNK_MAX_TOKENS` để xác nhận trên dữ liệu thật.
5. **Thay bigram âm tiết bằng phân từ thật** và đo xem BM25 lợi bao nhiêu. Hiện
   tránh `pyvi`/`underthesea` vì chúng kéo theo scikit-learn; nếu chấp nhận thêm
   dependency thì đây là phép đo đáng làm.
6. **Chấm lại judge bằng người.** Lấy ~30 mệnh đề, để người gán `supported` /
   `not supported`, rồi đo agreement với judge. Không có con số này thì
   `support_ratio` là một chỉ số chưa ai kiểm.
7. **Xử lý văn bản sửa đổi bổ sung.** Corpus hiện chỉ có quan hệ "thay thế"
   (NĐ 356 thay NĐ 13). Quan hệ "sửa đổi một số điều" khó hơn nhiều vì hiệu lực
   được áp ở mức từng Điều chứ không mức văn bản.

## Trạng thái chạy thật

**Toàn bộ pipeline đã chạy thật đầu-cuối.** Mọi con số trong README này đến từ
lời gọi API thật, và cache của cả bốn loại lời gọi (`embed`, `rerank`, `generate`,
`judge`) đã được commit trong `outputs/cache/`.

Hệ quả: `python run.py all --arm all --offline` dựng lại được **mọi bảng** trong
tài liệu này mà không cần một API key nào. Chỉ cần key khi hỏi một câu **mới**
chưa có trong cache.

Không có nhánh nào coi "không gọi được judge" là "đã chấm và đạt": Check B lỗi
kết nối thì `with_backoff` thử lại rồi raise, `check_b` giữ nguyên `None`, và
`abstain_stage` ghi rõ tầng chết.
