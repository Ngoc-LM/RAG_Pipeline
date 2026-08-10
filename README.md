# Pipeline RAG cho văn bản quy phạm pháp luật tiếng Việt

Truy xuất lai (BM25 + embedding), hợp nhất RRF, rerank bằng LLM, sinh câu trả lời
có trích dẫn, và một tầng kiểm chứng quyết định chấp nhận / sinh lại / từ chối trả lời.

Python thuần + numpy + rank_bm25 + google-genai + groq. Không LangChain, không
LlamaIndex, không FAISS — corpus cỡ này thì cosine brute-force bằng một phép nhân
ma trận numpy vừa đủ nhanh vừa giải thích được từng bước.

| Vai trò | Model |
|---|---|
| Embedding | `gemini-embedding-001` (768 chiều, cắt Matryoshka + chuẩn hoá L2) |
| Sinh câu trả lời | `gemini-2.5-flash` |
| Rerank | `gemini-2.5-flash-lite` (listwise, một lời gọi cho cả danh sách) |
| LLM judge | `llama-3.3-70b-versatile` trên Groq |

Judge cố ý khác họ model với bộ sinh: dùng chính Gemini để chấm câu trả lời của
Gemini thì sẽ có thiên lệch tự ưu ái.

## Cài đặt

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # điền GEMINI_API_KEY và GROQ_API_KEY
```

`.env` nằm trong `.gitignore`. Không commit key.

## Chuẩn bị corpus

Mỗi văn bản là một file `.md` hoặc `.txt` trong `data/corpus/`, mở đầu bằng khối
frontmatter phẳng dạng `key: value`:

```markdown
---
doc_id: nd-135-2020
title: Nghị định 135/2020/NĐ-CP về tuổi nghỉ hưu
number: 135/2020/NĐ-CP
doc_type: nghi_dinh
issuer: Chính phủ
issued_date: 2020-11-18
effective_date: 2021-01-01
source_url: https://...
---

Chương I

QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
...
```

`doc_id` và `title` bắt buộc, phải duy nhất trong corpus; các key khác tuỳ ý và
được giữ nguyên vào metadata. Chỉ hỗ trợ `key: value` phẳng, không lồng nhau —
để không phải kéo thêm PyYAML.

Bộ chunk nhận diện `Chương`, `Mục`, `Điều`, `Khoản` ở đầu dòng (cho phép có tiền
tố `#` của markdown). Cắt ở ranh giới `Điều` trước; chỉ khi một `Điều` vượt
`CHUNK_MAX_CHARS` mới cắt tiếp ở ranh giới `Khoản`; cắt cứng giữa câu là phương
án cuối.

## Trục ký tự — đọc trước khi soạn gold set

`gold_spans` trỏ tới **offset ký tự trong body đã chuẩn hoá**, không phải trong
file gốc. Body = phần sau frontmatter, đã đổi CRLF/CR thành LF, đã chuẩn hoá NFC,
đã strip hai đầu. Đừng đếm tay trên file gốc.

```bash
python run.py ingest                  # ghi outputs/bodies/<doc_id>.txt
python run.py show-span --doc-id nd-135-2020 --start 1070 --end 1154
```

`outputs/bodies/<doc_id>.txt` chính là chuỗi mà offset tham chiếu tới; `show-span`
in lại đúng đoạn để đối chiếu trước khi chốt.

## Gold set

`eval/questions.json` là một mảng:

```json
{
  "qid": "q01",
  "question": "...",
  "type": "factoid_1hop | multihop | distractor | unanswerable_oos | unanswerable_nearmiss",
  "answerable": true,
  "gold_spans": [{ "doc_id": "nd-135-2020", "char_start": 1240, "char_end": 1533 }],
  "gold_answer": "câu trả lời ngắn 1-2 câu"
}
```

Câu `unanswerable_*` có `answerable: false` và `gold_spans: []`; chúng không tham
gia metric truy xuất, chỉ dùng để đo chất lượng abstain.

QC trước khi tin bất kỳ con số nào:

```bash
python run.py check-leakage
```

Tính Jaccard unigram âm tiết giữa mỗi câu hỏi và text `gold_span` của nó, cảnh báo
khi vượt `LEAKAGE_JACCARD_MAX` (0.3). Câu hỏi copy từ vựng của chính đoạn nguồn sẽ
làm BM25 thắng giả tạo và thổi phồng Recall của mọi arm. Lệnh trả exit code khác 0
khi có span bị gắn cờ, để cắm vào CI được.

## Chạy

```bash
python run.py index                       # chunk + embed + dựng BM25
python run.py ask "Tuổi nghỉ hưu năm 2028 của lao động nam là bao nhiêu?"
python run.py answer                      # chạy cả gold set -> outputs/answers.json
python run.py evaluate --retrieval-only   # mọi arm, không sinh câu trả lời
python run.py evaluate --full --arms bm25,hybrid_rerank
python run.py calibrate                   # quét lưới hai ngưỡng abstain
```

Thêm `--offline` vào bất kỳ lệnh nào để chạy hoàn toàn từ cache.

## Đo truy xuất bằng coverage, không bằng "chunk gold"

Repo này **không** có khái niệm "chunk nào là chunk gold". Gán nhãn ở mức chunk
buộc phải chọn một ngưỡng overlap, mà mọi ngưỡng như vậy đều thiên lệch theo
`chunk_size`: lấy `overlap >= 0.5 · |span|` thì chunk nhỏ không bao giờ đạt; đổi
sang `0.5 · min(|span|, |chunk|)` thì chunk 50 ký tự chỉ cần phủ 25 ký tự của một
span 1000 ký tự cũng thành gold. Cả hai đều làm hỏng bảng ablation theo chunk_size.

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
và coverage vọt quá 1. `src/intervals.py` tách riêng phép hợp khoảng và có test
cho các ca chồng lấn, lồng nhau, kề nhau, trùng lặp.

Câu multi-hop báo cáo hai cột: **strict** (mọi `gold_span` đều đạt ngưỡng) và
**any** (ít nhất một span đạt). Strict là cột chính.

Không báo cáo nDCG: với relevance dạng coverage thì IDCG phải dựng từ một lời giải
phủ tối ưu, và con số đó nói về bộ giải set-cover nhiều hơn là về retriever. Thay
bằng `mean_cov@k`, vốn đã là metric có thứ bậc và không cần chuẩn hoá tuỳ tiện.

## Bốn arm ablation

| arm | mô tả |
|---|---|
| `bm25` | chỉ từ khoá |
| `dense` | chỉ embedding |
| `hybrid` | RRF hợp nhất hai danh sách trên |
| `hybrid_rerank` | hybrid rồi rerank listwise — arm mặc định của pipeline |

RRF hợp nhất theo **hạng** chứ không theo điểm, vì điểm BM25 và cosine không cùng
thang đo và mọi cách chuẩn hoá về một thang đều tuỳ tiện.

Bảng tách theo `type` là chỗ đáng nhìn nhất: câu `distractor` (nhiều văn bản cùng
bàn một chủ đề, chỉ một chứa đáp án) là nơi rerank phải chứng minh nó đáng tiền.

### Tokenizer tiếng Việt cho BM25

Tiếng Việt tách âm tiết bằng khoảng trắng, nên tách theo whitespace chỉ cho ra âm
tiết rời — "bảo", "hiểm", "xã", "hội" khớp lung tung với mọi văn bản có chữ "xã".
Bù bằng **bigram âm tiết liền kề** (`bảo_hiểm`, `xã_hội`) thay vì kéo `pyvi` hay
`underthesea` về (chúng kéo theo scikit-learn). Chữ số được giữ và tách riêng vì
"Điều 60", "12 tháng" là tín hiệu mạnh; bigram `điều_60` còn mạnh hơn.

## Sinh câu trả lời và kiểm chứng

Bộ sinh trả về **từng câu kèm đúng tập citation của riêng câu đó**, không trả một
khối text rồi tách claim ở bước sau — tách câu tiếng Việt bằng heuristic sẽ vỡ ở
"Điều 1.", "khoản 2.", "0,5%", mà claim tách sai thì mọi con số groundedness đều
sai theo.

Hai tầng kiểm chứng, chi phí khác hẳn nhau:

- **Check A — cú pháp, miễn phí.** Mọi citation phải trỏ tới chunk thực sự có
  trong prompt, và mọi claim phải có ít nhất một citation. Fail thì sinh lại ngay,
  không tốn một lời gọi judge nào.
- **Check B — ngữ nghĩa, đúng một lời gọi Groq.** Toàn bộ claim đi trong một
  request; judge trả `[{claim_id, supported, reason}]`. `support_ratio` = số claim
  được hỗ trợ / tổng số claim. Claim mà judge không nhắc tới bị tính là **không**
  được hỗ trợ — im lặng không phải bằng chứng ủng hộ, và cách tính này không
  thưởng cho judge trả thiếu.

Sinh lại tối đa một lần, và prompt lượt hai **bắt buộc khác** lượt một: ở
`temperature=0`, prompt y hệt sẽ cho output y hệt. Lượt hai chèn danh sách claim
đã bị bác kèm yêu cầu sửa hoặc bỏ. Vẫn fail thì trả `"Không đủ căn cứ trong tài liệu."`

`outputs/verify.json` ghi cả hai lượt:

```json
{
  "qid": "q01",
  "attempt_1": { "answer": "...", "support_ratio": 0.5, "failed_claims": ["..."] },
  "attempt_2": { "answer": "...", "support_ratio": 1.0, "failed_claims": [] },
  "final_action": "retry_accept"
}
```

`evaluate --full` báo cáo faithfulness **trước** và **sau** tầng verify, nên đọc
được tầng này thực sự thêm bao nhiêu giá trị chứ không chỉ thêm bao nhiêu chi phí.

## Hai ngưỡng abstain

Tách đôi vì chúng bắt hai loại lỗi khác nhau:

- `TAU_RETRIEVE` — gác **trước** generate, so với điểm rerank cao nhất. Bắt câu
  hỏi ngoài phạm vi corpus, và chặn trước khi tốn lượt sinh nào.
- `TAU_GROUND` — gác **sau** generate, so với `support_ratio`. Bắt trường hợp
  chunk trông hợp lý nhưng model bịa thêm chi tiết.

```bash
python run.py calibrate
```

Quét lưới cả hai ngưỡng trên gold set, tối ưu F1 giữa abstain đúng (các câu
`unanswerable_*`) và không abstain nhầm (các câu answerable), ghi
`outputs/calibration.json`. Tín hiệu được thu **một lần** với cả hai ngưỡng bằng 0
rồi mô phỏng lại toàn lưới — nếu không thì mỗi ô lại tốn một lượt generate và
judge. Đánh đổi: nhánh sinh lại không được mô phỏng, nên F1 báo cáo là **cận dưới**
của cấu hình có retry.

## Cache và `--offline`

Cache **luôn bật**, kể cả khi chạy online. `--offline` không phải một nhánh code
riêng — nó chỉ đổi hành vi lúc cache miss: online thì gọi API rồi ghi cache,
offline thì raise `CacheMiss` kèm key, provider, model, task và 200 ký tự đầu của
input. Nhờ vậy đường đi dữ liệu ở hai chế độ là một, không có nhánh nào chỉ chạy
lúc demo mà chưa từng chạy lúc thật.

Cache key = SHA-256 của canonical JSON:

```json
{"provider": "...", "model": "...", "task": "...", "input": ..., "params": {...}, "prompt_version": 1}
```

`params` chứa mọi thứ ảnh hưởng tới output: `temperature`, `max_output_tokens`,
`output_dimensionality`, `task_type`, `thinking_budget`. Băm **payload có cấu
trúc** chứ không băm chuỗi prompt đã render, để đổi cách trình bày prompt không
vứt sạch cache; bù lại template đổi mà payload không đổi sẽ không tự phát hiện
được — đó là việc của `PROMPT_VERSION` trong `config.py`, **tăng tay** mỗi khi sửa
template.

Embedding cache theo **từng text** (không theo batch), nên đổi thứ tự hay đổi kích
thước batch không làm hỏng cache; batching chỉ tồn tại ở tầng gọi API, gom đúng
những text đang miss rồi tách kết quả ghi cache từng cái.

Layout: `outputs/cache/{embed,generate,rerank,judge}/<sha256>.json`, mỗi file lưu
`{"key_payload": {...}, "response": ..., "created_at": ...}`. Lưu cả `key_payload`
để khi hai lời gọi tưởng giống nhau mà ra key khác thì diff được ngay.

Dùng 768 chiều thay vì 3072 để cache commit vào repo nhỏ đi khoảng bốn lần.
`gemini-embedding-001` chỉ chuẩn hoá sẵn ở 3072 chiều, nên sau khi cắt chiều bắt
buộc chuẩn hoá L2 lại — có `assert ||v||₂ ≈ 1.0` chặn ngay tại chỗ.

Mọi lời gọi API đều qua `with_backoff`: luỹ thừa cơ số 2 + jitter, tối đa 6 lần,
cho 429 / quota / 5xx / lỗi mạng tạm thời.

## Test

```bash
.venv/bin/python -m pytest tests/ -q
```

Không test nào chạm mạng. Đáng chú ý:

- `test_intervals.py` — hợp khoảng: chồng lấn, lồng nhau, kề nhau, trùng lặp, rỗng.
- `test_metrics.py` — coverage bất biến với `chunk_size`; chunk nhỏ không bị thổi
  phồng; multi-hop strict lấy hạng muộn nhất.
- `test_chunk.py` — `chunk.text == body[char_start:char_end]` ở mọi `chunk_size`.
  Test này fail thì mọi `gold_span` đều vô nghĩa.
- `test_cache.py` — cái gì làm đổi key, và `CacheMiss` lúc offline.
- `test_verify.py` — check A, và claim bị judge bỏ sót tính là không được hỗ trợ.

`tests/fixtures/` chứa hai văn bản **giả lập** (Nghị định và Thông tư 00/2099 hư
cấu) cùng 5 câu hỏi, để chạy được toàn bộ pipeline mà không đụng corpus thật.
Chúng không phải văn bản pháp luật có thật và không được dùng làm nguồn tra cứu.

## Giới hạn đã biết

- `calibrate` mô phỏng ngưỡng trên tín hiệu thu ở `tau=0`, không mô phỏng nhánh
  sinh lại; F1 báo cáo là cận dưới.
- `evaluate --retrieval-only` không sinh câu trả lời và không gọi judge, nhưng vẫn
  gọi embed và rerank (cả hai đều đi qua cache). "Không gọi LLM" ở đây nghĩa là
  không có generate và không có judge.
- Reranker chấm pointwise trong một lời gọi listwise: model thấy cả danh sách nên
  điểm không hoàn toàn độc lập giữa các ứng viên.
- Bigram âm tiết chỉ xấp xỉ ranh giới từ ghép, không thay được phân từ thật; từ
  ghép ba âm tiết trở lên chỉ được bắt một phần.
