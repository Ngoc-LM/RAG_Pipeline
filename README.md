# Pipeline RAG cho văn bản quy phạm pháp luật tiếng Việt

Truy xuất lai (BM25 + embedding), hợp nhất RRF, rerank bằng LLM, sinh câu trả lời
có trích dẫn, và một tầng kiểm chứng quyết định chấp nhận / sinh lại / từ chối trả lời.

Python thuần + numpy + rank_bm25 + google-genai + groq. Không LangChain, không
LlamaIndex, không FAISS — corpus cỡ này thì cosine brute-force bằng một phép nhân
ma trận numpy vừa đủ nhanh vừa giải thích được từng bước.

## Trạng thái

Repo đang dựng lại tầng truy xuất trên nền móng Chương-Điều-Khoản. Phần đã có
chạy được và có test; phần chưa có là thiết kế đã chốt nhưng chưa viết mã.

| Thành phần | File | Trạng thái |
|---|---|---|
| Hằng số tập trung | `src/config.py` | ✅ |
| Parse frontmatter + Chương/Điều/Khoản | `src/ingest.py` | ✅ |
| Chia chunk theo khoản | `src/chunk.py` | ✅ |
| Số học khoảng cho coverage | `src/intervals.py` | ✅ |
| Tokenizer tiếng Việt cho BM25 | `src/tokenize_vi.py` | ✅ |
| Cache + backoff cho API | `src/llm.py` | ✅ |
| CLI gán nhãn gold_span | `tools/annotate.py` | ✅ |
| Gold set 20 câu | `eval/questions.json` | ✅ |
| Index (embed + BM25) | — | ⬜ chưa dựng |
| Retrieve (RRF + rerank) | — | ⬜ chưa dựng |
| Generate + verify hai tầng | — | ⬜ chưa dựng |
| Evaluate + calibrate | — | ⬜ chưa dựng |

Lệnh duy nhất chạy được đầu-cuối hôm nay:

```bash
python -m src.chunk --corpus tests/fixtures/corpus
```

In số document, số điều, số chunk, phân phối `n_tokens` và tỉ lệ chunk theo
`status`. Đổi `--corpus data/corpus` khi đã có corpus thật.

Ngoài ra: `python -m tools.annotate --help` và `python -m pytest`.

Bảng model dùng khi tầng LLM được dựng lại:

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

`.env` nằm trong `.gitignore`. Không commit key. Chưa cần key để chạy `src.chunk`,
`tools.annotate` hay test — không phần nào trong số đó chạm mạng.

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

> Thiết kế đã chốt; tầng evaluate chưa dựng lại. Phần số học khoảng đã có trong
> `src/intervals.py` và có test cho các ca chồng lấn, lồng nhau, kề nhau, trùng lặp.

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
và coverage vọt quá 1.

Câu multi-hop báo cáo hai cột: **strict** (mọi `gold_span` đều đạt ngưỡng) và
**any** (ít nhất một span đạt). Strict là cột chính.

Không báo cáo nDCG: với relevance dạng coverage thì IDCG phải dựng từ một lời giải
phủ tối ưu, và con số đó nói về bộ giải set-cover nhiều hơn là về retriever. Thay
bằng `mean_cov@k`, vốn đã là metric có thứ bậc và không cần chuẩn hoá tuỳ tiện.

## Bốn arm ablation

> Thiết kế đã chốt; chưa dựng lại.

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

Rerank chấm cả danh sách trong **đúng một** lời gọi LLM. Pointwise mỗi ứng viên một
call sẽ tốn n lần quota; free tier giới hạn RPM thấp nên listwise là bắt buộc chứ
không chỉ là tối ưu.

### Tokenizer tiếng Việt cho BM25

Tiếng Việt tách âm tiết bằng khoảng trắng, nên tách theo whitespace chỉ cho ra âm
tiết rời — "bảo", "hiểm", "xã", "hội" khớp lung tung với mọi văn bản có chữ "xã".
Bù bằng **bigram âm tiết liền kề** (`bảo_hiểm`, `xã_hội`) thay vì kéo `pyvi` hay
`underthesea` về (chúng kéo theo scikit-learn). Chữ số được giữ và tách riêng vì
"Điều 60", "12 tháng" là tín hiệu mạnh; bigram `điều_60` còn mạnh hơn.

## Sinh câu trả lời và kiểm chứng

> Thiết kế đã chốt; chưa dựng lại.

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

Báo cáo faithfulness **trước** và **sau** tầng verify, để đọc được tầng này thực sự
thêm bao nhiêu giá trị chứ không chỉ thêm bao nhiêu chi phí.

### Hai ngưỡng abstain

Tách đôi vì chúng bắt hai loại lỗi khác nhau: một ngưỡng gác **trước** generate
(so với điểm rerank cao nhất) bắt câu hỏi ngoài phạm vi corpus và chặn trước khi
tốn lượt sinh nào; một ngưỡng gác **sau** generate (so với `support_ratio`) bắt
trường hợp chunk trông hợp lý nhưng model bịa thêm chi tiết. Cả hai sẽ được quét
lưới trên gold set để tối ưu F1 giữa abstain đúng và abstain nhầm, và sẽ vào
`src/config.py` khi tầng đó được dựng.

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

`tests/fixtures/corpus/` chứa hai văn bản **giả lập** (Quy chế 99/2099 và Thông tư
88/2088 hư cấu, một `active` một `expired`) để chạy được toàn bộ pipeline mà không
đụng corpus thật. Chúng không phải văn bản pháp luật có thật và không được dùng làm
nguồn tra cứu.

## Giới hạn đã biết

- Reranker chấm trong một lời gọi listwise: model thấy cả danh sách nên điểm không
  hoàn toàn độc lập giữa các ứng viên.
- Bigram âm tiết chỉ xấp xỉ ranh giới từ ghép, không thay được phân từ thật; từ
  ghép ba âm tiết trở lên chỉ được bắt một phần.
- Nếu một câu đơn lẻ dài hơn trần cắt thì hai part liền kề kề nhau chứ không chồng
  lấn — không thể vừa chồng lấn vừa tiến. Hợp các span vẫn đúng bằng span gốc.
- `verify_coverage` sẽ raise nếu văn bản có phần mở đầu (lời nói đầu, căn cứ ban
  hành) nằm trước `Chương`/`Điều` đầu tiên, vì phần đó chưa thuộc chunk nào. Gặp
  trên corpus thật thì cần thêm một chunk cấp document cho phần mở đầu.
