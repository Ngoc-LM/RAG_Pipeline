# Năm câu trả lời mẫu

Sinh bằng `python run.py answer` (arm `hybrid_rerank`), đọc lại từ `outputs/eval/answers.json`. Không viết tay câu nào.

Ngưỡng đang dùng: `TAU_RETRIEVE = 0.0`, `TAU_GROUND = 1.0`.

Mỗi mệnh đề mang đúng tập trích dẫn của riêng nó. Số `[n]` trỏ tới mục tương ứng trong phần **Căn cứ** ngay dưới câu trả lời.

---

## 1. q01 — `factoid_1hop`

> Truy xuất một khoản duy nhất và trả lời trực tiếp.

**Câu hỏi.** Công ty bị phát hiện bán danh sách khách hàng ra ngoài thì trần tiền phạt được xác định dựa trên cái gì?

**Trả lời.** Mức phạt tiền tối đa đối với hành vi mua, bán dữ liệu cá nhân là 10 lần khoản thu có được từ hành vi vi phạm. [1] Trường hợp không có khoản thu từ hành vi vi phạm hoặc mức phạt tính theo khoản thu thấp hơn mức phạt tiền tối đa quy định tại khoản 5 Điều 8 thì áp dụng mức phạt tiền theo quy định tại khoản 5 Điều 8. [1] Mức phạt tiền tối đa đối với các hành vi vi phạm khác trong lĩnh vực bảo vệ dữ liệu cá nhân là 03 tỷ đồng. [2]

**Căn cứ.**
- `[1] Điều 8 Khoản 3-4 Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15`
- `[2] Điều 8 Khoản 5-7 Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15`

<sub>quyết định: **đã trả lời** · support_ratio trước verify: 1.000 · sau verify: 1.000 · số lượt sinh: 1</sub>

---

## 2. q09 — `multihop`

> Cần ghép căn cứ từ HAI văn bản khác nhau.

**Câu hỏi.** Hai công ty sáp nhập, bên tôi tiếp nhận toàn bộ danh sách khách của bên kia. Có phải ký kết văn bản riêng cho việc bàn giao đó không, và trong đó cần ghi những gì?

**Trả lời.** Tổ chức chuyển giao dữ liệu cá nhân trong trường hợp sáp nhập phải xác lập thỏa thuận về việc chuyển giao dữ liệu cá nhân với bên nhận. [1][2] Thỏa thuận chuyển giao dữ liệu cá nhân phải nêu rõ mục đích chuyển giao. [1] Thỏa thuận phải nêu rõ đối tượng chủ thể dữ liệu cá nhân và loại dữ liệu cá nhân được chuyển giao phù hợp với mục đích chuyển giao. [1] Thỏa thuận phải nêu rõ thời hạn xử lý dữ liệu cá nhân, yêu cầu về xóa, hủy dữ liệu cá nhân sau khi hoàn thành mục đích chuyển giao. [1] Thỏa thuận phải nêu rõ cơ sở pháp lý của việc chuyển giao dữ liệu cá nhân. [1] Thỏa thuận phải nêu rõ trách nhiệm bảo vệ dữ liệu cá nhân trong quá trình chuyển giao, xử lý dữ liệu cá nhân. [1] Thỏa thuận phải nêu rõ trách nhiệm thực hiện các quyền của chủ thể dữ liệu cá nhân. [1] Thỏa thuận phải nêu rõ trách nhiệm phối hợp và tuân thủ của các bên trong trường hợp phát hiện vi phạm quy định bảo vệ dữ liệu cá nhân. [1]

**Căn cứ.**
- `[1] Điều 7 Khoản 1 Nghị định 356/2025/NĐ-CP quy định chi tiết một số điều và biện pháp thi hành Luật Bảo vệ dữ liệu cá nhân`
- `[2] Điều 17 Khoản 1-3 Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15`

<sub>quyết định: **đã trả lời** · support_ratio trước verify: 0.889 · sau verify: 1.000 · số lượt sinh: 2</sub>

---

## 3. q14 — `distractor`

> Corpus chứa một Nghị định đã hết hiệu lực bàn đúng chủ đề này. Câu trả lời phải viện dẫn văn bản còn hiệu lực.

**Câu hỏi.** Công ty tôi vừa xác định có một vụ rò rỉ dữ liệu nghiêm trọng ảnh hưởng tới tài sản của khách. Phải gửi thông báo tới đâu và trong bao lâu?

**Trả lời.** Khi phát hiện vi phạm quy định về bảo vệ dữ liệu cá nhân có thể gây tổn hại đến tài sản của chủ thể dữ liệu, bên kiểm soát dữ liệu cá nhân, bên kiểm soát và xử lý dữ liệu cá nhân, bên thứ ba phải thông báo cho cơ quan chuyên trách bảo vệ dữ liệu cá nhân. [1] Thời hạn thông báo cho cơ quan chuyên trách bảo vệ dữ liệu cá nhân là chậm nhất 72 giờ kể từ khi phát hiện hành vi vi phạm. [1] Nếu là dữ liệu nhạy cảm trong lĩnh vực tài chính, ngân hàng, hoạt động thông tin tín dụng, tổ chức hoặc cá nhân thu thập dữ liệu phải thông báo cho cơ quan chuyên trách bảo vệ dữ liệu cá nhân và chủ thể dữ liệu cá nhân trong thời hạn không quá 72 giờ sau khi phát hiện lộ, mất dữ liệu. [2]

**Căn cứ.**
- `[1] Điều 23 Khoản 1 Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15`
- `[2] Điều 8 Khoản 3 Nghị định 356/2025/NĐ-CP quy định chi tiết một số điều và biện pháp thi hành Luật Bảo vệ dữ liệu cá nhân`

<sub>quyết định: **đã trả lời** · support_ratio trước verify: 1.000 · sau verify: 1.000 · số lượt sinh: 1</sub>

---

## 4. q17 — `unanswerable_oos`

> Chủ đề nằm ngoài corpus — phải từ chối trả lời.

**Câu hỏi.** Doanh nghiệp phải giữ hóa đơn điện tử trong bao nhiêu năm trước khi được huỷ?

**Trả lời.** Không đủ căn cứ trong tài liệu.

<sub>quyết định: **TỪ CHỐI TRẢ LỜI** · chặn ở tầng `model` · số lượt sinh: 1</sub>

---

## 5. q19 — `unanswerable_nearmiss`

> Corpus bàn đúng chủ đề nhưng KHÔNG có con số được hỏi. Đây là ca dễ bịa nhất.

**Câu hỏi.** Dữ liệu khách hàng được phép giữ tối đa bao nhiêu tháng trước khi buộc phải xoá?

**Trả lời.** Không đủ căn cứ trong tài liệu.

<sub>quyết định: **TỪ CHỐI TRẢ LỜI** · chặn ở tầng `model` · số lượt sinh: 1</sub>
