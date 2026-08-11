"""Sinh `outputs/demo_answers.md` — 5 câu trả lời mẫu kèm trích dẫn.

Đề bài yêu cầu "trả lời 3 đến 5 câu hỏi kèm trích dẫn nguồn". File này là chính
deliverable đó, đọc thẳng từ `outputs/eval/answers.json` chứ không gọi lại API,
nên nó luôn khớp với con số trong báo cáo đánh giá.

Năm câu được chọn để phủ đủ năm hành vi khác nhau của pipeline, không phải chọn
năm câu dễ nhất.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import config

# Mỗi mục: (loại câu, vì sao chọn câu này)
SHOWCASE: tuple[tuple[str, str], ...] = (
    ("factoid_1hop", "Truy xuất một khoản duy nhất và trả lời trực tiếp."),
    ("multihop", "Cần ghép căn cứ từ HAI văn bản khác nhau."),
    (
        "distractor",
        "Corpus chứa một Nghị định đã hết hiệu lực bàn đúng chủ đề này. "
        "Câu trả lời phải viện dẫn văn bản còn hiệu lực.",
    ),
    ("unanswerable_oos", "Chủ đề nằm ngoài corpus — phải từ chối trả lời."),
    (
        "unanswerable_nearmiss",
        "Corpus bàn đúng chủ đề nhưng KHÔNG có con số được hỏi. "
        "Đây là ca dễ bịa nhất.",
    ),
)


def pick(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Lấy câu đầu tiên của mỗi loại, giữ nguyên thứ tự trong SHOWCASE."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in report["questions"]:
        by_type.setdefault(item["type"], []).append(item)
    return [by_type[kind][0] for kind, _ in SHOWCASE if by_type.get(kind)]


def render(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Năm câu trả lời mẫu",
        "",
        f"Sinh bằng `python run.py answer` (arm `{report['arm']}`), đọc lại từ "
        "`outputs/eval/answers.json`. Không viết tay câu nào.",
        "",
        f"Ngưỡng đang dùng: `TAU_RETRIEVE = {report['tau_retrieve']}`, "
        f"`TAU_GROUND = {report['tau_ground']}`.",
        "",
        "Mỗi mệnh đề mang đúng tập trích dẫn của riêng nó. Số `[n]` trỏ tới mục "
        "tương ứng trong phần **Căn cứ** ngay dưới câu trả lời.",
        "",
    ]

    reasons = dict(SHOWCASE)
    for order, item in enumerate(pick(report), start=1):
        lines += [
            "---",
            "",
            f"## {order}. {item['qid']} — `{item['type']}`",
            "",
            f"> {reasons[item['type']]}",
            "",
            f"**Câu hỏi.** {item['question']}",
            "",
            f"**Trả lời.** {item['answer']}",
            "",
        ]
        if item["citations"]:
            lines.append("**Căn cứ.**")
            lines += [f"- `{label}`" for label in item["citations"]]
            lines.append("")

        verdict = "TỪ CHỐI TRẢ LỜI" if item["abstained"] else "đã trả lời"
        detail = [f"quyết định: **{verdict}**"]
        if item["abstained"]:
            detail.append(f"chặn ở tầng `{item['abstain_stage']}`")
        if item["support_ratio_first"] is not None:
            detail.append(f"support_ratio trước verify: {item['support_ratio_first']:.3f}")
        if item["support_ratio_final"] is not None:
            detail.append(f"sau verify: {item['support_ratio_final']:.3f}")
        detail.append(f"số lượt sinh: {len(item['attempts'])}")
        lines += ["<sub>" + " · ".join(detail) + "</sub>", ""]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sinh file 5 câu trả lời mẫu")
    parser.add_argument("--answers", default=str(config.EVAL_DIR / "answers.json"))
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "demo_answers.md"))
    args = parser.parse_args()

    source = Path(args.answers)
    if not source.is_file():
        raise SystemExit(f"Chưa có {source}. Chạy `python run.py answer` trước.")

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"Đã ghi {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
