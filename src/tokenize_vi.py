"""Tokenizer tiếng Việt cho BM25, không kéo thêm dependency phân từ.

Tiếng Việt tách âm tiết bằng khoảng trắng nên tách theo whitespace chỉ cho ra
âm tiết rời: "bảo", "hiểm", "xã", "hội" khớp lung tung với mọi văn bản có chữ
"xã". Bù lại bằng bigram âm tiết liền kề để xấp xỉ từ ghép, thay vì dùng pyvi
hay underthesea (kéo theo scikit-learn, phá ràng buộc dependency của đề bài).

Chữ số được giữ và tách riêng vì trong văn bản QPPL "Điều 60", "12 tháng",
"6%" là tín hiệu truy xuất mạnh; bigram "điều_60" còn mạnh hơn nữa.
"""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)


def syllables(text: str) -> list[str]:
    """Âm tiết và cụm chữ số đã hạ hoa, bỏ dấu câu, giữ dấu thanh."""
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def tokenize(text: str) -> list[str]:
    """Unigram âm tiết + bigram âm tiết liền kề, dùng cho cả index và query."""
    units = syllables(text)
    bigrams = [f"{a}_{b}" for a, b in zip(units, units[1:])]
    return units + bigrams
