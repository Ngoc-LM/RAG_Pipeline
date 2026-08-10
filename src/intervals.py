"""Số học khoảng trên trục ký tự của document.

Coverage phải tính trên hợp các khoảng [char_start, char_end) rồi mới giao với
gold_span. Nếu cộng dồn độ dài text của từng chunk thì phần chồng lấn giữa các
chunk bị đếm hai lần và coverage vượt quá 1.
"""

from __future__ import annotations

from typing import Iterable

Interval = tuple[int, int]


def interval_union(intervals: Iterable[Interval]) -> list[Interval]:
    """Hợp các khoảng nửa mở thành danh sách rời nhau, sắp tăng dần.

    Khoảng rỗng hoặc nghịch đảo (start >= end) bị bỏ. Hai khoảng kề nhau
    ([0,5) và [5,9)) được gộp vì chúng liền mạch trên trục ký tự.
    """
    valid = sorted((s, e) for s, e in intervals if s < e)
    if not valid:
        return []
    merged: list[Interval] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def intersection_length(union: Iterable[Interval], span: Interval) -> int:
    """Tổng độ dài phần giao giữa một tập khoảng rời nhau và một khoảng."""
    span_start, span_end = span
    if span_start >= span_end:
        return 0
    total = 0
    for start, end in union:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > 0:
            total += overlap
    return total


def coverage(union: Iterable[Interval], span: Interval) -> float:
    """Tỉ lệ span được phủ bởi tập khoảng, trong [0, 1]."""
    span_start, span_end = span
    length = span_end - span_start
    if length <= 0:
        return 0.0
    return intersection_length(union, span) / length
