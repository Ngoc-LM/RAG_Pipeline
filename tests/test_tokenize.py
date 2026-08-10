"""Test tokenizer tiếng Việt cho BM25."""

from __future__ import annotations

from src.tokenize_vi import syllables, tokenize


def test_giu_dau_thanh():
    assert "phụ" in syllables("Phụ cấp")
    assert "phu" not in syllables("Phụ cấp")


def test_giu_chu_so_va_tach_khoi_chu():
    assert syllables("Điều 60 quy định") == ["điều", "60", "quy", "định"]


def test_bo_dau_cau():
    assert syllables("0,7 lần; mức lương.") == ["0", "7", "lần", "mức", "lương"]


def test_bigram_xap_xi_tu_ghep():
    tokens = tokenize("bảo hiểm xã hội")
    assert "bảo_hiểm" in tokens
    assert "xã_hội" in tokens
    assert "bảo" in tokens


def test_bigram_bat_duoc_so_dieu():
    assert "điều_60" in tokenize("Điều 60")


def test_chuoi_rong():
    assert tokenize("") == []
    assert tokenize("!!!") == []
