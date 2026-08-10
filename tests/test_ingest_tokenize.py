"""Test parse frontmatter, chuẩn hoá trục ký tự, và tokenizer tiếng Việt."""

from __future__ import annotations

import unicodedata

import pytest

from src.ingest import FrontmatterError, normalize_text, parse_frontmatter
from src.tokenize_vi import syllables, tokenize

DOC = """---
doc_id: abc
title: Văn bản mẫu
issuer: Cơ quan X
---

Điều 1. Nội dung

Nội dung của điều 1.
"""


def test_parse_frontmatter_co_ban():
    meta, body = parse_frontmatter(DOC, "abc.md")
    assert meta["doc_id"] == "abc"
    assert meta["title"] == "Văn bản mẫu"
    assert meta["issuer"] == "Cơ quan X"
    assert body.startswith("\nĐiều 1.")


def test_thieu_key_bat_buoc_thi_bao_loi():
    with pytest.raises(FrontmatterError, match="thiếu key bắt buộc"):
        parse_frontmatter("---\ntitle: X\n---\nnội dung\n", "x.md")


def test_khong_co_frontmatter_thi_bao_loi():
    with pytest.raises(FrontmatterError, match="thiếu khối frontmatter"):
        parse_frontmatter("Điều 1. Nội dung\n", "x.md")


def test_frontmatter_khong_dong_thi_bao_loi():
    with pytest.raises(FrontmatterError, match="không có '---' đóng"):
        parse_frontmatter("---\ndoc_id: a\ntitle: b\nnội dung\n", "x.md")


def test_crlf_va_lf_cho_cung_mot_truc_ky_tu():
    """Cùng nội dung, khác kiểu xuống dòng, offset phải trùng khớp."""
    lf = normalize_text("Điều 1.\n\nNội dung.\n")
    crlf = normalize_text("Điều 1.\r\n\r\nNội dung.\r\n")
    assert lf == crlf


def test_normalize_dua_ve_nfc():
    decomposed = unicodedata.normalize("NFD", "phụ cấp")
    assert normalize_text(decomposed) == "phụ cấp"
    assert len(normalize_text(decomposed)) == len("phụ cấp")


def test_normalize_strip_hai_dau():
    assert normalize_text("\n\n  Nội dung.  \n\n") == "Nội dung."


def test_tokenize_giu_dau_thanh():
    assert "phụ" in syllables("Phụ cấp")
    assert "phu" not in syllables("Phụ cấp")


def test_tokenize_giu_chu_so_va_tach_khoi_chu():
    assert syllables("Điều 60 quy định") == ["điều", "60", "quy", "định"]


def test_tokenize_bo_dau_cau():
    assert syllables("0,7 lần; mức lương.") == ["0", "7", "lần", "mức", "lương"]


def test_bigram_xap_xi_tu_ghep():
    tokens = tokenize("bảo hiểm xã hội")
    assert "bảo_hiểm" in tokens
    assert "xã_hội" in tokens
    assert "bảo" in tokens


def test_bigram_bat_duoc_so_dieu():
    assert "điều_60" in tokenize("Điều 60")


def test_tokenize_chuoi_rong():
    assert tokenize("") == []
    assert tokenize("!!!") == []
