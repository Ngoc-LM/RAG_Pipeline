"""Test chunk: điều kiện sống còn là char offset phải khớp body tuyệt đối."""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from src.chunk import chunk_corpus, chunk_document
from src.ingest import load_corpus
from src.intervals import interval_union

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE_DIR)


def test_offset_khop_tuyet_doi_voi_body(docs):
    """Nếu test này fail thì mọi gold_span trong eval/questions.json đều vô nghĩa."""
    for doc in docs:
        for chunk in chunk_document(doc, 1200, 1800, 150):
            assert chunk.text == doc.body[chunk.char_start : chunk.char_end]


def test_offset_van_khop_o_moi_chunk_size(docs):
    for size in (200, 600, 1200, 5000):
        for doc in docs:
            for chunk in chunk_document(doc, size, int(size * 1.5), 150):
                assert chunk.text == doc.body[chunk.char_start : chunk.char_end]


def test_chunk_khong_rong_va_khong_co_khoang_trang_thua(docs):
    for doc in docs:
        for chunk in chunk_document(doc, 1200, 1800, 150):
            assert chunk.text
            assert chunk.text == chunk.text.strip()


def test_so_chunk_khong_giam_khi_target_nho_di(docs):
    """Cắt theo cấu trúc nên số chunk bị chặn dưới bởi số Điều, không giảm đơn điệu
    theo target — chỉ bảo đảm không bao giờ ít đi khi target nhỏ lại."""
    counts = [len(chunk_corpus(docs, size, int(size * 1.5), 150))
              for size in (5000, 1200, 600, 150)]
    assert all(a <= b for a, b in zip(counts, counts[1:]))


def test_target_du_nho_thi_mot_dieu_bi_tach_lam_nhieu_chunk(docs):
    theo_dieu = chunk_corpus(docs, 5000, 7500, 150)
    that_nho = chunk_corpus(docs, 120, 180, 50)
    assert len(that_nho) > len(theo_dieu)


def test_hop_cac_chunk_phu_gan_het_body(docs):
    """Chỉ mất khoảng trắng giữa các đoạn, không được mất nội dung."""
    for doc in docs:
        chunks = chunk_document(doc, 600, 900, 150)
        union = interval_union([(c.char_start, c.char_end) for c in chunks])
        covered = sum(end - start for start, end in union)
        assert covered >= len(doc.body) * 0.95


def test_breadcrumb_mang_dieu_va_chuong(docs):
    chunks = chunk_document(docs[0], 1200, 1800, 150)
    with_article = [c for c in chunks if "Điều" in c.breadcrumb]
    assert with_article
    assert any("Chương" in c.breadcrumb for c in chunks)
    assert all(c.breadcrumb.startswith(docs[0].title) for c in chunks)


def test_cat_cung_tao_chunk_chong_lan(docs):
    """Khoản dài quá max bị cắt cứng có overlap — ca mà coverage phải xử lý đúng."""
    doc = docs[0]
    chunks = chunk_document(doc, 120, 150, 50)
    pairs = [(c.char_start, c.char_end) for c in chunks]
    assert any(
        b_start < a_end for (a_start, a_end), (b_start, b_end) in zip(pairs, pairs[1:])
    )


def test_chunk_id_duy_nhat(docs):
    chunks = chunk_corpus(docs, config.CHUNK_TARGET_CHARS, config.CHUNK_MAX_CHARS, 150)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_indexed_text_co_breadcrumb(docs):
    chunk = chunk_document(docs[0], 1200, 1800, 150)[0]
    assert chunk.indexed_text.startswith(chunk.breadcrumb)
    assert chunk.text in chunk.indexed_text
