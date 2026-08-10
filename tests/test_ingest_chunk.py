"""Test parse cấu trúc và chia chunk trên fixture."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src import config
from src.chunk import Chunk, chunk_corpus, chunk_document, split_sentences
from src.ingest import ParseError, load_corpus, parse_body, parse_frontmatter

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE_DIR)


@pytest.fixture(scope="module")
def chunks(docs):
    return chunk_corpus(docs)


def article(docs, doc_id: str, article_no: int):
    document = next(d for d in docs if d.meta.doc_id == doc_id)
    return document, next(a for a in document.articles if a.article_no == article_no)


# --- offset ---------------------------------------------------------------
def test_char_offsets_exact(docs, chunks):
    """Bất biến nền móng: mất nó thì mọi gold_span và mọi trích dẫn đều vô nghĩa."""
    by_id = {d.meta.doc_id: d for d in docs}
    for document in docs:
        for art in document.articles:
            for clause in art.clauses:
                assert document.body[clause.char_start : clause.char_end] == clause.text
    for chunk in chunks:
        body = by_id[chunk.doc_id].body
        assert body[chunk.char_start : chunk.char_end] == chunk.text


def test_offset_lech_thi_raise_kem_context():
    """Thông báo lỗi phải đủ để lần ra chỗ hỏng ngay."""
    from src.ingest import Article, Clause, DocMeta, Document, _verify_offsets

    document = Document(
        meta=DocMeta("d1", "T", "luat", "2099-01-01", "2099-01-01", None, "active"),
        body="nội dung thật",
        articles=[Article(None, 7, "tiêu đề", [Clause(3, "sai lệch", 0, 5)])],
    )
    with pytest.raises(ParseError) as excinfo:
        _verify_offsets(document)
    message = str(excinfo.value)
    assert "d1" in message and "article_no=7" in message and "clause_no=3" in message


# --- ranh giới cấu trúc ---------------------------------------------------
def test_no_cross_article_merge(docs, chunks):
    """Không chunk nào chứa nội dung của hai Điều khác nhau."""
    for chunk in chunks:
        document = next(d for d in docs if d.meta.doc_id == chunk.doc_id)
        for other in document.articles:
            if other.article_no == chunk.article_no:
                continue
            for clause in other.clauses:
                overlap = min(chunk.char_end, clause.char_end) - max(
                    chunk.char_start, clause.char_start
                )
                assert overlap <= 0, (
                    f"{chunk.chunk_id} lấn sang Điều {other.article_no} khoản {clause.clause_no}"
                )


def test_cross_reference_not_parsed(docs):
    """"quy định tại khoản 2 Điều 3" nằm giữa đoạn không được tạo Article/Clause."""
    document, art = article(docs, "qc_99_2099", 1)
    assert [a.article_no for a in document.articles] == [1, 2, 3, 4, 5]

    reference = "quy định tại khoản 2 Điều 3"
    assert reference in document.body
    assert len(art.clauses) == 1
    assert art.clauses[0].clause_no == 0
    assert reference in art.clauses[0].text

    position = document.body.index(reference)
    starts = {c.char_start for a in document.articles for c in a.clauses}
    assert position not in starts


def test_tham_chieu_giua_dong_khong_cat_khoan(docs):
    """Điều 3 khoản 3 chứa "khoản 2 Điều 4" — vẫn phải là một khoản liền mạch."""
    _, art = article(docs, "qc_99_2099", 3)
    assert [c.clause_no for c in art.clauses] == [1, 2, 3]
    assert "khoản 2 Điều 4" in art.clauses[2].text


def test_diem_khong_tach_thanh_khoan_rieng(docs):
    """Điểm a) b) c) thuộc về khoản đang mở."""
    _, art = article(docs, "qc_99_2099", 2)
    assert [c.clause_no for c in art.clauses] == [1, 2, 3]
    clause_two = art.clauses[1]
    assert "a) dữ liệu về tình trạng sức khoẻ" in clause_two.text
    assert "c) dữ liệu định danh sinh trắc học" in clause_two.text


def test_dieu_mot_doan_thi_clause_no_bang_khong(docs):
    _, art = article(docs, "tt_88_2088", 1)
    assert [c.clause_no for c in art.clauses] == [0]


def test_chapter_gan_dung_cho_tung_dieu(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    chapters = {a.article_no: a.chapter for a in document.articles}
    assert chapters[1] == "Chương I. QUY ĐỊNH CHUNG"
    assert chapters[4] == "Chương II. TIẾP NHẬN VÀ LƯU TRỮ"


def test_so_khoan_nhay_coc_thi_canh_bao_va_khong_tach(caplog):
    body = "Điều 9. Thử nghiệm\n1. Khoản một.\n3. Dòng này nhảy cóc nên là nội dung thường.\n"
    with caplog.at_level(logging.WARNING):
        articles = parse_body(body, "d1")
    assert [c.clause_no for c in articles[0].clauses] == [1]
    assert "nhảy cóc" in articles[0].clauses[0].text
    assert any("bỏ qua ranh giới khoản" in r.message for r in caplog.records)


# --- frontmatter ----------------------------------------------------------
def test_frontmatter_parse_du_truong(docs):
    meta = next(d for d in docs if d.meta.doc_id == "tt_88_2088").meta
    assert meta.doc_type == "thong_tu"
    assert meta.status == "expired"
    assert meta.effective_to == "2099-04-30"
    assert meta.title == "Thông tư 88/2088/TT-GL hướng dẫn báo cáo kho dữ liệu thử nghiệm"


def test_effective_to_null_thanh_none(docs):
    meta = next(d for d in docs if d.meta.doc_id == "qc_99_2099").meta
    assert meta.effective_to is None
    assert meta.status == "active"


def test_thieu_key_bat_buoc_thi_raise():
    with pytest.raises(ParseError, match="thiếu key bắt buộc"):
        parse_frontmatter('---\ndoc_id: a\ntitle: "b"\n---\nnội dung\n', "x.txt")


def test_status_khong_hop_le_thi_raise():
    raw = (
        '---\ndoc_id: a\ntitle: "b"\ndoc_type: luat\nissued_date: "2099-01-01"\n'
        'effective_from: "2099-01-01"\nstatus: draft\n---\nnội dung\n'
    )
    with pytest.raises(ParseError, match="status="):
        parse_frontmatter(raw, "x.txt")


def test_khong_co_frontmatter_thi_raise():
    with pytest.raises(ParseError, match="frontmatter"):
        parse_frontmatter("Điều 1. Nội dung\n", "x.txt")


# --- citation label -------------------------------------------------------
def test_citation_label_format(docs, chunks):
    labels = {c.chunk_id: c.citation_label for c in chunks}
    title = "Quy chế quản lý kho dữ liệu thử nghiệm số 99/2099/QC-GL"
    thong_tu = "Thông tư 88/2088/TT-GL hướng dẫn báo cáo kho dữ liệu thử nghiệm"

    assert labels["qc_99_2099#a4#c1#p0"] == f"Điều 4 Khoản 1 {title}"
    assert labels["qc_99_2099#a5#c1-4#p0"] == f"Điều 5 Khoản 1-4 {title}"
    assert labels["qc_99_2099#a1#c0#p0"] == f"Điều 1 {title}"
    assert labels["tt_88_2088#a3#c0#p0"] == f"Điều 3 (đoạn mở đầu) {thong_tu}"


def test_lead_in_chi_ap_dung_khi_dieu_co_khoan_danh_so(chunks):
    """Điều không chia khoản vẫn dùng nhãn nguyên Điều, không phải (đoạn mở đầu)."""
    by_id = {c.chunk_id: c for c in chunks}
    assert by_id["tt_88_2088#a3#c0#p0"].is_lead_in is True
    assert by_id["tt_88_2088#a5#c0#p0"].is_lead_in is False
    assert "(đoạn mở đầu)" not in by_id["tt_88_2088#a5#c0#p0"].citation_label


# --- cắt khoản dài --------------------------------------------------------
def test_long_clause_split(docs):
    from src.intervals import interval_union

    document, art = article(docs, "qc_99_2099", 4)
    clause = art.clauses[0]
    assert clause.clause_no == 1
    assert len(clause.text) / config.CHARS_PER_TOKEN > config.CHUNK_MAX_TOKENS

    parts = [c for c in chunk_document(document) if c.chunk_id.startswith("qc_99_2099#a4#c1#")]
    assert len(parts) > 1

    for left, right in zip(parts, parts[1:]):
        assert left.char_end - right.char_start > 0, "hai phần liền kề phải chồng lấn"

    union = interval_union([(c.char_start, c.char_end) for c in parts])
    assert union == [(clause.char_start, clause.char_end)]


def test_moi_phan_khong_vuot_tran_qua_nhieu(docs):
    document, _ = article(docs, "qc_99_2099", 4)
    parts = [c for c in chunk_document(document) if c.chunk_id.startswith("qc_99_2099#a4#c1#")]
    assert all(p.n_tokens <= config.CHUNK_MAX_TOKENS for p in parts)


# --- gộp khoản ngắn -------------------------------------------------------
def test_gop_khoan_ngan_trong_cung_dieu(docs, chunks):
    """Điều 5 có bốn khoản đều ngắn -> gộp thành một chunk duy nhất."""
    merged = [c for c in chunks if c.doc_id == "qc_99_2099" and c.article_no == 5]
    assert len(merged) == 1
    assert merged[0].clause_range == "1-4"


def test_khong_gop_xuyen_document(chunks):
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"{chunk.doc_id}#")


# --- tách câu -------------------------------------------------------------
def test_khong_cat_tai_viet_tat():
    text = "Đơn vị đặt tại TP. Đà Nẵng thực hiện theo hướng dẫn. Câu sau là câu mới."
    spans = split_sentences(text, 0)
    assert len(spans) == 2
    assert "TP. Đà Nẵng" in text[spans[0][0] : spans[0][1]]


def test_khong_cat_tai_so_thu_tu():
    text = "Mã số 13/2023 được cấp theo quy định. Câu thứ hai."
    assert len(split_sentences(text, 0)) == 2


def test_span_cau_phu_lien_mach():
    text = "Câu một. Câu hai! Câu ba?"
    spans = split_sentences(text, 100)
    assert spans[0][0] == 100
    assert spans[-1][1] == 100 + len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_metadata_hieu_luc_di_theo_chunk(chunks):
    active = [c for c in chunks if c.doc_id == "qc_99_2099"]
    expired = [c for c in chunks if c.doc_id == "tt_88_2088"]
    assert all(c.status == "active" and c.effective_to is None for c in active)
    assert all(c.status == "expired" and c.effective_to == "2099-04-30" for c in expired)


def test_chunk_id_duy_nhat(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_n_tokens_khop_uoc_luong(chunks: list[Chunk]):
    for chunk in chunks:
        assert chunk.n_tokens == max(1, round(len(chunk.text) / config.CHARS_PER_TOKEN))


def test_indexed_text_khong_dung_cho_offset(chunks):
    """Offset đo trên `text`, không phải `indexed_text`.

    indexed_text gắn thêm nhãn trích dẫn vào đầu để đánh chỉ mục. Nếu ở bước sau
    có chỗ nào lỡ đo độ dài trên nó, coverage sẽ tính dư đúng bằng độ dài nhãn
    và mọi Cov@k đều sai lệch một cách âm thầm.
    """
    for chunk in chunks:
        assert chunk.char_end - chunk.char_start == len(chunk.text)
        assert chunk.char_end - chunk.char_start != len(chunk.indexed_text)


def test_khong_bao_gio_vua_gop_vua_cat(docs, chunks):
    """Mỗi chunk hoặc là part của đúng một khoản, hoặc là hợp các khoản nguyên vẹn."""
    for chunk in chunks:
        document = next(d for d in docs if d.meta.doc_id == chunk.doc_id)
        art = next(a for a in document.articles if a.article_no == chunk.article_no)
        touching = [
            c
            for c in art.clauses
            if min(chunk.char_end, c.char_end) - max(chunk.char_start, c.char_start) > 0
        ]
        assert touching, f"{chunk.chunk_id} không chạm khoản nào"

        if len(touching) == 1:
            clause = touching[0]
            assert clause.char_start <= chunk.char_start
            assert chunk.char_end <= clause.char_end
            assert chunk.clause_range == str(clause.clause_no)
            continue

        assert chunk.char_start == touching[0].char_start
        assert chunk.char_end == touching[-1].char_end
        for clause in touching:
            assert chunk.char_start <= clause.char_start and clause.char_end <= chunk.char_end, (
                f"{chunk.chunk_id} cắt ngang khoản {clause.clause_no} trong khi đang gộp"
            )


def test_khoan_bi_cat_khong_bao_gio_bi_gop(docs):
    """Khoản dài của Điều 4 bị cắt -> khoản 2 liền kề phải đứng riêng."""
    document, art = article(docs, "qc_99_2099", 4)
    ranges = {c.clause_range for c in chunk_document(document) if c.article_no == 4}
    assert ranges == {"1", "2"}
    assert not any("-" in r for r in ranges)


def test_doan_mo_dau_khong_gop_voi_khoan_danh_so(docs):
    """Gộp lead-in vào khoản 1 sẽ tạo clause_range '0-1', một nhãn vô nghĩa."""
    document, _ = article(docs, "tt_88_2088", 3)
    ranges = [c.clause_range for c in chunk_document(document) if c.article_no == 3]
    assert ranges == ["0", "1-4"]
