"""Test công cụ gán nhãn. Sai ở đây thì gold set sai theo, nên phải có test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest import load_corpus
from tools.annotate import (
    article_span,
    cmd_validate,
    excerpt,
    find_matches,
    fold,
    locate,
    short_label,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE_DIR)


def write_questions(tmp_path: Path, spans: list[dict]) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps([{"qid": "q01", "gold_spans": spans}], ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# --- fold + ánh xạ offset -------------------------------------------------
def test_fold_giu_dung_do_dai_khi_con_dau():
    folded, positions = fold("Thời Hạn", no_accent=False)
    assert folded == "thời hạn"
    assert positions == list(range(len("Thời Hạn")))


def test_fold_bo_dau_van_anh_xa_nguoc_dung():
    text = "Điều đặc thù"
    folded, positions = fold(text, no_accent=True)
    assert folded == "dieu dac thu"
    assert len(folded) == len(positions)
    for index, character in enumerate(folded):
        if character != " ":
            assert text[positions[index]] != " "


def test_find_matches_khong_phan_biet_hoa_thuong(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    spans = find_matches(document, "THỜI HẠN LƯU TRỮ", no_accent=False)
    assert spans
    for start, end in spans:
        assert document.body[start:end].lower() == "thời hạn lưu trữ"


def test_find_matches_bo_dau_tra_ve_offset_tren_body_goc(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    spans = find_matches(document, "thoi han luu tru", no_accent=True)
    assert spans
    start, end = spans[0]
    assert document.body[start:end] == "thời hạn lưu trữ"


def test_find_matches_khong_bo_dau_thi_khong_khop(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    assert find_matches(document, "thoi han luu tru", no_accent=False) == []


def test_find_matches_chuoi_rong():
    document = load_corpus(FIXTURE_DIR)[0]
    assert find_matches(document, "", no_accent=False) == []


# --- định vị --------------------------------------------------------------
def test_locate_tra_ve_dung_khoan(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    art = next(a for a in document.articles if a.article_no == 5)
    clause = art.clauses[1]
    found = locate(document, clause.char_start + 3)
    assert found is not None
    assert found[0].article_no == 5
    assert found[1].clause_no == clause.clause_no


def test_locate_ngoai_khoan_tra_ve_none(docs):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    assert locate(document, 0) is None


def test_short_label_doan_mo_dau(docs):
    document = next(d for d in docs if d.meta.doc_id == "tt_88_2088")
    art = next(a for a in document.articles if a.article_no == 3)
    assert short_label(art, art.clauses[0]) == "Điều 3 (đoạn mở đầu)"
    assert short_label(art, art.clauses[1]) == "Điều 3 Khoản 1"


def test_excerpt_cat_va_highlight():
    text = "x" * 500 + "MỤC TIÊU" + "y" * 500
    rendered = excerpt(text, 500, 508)
    assert "»MỤC TIÊU«" in rendered
    assert rendered.startswith("…") and rendered.endswith("…")


def test_excerpt_ngan_thi_khong_cat():
    rendered = excerpt("Một câu ngắn.", 4, 7)
    assert rendered == "Một »câu« ngắn."


# --- validate -------------------------------------------------------------
def test_validate_span_hop_le(docs, tmp_path, capsys):
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    art = next(a for a in document.articles if a.article_no == 4)
    clause = art.clauses[0]
    path = write_questions(
        tmp_path,
        [{"doc_id": "qc_99_2099", "char_start": clause.char_start, "char_end": clause.char_end}],
    )
    assert cmd_validate(docs, path) == 0
    assert "0 lỗi" in capsys.readouterr().out


def test_validate_bat_span_ngoai_pham_vi(docs, tmp_path, capsys):
    path = write_questions(
        tmp_path, [{"doc_id": "qc_99_2099", "char_start": 10, "char_end": 10**7}]
    )
    assert cmd_validate(docs, path) == 1
    assert "ngoài phạm vi" in capsys.readouterr().out


def test_validate_bat_span_dao_nguoc(docs, tmp_path):
    path = write_questions(
        tmp_path, [{"doc_id": "qc_99_2099", "char_start": 900, "char_end": 400}]
    )
    assert cmd_validate(docs, path) == 1


def test_validate_bat_doc_id_khong_ton_tai(docs, tmp_path, capsys):
    path = write_questions(tmp_path, [{"doc_id": "khong_co", "char_start": 0, "char_end": 10}])
    assert cmd_validate(docs, path) == 1
    assert "không có trong corpus" in capsys.readouterr().out


def test_validate_bat_span_vat_qua_hai_dieu(docs, tmp_path, capsys):
    """Dấu hiệu gán nhầm ranh giới: span bắt đầu ở Điều này, kết ở Điều khác."""
    document = next(d for d in docs if d.meta.doc_id == "qc_99_2099")
    first = article_span(next(a for a in document.articles if a.article_no == 1))
    second = article_span(next(a for a in document.articles if a.article_no == 2))
    assert first is not None and second is not None

    path = write_questions(
        tmp_path, [{"doc_id": "qc_99_2099", "char_start": first[0], "char_end": second[1]}]
    )
    assert cmd_validate(docs, path) == 1
    assert "không nằm trọn trong một Điều" in capsys.readouterr().out


def test_validate_in_preview_de_mat_thuong_xac_nhan(docs, tmp_path, capsys):
    document = next(d for d in docs if d.meta.doc_id == "tt_88_2088")
    art = next(a for a in document.articles if a.article_no == 4)
    clause = art.clauses[0]
    path = write_questions(
        tmp_path,
        [{"doc_id": "tt_88_2088", "char_start": clause.char_start, "char_end": clause.char_end}],
    )
    cmd_validate(docs, path)
    output = capsys.readouterr().out
    assert "Điều 4 Khoản 1" in output
    assert clause.text[:40].replace("\n", " ") in output
