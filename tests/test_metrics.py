"""Test metric coverage: bất biến với chunk_size, và MRR theo hạng đủ căn cứ."""

from __future__ import annotations

import pytest

from src.metrics import (
    abstain_scores,
    coverage_curve,
    first_rank_reaching,
    question_metrics,
)
from src.schema import Chunk, Question, Span

THETA = 0.8


def make_chunk(chunk_id: str, doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        breadcrumb="",
        text="x" * (end - start),
        char_start=start,
        char_end=end,
    )


def make_question(spans: list[Span], qtype: str = "factoid_1hop") -> Question:
    return Question(
        qid="q01",
        question="?",
        type=qtype,  # type: ignore[arg-type]
        answerable=True,
        gold_spans=tuple(spans),
        gold_answer="",
    )


def test_curve_don_dieu_khong_giam():
    span = Span("d1", 0, 100)
    ranked = [
        make_chunk("c1", "d1", 0, 20),
        make_chunk("c2", "d1", 20, 60),
        make_chunk("c3", "d1", 60, 100),
    ]
    curve = coverage_curve(span, ranked)
    assert curve == pytest.approx([0.2, 0.6, 1.0])
    assert all(b >= a for a, b in zip(curve, curve[1:]))


def test_chunk_khac_doc_khong_dong_gop():
    span = Span("d1", 0, 100)
    ranked = [make_chunk("c1", "d2", 0, 100), make_chunk("c2", "d1", 0, 80)]
    assert coverage_curve(span, ranked) == pytest.approx([0.0, 0.8])


def test_span_can_nhieu_chunk_moi_du_nguong():
    """Không chunk nào một mình đạt 0.8 — chỉ hợp lại mới đủ."""
    span = Span("d1", 0, 100)
    ranked = [
        make_chunk("c1", "d1", 0, 50),
        make_chunk("c2", "d1", 50, 90),
    ]
    curve = coverage_curve(span, ranked)
    assert first_rank_reaching(curve, THETA) == 2


def test_bat_bien_voi_chunk_size():
    """Cùng vùng văn bản, chia thô hay chia mịn đều cho coverage như nhau."""
    span = Span("d1", 100, 400)
    tho = [make_chunk("a", "d1", 0, 600)]
    min_ = [make_chunk(f"b{i}", "d1", s, s + 50) for i, s in enumerate(range(0, 600, 50))]
    assert coverage_curve(span, tho)[-1] == pytest.approx(1.0)
    assert coverage_curve(span, min_)[-1] == pytest.approx(1.0)


def test_chunk_nho_khong_bi_thoi_phong():
    """Chunk 50 ký tự so với span 1000 ký tự chỉ phủ 5%, không thành 'gold'."""
    span = Span("d1", 0, 1000)
    ranked = [make_chunk("c1", "d1", 0, 50)]
    assert first_rank_reaching(coverage_curve(span, ranked), THETA) is None


def test_multihop_strict_lay_hang_muon_nhat():
    q = make_question([Span("d1", 0, 100), Span("d2", 0, 100)], "multihop")
    ranked = [
        make_chunk("c1", "d1", 0, 100),
        make_chunk("c2", "d3", 0, 100),
        make_chunk("c3", "d2", 0, 100),
    ]
    row = question_metrics(q, ranked, (1, 3), THETA)
    assert row["first_rank_any"] == 1
    assert row["first_rank_strict"] == 3
    assert row["rr_strict"] == pytest.approx(1 / 3)
    assert row["per_k"]["1"]["recall_strict"] == 0.0
    assert row["per_k"]["1"]["recall_any"] == 1.0
    assert row["per_k"]["3"]["recall_strict"] == 1.0


def test_multihop_thieu_mot_span_thi_strict_bang_khong():
    q = make_question([Span("d1", 0, 100), Span("d2", 0, 100)], "multihop")
    ranked = [make_chunk("c1", "d1", 0, 100)]
    row = question_metrics(q, ranked, (1,), THETA)
    assert row["first_rank_strict"] is None
    assert row["rr_strict"] == 0.0
    assert row["rr_any"] == 1.0


def test_ket_qua_rong_khong_no():
    q = make_question([Span("d1", 0, 100)])
    row = question_metrics(q, [], (1, 5), THETA)
    assert row["rr_strict"] == 0.0
    assert row["per_k"]["5"]["mean_cov"] == 0.0


def test_abstain_f1():
    answerable = make_question([Span("d1", 0, 10)])
    unanswerable = Question("q02", "?", "unanswerable_oos", False, (), "")
    scores = abstain_scores(
        [(answerable, False), (unanswerable, True), (unanswerable, False)]
    )
    assert scores["abstain_precision"] == 1.0
    assert scores["abstain_recall"] == 0.5
    assert scores["abstain_f1"] == pytest.approx(2 / 3)
