"""Test tầng đo truy xuất.

Test quan trọng nhất ở đây là `test_coverage_bat_bien_voi_chunk_size`: nó chính
là lý do repo này bỏ nhãn "chunk gold" để đo bằng coverage. Nếu nó fail thì mọi
bảng ablation theo kích thước chunk đều vô nghĩa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.chunk import Chunk
from src.evaluate import (
    GoldSpan,
    Question,
    QuestionResult,
    SpanResult,
    aggregate,
    check_questions,
    coverage_curve,
    evaluate_question,
    first_reaching,
    load_questions,
    run,
    write_report,
)
from src.index import build_index
from src.ingest import load_corpus
from src.retrieve import Retrieved

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


def make_chunk(doc_id: str, start: int, end: int, suffix: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{start}-{end}{suffix}",
        doc_id=doc_id,
        doc_title="Văn bản thử",
        doc_type="luat",
        status="active",
        effective_from="2026-01-01",
        effective_to=None,
        chapter=None,
        article_no=1,
        article_title="Điều thử",
        clause_range="1",
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        n_tokens=max(1, (end - start) // 4),
    )


def as_results(chunks) -> list[Retrieved]:
    return [Retrieved(c, rank=i, score=1.0) for i, c in enumerate(chunks, start=1)]


# --- Coverage curve -------------------------------------------------------
def test_coverage_curve_don_dieu_khong_giam():
    span = GoldSpan("d1", 0, 100)
    ranked = [make_chunk("d1", 0, 20), make_chunk("d1", 20, 60), make_chunk("d1", 60, 100)]
    curve = coverage_curve(span, ranked)
    assert curve == [pytest.approx(0.2), pytest.approx(0.6), pytest.approx(1.0)]
    assert all(b >= a for a, b in zip(curve, curve[1:]))


def test_coverage_chi_tinh_chunk_cung_document():
    """Trục ký tự là trục riêng của từng văn bản.

    Chunk [0, 100) của văn bản khác chồng lên span về mặt số học nhưng không phủ
    một chữ nào của nó; đếm vào là tạo ra phần giao hoàn toàn ảo.
    """
    span = GoldSpan("d1", 0, 100)
    assert coverage_curve(span, [make_chunk("d2", 0, 100)]) == [0.0]


def test_coverage_bat_bien_voi_chunk_size():
    """Chia một chunk thành hai chunk kề nhau không đổi hợp, nên không đổi coverage.

    Đây là tính chất mà nhãn "chunk gold" không thể có: mọi ngưỡng overlap ở mức
    chunk đều đổi kết quả khi chunk to nhỏ khác đi.
    """
    span = GoldSpan("d1", 10, 90)
    tho = [make_chunk("d1", 0, 100)]
    min_ = [make_chunk("d1", 0, 25, "a"), make_chunk("d1", 25, 50, "b"),
            make_chunk("d1", 50, 75, "c"), make_chunk("d1", 75, 100, "d")]
    assert coverage_curve(span, tho)[-1] == coverage_curve(span, min_)[-1] == 1.0


def test_coverage_chunk_chong_lan_khong_vuot_qua_1():
    """Cộng độ dài text từng chunk sẽ ra 1.4; hợp khoảng thì đúng 1.0."""
    span = GoldSpan("d1", 0, 100)
    ranked = [make_chunk("d1", 0, 70, "a"), make_chunk("d1", 30, 100, "b")]
    assert coverage_curve(span, ranked)[-1] == pytest.approx(1.0)


def test_coverage_span_khong_duoc_cham_toi():
    assert coverage_curve(GoldSpan("d1", 0, 50), [make_chunk("d1", 200, 300)]) == [0.0]


# --- r* -------------------------------------------------------------------
def test_first_reaching_lay_hang_nho_nhat():
    assert first_reaching([0.1, 0.5, 0.8, 0.9], 0.8) == 3


def test_first_reaching_khong_bao_gio_dat():
    assert first_reaching([0.1, 0.2], 0.8) is None


def test_first_reaching_dat_ngay_hang_1():
    assert first_reaching([1.0], 0.8) == 1


# --- strict vs any --------------------------------------------------------
def multihop_result(r_first: int | None, r_second: int | None) -> QuestionResult:
    def span_result(rank: int | None) -> SpanResult:
        curve = tuple(1.0 if rank is not None and k >= rank else 0.0 for k in range(1, 11))
        return SpanResult(GoldSpan("d1", 0, 10), curve, rank)

    return QuestionResult(
        qid="q", type="multihop",
        spans=(span_result(r_first), span_result(r_second)),
        ranked_chunk_ids=(),
    )


def test_strict_doi_moi_span_deu_dat():
    result = multihop_result(2, 7)
    assert result.any_hit(2) is True
    assert result.strict_hit(2) is False
    assert result.strict_hit(7) is True


def test_r_star_strict_la_span_cham_nhat():
    assert multihop_result(2, 7).r_star_strict == 7
    assert multihop_result(2, 7).mrr_strict == pytest.approx(1 / 7)


def test_mot_span_khong_bao_gio_dat_thi_strict_that_bai():
    result = multihop_result(2, None)
    assert result.r_star_strict is None
    assert result.mrr_strict == 0.0
    assert result.any_hit(10) is True


def test_mean_cov_trung_binh_trong_cau_truoc():
    """Câu 2 span không được đếm trọng số đôi so với câu 1 span."""
    result = multihop_result(1, None)
    assert result.mean_cov(5) == pytest.approx(0.5)


# --- Tổng hợp -------------------------------------------------------------
def test_aggregate_rong():
    assert aggregate([]) == {"n": 0}


def test_aggregate_trung_binh_theo_cau():
    results = [multihop_result(1, 1), multihop_result(1, None)]
    stats = aggregate(results)
    assert stats["n"] == 2
    assert stats["recall_strict"]["5"] == pytest.approx(0.5)
    assert stats["recall_any"]["5"] == pytest.approx(1.0)
    assert stats["mrr_strict"] == pytest.approx(0.5)


# --- Kiểm dữ liệu đầu vào -------------------------------------------------
@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE_DIR)


def question(**kwargs) -> Question:
    base = {
        "qid": "q01", "question": "Câu hỏi thử.", "type": "factoid_1hop",
        "answerable": True, "gold_spans": (GoldSpan("qc_99_2099", 0, 50),),
    }
    return Question(**{**base, **kwargs})


def test_check_bat_doc_id_khong_ton_tai(docs):
    bad = question(gold_spans=(GoldSpan("khong_co", 0, 10),))
    with pytest.raises(ValueError, match="không có trong corpus"):
        check_questions([bad], docs)


def test_check_bat_span_ngoai_pham_vi(docs):
    bad = question(gold_spans=(GoldSpan("qc_99_2099", 0, 10**7),))
    with pytest.raises(ValueError, match="ngoài phạm vi"):
        check_questions([bad], docs)


def test_check_bat_answerable_thieu_span(docs):
    with pytest.raises(ValueError, match="answerable nhưng không có gold_span"):
        check_questions([question(gold_spans=())], docs)


def test_check_bat_unanswerable_thua_span(docs):
    with pytest.raises(ValueError, match="unanswerable nhưng vẫn có gold_span"):
        check_questions([question(answerable=False)], docs)


def test_check_cho_qua_bo_hop_le(docs):
    check_questions([question(), question(qid="q02", answerable=False, gold_spans=())], docs)


def test_load_questions_doc_dung_schema(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps([{
            "qid": "q01", "question": "Hỏi?", "type": "multihop", "answerable": True,
            "gold_spans": [{"doc_id": "d1", "char_start": 5, "char_end": 9}],
            "gold_answer": "Đáp.",
        }], ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = load_questions(path)
    assert loaded[0].gold_spans[0].interval == (5, 9)
    assert loaded[0].type == "multihop"


# --- Đường chạy đầy đủ ----------------------------------------------------
def test_evaluate_question_tren_chunk_that(docs):
    index = build_index(docs)
    target = index.chunks[3]
    span = GoldSpan(target.doc_id, target.char_start, target.char_end)
    result = evaluate_question(
        question(gold_spans=(span,)), as_results([target] + list(index.chunks[:3]))
    )
    assert result.spans[0].r_star == 1
    assert result.mrr_strict == 1.0


def test_run_bo_qua_arm_thieu_cache(docs, tmp_path, monkeypatch):
    """Cache rỗng thì bm25 vẫn ra bảng, dense bị đánh dấu bỏ qua."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    index = build_index(docs)
    questions = [question(gold_spans=(GoldSpan("qc_99_2099", 100, 200),))]

    report, skipped = run(index, questions, arms=["bm25", "dense"], offline=True)
    assert "bm25" in report
    assert "dense" not in report
    assert "dense" in skipped and "Cache miss" in skipped["dense"]


def test_write_report_ghi_du_ngu_canh(docs, tmp_path):
    index = build_index(docs)
    questions = [
        question(gold_spans=(GoldSpan("qc_99_2099", 100, 200),)),
        question(qid="q02", type="unanswerable_oos", answerable=False, gold_spans=()),
    ]
    report, skipped = run(index, questions, arms=["bm25"], offline=True)
    path = write_report(report, questions, skipped, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["theta_coverage"] == config.THETA_COVERAGE
    assert payload["n_questions"] == 2 and payload["n_unanswerable"] == 1
    assert payload["arms"]["bm25"]["overall"]["n"] == 1
    assert payload["arms"]["bm25"]["questions"][0]["ranked_chunk_ids"]


def test_unanswerable_khong_vao_metric_truy_xuat(docs):
    index = build_index(docs)
    questions = [
        question(),
        question(qid="q02", type="unanswerable_oos", answerable=False, gold_spans=()),
    ]
    report, _ = run(index, questions, arms=["bm25"], offline=True)
    assert report["bm25"]["overall"]["n"] == 1
