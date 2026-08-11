"""Test quét lưới ngưỡng abstain và lớp nối dây CLI.

Lưới được chạy qua pipeline thật (chỉ thay bộ sinh và judge bằng kết quả định
sẵn), nên test bao luôn logic hai ngưỡng trong `answer_question` chứ không chỉ
bao phép số học tổng hợp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run as run_cli
from src import config
from src.calibrate import GridPoint, best_point, plateau, sweep, write_report
from src.evaluate import (
    GoldSpan,
    Question,
    abstain_stats,
    faithfulness_stats,
)
from src.generate import Attempt, Claim, Draft, answer_question
from src.index import build_index
from src.ingest import load_corpus
from src.retrieve import Retrieved
from src.verify import CheckA, CheckB, ClaimVerdict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return build_index(load_corpus(FIXTURE_DIR)).chunks[:3]


def question(qid: str, answerable: bool) -> Question:
    return Question(
        qid=qid,
        question=f"Câu hỏi {qid}?",
        type="factoid_1hop" if answerable else "unanswerable_oos",
        answerable=answerable,
        gold_spans=(GoldSpan("qc_99_2099", 100, 200),) if answerable else (),
    )


def retrievals_for(chunks, scores):
    return [
        [Retrieved(c, rank=i, score=1.0, rerank_score=s) for i, c in enumerate(chunks, 1)]
        for s in scores
    ]


def stub_pipeline(monkeypatch, questions, support_ratios):
    """Bộ sinh luôn ra một mệnh đề hợp lệ; judge trả support_ratio định sẵn.

    Tra theo NỘI DUNG câu hỏi chứ không theo thứ tự lời gọi: lưới chạy lại toàn
    bộ tập câu hỏi ở mỗi điểm, và số lượt sinh mỗi câu còn đổi theo ngưỡng.
    """
    ratios = {q.question: r for q, r in zip(questions, support_ratios)}

    def fake_draft(question, chunks, *, attempt, feedback, offline):
        return Draft((Claim("Mệnh đề.", (1,)),), False, "")

    def fake_grounding(question, texts, citations, chunks, *, offline):
        ratio = ratios[question]
        return CheckB((ClaimVerdict(1, ratio > 0, "lý do"),), ratio)

    monkeypatch.setattr("src.generate.draft_answer", fake_draft)
    monkeypatch.setattr("src.generate.check_grounding", fake_grounding)


def answer_result(abstained: bool, stage=None, ratios=(1.0,), retried=False):
    attempts = tuple(
        Attempt(i + 1, Draft((Claim("A", (1,)),), False, ""), CheckA(True, ()),
                CheckB((ClaimVerdict(1, r > 0, ""),), r))
        for i, r in enumerate(ratios)
    )
    from src.generate import AnswerResult

    return AnswerResult(
        question="Hỏi?",
        answer=config.ABSTAIN_TEXT if abstained else "Trả lời. [1]",
        citations=() if abstained else ("[1] nhãn",),
        abstained=abstained,
        abstain_stage=stage,
        attempts=attempts,
        retrieval_score=1.0,
    )


# --- Ma trận nhầm lẫn của abstain -----------------------------------------
def test_abstain_hoan_hao():
    questions = [question("q1", True), question("q2", False)]
    answers = [answer_result(False), answer_result(True, "check_b")]
    stats = abstain_stats(questions, answers)
    assert stats["f1"] == 1.0
    assert stats["wrong_abstain"] == 0 and stats["missed_abstain"] == 0


def test_khong_bao_gio_abstain_thi_f1_bang_0():
    """Hệ thống né bài toán không được thưởng."""
    questions = [question("q1", True), question("q2", False)]
    stats = abstain_stats(questions, [answer_result(False), answer_result(False)])
    assert stats["precision"] == 0.0 and stats["f1"] == 0.0
    assert stats["missed_abstain"] == 1


def test_luon_abstain_thi_recall_1_precision_thap():
    questions = [question("q1", True), question("q2", True), question("q3", False)]
    answers = [answer_result(True, "check_b")] * 3
    stats = abstain_stats(questions, answers)
    assert stats["recall"] == 1.0
    assert stats["precision"] == round(1 / 3, 4)
    assert stats["wrong_abstain"] == 2


def test_hai_loai_loi_duoc_tach_rieng():
    questions = [question("q1", True), question("q2", False)]
    answers = [answer_result(True, "check_b"), answer_result(False)]
    stats = abstain_stats(questions, answers)
    assert stats["wrong_abstain"] == 1 and stats["missed_abstain"] == 1


# --- Faithfulness ---------------------------------------------------------
def test_faithfulness_truoc_va_sau_verify():
    stats = faithfulness_stats([answer_result(False, ratios=(0.0, 1.0))])
    assert stats["support_ratio_first"] == 0.0
    assert stats["support_ratio_final"] == 1.0
    assert stats["n_retry"] == 1


def test_faithfulness_dem_tang_abstain():
    answers = [answer_result(True, "retrieve"), answer_result(True, "check_b"),
               answer_result(False)]
    stats = faithfulness_stats(answers)
    assert stats["n_answered"] == 1 and stats["n_abstained"] == 2
    assert stats["abstain_stage"] == {"check_b": 1, "retrieve": 1}


# --- Ngưỡng thật sự tách được hai lớp -------------------------------------
def test_tau_retrieve_tach_duoc_cau_ngoai_pham_vi(chunks, monkeypatch):
    """Câu unanswerable có điểm rerank thấp; ngưỡng đúng chỗ cho F1 = 1."""
    questions = [question("q1", True), question("q2", True),
                 question("q3", False), question("q4", False)]
    retrievals = retrievals_for(chunks, [1.0, 1.0, 1 / 3, 1 / 3])
    stub_pipeline(monkeypatch, questions, [1.0, 1.0, 1.0, 1.0])

    points = sweep(questions, retrievals, offline=True,
                   tau_retrieve_grid=(0.0, 0.67), tau_ground_grid=(0.0,))
    by_tau = {p.tau_retrieve: p for p in points}
    assert by_tau[0.0].abstain["f1"] == 0.0
    assert by_tau[0.67].abstain["f1"] == 1.0


def test_tau_ground_tach_duoc_cau_khong_co_can_cu(chunks, monkeypatch):
    questions = [question("q1", True), question("q2", False)]
    retrievals = retrievals_for(chunks, [1.0, 1.0])
    stub_pipeline(monkeypatch, questions, [1.0, 0.0])

    points = sweep(questions, retrievals, offline=True,
                   tau_retrieve_grid=(0.0,), tau_ground_grid=(0.0, 0.5))
    by_tau = {p.tau_ground: p for p in points}
    assert by_tau[0.0].abstain["missed_abstain"] == 1
    assert by_tau[0.5].abstain["f1"] == 1.0


def test_sweep_phu_het_luoi(chunks, monkeypatch):
    questions = [question("q1", True)]
    stub_pipeline(monkeypatch, questions, [1.0])
    points = sweep(questions, retrievals_for(chunks, [1.0]), offline=True,
                   tau_retrieve_grid=(0.0, 0.5), tau_ground_grid=(0.0, 0.5, 1.0))
    assert len(points) == 6


# --- Chọn điểm ------------------------------------------------------------
def grid_point(f1: float, missed: int, tau_r: float, tau_g: float,
               support: float = 0.0) -> GridPoint:
    return GridPoint(
        tau_retrieve=tau_r, tau_ground=tau_g,
        abstain={"f1": f1, "missed_abstain": missed, "precision": 0.0, "recall": 0.0,
                 "wrong_abstain": 0, "true_abstain": 0, "true_answer": 0, "n": 0},
        faithfulness={"support_ratio_final": support},
    )


def test_best_point_hoa_f1_thi_uu_tien_faithfulness_cao_hon():
    """Lưới phẳng ở F1 không có nghĩa mọi điểm tương đương — đã gặp thật."""
    kem = grid_point(1.0, 0, 0.0, 0.0, support=0.9597)
    tot = grid_point(1.0, 0, 0.0, 1.0, support=1.0)
    assert best_point([kem, tot]) is tot


def test_best_point_faithfulness_khong_lan_at_missed_abstain():
    """An toàn vẫn đứng trên groundedness."""
    risky = grid_point(1.0, 2, 0.0, 1.0, support=1.0)
    safe = grid_point(1.0, 0, 0.0, 0.0, support=0.5)
    assert best_point([risky, safe]) is safe


def test_best_point_lay_f1_cao_nhat():
    points = [grid_point(0.5, 0, 0.0, 0.0), grid_point(0.9, 2, 0.5, 0.5)]
    assert best_point(points).f1 == 0.9


def test_best_point_hoa_thi_uu_tien_it_bo_sot_abstain():
    """Trả lời câu không có căn cứ nguy hiểm hơn từ chối nhầm."""
    risky = grid_point(0.8, 3, 0.0, 0.0)
    safe = grid_point(0.8, 0, 0.5, 0.5)
    assert best_point([risky, safe]) is safe


def test_best_point_hoa_tiep_thi_lay_nguong_thap_hon():
    """Không siết chặt hơn mức dữ liệu biện minh được."""
    low = grid_point(0.8, 0, 0.0, 0.25)
    high = grid_point(0.8, 0, 0.5, 0.25)
    assert best_point([low, high]) is low


def test_best_point_luoi_rong_thi_raise():
    with pytest.raises(ValueError, match="Lưới rỗng"):
        best_point([])


def test_plateau_gom_moi_diem_cung_f1():
    points = [grid_point(0.8, 0, 0.0, 0.0), grid_point(0.8, 0, 0.5, 0.5),
              grid_point(0.2, 0, 1.0, 1.0)]
    assert len(plateau(points, best_point(points))) == 2


def test_write_report_ghi_du_luoi(tmp_path):
    points = [grid_point(0.8, 0, 0.0, 0.0), grid_point(0.2, 0, 0.5, 0.5)]
    path = write_report(points, best_point(points), tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["best"]["tau_retrieve"] == 0.0
    assert len(payload["grid"]) == 2
    assert payload["plateau_size"] == 1


# --- CLI ------------------------------------------------------------------
def test_cli_ask_thieu_cau_hoi_thi_bao_loi():
    with pytest.raises(SystemExit):
        run_cli.main(["ask"])


def test_cli_lenh_khong_hop_le():
    with pytest.raises(SystemExit):
        run_cli.main(["khong-co-lenh"])


def test_cli_index_ghi_manifest(tmp_path):
    code = run_cli.main(["index", "--corpus", str(FIXTURE_DIR), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "index" / "chunks.json").is_file()


def test_cli_cache_miss_tra_exit_code_2(tmp_path, monkeypatch):
    """--offline với cache rỗng phải báo rõ chứ không quăng traceback."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    code = run_cli.main([
        "ask", "Hỏi gì đó?", "--corpus", str(FIXTURE_DIR), "--offline"
    ])
    assert code == 2


def test_cli_eval_chay_duoc_arm_bm25(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps([{
            "qid": "q01", "question": "thời hạn lưu trữ", "type": "factoid_1hop",
            "answerable": True,
            "gold_spans": [{"doc_id": "qc_99_2099", "char_start": 100, "char_end": 200}],
            "gold_answer": "…",
        }], ensure_ascii=False),
        encoding="utf-8",
    )
    code = run_cli.main([
        "eval", "--corpus", str(FIXTURE_DIR), "--questions", str(questions),
        "--arm", "bm25", "--out", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "eval" / "retrieval.json").is_file()


# --- File demo 5 câu trả lời ----------------------------------------------
def demo_report(types_present) -> dict:
    return {
        "arm": "hybrid_rerank",
        "tau_retrieve": config.TAU_RETRIEVE,
        "tau_ground": config.TAU_GROUND,
        "questions": [
            {
                "qid": f"q{i:02d}",
                "type": kind,
                "answerable": not kind.startswith("unanswerable"),
                "question": f"Câu hỏi {kind}?",
                "answer": config.ABSTAIN_TEXT if kind.startswith("unanswerable") else "Trả lời. [1]",
                "citations": [] if kind.startswith("unanswerable") else ["[1] Điều 1 Luật X"],
                "abstained": kind.startswith("unanswerable"),
                "abstain_stage": "check_b" if kind.startswith("unanswerable") else None,
                "support_ratio_first": 1.0,
                "support_ratio_final": 1.0,
                "attempts": [{}],
            }
            for i, kind in enumerate(types_present, start=1)
        ],
    }


def test_demo_chon_du_nam_hanh_vi():
    from tools.demo_answers import SHOWCASE, pick

    report = demo_report([kind for kind, _ in SHOWCASE])
    assert [item["type"] for item in pick(report)] == [kind for kind, _ in SHOWCASE]


def test_demo_bo_qua_loai_khong_co_trong_bao_cao():
    from tools.demo_answers import pick

    assert [i["type"] for i in pick(demo_report(["factoid_1hop", "multihop"]))] == [
        "factoid_1hop",
        "multihop",
    ]


def test_demo_render_co_trich_dan_va_quyet_dinh():
    from tools.demo_answers import SHOWCASE, render

    text = render(demo_report([kind for kind, _ in SHOWCASE]))
    assert "[1] Điều 1 Luật X" in text
    assert "TỪ CHỐI TRẢ LỜI" in text
    assert "**Căn cứ.**" in text
    assert config.ABSTAIN_TEXT in text
