"""Test tầng sinh và hai tầng kiểm chứng.

Hai tính chất được canh kỹ nhất:
- Check A phải bắt được đúng những lỗi mà nó sinh ra để bắt, nên bộ parse của
  generate cố ý KHÔNG lọc trước mệnh đề hỏng.
- Lượt sinh lại phải có cache key khác lượt đầu. Ở temperature = 0, prompt y hệt
  cho output y hệt, nên một vòng sinh lại dùng chung key là vòng lặp vô nghĩa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.chunk import Chunk
from src.generate import (
    Attempt,
    Claim,
    Draft,
    _parse_draft,
    answer_question,
    draft_answer,
    render_answer,
)
from src.index import build_index
from src.ingest import load_corpus
from src.llm import CacheMiss, CallKey
from src.retrieve import Retrieved
from src.verify import (
    CheckA,
    CheckB,
    ClaimVerdict,
    _parse_verdicts,
    check_citations,
    check_grounding,
    grounded,
    render_context,
    retry_feedback,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return build_index(load_corpus(FIXTURE_DIR)).chunks[:4]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def write_cache(key: CallKey, response) -> None:
    key.path.parent.mkdir(parents=True, exist_ok=True)
    key.path.write_text(
        json.dumps({"key_payload": key.payload, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )


def as_results(chunks, rerank_score=None) -> list[Retrieved]:
    return [
        Retrieved(c, rank=i, score=1.0, rerank_score=rerank_score)
        for i, c in enumerate(chunks, start=1)
    ]


def generate_key(question: str, chunks, attempt: int, feedback: str) -> CallKey:
    """Dựng lại cache key đúng như src.generate.draft_answer."""
    return CallKey(
        task="generate",
        provider="google",
        model=config.GEN_MODEL,
        input={
            "question": question,
            "context": [c.chunk_id for c in chunks],
            "attempt": attempt,
            "feedback": feedback,
        },
        params={
            "temperature": config.GEN_TEMPERATURE,
            "max_output_tokens": config.GEN_MAX_TOKENS,
            "thinking_budget": config.GEN_THINKING_BUDGET,
            "response_mime_type": "application/json",
        },
    )


# --- Check A: cú pháp trích dẫn -------------------------------------------
def test_check_a_cho_qua_trich_dan_hop_le():
    assert check_citations([[1, 3], [2]], n_context=4).ok


def test_check_a_bat_menh_de_khong_co_trich_dan():
    result = check_citations([[1], []], n_context=4)
    assert not result.ok
    assert "mệnh đề 2 không có trích dẫn" in result.problems


def test_check_a_bat_trich_dan_ngoai_danh_sach():
    """Trích dẫn trỏ ra ngoài prompt là bằng chứng chắc chắn model bịa nguồn."""
    result = check_citations([[9]], n_context=4)
    assert not result.ok
    assert "ngoài danh sách" in result.problems[0]


def test_check_a_bat_trich_dan_so_0():
    assert not check_citations([[0]], n_context=4).ok


def test_check_a_bat_khong_co_menh_de_nao():
    result = check_citations([], n_context=4)
    assert not result.ok
    assert "không có mệnh đề nào" in result.problems


# --- Check B: parse verdict -----------------------------------------------
def test_parse_verdicts_doc_dung_dang():
    found = _parse_verdicts('{"verdicts": [{"claim_id": 1, "supported": true, "reason": "ok"}]}', 2)
    assert found[1].supported is True


def test_parse_verdicts_nhan_mang_tran():
    assert _parse_verdicts('[{"claim_id": 1, "supported": false, "reason": "x"}]', 1)[1].supported is False


def test_parse_verdicts_bo_claim_id_ngoai_pham_vi():
    assert _parse_verdicts('{"verdicts": [{"claim_id": 7, "supported": true}]}', 2) == {}


def test_parse_verdicts_json_hong_thi_raise():
    with pytest.raises(ValueError, match="JSON hỏng"):
        _parse_verdicts("không phải json", 1)


# --- Check B: chấm điểm ---------------------------------------------------
def judge_key(question, claim_texts, citations, chunks) -> CallKey:
    return CallKey(
        task="judge",
        provider="groq",
        model=config.JUDGE_MODEL,
        input={
            "question": question,
            "claims": list(claim_texts),
            "citations": [list(c) for c in citations],
            "context": [c.chunk_id for c in chunks],
        },
        params={
            "temperature": config.JUDGE_TEMPERATURE,
            "max_tokens": config.JUDGE_MAX_TOKENS,
            "response_format": "json_object",
        },
    )


def test_check_b_tinh_support_ratio(chunks, cache_dir):
    question, texts, citations = "Hỏi?", ["A", "B"], [(1,), (2,)]
    write_cache(
        judge_key(question, texts, citations, chunks),
        json.dumps({"verdicts": [
            {"claim_id": 1, "supported": True, "reason": "ok"},
            {"claim_id": 2, "supported": False, "reason": "vượt quá trích đoạn"},
        ]}),
    )
    result = check_grounding(question, texts, citations, chunks, offline=True)
    assert result.support_ratio == pytest.approx(0.5)
    assert len(result.rejected) == 1


def test_check_b_menh_de_judge_bo_qua_tinh_la_khong_duoc_ho_tro(chunks, cache_dir):
    """Im lặng không phải bằng chứng ủng hộ."""
    question, texts, citations = "Hỏi?", ["A", "B"], [(1,), (2,)]
    write_cache(
        judge_key(question, texts, citations, chunks),
        json.dumps({"verdicts": [{"claim_id": 1, "supported": True, "reason": "ok"}]}),
    )
    result = check_grounding(question, texts, citations, chunks, offline=True)
    assert result.support_ratio == pytest.approx(0.5)
    assert result.verdicts[1].reason == "judge không chấm mệnh đề này"


def test_check_b_khong_co_menh_de_thi_khong_goi_api(chunks, cache_dir):
    """Cache rỗng + offline mà không raise: chứng minh không có lời gọi nào."""
    result = check_grounding("Hỏi?", [], [], chunks, offline=True)
    assert result.support_ratio == 0.0
    assert not cache_dir.exists()


def test_grounded_theo_nguong():
    assert grounded(CheckB((), config.TAU_GROUND))
    assert not grounded(CheckB((), config.TAU_GROUND - 0.01))


# --- Feedback cho lượt sinh lại -------------------------------------------
def test_feedback_neu_ro_menh_de_bi_bac():
    check_b = CheckB((ClaimVerdict(1, False, "không có trong trích đoạn"),), 0.0)
    feedback = retry_feedback(CheckA(True, ()), check_b, ["Trần phạt là 20 lần."])
    assert "Trần phạt là 20 lần." in feedback
    assert "không có trong trích đoạn" in feedback
    assert "abstain" in feedback


def test_feedback_neu_ro_loi_trich_dan():
    feedback = retry_feedback(CheckA(False, ("mệnh đề 1 không có trích dẫn",)), None, ["A"])
    assert "mệnh đề 1 không có trích dẫn" in feedback


# --- Parse bộ sinh --------------------------------------------------------
def test_parse_draft_abstain():
    draft = _parse_draft('{"abstain": true, "reason": "không đủ căn cứ"}')
    assert draft.abstain and draft.claims == ()


def test_parse_draft_giu_nguyen_menh_de_thieu_trich_dan():
    """Lọc ở đây sẽ giấu mất đúng lỗi mà Check A sinh ra để bắt."""
    draft = _parse_draft('{"claims": [{"text": "A", "citations": []}]}')
    assert draft.claims[0].citations == ()
    assert not check_citations(draft.citations, 3).ok


def test_parse_draft_giu_nguyen_trich_dan_ngoai_pham_vi():
    draft = _parse_draft('{"claims": [{"text": "A", "citations": [99]}]}')
    assert draft.claims[0].citations == (99,)


def test_parse_draft_bo_menh_de_rong():
    draft = _parse_draft('{"claims": [{"text": "  ", "citations": [1]}, {"text": "B", "citations": [2]}]}')
    assert draft.texts == ("B",)


def test_parse_draft_json_hong_thi_raise():
    with pytest.raises(ValueError, match="JSON hỏng"):
        _parse_draft("<html>")


# --- Kết xuất -------------------------------------------------------------
def test_render_answer_gan_so_hieu_va_nhan(chunks):
    draft = Draft((Claim("Mệnh đề một.", (1, 2)), Claim("Mệnh đề hai.", (2,))), False, "")
    text, labels = render_answer(draft, chunks)
    assert text == "Mệnh đề một. [1][2] Mệnh đề hai. [2]"
    assert len(labels) == 2
    assert labels[0].startswith("[1] ") and chunks[0].citation_label in labels[0]


def test_render_context_danh_so_tu_1(chunks):
    rendered = render_context(chunks)
    assert rendered.startswith("[1] ")
    assert "[2] " in rendered


# --- Điều phối ------------------------------------------------------------
def stub(monkeypatch, drafts, support_ratios):
    """Thay bộ sinh và judge bằng chuỗi kết quả định sẵn cho từng lượt."""
    calls = {"n": 0}

    def fake_draft(question, chunks, *, attempt, feedback, offline):
        calls["n"] += 1
        return drafts[attempt - 1]

    def fake_grounding(question, texts, citations, chunks, *, offline):
        ratio = support_ratios.pop(0)
        return CheckB(
            tuple(
                ClaimVerdict(i, i == 1 and ratio > 0, "lý do")
                for i in range(1, len(texts) + 1)
            ),
            ratio,
        )

    monkeypatch.setattr("src.generate.draft_answer", fake_draft)
    monkeypatch.setattr("src.generate.check_grounding", fake_grounding)
    return calls


def test_khong_co_ket_qua_truy_xuat_thi_abstain(chunks):
    result = answer_question("Hỏi?", [], offline=True)
    assert result.abstained and result.abstain_stage == "retrieve"
    assert result.answer == config.ABSTAIN_TEXT


def test_diem_rerank_duoi_nguong_thi_abstain_truoc_khi_sinh(chunks, cache_dir):
    """Cache rỗng + offline mà không raise: chứng minh không tốn lượt sinh nào."""
    results = as_results(chunks, rerank_score=config.TAU_RETRIEVE - 0.1)
    result = answer_question("Hỏi?", results, offline=True)
    assert result.abstained and result.abstain_stage == "retrieve"
    assert result.attempts == ()
    assert not cache_dir.exists()


def test_arm_khong_co_diem_rerank_thi_khong_gac(chunks, monkeypatch):
    """Điểm BM25 không cùng thang với ngưỡng, nên không gác còn hơn gác sai."""
    stub(monkeypatch, [Draft((Claim("A", (1,)),), False, "")], [1.0])
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert not result.abstained


def test_model_tu_abstain(chunks, monkeypatch):
    stub(monkeypatch, [Draft((), True, "không đủ căn cứ")], [])
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert result.abstained and result.abstain_stage == "model"
    assert len(result.attempts) == 1


def test_duong_thanh_cong_mot_luot(chunks, monkeypatch):
    stub(monkeypatch, [Draft((Claim("Mệnh đề.", (1,)),), False, "")], [1.0])
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert not result.abstained
    assert result.answer == "Mệnh đề. [1]"
    assert result.support_ratio_first == 1.0 == result.support_ratio_final
    assert len(result.attempts) == 1


def test_check_a_fail_thi_khong_goi_judge(chunks, monkeypatch):
    """Tầng rẻ chặn trước tầng đắt: Check A hỏng thì không tốn lời gọi judge nào."""
    bad = Draft((Claim("Mệnh đề.", (99,)),), False, "")
    stub(monkeypatch, [bad, bad], [])
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert result.abstained and result.abstain_stage == "check_a"
    assert all(a.check_b is None for a in result.attempts)


def test_sinh_lai_cuu_duoc_luot_dau_hong(chunks, monkeypatch):
    stub(
        monkeypatch,
        [Draft((Claim("Sai.", (1,)),), False, ""), Draft((Claim("Đúng.", (2,)),), False, "")],
        [0.0, 1.0],
    )
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert not result.abstained
    assert result.answer == "Đúng. [2]"
    assert result.support_ratio_first == 0.0
    assert result.support_ratio_final == 1.0
    assert len(result.attempts) == config.MAX_GENERATE_ATTEMPTS


def test_hai_luot_deu_khong_du_can_cu_thi_abstain(chunks, monkeypatch):
    draft = Draft((Claim("Mệnh đề.", (1,)),), False, "")
    stub(monkeypatch, [draft, draft], [0.0, 0.0])
    result = answer_question("Hỏi?", as_results(chunks), offline=True)
    assert result.abstained and result.abstain_stage == "check_b"
    assert result.answer == config.ABSTAIN_TEXT
    assert len(result.attempts) == config.MAX_GENERATE_ATTEMPTS


def test_luot_sinh_lai_dung_cache_key_khac(chunks, cache_dir):
    """Ở temperature = 0, prompt y hệt cho output y hệt.

    Gieo cache CHỈ cho lượt 1 với một mệnh đề trích dẫn hỏng. Nếu lượt 2 dùng
    chung key thì nó đọc lại đúng câu trả lời vừa bị bác và không bao giờ raise;
    CacheMiss ở đây chính là bằng chứng lượt 2 là một lời gọi khác.
    """
    question = "Hỏi?"
    write_cache(
        generate_key(question, chunks, attempt=1, feedback=""),
        json.dumps({"abstain": False, "claims": [{"text": "A", "citations": [99]}]}),
    )
    assert draft_answer(
        question, chunks, attempt=1, feedback="", offline=True
    ).claims[0].citations == (99,)

    with pytest.raises(CacheMiss):
        answer_question(question, as_results(chunks), offline=True)


def test_to_dict_du_de_tai_hien(chunks, monkeypatch):
    stub(monkeypatch, [Draft((Claim("Mệnh đề.", (1,)),), False, "")], [1.0])
    record = answer_question("Hỏi?", as_results(chunks), offline=True).to_dict()
    assert record["answer"] == "Mệnh đề. [1]"
    assert record["support_ratio_final"] == 1.0
    assert record["attempts"][0]["check_a"]["ok"] is True
    assert record["attempts"][0]["check_b"]["support_ratio"] == 1.0
