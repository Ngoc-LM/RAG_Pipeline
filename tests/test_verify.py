"""Test tầng kiểm chứng: check A, cách tính support_ratio, và parse output."""

from __future__ import annotations

from src.generate import parse_answer
from src.schema import Answer, Chunk, Claim
from src.verify import check_citations, support_ratio

CHUNKS = [
    Chunk("d1#000", "d1", "b", "nội dung một", 0, 12),
    Chunk("d1#001", "d1", "b", "nội dung hai", 12, 24),
]


def make_answer(*claims: Claim) -> Answer:
    return Answer(qid="q", question="?", claims=claims, abstained=False)


def test_citation_hop_le_thi_khong_bao_loi():
    answer = make_answer(Claim(1, "một khẳng định", ("d1#000",)))
    assert check_citations(answer, CHUNKS) == []


def test_citation_tro_toi_chunk_khong_co_trong_prompt():
    answer = make_answer(Claim(1, "bịa", ("d9#999",)))
    assert check_citations(answer, CHUNKS) == ["d9#999"]


def test_claim_khong_co_citation_tinh_la_loi():
    answer = make_answer(Claim(1, "không dẫn nguồn", ()))
    assert "claim 1: không có citation" in check_citations(answer, CHUNKS)


def test_support_ratio_day_du():
    answer = make_answer(Claim(1, "a", ("d1#000",)), Claim(2, "b", ("d1#001",)))
    results = [
        {"claim_id": 1, "supported": True},
        {"claim_id": 2, "supported": True},
    ]
    assert support_ratio(answer, results) == (1.0, [])


def test_support_ratio_mot_nua():
    answer = make_answer(Claim(1, "a", ("d1#000",)), Claim(2, "b", ("d1#001",)))
    results = [
        {"claim_id": 1, "supported": True},
        {"claim_id": 2, "supported": False},
    ]
    ratio, failed = support_ratio(answer, results)
    assert ratio == 0.5
    assert failed == ["b"]


def test_claim_judge_bo_sot_tinh_la_khong_duoc_ho_tro():
    """Im lặng không phải bằng chứng ủng hộ — nếu không, judge trả thiếu sẽ được thưởng."""
    answer = make_answer(Claim(1, "a", ("d1#000",)), Claim(2, "b", ("d1#001",)))
    ratio, failed = support_ratio(answer, [{"claim_id": 1, "supported": True}])
    assert ratio == 0.5
    assert failed == ["b"]


def test_judge_tra_rong_thi_ratio_bang_khong():
    answer = make_answer(Claim(1, "a", ("d1#000",)))
    assert support_ratio(answer, []) == (0.0, ["a"])


def test_parse_answer_binh_thuong():
    raw = '{"answerable": true, "claims": [{"text": "Hệ số là 0,7.", "citations": ["d1#000"]}]}'
    answer = parse_answer("q", "?", raw)
    assert not answer.abstained
    assert answer.claims[0].citations == ("d1#000",)
    assert answer.text == "Hệ số là 0,7. [d1#000]"


def test_parse_answer_khong_tra_loi_duoc_thi_abstain():
    answer = parse_answer("q", "?", '{"answerable": false, "claims": []}')
    assert answer.abstained


def test_parse_answer_json_hong_thi_abstain():
    assert parse_answer("q", "?", "không phải JSON").abstained


def test_parse_answer_claims_rong_thi_abstain():
    assert parse_answer("q", "?", '{"answerable": true, "claims": []}').abstained


def test_parse_answer_bo_qua_claim_rong():
    raw = (
        '{"answerable": true, "claims": ['
        '{"text": "   ", "citations": ["d1#000"]},'
        '{"text": "Có nội dung.", "citations": ["d1#001"]}]}'
    )
    answer = parse_answer("q", "?", raw)
    assert len(answer.claims) == 1
    assert answer.claims[0].text == "Có nội dung."


def test_text_gan_marker_cho_tung_cau():
    answer = make_answer(
        Claim(1, "Câu một.", ("d1#000",)),
        Claim(2, "Câu hai.", ("d1#000", "d1#001")),
    )
    assert answer.text == "Câu một. [d1#000] Câu hai. [d1#000][d1#001]"
