"""Test chỉ mục và bốn arm truy xuất. Không test nào chạm mạng.

Phía dense và rerank được kiểm bằng cách GIEO SẴN cache rồi chạy đúng đường dữ
liệu thật, chứ không monkeypatch hàm gọi API. Nhờ vậy test bao luôn cả hình dạng
cache key — thứ mà một stub sẽ bỏ lọt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.chunk import Chunk
from src.index import build_index, write_manifest
from src.ingest import load_corpus
from src.llm import CacheMiss, CallKey, _embed_key
from src.retrieve import (
    RERANK_SCHEMA,
    Retrieved,
    _parse_rerank,
    rank_by_score,
    rerank,
    retrieve,
    rrf_fuse,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(FIXTURE_DIR)


@pytest.fixture
def index(docs):
    return build_index(docs)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Cache rỗng, tách khỏi outputs/cache của repo."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def write_cache(key: CallKey, response) -> None:
    key.path.parent.mkdir(parents=True, exist_ok=True)
    key.path.write_text(
        json.dumps({"key_payload": key.payload, "response": response}, ensure_ascii=False),
        encoding="utf-8",
    )


def unit_vector(position: int) -> list[float]:
    """Vector one-hot đúng EMBED_DIM chiều: cosine giữa hai vector khác vị trí là 0."""
    values = [0.0] * config.EMBED_DIM
    values[position] = 1.0
    return values


def seed_embeddings(index, question: str, target_index: int) -> None:
    """Gieo embedding sao cho câu hỏi trùng hướng đúng một chunk."""
    for position, chunk in enumerate(index.chunks):
        write_cache(
            _embed_key(chunk.indexed_text, config.EMBED_TASK_DOCUMENT),
            unit_vector(position + 1),
        )
    write_cache(
        _embed_key(question, config.EMBED_TASK_QUERY), unit_vector(target_index + 1)
    )


def seed_rerank(question: str, chunks, scores: list[dict]) -> None:
    """Gieo cache reranker, dựng lại key đúng như src.retrieve.rerank."""
    import hashlib

    digest = hashlib.sha256(
        "\n".join(c.indexed_text for c in chunks).encode("utf-8")
    ).hexdigest()
    key = CallKey(
        task="rerank",
        provider="google",
        model=config.RERANK_MODEL,
        input={
            "question": question,
            "candidates": [c.chunk_id for c in chunks],
            "candidates_digest": digest,
        },
        params={
            "temperature": config.RERANK_TEMPERATURE,
            "max_output_tokens": config.RERANK_MAX_TOKENS,
            "thinking_budget": config.RERANK_THINKING_BUDGET,
            "response_mime_type": "application/json",
            "response_schema": RERANK_SCHEMA,
        },
    )
    write_cache(key, json.dumps({"scores": scores}, ensure_ascii=False))


# --- Chỉ mục --------------------------------------------------------------
def test_index_giu_du_moi_chunk(index, docs):
    from src.chunk import chunk_corpus

    assert len(index) == len(chunk_corpus(docs))
    assert len(set(index.chunk_ids)) == len(index)


def test_index_corpus_rong_thi_raise():
    with pytest.raises(ValueError, match="Corpus rỗng"):
        build_index([])


def test_bm25_dua_dung_khoan_len_dau(index):
    scores = index.bm25_scores("thời hạn lưu trữ dữ liệu")
    best = index.chunks[int(np.argmax(scores))]
    assert "lưu trữ" in best.indexed_text.lower()


def test_manifest_ghi_du_truong_va_doc_lai(index, tmp_path):
    path = write_manifest(index, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_chunks"] == len(index)
    assert payload["embed_dim"] == config.EMBED_DIM
    first = payload["chunks"][0]
    assert first["chunk_id"] == index.chunks[0].chunk_id
    assert first["citation_label"] == index.chunks[0].citation_label
    assert first["char_end"] > first["char_start"]


# --- Xếp hạng và hợp nhất -------------------------------------------------
def test_rank_by_score_pha_hoa_bang_chi_so():
    """BM25 trả 0.0 cho rất nhiều chunk cùng lúc; thứ tự ở đó phải tất định."""
    assert rank_by_score(np.array([0.0, 0.0, 0.0, 0.0]), 3) == [0, 1, 2]


def test_rank_by_score_giam_dan(index):
    scores = np.array([0.1, 5.0, 2.0])
    assert rank_by_score(scores, 3) == [1, 2, 0]


def test_rrf_cong_theo_hang_tu_1():
    fused = rrf_fuse([[7, 3], [3]], k=60)
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[3] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_uu_tien_chunk_co_mat_o_ca_hai_danh_sach():
    """Hạng 2 ở cả hai danh sách thắng hạng 1 chỉ ở một danh sách."""
    fused = rrf_fuse([[0, 1], [2, 1]], k=60)
    assert fused[1] > fused[0] and fused[1] > fused[2]


def test_rrf_khong_phu_thuoc_thang_diem():
    """Đổi thang điểm mà giữ nguyên thứ hạng thì kết quả hợp nhất không đổi.

    Đây là lý do RRF hợp nhất theo hạng: điểm BM25 không chặn trên còn cosine
    nằm trong [-1, 1], mọi cách chuẩn hoá về một thang đều tuỳ tiện.
    """
    small = rank_by_score(np.array([0.9, 0.8, 0.1]), 3)
    huge = rank_by_score(np.array([900.0, 800.0, 100.0]), 3)
    assert rrf_fuse([small]) == rrf_fuse([huge])


# --- Arm bm25: không cần mạng --------------------------------------------
def test_arm_bm25_chay_duoc_offline_voi_cache_rong(index, cache_dir):
    """Lười hoá embedding: arm thuần từ khoá không đụng tới một vector nào."""
    results = retrieve(index, "thời hạn lưu trữ", arm="bm25", top_k=3, offline=True)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.rerank_score is None for r in results)
    assert not cache_dir.exists()


def test_arm_dense_offline_cache_rong_thi_bao_cache_miss(index, cache_dir):
    with pytest.raises(CacheMiss, match="offline"):
        retrieve(index, "thời hạn lưu trữ", arm="dense", offline=True)


# --- Arm dense và hybrid --------------------------------------------------
def test_arm_dense_dung_cache_da_gieo(index, cache_dir):
    question = "câu hỏi thử nghiệm"
    seed_embeddings(index, question, target_index=4)
    results = retrieve(index, question, arm="dense", top_k=3, offline=True)
    assert results[0].chunk.chunk_id == index.chunks[4].chunk_id
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


def test_embedding_chi_nhung_mot_lan(index, cache_dir):
    question = "câu hỏi thử nghiệm"
    seed_embeddings(index, question, target_index=0)
    retrieve(index, question, arm="dense", offline=True)
    first = index.embeddings(offline=True)
    assert index.embeddings(offline=True) is first


def test_arm_hybrid_gop_ca_hai_tin_hieu(index, cache_dir):
    question = "thời hạn lưu trữ"
    dense_target = len(index) - 1
    seed_embeddings(index, question, target_index=dense_target)
    results = retrieve(index, question, arm="hybrid", top_k=5, offline=True)

    ids = [r.chunk.chunk_id for r in results]
    assert index.chunks[dense_target].chunk_id in ids
    top_bm25 = index.chunks[int(np.argmax(index.bm25_scores(question)))]
    assert top_bm25.chunk_id in ids
    assert all(r.fusion_score is not None for r in results)


# --- Rerank ---------------------------------------------------------------
def test_rerank_chuan_hoa_ve_0_1(index, cache_dir):
    question = "câu hỏi thử nghiệm"
    chunks = index.chunks[:3]
    seed_rerank(question, chunks, [{"id": 1, "score": 0}, {"id": 2, "score": 3}, {"id": 3, "score": 2}])
    assert rerank(question, chunks, offline=True) == [0.0, 1.0, pytest.approx(2 / 3)]


def test_rerank_ung_vien_khong_duoc_cham_nhan_0(index, cache_dir):
    """Im lặng không phải bằng chứng ủng hộ, và không thưởng reranker trả thiếu."""
    question = "câu hỏi thử nghiệm"
    chunks = index.chunks[:3]
    seed_rerank(question, chunks, [{"id": 2, "score": 3}])
    assert rerank(question, chunks, offline=True) == [0.0, 1.0, 0.0]


def test_rerank_bo_id_ngoai_pham_vi():
    """Model bịa id thì bỏ, không ánh xạ đại sang một chunk nào đó."""
    assert _parse_rerank('{"scores": [{"id": 99, "score": 3}, {"id": 2, "score": 1}]}', 3) == {2: 1}


def test_rerank_ket_diem_ve_khoang_hop_le():
    assert _parse_rerank('{"scores": [{"id": 1, "score": 9}, {"id": 2, "score": -4}]}', 2) == {
        1: config.RERANK_MAX_SCORE,
        2: 0,
    }


def test_rerank_nhan_ca_mang_tran():
    assert _parse_rerank('[{"id": 1, "score": 2}]', 1) == {1: 2}


def test_rerank_json_hong_thi_raise():
    with pytest.raises(ValueError, match="JSON hỏng"):
        _parse_rerank("không phải json", 3)


def test_rerank_khong_cham_duoc_gi_thi_raise():
    """Thất bại phải nhìn thấy được, không âm thầm rơi về thứ tự RRF."""
    with pytest.raises(ValueError, match="không chấm được"):
        _parse_rerank('{"scores": [{"id": 42, "score": 3}]}', 3)


def test_rerank_prompt_co_trang_thai_hieu_luc(index):
    from src.retrieve import _render_candidates

    expired = next(c for c in index.chunks if c.status == "expired")
    rendered = _render_candidates([expired])
    assert "expired" in rendered
    assert "hết hiệu lực" in rendered


# --- Arm hybrid_rerank ----------------------------------------------------
def _stub_rerank(monkeypatch, scores_by_chunk_id: dict[str, float]):
    def fake(question, chunks, *, offline):
        return [scores_by_chunk_id.get(c.chunk_id, 0.0) for c in chunks]

    monkeypatch.setattr("src.retrieve.rerank", fake)


def test_arm_rerank_dua_chunk_diem_cao_len_dau(index, cache_dir, monkeypatch):
    question = "thời hạn lưu trữ"
    seed_embeddings(index, question, target_index=0)
    chosen = index.chunks[-2]
    _stub_rerank(monkeypatch, {chosen.chunk_id: 1.0})

    results = retrieve(index, question, arm="hybrid_rerank", top_k=4, offline=True)
    assert results[0].chunk.chunk_id == chosen.chunk_id
    assert results[0].rerank_score == 1.0
    assert results[0].fusion_score is not None


def test_arm_rerank_hoa_diem_thi_giu_thu_tu_rrf(index, cache_dir, monkeypatch):
    """Thang 0-3 hoà rất nhiều; rơi về thứ tự tuỳ ý ở đó là vứt bỏ tầng RRF."""
    question = "thời hạn lưu trữ"
    seed_embeddings(index, question, target_index=0)
    _stub_rerank(monkeypatch, {})

    fusion = retrieve(index, question, arm="hybrid", top_k=5, offline=True)
    reranked = retrieve(index, question, arm="hybrid_rerank", top_k=5, offline=True)
    assert [r.chunk.chunk_id for r in reranked] == [r.chunk.chunk_id for r in fusion]


# --- Biên và lỗi đầu vào --------------------------------------------------
def test_arm_khong_hop_le(index):
    with pytest.raises(ValueError, match="arm không hợp lệ"):
        retrieve(index, "câu hỏi", arm="magic")


def test_cau_hoi_rong(index):
    with pytest.raises(ValueError, match="Câu hỏi rỗng"):
        retrieve(index, "   ", arm="bm25")


def test_top_k_lon_hon_so_chunk(index):
    results = retrieve(index, "dữ liệu", arm="bm25", top_k=999)
    assert len(results) == len(index)


def test_to_dict_du_de_dung_lai_offline(index):
    item = retrieve(index, "dữ liệu", arm="bm25", top_k=1)[0]
    record = item.to_dict()
    assert record["rank"] == 1
    assert record["chunk_id"] == item.chunk.chunk_id
    assert record["char_end"] > record["char_start"]
    assert record["rerank_score"] is None
