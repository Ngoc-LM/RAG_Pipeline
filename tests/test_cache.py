"""Test lớp cache: điều gì tạo ra key khác nhau, và hành vi lúc miss."""

from __future__ import annotations

import json

import pytest

import config
from src.llm import CacheMiss, CallKey, cached_call


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def make_key(**overrides) -> CallKey:
    base = {
        "task": "generate",
        "provider": "google",
        "model": "gemini-2.5-flash",
        "input": {"question": "abc"},
        "params": {"temperature": 0.0},
    }
    return CallKey(**{**base, **overrides})


def test_miss_online_thi_goi_ham_va_ghi_cache():
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "kết quả"

    assert cached_call(make_key(), fn, offline=False) == "kết quả"
    assert len(calls) == 1
    assert make_key().path.is_file()


def test_hit_thi_khong_goi_lai_ham():
    cached_call(make_key(), lambda: "một", offline=False)
    assert cached_call(make_key(), lambda: "hai", offline=False) == "một"


def test_miss_offline_thi_raise_cachemiss():
    with pytest.raises(CacheMiss) as excinfo:
        cached_call(make_key(), lambda: "x", offline=True)
    message = str(excinfo.value)
    assert make_key().digest in message
    assert "google" in message and "gemini-2.5-flash" in message and "generate" in message


def test_hit_offline_thi_doc_duoc():
    cached_call(make_key(), lambda: "đã lưu", offline=False)
    assert cached_call(make_key(), lambda: "không được gọi", offline=True) == "đã lưu"


def test_file_cache_luu_ca_key_payload():
    cached_call(make_key(), lambda: "x", offline=False)
    saved = json.loads(make_key().path.read_text(encoding="utf-8"))
    assert saved["key_payload"]["model"] == "gemini-2.5-flash"
    assert saved["key_payload"]["prompt_version"] == config.PROMPT_VERSION
    assert "created_at" in saved


def test_key_doi_khi_input_doi():
    assert make_key().digest != make_key(input={"question": "khác"}).digest


def test_key_doi_khi_params_doi():
    assert make_key().digest != make_key(params={"temperature": 0.7}).digest


def test_key_doi_khi_model_doi():
    assert make_key().digest != make_key(model="gemini-2.5-flash-lite").digest


def test_key_doi_khi_task_doi():
    assert make_key().digest != make_key(task="rerank").digest


def test_key_doi_khi_prompt_version_doi(monkeypatch):
    truoc = make_key().digest
    monkeypatch.setattr(config, "PROMPT_VERSION", config.PROMPT_VERSION + 1)
    assert make_key().digest != truoc


def test_key_on_dinh_khi_thu_tu_field_doi():
    """Canonical JSON sort_keys nên thứ tự khai báo không ảnh hưởng."""
    a = make_key(input={"x": 1, "y": 2})
    b = make_key(input={"y": 2, "x": 1})
    assert a.digest == b.digest


def test_key_phan_biet_dau_tieng_viet():
    assert make_key(input="phụ cấp").digest != make_key(input="phu cap").digest


def test_moi_task_ghi_vao_thu_muc_rieng():
    for task in ("embed", "generate", "rerank", "judge"):
        key = make_key(task=task)
        cached_call(key, lambda: task, offline=False)
        assert key.path.parent.name == task
