"""Lớp gọi API dùng chung: cache write-through, backoff 429, chuẩn hoá embedding.

Cache luôn bật. `offline` không phải một nhánh code riêng, nó chỉ đổi hành vi
lúc cache miss: online thì gọi API rồi ghi cache, offline thì raise CacheMiss.
Nhờ vậy đường đi của dữ liệu ở hai chế độ là một, không có nhánh nào chỉ chạy
lúc demo mà chưa từng chạy lúc thật.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np

from src import config

RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_STATUS_ATTRS: Final[tuple[str, ...]] = ("status_code", "code", "status")
_sdk_transport_types: tuple[type[BaseException], ...] | None = None


class CacheMiss(RuntimeError):
    """Chạy --offline nhưng lời gọi này chưa có trong cache."""


class MissingAPIKey(RuntimeError):
    """Cần gọi API thật nhưng biến môi trường key chưa được set."""


def load_dotenv(path: Path | None = None) -> None:
    """Nạp .env vào os.environ, không ghi đè biến đã có sẵn."""
    env_path = path or config.ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CallKey:
    """Danh tính của một lời gọi API.

    Băm payload có cấu trúc chứ không băm chuỗi prompt đã render: đổi cách trình
    bày prompt mà logic không đổi thì không nên vứt cache. Bù lại, template đổi
    mà payload không đổi sẽ KHÔNG tự phát hiện được — đó là việc của
    config.PROMPT_VERSION, tăng tay khi sửa template.
    """

    task: str
    provider: str
    model: str
    input: Any
    params: dict[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "task": self.task,
            "input": self.input,
            "params": self.params,
            "prompt_version": config.PROMPT_VERSION,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.payload).encode("utf-8")).hexdigest()

    @property
    def path(self) -> Path:
        return config.CACHE_DIR / self.task / f"{self.digest}.json"

    def preview(self) -> str:
        return _canonical(self.input)[:200]


def _transport_types() -> tuple[type[BaseException], ...]:
    """Lỗi tầng vận chuyển của hai SDK, nạp trễ để không kéo SDK vào lúc import."""
    global _sdk_transport_types
    if _sdk_transport_types is None:
        collected: list[type[BaseException]] = []
        try:
            import groq

            collected.extend([groq.APIConnectionError, groq.APITimeoutError])
        except Exception:  # noqa: BLE001 - thiếu SDK thì bỏ qua
            pass
        _sdk_transport_types = tuple(collected)
    return _sdk_transport_types


def _status_code(exc: BaseException) -> int | None:
    """Đọc mã HTTP từ exception của SDK.

    google-genai đặt ở `.code`, groq ở `.status_code`. Không suy ra từ nội dung
    message: chuỗi "max_output_tokens 500" sẽ khớp marker "500" và biến một lỗi
    tham số vĩnh viễn thành sáu lần thử lại vô ích.
    """
    for attribute in _STATUS_ATTRS:
        value = getattr(exc, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, *_transport_types())):
        return True
    code = _status_code(exc)
    return code is not None and code in RETRYABLE_STATUS


def with_backoff(fn: Callable[[], Any], *, what: str) -> Any:
    """Thử lại với backoff luỹ thừa + jitter cho 429 và lỗi mạng tạm thời."""
    last: Exception | None = None
    for attempt in range(config.RETRY_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - phân loại lại bằng is_retryable
            last = exc
            if not is_retryable(exc):
                raise
            if attempt == config.RETRY_MAX_ATTEMPTS - 1:
                break
            sleep = min(
                config.RETRY_BASE_SECONDS * (2**attempt), config.RETRY_MAX_SLEEP_SECONDS
            )
            sleep *= 1.0 + random.uniform(0.0, config.RETRY_JITTER)
            print(f"[backoff] {what}: {type(exc).__name__}, chờ {sleep:.1f}s "
                  f"(lần {attempt + 1}/{config.RETRY_MAX_ATTEMPTS})")
            time.sleep(sleep)
    raise RuntimeError(f"Hết số lần thử cho {what}: {last}") from last


def cached_call(key: CallKey, fn: Callable[[], Any], offline: bool) -> Any:
    """Hàm cache duy nhất cho cả embed, generate, rerank và judge."""
    path = key.path
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    if offline:
        raise CacheMiss(
            "Cache miss ở chế độ --offline.\n"
            f"  key      : {key.digest}\n"
            f"  provider : {key.provider}\n"
            f"  model    : {key.model}\n"
            f"  task     : {key.task}\n"
            f"  input    : {key.preview()}\n"
            "Chạy lại không có --offline (cần API key) để sinh entry này."
        )

    response = with_backoff(fn, what=f"{key.provider}/{key.task}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "key_payload": key.payload,
                "response": response,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)
    return response


# --- Client ---------------------------------------------------------------
_gemini_client: Any = None
_groq_client: Any = None


def gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise MissingAPIKey("Chưa set GEMINI_API_KEY (xem .env.example)")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def groq_client() -> Any:
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise MissingAPIKey("Chưa set GROQ_API_KEY (xem .env.example)")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# --- Generation -----------------------------------------------------------
def gemini_generate(
    *,
    task: str,
    model: str,
    input_obj: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    offline: bool,
) -> str:
    """Sinh text bằng Gemini, có cache. `input_obj` mới là thứ đi vào cache key."""
    key = CallKey(
        task=task,
        provider="google",
        model=model,
        input=input_obj,
        params={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "thinking_budget": thinking_budget,
            "response_mime_type": "application/json",
        },
    )

    def call() -> str:
        from google.genai import types

        response = gemini_client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError(f"Gemini trả về rỗng (finish={response.candidates})")
        return text

    return cached_call(key, call, offline)


def groq_generate(*, task: str, input_obj: Any, prompt: str, offline: bool) -> str:
    """Sinh text bằng Groq cho LLM judge, có cache."""
    key = CallKey(
        task=task,
        provider="groq",
        model=config.JUDGE_MODEL,
        input=input_obj,
        params={
            "temperature": config.JUDGE_TEMPERATURE,
            "max_tokens": config.JUDGE_MAX_TOKENS,
            "response_format": "json_object",
        },
    )

    def call() -> str:
        response = groq_client().chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq trả về rỗng")
        return content

    return cached_call(key, call, offline)


# --- Embedding ------------------------------------------------------------
def _embed_key(text: str, task_type: str) -> CallKey:
    return CallKey(
        task="embed",
        provider="google",
        model=config.EMBED_MODEL,
        input=text,
        params={
            "task_type": task_type,
            "output_dimensionality": config.EMBED_DIM,
        },
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 từng hàng.

    Bắt buộc sau khi cắt chiều Matryoshka: gemini-embedding-001 chỉ chuẩn hoá
    sẵn ở 3072 chiều, vector 768 chiều cắt ra có norm != 1 nên cosine tính bằng
    tích vô hướng sẽ sai nếu bỏ bước này.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Có embedding vector rỗng (norm = 0)")
    normalized = matrix / norms
    result = np.linalg.norm(normalized, axis=1)
    if not np.allclose(result, 1.0, atol=1e-5):
        raise ValueError(
            f"Chuẩn hoá L2 thất bại: norm nằm trong [{result.min():.6f}, {result.max():.6f}]"
        )
    return normalized


def embed_texts(texts: Sequence[str], *, task_type: str, offline: bool) -> np.ndarray:
    """Embed danh sách text, trả về ma trận (n, EMBED_DIM) đã chuẩn hoá L2.

    Cache theo TỪNG text để đổi thứ tự hay đổi kích thước batch không làm hỏng
    cache; batching chỉ tồn tại ở tầng gọi API, gom đúng những text đang miss.
    """
    keys = [_embed_key(text, task_type) for text in texts]
    vectors: list[list[float] | None] = [None] * len(texts)

    # Gom theo digest: chunk trùng nội dung (rất hay gặp với các khoản dẫn chiếu
    # giống hệt nhau) chỉ tốn đúng một lời gọi API, không phải một lời gọi mỗi lần
    # xuất hiện.
    pending: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        if key.path.is_file():
            vectors[index] = json.loads(key.path.read_text(encoding="utf-8"))["response"]
        else:
            pending.setdefault(key.digest, []).append(index)

    if pending and offline:
        sample = keys[next(iter(pending.values()))[0]]
        raise CacheMiss(
            f"Cache miss ở chế độ --offline: {len(pending)} text chưa embed.\n"
            f"  ví dụ key : {sample.digest}\n"
            f"  model     : {config.EMBED_MODEL} ({task_type}, dim={config.EMBED_DIM})\n"
            f"  input     : {sample.preview()}\n"
            "Chạy lại không có --offline (cần GEMINI_API_KEY) để sinh entry này."
        )

    unique = list(pending.values())
    for start in range(0, len(unique), config.EMBED_BATCH_SIZE):
        batch = unique[start : start + config.EMBED_BATCH_SIZE]
        batch_texts = [texts[group[0]] for group in batch]

        def call(batch_texts: list[str] = batch_texts) -> list[list[float]]:
            from google.genai import types

            response = gemini_client().models.embed_content(
                model=config.EMBED_MODEL,
                contents=batch_texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBED_DIM,
                ),
            )
            return [list(e.values) for e in response.embeddings]

        batch_vectors = with_backoff(call, what="google/embed")
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"Gemini trả {len(batch_vectors)} vector cho {len(batch)} text"
            )
        for group, vector in zip(batch, batch_vectors):
            cached_call(keys[group[0]], lambda v=vector: v, offline=False)
            for index in group:
                vectors[index] = vector

    return _l2_normalize(np.asarray(vectors, dtype=np.float32))
