"""OpenAI-compatible judge client supporting text and vision calls.

Uses bare `requests` (same pattern as test_gemini.py / test_gpt_image.py) —
no openai SDK dependency. The base_url + /chat/completions path mirrors the
OpenAI Chat Completions API; configure to point at matrixllm.alipay.com or
any other compatible gateway.

Configured per-section via yaml:

    judge:
      vision:
        base_url: "https://matrixllm.alipay.com/v1"
        model: "gemini-3.1-pro-preview"
        api_key_env: "MATRIXCUBE_API_KEY"
      text:
        base_url: "https://matrixllm.alipay.com/v1"
        model: "gemini-3-flash-preview"
        api_key_env: "MATRIXCUBE_API_KEY"
      cache_dir: "./results/judge_cache"
      request_timeout: 120
      max_retries: 4
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    api_key: str
    timeout: int = 120
    max_retries: int = 4
    # gemini-3.x (and other reasoning models) burn the whole token budget on
    # reasoning at default effort and return empty content. "low" caps that so
    # the answer surfaces — see configs/gemini.yaml:chat reasoning_effort note.
    reasoning_effort: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, default_timeout: int = 120, default_retries: int = 4) -> "JudgeConfig":
        api_key_env = d.get("api_key_env", "JUDGE_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        return cls(
            base_url=d["base_url"],
            model=d["model"],
            api_key=api_key,
            timeout=d.get("timeout", default_timeout),
            max_retries=d.get("max_retries", default_retries),
            reasoning_effort=d.get("reasoning_effort"),
        )


def _encode_image(path: str, max_side: int = 1280) -> str:
    """Encode an image as a base64 data URL. Downscales if any side > max_side."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


class RequestRateLimiter:
    """Thread-safe, process-local request-start rate limiter.

    A shared instance is attached to both text and vision judge clients, so
    ``qps`` caps their combined request start rate (including retries).  Slots
    are evenly spaced instead of allowing an initial burst.
    """

    def __init__(self, qps: float):
        if qps <= 0:
            raise ValueError(f"qps must be > 0, got {qps}")
        self.qps = float(qps)
        self._interval = 1.0 / self.qps
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed)
            self._next_allowed = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


class JudgeClient:
    """OpenAI-compatible Chat Completions client built on bare requests.

    Same call pattern as test_gemini.py: POST {base_url}/chat/completions
    with a Bearer auth header and a JSON body.
    """

    def __init__(self, cfg: JudgeConfig, rate_limiter: RequestRateLimiter | None = None):
        self.cfg = cfg
        self._rate_limiter = rate_limiter
        # Tolerate both "https://host/v1" and "https://host/v1/" forms.
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"

    def chat(
        self,
        system: str,
        user: str,
        images: Optional[list[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> str:
        """Single chat completion. `images` is a list of local image paths."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for p in images or []:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(p)}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            # gpt-5.x / o-series reject `max_tokens`; the new field is
            # `max_completion_tokens`. Gemini accepts both names harmlessly.
            "max_completion_tokens": max_tokens,
        }
        # Some reasoning models reject any non-default temperature. Only send
        # the field when the caller explicitly asks for non-zero sampling.
        if temperature is not None and temperature != 0.0:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = response_format
        # Forward the OpenAI-compatible reasoning control for any model whose
        # endpoint supports it. Omit it to use the provider/model default.
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()
                r = requests.post(self._url, headers=headers, json=body, timeout=self.cfg.timeout)
                if r.status_code >= 400:
                    raise RuntimeError(f"judge HTTP {r.status_code}: {r.text[:300]}")
                data = json.loads(r.text)
                try:
                    choice = data["choices"][0]
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(f"unexpected judge response: {data}") from e
                # message / content may be missing entirely when a thinking
                # model exhausts max_completion_tokens on reasoning.
                text = (choice.get("message") or {}).get("content") or ""
                if not text:
                    finish = choice.get("finish_reason")
                    usage = data.get("usage", {})
                    raise RuntimeError(
                        f"empty judge content (finish_reason={finish!r}, usage={usage}); "
                        "likely reasoning tokens exhausted max_completion_tokens — raise the limit"
                    )
                return text
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Judge call failed after {self.cfg.max_retries} retries: {last_err}")

    def chat_json(
        self,
        system: str,
        user: str,
        images: Optional[list[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Chat completion that must return a JSON object.

        Transport/API failures are retried inside :meth:`chat`. A non-empty but
        truncated or malformed response used to bypass those retries because
        parsing happened only after ``chat`` returned. Retry the full request
        when JSON parsing fails so transient length/format errors do not
        immediately turn an evaluation image into a skipped case.
        """
        last_parse_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            raw = self.chat(
                system=system,
                user=user,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                return _parse_json_lenient(raw)
            except Exception as e:
                last_parse_err = e
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"Judge JSON parse failed after {self.cfg.max_retries} attempts: "
            f"{last_parse_err}"
        ) from last_parse_err


def _parse_json_lenient(text: str) -> dict:
    """Tolerate ```json fences or stray prefix text around a JSON object."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block, scanning for braces and skipping
    # strings (so `}` inside quoted strings doesn't fool us). Tries each
    # candidate end position from outermost-in.
    start = s.find("{")
    if start < 0:
        raise RuntimeError(f"judge returned no JSON object; raw[:300]={s[:300]!r}")
    end = s.rfind("}")
    if end <= start:
        repaired = _repair_truncated_json(s[start:])
        if repaired is not None:
            return repaired
        raise RuntimeError(f"judge returned no JSON object (truncated); raw[:300]={s[:300]!r}")
    while end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            end = s.rfind("}", start, end)
    # Last-resort repair (gemini often emits real newlines / un-escaped
    # backslashes inside string values).
    inner = s[s.find("{") : s.rfind("}") + 1]
    try:
        return json.loads(_repair_json_strings(inner))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"judge returned malformed JSON ({e}); raw[:500]={inner[:500]!r}"
        ) from e


def _repair_truncated_json(s: str) -> Optional[dict]:
    """Attempt to recover a JSON object whose output was truncated (no closing brace).

    Finds the last comma at top-level depth, keeps everything before it, and
    closes the object.
    """
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    last_comma = -1
    i = 0
    while i < len(s):
        c = s[i]
        if in_str:
            if c == '\\' and i + 1 < len(s):
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1
            elif c == '[':
                depth_bracket += 1
            elif c == ']':
                depth_bracket -= 1
            elif c == ',' and depth_brace == 1 and depth_bracket == 0:
                last_comma = i
        i += 1
    if last_comma <= 0:
        return None
    candidate = s[:last_comma] + "}"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_json_strings(candidate))
    except (json.JSONDecodeError, Exception):
        return None


def _repair_json_strings(s: str) -> str:
    """Best-effort fix for un-escaped newlines / backslashes inside JSON
    string values. Same logic as the matrixcube backend version."""
    out = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        if c == '"':
            out.append(c)
            in_str = False
            i += 1
            continue
        if c == "\\":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt in '"\\/bfnrtu':
                out.append(c)
                out.append(nxt)
                i += 2
            else:
                out.append("\\\\")
                i += 1
            continue
        if c == "\n":
            out.append("\\n")
            i += 1
            continue
        if c == "\r":
            out.append("\\r")
            i += 1
            continue
        if c == "\t":
            out.append("\\t")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def load_judge_clients(
    yaml_path: str,
    cache_dir_override: str | None = None,
    qps_override: float | None = None,
) -> tuple[JudgeClient, JudgeClient, "JudgeCache"]:
    """Load vision + text judge clients and a shared cache from a config yaml.

    Args:
        cache_dir_override: If provided, use this as the cache directory instead
            of the global one in the yaml config. This ensures each experiment
            gets its own isolated cache.
        qps_override: Process-local combined QPS cap shared by text and vision
            judge clients. ``None``/``0`` disables rate limiting.
    """
    import yaml

    from .judge_cache import JudgeCache

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    jcfg = cfg["judge"]
    configured_qps = jcfg.get("qps", 0)
    qps = configured_qps if qps_override is None else qps_override
    qps = float(qps or 0)
    limiter = RequestRateLimiter(qps) if qps > 0 else None
    vision = JudgeClient(JudgeConfig.from_dict(jcfg["vision"]), rate_limiter=limiter)
    text = JudgeClient(JudgeConfig.from_dict(jcfg["text"]), rate_limiter=limiter)
    cache_dir = cache_dir_override or jcfg.get("cache_dir", "./results/judge_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = JudgeCache(cache_dir)
    return vision, text, cache
