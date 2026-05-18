"""
Doubao (Tencent tRPC gateway) LLM client for SearchCop.

Differences vs OpenAI:
  - HMAC-SHA1 auth (re-signed per request)
  - Endpoint: <HOST>/api/v1/data_eval
  - Body: {request_id, model_marker, messages, params}
  - messages[].content: list of {type, value} blocks
      type ∈ {"text", "image_url", "image_base64"}

Two model markers (route by `tier`):
  - "pro"  -> doubao-seed-2-0-pro-260215    (planner / critic / heavy reasoning)
  - "lite" -> doubao-seed-2-0-lite-260215   (aggregator / quality scorer / cheap calls)

Usage:
    from agent.llm_client import LLMClient
    cli = LLMClient(mock=True)                       # local dev, deterministic
    cli = LLMClient.from_env()                       # production, reads .env
    out = cli.chat([{"role": "user", "content": "Hi"}], tier="pro")
    out = cli.chat([{"role": "user",
                     "content": [
                         {"type": "text",      "value": "Describe this person"},
                         {"type": "image_url", "value": "https://.../crop.jpg"},
                     ]}], tier="lite")
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import diskcache
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


# --- Config -----------------------------------------------------------------
@dataclass
class LLMConfig:
    app_id: str = ""
    app_key: str = ""
    source: str = "searchcop"
    host: str = "http://trpc-gpt-eval.production.polaris:8080"
    api_path: str = "/api/v1/data_eval"
    api_version: str = "v2.03"
    timeout: int = 150
    # Two-tier routing
    model_pro: str = "doubao-seed-2-0-pro-260215"
    model_lite: str = "doubao-seed-2-0-lite-260215"
    default_tier: str = "pro"
    # Generation defaults
    temperature: float = 0.2
    max_tokens: int = 1024
    reasoning_effort: str = "medium"     # low / medium / high (pro 才生效)
    stream: bool = False
    # Cache
    cache_dir: str = "cache/llm"

    @classmethod
    def from_env(cls, **override) -> "LLMConfig":
        cfg = cls(
            app_id=os.getenv("DOUBAO_APP_ID", "").strip(),
            app_key=os.getenv("DOUBAO_APP_KEY", "").strip(),
            source=os.getenv("DOUBAO_SOURCE", "searchcop"),
            host=os.getenv("DOUBAO_HOST", "http://trpc-gpt-eval.production.polaris:8080"),
            model_pro=os.getenv("DOUBAO_MODEL_PRO", "doubao-seed-2-0-pro-260215"),
            model_lite=os.getenv("DOUBAO_MODEL_LITE", "doubao-seed-2-0-lite-260215"),
            default_tier=os.getenv("DOUBAO_DEFAULT_TIER", "pro"),
            cache_dir=os.getenv("LLM_CACHE_DIR", "cache/llm"),
        )
        for k, v in override.items():
            setattr(cfg, k, v)
        return cfg


# --- Errors -----------------------------------------------------------------
class LLMTransportError(RuntimeError):
    """Network / HTTP error worth retrying."""


class LLMBusinessError(RuntimeError):
    """Gateway returned a non-zero business code."""


# --- Helpers ----------------------------------------------------------------
def _sign(source: str, app_id: str, app_key: str) -> tuple[str, str]:
    """Reproduce the official HMAC-SHA1 signing scheme."""
    if not app_id or not app_key:
        raise RuntimeError("DOUBAO_APP_ID / DOUBAO_APP_KEY missing — set them in .env")
    date_time = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_str = f"date: {date_time}\nsource: {source}"
    sig = hmac.new(app_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sig_b64 = base64.b64encode(sig).decode()
    auth = f'hmac id="{app_id}", algorithm="hmac-sha1", headers="date source", signature="{sig_b64}"'
    return auth, date_time


def _to_doubao_content(content: Union[str, List[Dict]]) -> List[Dict]:
    """Coerce OpenAI-style content into Doubao [{type, value}, ...]."""
    if isinstance(content, str):
        return [{"type": "text", "value": content}]
    if isinstance(content, list):
        out = []
        for block in content:
            if not isinstance(block, dict):
                out.append({"type": "text", "value": str(block)})
                continue
            # already in doubao schema
            if "value" in block and "type" in block:
                out.append({"type": block["type"], "value": block["value"]})
                continue
            # openai-style: {"type": "text", "text": "..."}  /  {"type": "image_url", "image_url": {"url": "..."}}
            t = block.get("type", "text")
            if t == "text":
                out.append({"type": "text", "value": block.get("text", "")})
            elif t == "image_url":
                url = block.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                out.append({"type": "image_url", "value": url or ""})
            elif t == "image_base64":
                out.append({"type": "image_base64", "value": block.get("data") or block.get("value", "")})
            else:
                out.append({"type": "text", "value": json.dumps(block, ensure_ascii=False)})
        return out
    return [{"type": "text", "value": str(content)}]


def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    norm = []
    for m in messages:
        norm.append({
            "role": m.get("role", "user"),
            "content": _to_doubao_content(m.get("content", "")),
        })
    return norm


# --- Client -----------------------------------------------------------------
class LLMClient:
    """Doubao (tRPC) chat client with two-tier routing, diskcache, and retry."""

    def __init__(self, cfg: Optional[LLMConfig] = None, mock: bool = False):
        self.cfg = cfg or LLMConfig()
        self.mock = mock
        self.cache = diskcache.Cache(self.cfg.cache_dir)

    # ------------------ public ------------------
    @classmethod
    def from_env(cls, mock: Optional[bool] = None) -> "LLMClient":
        cfg = LLMConfig.from_env()
        if mock is None:
            mock = not (cfg.app_id and cfg.app_key)
        return cls(cfg=cfg, mock=mock)

    def chat(
        self,
        messages: List[Dict],
        *,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Synchronous chat. Returns the assistant text."""
        tier = tier or self.cfg.default_tier
        model_marker = self._pick_model(tier)
        params = {
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": self.cfg.stream,
            "reasoning_effort": reasoning_effort or self.cfg.reasoning_effort,
        }
        if temperature is not None:
            params["temperature"] = temperature

        body = {
            "request_id": str(uuid.uuid4()),
            "model_marker": model_marker,
            "messages": _normalize_messages(messages),
            "params": params,
            "timeout": 120,
        }

        key = self._cache_key(body)
        if use_cache and key in self.cache:
            return self.cache[key]

        if self.mock:
            text = self._mock_response(messages, tier)
        else:
            text = self._call(body)

        if use_cache:
            self.cache[key] = text
        return text

    # ------------------ internals ------------------
    def _pick_model(self, tier: str) -> str:
        if tier == "pro":
            return self.cfg.model_pro
        if tier == "lite":
            return self.cfg.model_lite
        # allow direct model_marker passthrough
        return tier

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception_type(LLMTransportError),
        reraise=True,
    )
    def _call(self, body: Dict) -> str:
        # request_id must be unique each retry, otherwise gateway may dedup
        body = dict(body, request_id=str(uuid.uuid4()))
        auth, date_time = _sign(self.cfg.source, self.cfg.app_id, self.cfg.app_key)
        headers = {
            "Apiversion": self.cfg.api_version,
            "Authorization": auth,
            "Date": date_time,
            "Source": self.cfg.source,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.cfg.host + self.cfg.api_path,
                headers=headers,
                json=body,
                timeout=self.cfg.timeout,
            )
        except requests.RequestException as e:
            raise LLMTransportError(f"network error: {e}") from e

        if resp.status_code >= 500:
            raise LLMTransportError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            # 4xx is usually auth/payload — don't retry
            raise LLMBusinessError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise LLMTransportError(f"non-JSON response: {resp.text[:200]}") from e

        code = data.get("code")
        if code not in (0, "0", None):
            raise LLMBusinessError(f"gateway code={code}, msg={data.get('msg')}, raw={str(data)[:300]}")

        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: Dict) -> str:
        """Doubao gateway answer extraction. Tolerant to schema drift."""
        # most common: {"answer": "..."}
        if isinstance(data.get("answer"), str):
            return data["answer"]
        # alt: {"answer": {"content": [{"type":"text","value":"..."}]}}
        ans = data.get("answer")
        if isinstance(ans, dict):
            if isinstance(ans.get("content"), list):
                texts = [b.get("value", "") for b in ans["content"] if b.get("type") == "text"]
                if texts:
                    return "\n".join(texts)
            if isinstance(ans.get("text"), str):
                return ans["text"]
        # fallback
        if "data" in data and isinstance(data["data"], dict) and "answer" in data["data"]:
            return data["data"]["answer"]
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _mock_response(messages: List[Dict], tier: str) -> str:
        last = messages[-1].get("content", "") if messages else ""
        if isinstance(last, list):
            last = " ".join(b.get("value", b.get("text", "")) for b in last if isinstance(b, dict))
        return f"[MOCK-DOUBAO/{tier}] {str(last)[:80]}"

    @staticmethod
    def _cache_key(body: Dict) -> str:
        # exclude request_id from hash (retries shouldn't break cache)
        canonical = {k: v for k, v in body.items() if k != "request_id"}
        s = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
