"""
GPT-4o client with disk cache + retry. Mock-friendly for local dev.

Usage:
    from agent.llm_client import LLMClient
    cli = LLMClient(mock=True)   # local dev: no real API call
    resp = cli.chat([{"role": "user", "content": "hello"}])
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import diskcache
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    cache_dir: str = "cache/llm"


class LLMClient:
    """Thin wrapper around OpenAI ChatCompletion with diskcache + retry."""

    def __init__(self, cfg: Optional[LLMConfig] = None, mock: bool = False):
        self.cfg = cfg or LLMConfig()
        self.mock = mock
        self.cache = diskcache.Cache(self.cfg.cache_dir)
        if not mock:
            from openai import OpenAI  # lazy import
            self.client = OpenAI(
                api_key=self.cfg.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.cfg.base_url or os.getenv("OPENAI_BASE_URL"),
            )

    # --- Public API ---
    def chat(self, messages: List[Dict], **kwargs) -> str:
        key = self._cache_key(messages, kwargs)
        if key in self.cache:
            return self.cache[key]
        if self.mock:
            out = self._mock_response(messages)
        else:
            out = self._real_chat(messages, **kwargs)
        self.cache[key] = out
        return out

    # --- Internals ---
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
    def _real_chat(self, messages: List[Dict], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=kwargs.get("model", self.cfg.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
        )
        return resp.choices[0].message.content

    def _mock_response(self, messages: List[Dict]) -> str:
        """Deterministic mock for unit tests."""
        last = messages[-1]["content"] if messages else ""
        return f"[MOCK-LLM] You said: {last[:64]}"

    def _cache_key(self, messages, kwargs) -> str:
        payload = json.dumps([messages, kwargs, self.cfg.model], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
