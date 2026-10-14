"""OpenAI-compatible chat client for the decomposer and the LLM baselines, with optional caching."""
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .cache import request_hash

# USD per million tokens: (input, cached input, output); reasoning tokens are billed as output (public list prices).
PRICES = {"gpt-5.5": (5.00, 0.50, 30.00), "gpt-5.4-mini": (0.75, 0.075, 4.50)}


class CacheMiss(RuntimeError):
    """A cache-only client was asked for a call that is not in its cache."""


def price_of(model):
    """(input, cached input, output) USD per million tokens of `model`, or None if unknown."""
    if model in PRICES:
        return PRICES[model]
    for name in sorted(PRICES, key=len, reverse=True):  # dated snapshots, e.g. gpt-5.5-2026-04-23
        if model.startswith(name):
            return PRICES[name]
    return None


def usd(model, usage):
    """Cost of one call from its token usage; 0 for a model without a known price."""
    p = price_of(model)
    if p is None:
        return 0.0
    pin, pcached, pout = p
    return ((usage.get("prompt", 0) - usage.get("cached", 0)) * pin + usage.get("cached", 0) * pcached
            + usage.get("completion", 0) * pout) / 1e6


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=dict)  # prompt, cached, completion, reasoning tokens
    wall_s: float = 0.0
    usd: float = 0.0


class LLM:
    """Chat completions against any OpenAI-compatible endpoint (OPENAI_BASE_URL / OPENAI_API_KEY by default).

    The request is {"model", "reasoning_effort", "max_completion_tokens", "messages"}; with a cache, an identical
    request is answered from the cache. `cache_only` raises CacheMiss instead of calling the API. Retries on rate
    limits and server errors are left to the openai client (`max_retries`). Safe to share between threads."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, cache=None,
                 cache_only: bool = False, timeout: float = 900, max_retries: int = 8):
        self.base_url, self.api_key = base_url, api_key
        self.cache, self.cache_only = cache, cache_only
        self.timeout, self.max_retries = timeout, max_retries
        self._client = None
        self._lock = threading.Lock()

    def _api(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
                                      api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                                      timeout=self.timeout, max_retries=self.max_retries)
            return self._client

    def chat(self, model: str, system: str, user: str, reasoning_effort: Optional[str] = "medium",
             max_completion_tokens: int = 8000) -> Reply:
        """One chat call. `reasoning_effort=None` leaves the field out (models without a reasoning setting)."""
        payload = {"model": model, "reasoning_effort": reasoning_effort, "max_completion_tokens": max_completion_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if reasoning_effort is None:
            del payload["reasoning_effort"]
        h = request_hash(payload)
        rec = self.cache.get(h) if self.cache else None
        if rec is None:
            if self.cache_only:
                raise CacheMiss(model)
            t0 = time.monotonic()
            resp = self._api().chat.completions.create(**payload)
            if not resp.choices:
                raise RuntimeError("%s returned no choices" % model)
            u = resp.usage
            details = getattr(u, "prompt_tokens_details", None)
            cdetails = getattr(u, "completion_tokens_details", None)
            rec = {"h": h, "text": resp.choices[0].message.content or "", "model": resp.model,
                   "wall_ms": round((time.monotonic() - t0) * 1000),
                   "usage": {"prompt": getattr(u, "prompt_tokens", 0) or 0,
                             "cached": getattr(details, "cached_tokens", 0) or 0,
                             "completion": getattr(u, "completion_tokens", 0) or 0,
                             "reasoning": getattr(cdetails, "reasoning_tokens", 0) or 0}}
            if self.cache:
                self.cache.put(rec)
        usage = rec.get("usage") or {}
        return Reply(rec.get("text") or "", usage, (rec.get("wall_ms") or 0) / 1000, usd(model, usage))
