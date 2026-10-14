"""Clients for decision models: Jev (TypeSafe, served on OpenRouter) and the open-weight Kev-27B (self-served).

A decision model reads a state (named texts, e.g. {"paper": ..., "proposition": ...}) and answers yes/no questions
about it with probabilities. Both models take the same request:
    {"model": ..., "state": {...}, "questions": {qid: {"type": "noul", "instructions": question}}}
and return {"answers": {qid: {"noul": p}}, "usage": {...}}.
"""
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from .cache import request_hash
from .llm import CacheMiss

QUESTIONS_PER_CALL = 16

JEV_MODEL = "typesafe/jev-1.13-20260917"
KEV_RUN = "jaredpalmer/kev-27b@v1.0"

log = logging.getLogger(__name__)


def _probs(data, questions):
    """{qid: probability} of a decision-model response, or None unless it answers every question."""
    try:
        p = {q: float(a["noul"]) for q, a in data["answers"].items()}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return p if all(q in p for q in questions) else None


@dataclass
class Answer:
    p: dict           # {qid: probability}
    usd: float = 0.0  # cost reported by the API (Jev); Kev reports none
    calls: int = 0
    wall_s: float = 0.0
    gpu_ms: float = 0.0


class DecisionModel:
    """Client of a decision-model endpoint (`base_url` + `route`); use jev() or kev() to build one.

    `name` and `model` identify the model in the cache keys; `request_model` is the model name sent to the server
    (default: `model`). Requests are retried with exponential backoff on timeouts, HTTP 408/429 and 5xx for up to
    `max_wait_s` seconds; other HTTP errors raise at once. Safe to share between threads."""

    def __init__(self, name: str, model: str, base_url: str, route: str, api_key: Optional[str] = None,
                 request_model: Optional[str] = None, cache=None, cache_only: bool = False, timeout: float = 300,
                 max_wait_s: float = 3600):
        self.name, self.model, self.request_model = name, model, request_model or model
        self.url, self.api_key = base_url.rstrip("/") + route, api_key
        self.cache, self.cache_only = cache, cache_only
        self.timeout, self.max_wait_s = timeout, max_wait_s

    def _post(self, payload):
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        start, attempt = time.monotonic(), 0
        while True:
            attempt += 1
            err = None
            try:
                r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (408, 429) and r.status_code < 500:
                    raise RuntimeError("%s HTTP %d: %s" % (self.name, r.status_code, r.text[:300]))
                err = "HTTP %d" % r.status_code
            except requests.RequestException as e:
                err = str(e)
            if time.monotonic() - start > self.max_wait_s:
                raise RuntimeError("%s: giving up after %d attempts: %s" % (self.name, attempt, err))
            if attempt == 1 or attempt % 10 == 0:
                log.warning("%s (%s): %s; retrying for up to %d s", self.name, self.url, err, self.max_wait_s)
            time.sleep(min(120.0, 2.0 ** min(attempt, 7)) * (0.5 + random.random()))

    def ask(self, state: dict, questions: dict) -> Answer:
        """Answer {qid: question} on one state, QUESTIONS_PER_CALL questions per call (sorted by question text)."""
        out = Answer({})
        qids = sorted(questions, key=lambda q: questions[q])
        for i in range(0, len(qids), QUESTIONS_PER_CALL):
            chunk = {q: questions[q] for q in qids[i:i + QUESTIONS_PER_CALL]}
            h = request_hash({"checker": self.name, "model": self.model, "state": state, "q": chunk})
            rec = self.cache.get(h) if self.cache else None
            if rec is not None and rec.get("error"):
                rec = None
            if rec is None:
                if self.cache_only:
                    raise CacheMiss(self.name)
                t0 = time.monotonic()
                data = self._post({"model": self.request_model, "state": state,
                                   "questions": {q: {"type": "noul", "instructions": s} for q, s in chunk.items()}})
                p = _probs(data, chunk)
                if p is None:  # not cached, so a rerun asks again
                    raise RuntimeError("%s: malformed answer for %s: %.300s" % (self.name, sorted(chunk), data))
                usage = data.get("usage") or {}
                rec = {"h": h, "p": p,
                       "cost": usage.get("cost", 0.0), "input_tokens": usage.get("input_tokens", 0),
                       "wall_ms": round((time.monotonic() - t0) * 1000)}
                if "latency_ms" in data:  # Kev: GPU time of the batch the request ran in
                    rec["gpu_ms"] = data["latency_ms"]
                if self.cache:
                    self.cache.put(rec)
            out.p.update(rec["p"])
            out.usd += rec.get("cost") or 0.0
            out.calls += 1
            out.wall_s += (rec.get("wall_ms") or 0) / 1000
            out.gpu_ms += rec.get("gpu_ms") or 0.0
        return out


def jev(api_key=None, cache=None, cache_only=False, model=JEV_MODEL):
    """Jev on OpenRouter's decisions endpoint (OPENROUTER_API_KEY)."""
    return DecisionModel("jev", model, "https://openrouter.ai", "/api/alpha/decisions",
                         api_key or os.environ.get("OPENROUTER_API_KEY"), cache=cache, cache_only=cache_only,
                         timeout=120)


def kev(base_url=None, api_key=None, cache=None, cache_only=False, run=KEV_RUN):
    """Kev-27B behind a TypeSafe-compatible /v1/systemone server (KEV_BASE_URL, KEV_API_KEY); see docs/kev.md."""
    base_url = base_url or os.environ.get("KEV_BASE_URL", "http://localhost:8000")
    return DecisionModel("kev", run, base_url, "/v1/systemone", api_key or os.environ.get("KEV_API_KEY"),
                         request_model="kev-latest", cache=cache, cache_only=cache_only)
