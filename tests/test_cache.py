import pytest

from premisecheck.cache import JsonlCache, request_hash
from premisecheck.llm import LLM, CacheMiss, price_of, usd


def test_request_hash_ignores_key_order():
    assert request_hash({"a": 1, "b": [1, 2]}) == request_hash({"b": [1, 2], "a": 1})
    assert request_hash({"a": 1}) != request_hash({"a": 2})
    assert len(request_hash({})) == 32


def test_round_trip(tmp_path):
    path = tmp_path / "sub" / "c.jsonl"
    c = JsonlCache(path)
    c.put({"h": "x", "text": "é"})
    c.put({"h": "y", "p": {"q": 0.5}})
    c.put({"h": "x", "text": "newer"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"h": "z", "text": "cut of')  # interrupted write
    d = JsonlCache(path)
    assert d.get("x") == {"h": "x", "text": "newer"} and d.get("y") == {"h": "y", "p": {"q": 0.5}}
    assert d.get("z") is None


def test_read_only(tmp_path):
    path = tmp_path / "c.jsonl"
    JsonlCache(path).put({"h": "a"})
    ro = JsonlCache(path, read_only=True)
    ro.put({"h": "b"})
    assert ro.get("b") == {"h": "b"}
    assert JsonlCache(path).get("b") is None


def test_llm_from_cache(tmp_path):
    cache = JsonlCache(tmp_path / "llm.jsonl")
    payload = {"model": "gpt-5.5", "reasoning_effort": "none", "max_completion_tokens": 8000,
               "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]}
    cache.put({"h": request_hash(payload), "text": '{"premises": []}', "wall_ms": 1500,
               "usage": {"prompt": 1000, "cached": 400, "completion": 100, "reasoning": 0}})
    llm = LLM(cache=cache, cache_only=True)
    r = llm.chat("gpt-5.5", "S", "U", "none")
    assert r.text == '{"premises": []}' and r.wall_s == 1.5
    assert r.usd == pytest.approx((600 * 5.0 + 400 * 0.5 + 100 * 30.0) / 1e6)
    with pytest.raises(CacheMiss):
        llm.chat("gpt-5.5", "S", "U", "low")


def test_prices():
    assert price_of("gpt-5.5-2026-04-23") == price_of("gpt-5.5")
    assert price_of("gpt-5.4-mini-2026-01-01") == price_of("gpt-5.4-mini")
    assert price_of("some-local-model") is None
    assert usd("some-local-model", {"prompt": 10 ** 6}) == 0.0


def test_append_after_cut_off_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"h": "a"}\n{"h": "b", "text": "cut of')
    c = JsonlCache(path)
    c.put({"h": "c"})
    d = JsonlCache(path)
    assert d.get("a") and d.get("c") and d.get("b") is None
