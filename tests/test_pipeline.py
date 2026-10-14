import json

import pytest

from fakes import FakeDecision, FakeLLM, premises_json
from premisecheck import prompts as P
from premisecheck.decision import DecisionModel
from premisecheck.pipeline import PremiseCheck

PAPER = "We evaluate BM25 on Natural Questions."
PAPER_PREM = "The paper does not evaluate on TriviaQA."
KNOW_PREM = "BM25 is a sparse retriever."


def nlp_decision(state, q):
    if q == P.ROUTE_Q:
        return 0.9 if state["statement"].startswith("The paper") else 0.2
    return {P.VERIFY_PAPER_Q: 0.8, P.VERIFY_KNOWLEDGE_Q: 0.3}[q]


def test_nlp_routes_and_takes_min():
    llm = FakeLLM(lambda u: premises_json(PAPER_PREM, KNOW_PREM), usd=0.01, wall_s=2.0)
    dm = FakeDecision(nlp_decision, usd=0.001, wall_s=0.5)
    r = PremiseCheck(llm, dm, "nlp", effort="none").score("c1", "No TriviaQA results.", PAPER)
    assert llm.calls[0]["system"] == P.DECOMP_NLP and llm.calls[0]["effort"] == "none"
    assert "No TriviaQA results." in llm.calls[0]["user"] and PAPER not in llm.calls[0]["user"]
    (a, b) = r.premises
    assert (a.kind, a.route, a.p) == ("paper", 0.9, 0.8)
    assert (b.kind, b.route, b.p) == ("knowledge", 0.2, 0.3)
    assert r.score == 0.3 and abs(r.error_score - 0.7) < 1e-12 and not r.fallback
    asked = {q for _, qs in dm.asks for q in qs}
    assert asked == {"c1#0#C1", "c1#0", "c1#1#C1", "c1#1#A3"}
    verify_states = [s for s, qs in dm.asks if "c1#0" in qs or "c1#1#A3" in qs]
    assert all(s["paper"] == PAPER for s in verify_states)
    assert r.usd == pytest.approx({"decompose": 0.01, "decision": 0.004, "total": 0.014})
    assert r.latency_s == pytest.approx(2.0 + 1.0)  # premises run in parallel: route + verify of one


def test_route_threshold():
    dm = FakeDecision(lambda s, q: 0.5 if q == P.ROUTE_Q else 0.6)
    r = PremiseCheck(FakeLLM(lambda u: premises_json(KNOW_PREM)), dm, "nlp").score("c", "x", PAPER)
    assert r.premises[0].kind == "paper"  # 0.5 is paper


def test_ml_prompt():
    llm = FakeLLM(lambda u: premises_json(PAPER_PREM))
    PremiseCheck(llm, FakeDecision(nlp_decision), "ml").score("c", "x", PAPER)
    assert llm.calls[0]["system"] == P.DECOMP_ML


@pytest.mark.parametrize("answer", ["not json", premises_json()])
def test_fallback_without_premises(answer):
    dm = FakeDecision(lambda s, q: 0.25)
    r = PremiseCheck(FakeLLM(lambda u: answer), dm, "nlp").score("c9", "Weaknesses:", PAPER)
    assert r.fallback and r.score == 0.25 and r.premises == []
    (state, qs), = dm.asks
    assert state == {"paper": PAPER, "review_item": "Weaknesses:"} and qs == {"c9#H": P.HOLISTIC_Q}
    assert r.usd["decision"] == 0.001 and r.usd["total"] == pytest.approx(0.011)


def typed(*pairs):
    return json.dumps({"premises": [{"text": t, "type": k} for t, k in pairs], "conclusion": "c"})


def test_science_types_and_contradiction():
    llm = FakeLLM(lambda u: typed(("The paper reports X.", "paper"), ("Y is standard.", "knowledge"),
                                  ("The paper should do Z.", "opinion"), ("The paper omits W.", "paper")))
    dm = FakeDecision(lambda s, q: {"The paper reports X.": 0.1, "The paper omits W.": 0.7}[s["proposition"]])
    r = PremiseCheck(llm, dm, "science").score("s1", "x", PAPER)
    assert llm.calls[0]["system"] == P.DECOMP_SCIENCE
    assert all(qs == {q: P.CONTRADICTION_Q} and q.endswith("#W3") for _, qs in dm.asks for q in qs)  # no routing
    assert sorted(q for _, qs in dm.asks for q in qs) == ["s1#0#W3", "s1#3#W3"]
    ps = r.premises
    assert ps[0].p == pytest.approx(0.9) and ps[3].p == pytest.approx(0.3)
    assert ps[1].p is None and ps[2].p is None
    assert r.score == pytest.approx(0.3)


def test_science_without_paper_premises():
    llm = FakeLLM(lambda u: typed(("Y is standard.", "knowledge"), ("Z is weak.", "opinion")))
    dm = FakeDecision(lambda s, q: 0.4)
    r = PremiseCheck(llm, dm, "science").score("s2", "x", PAPER)
    assert [q for _, qs in dm.asks for q in qs] == ["s2#0#W3"]
    assert r.score == pytest.approx(0.6)


def test_science_only_opinions_falls_back():
    llm = FakeLLM(lambda u: typed(("Z is weak.", "opinion"),))
    r = PremiseCheck(llm, FakeDecision(lambda s, q: 0.8), "science").score("s3", "x", PAPER)
    assert r.fallback and r.score == 0.8


def phrasing_prob(state, q):
    if q == P.ROUTE_Q:
        return 0.9 if state["statement"].startswith("The paper") else 0.1
    qs = P.PAPER_PHRASINGS if q in P.PAPER_PHRASINGS else P.KNOWLEDGE_PHRASINGS
    j = qs.index(q)
    return 0.2 if j in P.INVERTED else 0.9


def test_sixteen_phrasings():
    llm = FakeLLM(lambda u: premises_json(PAPER_PREM, KNOW_PREM))
    dm = FakeDecision(phrasing_prob)
    r = PremiseCheck(llm, dm, "nlp", phrasings=16).score("c", "x", PAPER)
    assert [p.p for p in r.premises] == pytest.approx([0.85, 0.85])  # (8 * 0.9 + 8 * (1 - 0.2)) / 16
    verify = [qs for _, qs in dm.asks if len(qs) == 16]
    assert len(verify) == 2
    texts = sorted(tuple(qs[k] for k in sorted(qs)) for qs in verify)
    assert texts == sorted([tuple(P.KNOWLEDGE_PHRASINGS), tuple(P.PAPER_PHRASINGS)])
    assert all(k.split("#")[-1] == "S%02d" % j for qs in verify for j, k in enumerate(sorted(qs)))


def test_sixteen_phrasings_mean():
    vals = [0.1 * (j % 10) for j in range(16)]
    dm = FakeDecision(lambda s, q: 0.9 if q == P.ROUTE_Q else vals[P.PAPER_PHRASINGS.index(q)])
    r = PremiseCheck(FakeLLM(lambda u: premises_json(PAPER_PREM)), dm, "nlp", phrasings=16).score("c", "x", PAPER)
    want = sum(1 - v if j >= 8 else v for j, v in enumerate(vals)) / 16
    assert r.premises[0].p == pytest.approx(want)


def test_phrasings_must_be_1_or_16():
    with pytest.raises(ValueError):
        PremiseCheck(FakeLLM(lambda u: ""), FakeDecision(lambda s, q: 0.5), phrasings=4)


def test_qid_tag():
    dm = FakeDecision(nlp_decision)
    PremiseCheck(FakeLLM(lambda u: premises_json(PAPER_PREM)), dm, "nlp", qid_tag="t1").score("c", "x", PAPER)
    assert {q for _, qs in dm.asks for q in qs} == {"c#t1#0#C1", "c#t1#0"}


def test_to_dict():
    r = PremiseCheck(FakeLLM(lambda u: premises_json(PAPER_PREM)), FakeDecision(nlp_decision)).score("c", "x", PAPER)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["id"] == "c" and d["error_score"] == pytest.approx(0.2) and d["premises"][0]["kind"] == "paper"


def test_decision_model_batches_and_caches(tmp_path):
    from premisecheck.cache import JsonlCache
    from premisecheck.llm import CacheMiss

    posts = []

    def post(payload):
        posts.append(payload)
        return {"answers": {q: {"noul": 0.5} for q in payload["questions"]}, "usage": {"cost": 0.002},
                "latency_ms": 12.0}

    dm = DecisionModel("kev", "run@v1", "http://x/", "/v1/systemone", request_model="kev-latest",
                       cache=JsonlCache(tmp_path / "kev.jsonl"))
    dm._post = post
    qs = {"q%02d" % i: "question %02d" % i for i in range(20)}
    a = dm.ask({"paper": "p", "proposition": "s"}, qs)
    assert len(posts) == 2 and [len(p["questions"]) for p in posts] == [16, 4]
    assert posts[0]["model"] == "kev-latest" and posts[0]["questions"]["q00"] == {"type": "noul",
                                                                                     "instructions": "question 00"}
    assert a.calls == 2 and a.usd == pytest.approx(0.004) and a.gpu_ms == 24.0 and set(a.p) == set(qs)
    again = DecisionModel("kev", "run@v1", "http://x", "/v1/systemone", cache=JsonlCache(tmp_path / "kev.jsonl"),
                          cache_only=True)
    assert again.ask({"paper": "p", "proposition": "s"}, qs).p == a.p
    with pytest.raises(CacheMiss):
        again.ask({"paper": "p", "proposition": "other"}, qs)


def test_unknown_field():
    with pytest.raises(ValueError, match="field"):
        PremiseCheck(FakeLLM(lambda u: ""), FakeDecision(lambda s, q: 0.5), field="physics")


def test_decision_model_rejects_incomplete_answer(tmp_path):
    from premisecheck.cache import JsonlCache

    dm = DecisionModel("kev", "run@v1", "http://x", "/v1/systemone", cache=JsonlCache(tmp_path / "kev.jsonl"))
    dm._post = lambda payload: {"answers": {"a": {"noul": 0.5}}}
    with pytest.raises(RuntimeError, match="malformed"):
        dm.ask({"proposition": "s"}, {"a": "question a", "b": "question b"})
    assert not (tmp_path / "kev.jsonl").exists()  # nothing cached, so a rerun asks again
