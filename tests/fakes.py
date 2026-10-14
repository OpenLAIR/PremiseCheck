"""Fake decomposer LLM and fake decision model: no network."""
import json

from premisecheck.decision import Answer
from premisecheck.llm import Reply


class FakeLLM:
    """Answers every chat with `answer(user)` (a JSON string or any text); records the calls."""

    def __init__(self, answer, usd=0.01, wall_s=1.0):
        self.answer, self.usd, self.wall_s, self.calls = answer, usd, wall_s, []

    def chat(self, model, system, user, reasoning_effort="medium", max_completion_tokens=8000):
        self.calls.append({"model": model, "system": system, "user": user, "effort": reasoning_effort})
        return Reply(self.answer(user), {}, self.wall_s, self.usd)


class FakeDecision:
    """Answers each question with `prob(state, question)`; records one entry per ask()."""

    def __init__(self, prob, usd=0.001, wall_s=0.5):
        self.prob, self.usd, self.wall_s, self.asks = prob, usd, wall_s, []

    def ask(self, state, questions):
        self.asks.append((state, dict(questions)))
        return Answer({q: self.prob(state, s) for q, s in questions.items()}, self.usd, 1, self.wall_s)


def premises_json(*premises):
    return json.dumps({"premises": list(premises), "conclusion": "c"})

