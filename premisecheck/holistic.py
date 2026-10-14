"""Holistic baseline: an LLM reads the paper and the whole comment and gives P(the comment is correct)."""
from typing import Optional

from . import prompts as P
from .parsing import json_obj, parse_probs
from .pipeline import Result


class HolisticLLM:
    """An LLM judges the whole comment against the paper (the baseline the paper compares with).

    The reference system of the paper is HolisticLLM(llm, "gpt-5.5", "medium") on the truncated paper. score() has
    the same signature and Result as PremiseCheck.score() (no premises)."""

    def __init__(self, llm, model: str = "gpt-5.5", effort: Optional[str] = "medium",
                 max_completion_tokens: int = 8000):
        self.llm, self.model, self.effort, self.max_tokens = llm, model, effort, max_completion_tokens

    def score(self, cid: str, comment: str, paper: str) -> Result:
        qid = cid + "#H"
        user = "### Evidence\npaper:\n%s\n\nreview_item:\n%s\n\n### Statements\n[%s] %s" % (
            paper, comment, qid, P.HOLISTIC_Q)
        r = self.llm.chat(self.model, P.LLM_JUDGE_SYSTEM, user, self.effort, self.max_tokens)
        p = parse_probs((json_obj(r.text) or {}).get("p"), [qid]).get(qid)
        if p is None:
            raise ValueError("%s: no probability in the answer for %s" % (self.model, cid))
        return Result(cid, p, usd={"total": r.usd}, latency_s=r.wall_s)
