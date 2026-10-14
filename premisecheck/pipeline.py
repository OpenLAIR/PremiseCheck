"""PremiseCheck: an LLM decomposes a review comment into premises without reading the paper; a decision model routes
each premise and verifies it against the paper; the comment score is the lowest premise probability (Section 4)."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from . import prompts as P
from .parsing import parse_premises


@dataclass(frozen=True)
class Field:
    """Per-field settings of the paper (Appendix B.1)."""
    decomposer_prompt: str
    route: bool          # the decision model routes each premise to the paper or the knowledge question
    contradiction: bool  # paper premises get the contradiction question (score = 1 - p) instead of VERIFY_PAPER_Q
    typed: bool          # the decomposer types its premises; only "paper" premises are verified


FIELDS = {
    "nlp": Field(P.DECOMP_NLP, route=True, contradiction=False, typed=False),
    "ml": Field(P.DECOMP_ML, route=True, contradiction=False, typed=False),
    "science": Field(P.DECOMP_SCIENCE, route=False, contradiction=True, typed=True),
}
ROUTE_THRESHOLD = 0.5


@dataclass
class Premise:
    text: str
    type: Optional[str] = None     # decomposer type: "paper", "knowledge" or "opinion" (natural-science prompt only)
    route: Optional[float] = None  # P(decidable from the paper alone); None when the field does not route
    kind: str = "paper"            # how it was verified: "paper" or "knowledge"
    p: Optional[float] = None      # P(premise holds); None for premises that were not verified


@dataclass
class Result:
    id: str
    score: float                   # P(the comment is correct) = min over verified premises
    premises: List[Premise] = field(default_factory=list)
    fallback: bool = False         # no premise to verify: the decision model judged the whole comment
    usd: dict = field(default_factory=dict)  # {"decompose", "decision", "total"} in USD (holistic: {"total"})
    latency_s: float = 0.0         # decomposer + slowest premise (premises are verified in parallel)

    @property
    def error_score(self):
        """Higher = more likely a factual error; the score AUROC is computed on."""
        return 1.0 - self.score

    def to_dict(self) -> dict:
        """JSON-ready dict, with `error_score` added."""
        d = asdict(self)
        d["error_score"] = self.error_score
        return d


class PremiseCheck:
    """Score review comments for factual errors about the paper.

    llm:        premisecheck.llm.LLM (the decomposer; any OpenAI-compatible endpoint)
    decision:   premisecheck.decision.DecisionModel (Jev or Kev-27B)
    field:      "nlp", "ml" or "science" (prompt and verification settings of each part of ReviewFact)
    effort:     reasoning effort of the decomposer ("medium", "low" or "none")
    phrasings:  1 = one verification question per premise; 16 = all 16 phrasings in one call, averaged (Section 7.3)
    qid_tag:    optional label inserted into question ids, to keep runs of different settings apart in a cache
    workers:    premises of one comment verified in parallel

    score() is safe to call from several threads at once.
    """

    def __init__(self, llm, decision, field: str = "nlp", decomposer: str = "gpt-5.5", effort: Optional[str] = "medium",
                 phrasings: int = 1, qid_tag: str = "", workers: int = 8):
        if phrasings not in (1, 16):
            raise ValueError("phrasings must be 1 or 16")
        if field not in FIELDS:
            raise ValueError("field must be one of %s, not %r" % (", ".join(FIELDS), field))
        self.llm, self.decision, self.field = llm, decision, FIELDS[field]
        self.decomposer, self.effort, self.phrasings = decomposer, effort, phrasings
        self.qid_tag, self.workers = qid_tag, workers

    # ------------------------------------------------------------------ steps -----
    def decompose(self, comment):
        r = self.llm.chat(self.decomposer, self.field.decomposer_prompt, P.DECOMP_USER % comment, self.effort)
        return [Premise(t, typ) for t, typ in parse_premises(r.text)], r

    def selected(self, premises):
        """Premises to verify: without types, all; with types, the "paper" ones (else all but opinions)."""
        if not self.field.typed:
            return [(i, p) for i, p in enumerate(premises) if p.type != "opinion"]
        sel = [(i, p) for i, p in enumerate(premises) if p.type == "paper"]
        return sel or [(i, p) for i, p in enumerate(premises) if p.type != "opinion"]

    def _qid(self, cid, i):
        return "%s#%s#%d" % (cid, self.qid_tag, i) if self.qid_tag else "%s#%d" % (cid, i)

    def verify(self, cid, i, prem, paper):
        """Route (if the field routes) and verify one premise; returns (premise, usd, wall_s)."""
        qid, usd, wall = self._qid(cid, i), 0.0, 0.0
        if self.field.route:
            a = self.decision.ask({"statement": prem.text}, {qid + "#C1": P.ROUTE_Q})
            prem.route = a.p[qid + "#C1"]
            prem.kind = "paper" if prem.route >= ROUTE_THRESHOLD else "knowledge"
            usd, wall = usd + a.usd, wall + a.wall_s
        state = {"paper": paper, "proposition": prem.text}
        if self.phrasings == 16:
            qs = P.PAPER_PHRASINGS if prem.kind == "paper" else P.KNOWLEDGE_PHRASINGS
            ids = ["%s#S%02d" % (qid, j) for j in range(16)]
            a = self.decision.ask(state, dict(zip(ids, qs)))
            prem.p = float(np.mean([1 - a.p[q] if j in P.INVERTED else a.p[q] for j, q in enumerate(ids)]))
        elif prem.kind == "knowledge":
            a = self.decision.ask(state, {qid + "#A3": P.VERIFY_KNOWLEDGE_Q})
            prem.p = a.p[qid + "#A3"]
        elif self.field.contradiction:
            a = self.decision.ask(state, {qid + "#W3": P.CONTRADICTION_Q})
            prem.p = 1 - a.p[qid + "#W3"]
        else:
            a = self.decision.ask(state, {qid: P.VERIFY_PAPER_Q})
            prem.p = a.p[qid]
        return prem, usd + a.usd, wall + a.wall_s

    def holistic(self, cid, comment, paper):
        """The decision model on the whole comment (used when there is no premise to verify)."""
        a = self.decision.ask({"paper": paper, "review_item": comment}, {cid + "#H": P.HOLISTIC_Q})
        return a.p[cid + "#H"], a

    # ------------------------------------------------------------------ score -----
    def score(self, cid: str, comment: str, paper: str) -> Result:
        """Score one comment. `cid` names its questions (e.g. "c1#0"); `paper` is the text the decision model
        reads, normally premisecheck.truncate(full_text)."""
        premises, r = self.decompose(comment)
        todo = self.selected(premises)
        usd = {"decompose": r.usd, "decision": 0.0}
        if todo:
            with ThreadPoolExecutor(min(self.workers, len(todo))) as pool:
                done = list(pool.map(lambda ip: self.verify(cid, ip[0], ip[1], paper), todo))
            usd["decision"] = sum(u for _, u, _ in done)
            res = Result(cid, min(p.p for p, _, _ in done), premises,
                         latency_s=r.wall_s + max(w for _, _, w in done))
        else:
            s, a = self.holistic(cid, comment, paper)
            usd["decision"] = a.usd
            res = Result(cid, s, premises, fallback=True, latency_s=r.wall_s + a.wall_s)
        usd["total"] = usd["decompose"] + usd["decision"]
        res.usd = usd
        return res
