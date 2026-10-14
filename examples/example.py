"""Check one review comment against one paper with PremiseCheck.

Needs an OpenAI-compatible endpoint for the decomposer (OPENAI_BASE_URL, OPENAI_API_KEY) and a Kev-27B server
(KEV_BASE_URL, default http://localhost:8000; see docs/kev.md). Use --decision jev for Jev (OPENROUTER_API_KEY).

  python examples/example.py [--decision kev|jev] [--effort none|low|medium]
"""
import argparse
import json

from premisecheck import LLM, PremiseCheck, jev, kev, truncate
from premisecheck.cache import JsonlCache

PAPER = """Title:
Sparse Retrieval Is Enough for Short-Answer Question Answering

Abstract:
We study whether a sparse retriever suffices for open-domain question answering with short answers. We pair BM25
with a 7B reader and evaluate on Natural Questions and TriviaQA.

1 Introduction:
Dense retrievers dominate recent open-domain QA systems, but they require large training sets and costly indexes.
We ask how far a sparse retriever can go when answers are short.

2 Method:
We retrieve the top 20 passages with BM25 (k1 = 0.9, b = 0.4) from the December 2018 Wikipedia dump and pass them to
the reader, which generates the answer. We do not fine-tune the reader.

3 Experiments:
Table 1 reports exact match on the test sets. BM25 with our reader reaches 41.2 on Natural Questions and 63.5 on
TriviaQA. With DPR as the retriever, the same reader reaches 44.0 and 64.1. We do not evaluate on WebQuestions.

Table 1: Exact match on Natural Questions (NQ) and TriviaQA (TQA).
Retriever | NQ | TQA
BM25 | 41.2 | 63.5
DPR | 44.0 | 64.1
"""

COMMENTS = {  # c1 is a factual error (Table 1 reports DPR); c2 is correct
    "c1": "The paper only uses BM25 and never compares against a dense retriever such as DPR.",
    "c2": "The authors do not report results on WebQuestions, which limits the scope of the conclusions.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision", choices=("kev", "jev"), default="kev")
    ap.add_argument("--effort", default="none", help="reasoning effort of the decomposer")
    ap.add_argument("--decomposer", default="gpt-5.5", help="decomposer model name")
    args = ap.parse_args()
    llm = LLM(cache=JsonlCache(".cache/example_llm.jsonl"))
    dm = (kev if args.decision == "kev" else jev)(cache=JsonlCache(".cache/example_%s.jsonl" % args.decision))
    pc = PremiseCheck(llm, dm, field="nlp", decomposer=args.decomposer, effort=args.effort)
    paper = truncate(PAPER)
    for cid, comment in COMMENTS.items():
        r = pc.score(cid, comment, paper)
        print("%s  P(error) = %.3f  %s" % (cid, r.error_score, comment))
        for p in r.premises:
            if p.p is not None:
                print("    %-9s P(true) = %.3f  %s" % (p.kind, p.p, p.text))
        if r.fallback:
            print("    (no premise: the decision model judged the whole comment)")
        print("    cost: %s" % json.dumps({k: round(v, 6) for k, v in r.usd.items()}))


if __name__ == "__main__":
    main()
