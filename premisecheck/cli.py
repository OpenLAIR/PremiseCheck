"""Command line: score ReviewFact (or your own comments) with PremiseCheck or the holistic baseline, and evaluate.

  premisecheck run      --data data/ReviewFact --source rc --decision kev --effort none --out runs/pc_rc.jsonl
  premisecheck holistic --data data/ReviewFact --source rc --model gpt-5.5 --effort medium --out runs/hol_rc.jsonl
  premisecheck eval     --data data/ReviewFact --source rc --scores runs/pc_rc.jsonl --ref runs/hol_rc.jsonl
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import data, decision
from .cache import JsonlCache
from .holistic import HolisticLLM
from .llm import LLM
from .metrics import auroc, paired_difference
from .papers import truncate
from .pipeline import FIELDS, PremiseCheck


def _system(args, cache_dir):
    llm = LLM(cache=JsonlCache(cache_dir / "llm.jsonl"))
    if args.cmd == "holistic":
        return HolisticLLM(llm, args.model, args.effort, args.max_tokens)
    make = decision.jev if args.decision == "jev" else decision.kev
    dm = make(cache=JsonlCache(cache_dir / ("%s.jsonl" % args.decision)))
    field = args.field or data.SOURCES[args.source]
    return PremiseCheck(llm, dm, field, args.decomposer, args.effort, args.phrasings)


def score(args):
    comments, papers = data.load(args.data, args.source)
    comments.sort(key=lambda c: (c["paper_id"], c["id"]))  # one paper's comments together: prompt-prefix cache hits
    if args.limit:
        comments = comments[:args.limit]
    out = Path(args.out)
    done = {r["id"] for r in data.read_jsonl(out)} if out.exists() else set()
    if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
        with open(out, "a", encoding="utf-8") as f:  # an interrupted write left a partial last line
            f.write("\n")
    todo = [c for c in comments if c["id"] not in done]
    missing = sorted({c["paper_id"] for c in todo} - set(papers))
    if missing:
        sys.exit("error: no paper text for paper_id %s" % ", ".join(missing[:5]))
    print("%d comments, %d already scored, %d to go" % (len(comments), len(done), len(todo)), flush=True)
    system = _system(args, Path(args.cache_dir))
    text = {}
    lock, n, failed = threading.Lock(), [0], [0]
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(c):
        pid = c["paper_id"]
        with lock:
            if pid not in text:
                text[pid] = truncate(papers[pid])
        try:
            r = system.score(c["id"], c["comment"], text[pid]).to_dict()
        except Exception as e:  # leave it unscored; a rerun retries it
            with lock:
                failed[0] += 1
            print("  %s failed: %s: %s" % (c["id"], type(e).__name__, e), flush=True)
            return
        r["paper_id"] = pid
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 200 == 0:
                print("  %d/%d" % (n[0], len(todo)), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    if failed[0]:
        print("%d of %d comments failed and were left unscored; rerun the same command to retry them"
              % (failed[0], len(todo)))
    print("done: %s" % out)


def evaluate(args):
    comments, _ = data.load(args.data, args.source)
    label = {c["id"]: c["label"] == "error" for c in comments}
    paper = {c["id"]: c["paper_id"] for c in comments}
    runs = {"scores": data.read_jsonl(args.scores)}
    if args.ref:
        runs["ref"] = data.read_jsonl(args.ref)
    runs = {k: {r["id"]: r for r in v} for k, v in runs.items()}
    ids = [i for i in label if all(i in v for v in runs.values())]
    if not ids:
        sys.exit("error: no comment of %s is scored by every run" % args.source)
    if len(ids) < len(label):
        print("note: %d of %d comments scored by every run; evaluating those" % (len(ids), len(label)))
    y = [label[i] for i in ids]
    print("%s: %d comments, %d factual errors" % (args.source, len(ids), sum(y)))
    usd = {}
    for k, v in runs.items():
        usd[k] = float(np.mean([v[i].get("usd", {}).get("total", 0.0) for i in ids]))
        lat = float(np.mean([v[i].get("latency_s", 0.0) for i in ids]))
        print("  %-6s AUROC %.1f   cost $%.4f / comment   latency %.2f s" % (
            k, 100 * auroc([v[i]["error_score"] for i in ids], y), usd[k], lat))
    if args.ref:
        rows = [{"paper_id": paper[i], "error": label[i], "a": runs["scores"][i]["error_score"],
                 "b": runs["ref"][i]["error_score"]} for i in ids]
        d, (lo, hi) = paired_difference(rows, B=args.bootstrap)
        share = "; cost %.1f%% of ref" % (100 * usd["scores"] / usd["ref"]) if usd["ref"] > 0 else ""
        print("  scores - ref: %+.1f [%+.1f, %+.1f] (paper bootstrap, %d resamples)%s" % (
            100 * d, 100 * lo, 100 * hi, args.bootstrap, share))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="premisecheck", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    helps = {"run": "score comments with PremiseCheck", "holistic": "score comments with the holistic LLM baseline"}
    for name in ("run", "holistic"):
        s = sub.add_parser(name, help=helps[name])
        s.add_argument("--data", required=True, help="ReviewFact directory (flat or Hugging Face layout)")
        s.add_argument("--source", required=True,
                       help="rc (NLP), rs (machine learning), prb (natural sciences), or the name of your own data")
        s.add_argument("--out", required=True, help="output JSONL; an existing file is resumed")
        s.add_argument("--effort", default="medium",
                       help="reasoning effort of the LLM (medium, low, none; default: medium)")
        s.add_argument("--cache-dir", default=".cache", help="request caches, for resumable runs (default: .cache)")
        s.add_argument("--workers", type=int, default=16, help="comments scored in parallel (default: 16)")
        s.add_argument("--limit", type=int, default=0, help="score only the first N comments")
    run_p, hol_p = sub.choices["run"], sub.choices["holistic"]
    run_p.add_argument("--decision", choices=("jev", "kev"), default="kev", help="decision model (default: kev)")
    run_p.add_argument("--field", choices=sorted(FIELDS),
                       help="prompt and verification settings (default: from --source; required for your own data)")
    run_p.add_argument("--decomposer", default="gpt-5.5", help="decomposer model name (default: gpt-5.5)")
    run_p.add_argument("--phrasings", type=int, choices=(1, 16), default=1,
                       help="verification questions per premise (default: 1)")
    hol_p.add_argument("--model", default="gpt-5.5", help="judge model name (default: gpt-5.5)")
    hol_p.add_argument("--max-tokens", type=int, default=8000, help="max_completion_tokens (default: 8000)")
    ev = sub.add_parser("eval", help="AUROC, cost and latency of a run")
    ev.add_argument("--data", required=True, help="ReviewFact directory (flat or Hugging Face layout)")
    ev.add_argument("--source", required=True, help="rc, rs, prb, or the name of your own data")
    ev.add_argument("--scores", required=True, help="output JSONL of run or holistic")
    ev.add_argument("--ref", help="reference run for the paired difference (e.g. holistic GPT-5.5)")
    ev.add_argument("--bootstrap", type=int, default=2000, help="paper-bootstrap resamples (default: 2000)")
    args = ap.parse_args(argv)
    if args.cmd == "run" and not args.field and args.source not in data.SOURCES:
        ap.error("--field is required for --source %s (one of %s)" % (args.source, ", ".join(sorted(FIELDS))))
    try:
        evaluate(args) if args.cmd == "eval" else score(args)
    except FileNotFoundError as e:
        sys.exit("error: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
