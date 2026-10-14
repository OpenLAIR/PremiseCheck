"""Rebuild the machine-learning part of ReviewFact (rs.jsonl, papers_rs.jsonl) from the ReviewScore repository.

ReviewScore states no license, so ReviewFact does not redistribute it. Clone the repository and run

  git clone https://github.com/Hyun-Ryu/ReivewScore
  python scripts/build_reviewscore.py --reviewscore ReivewScore --out data/ReviewFact

Comments are ReviewScore's argument items. The label follows ReviewScore's definition of a misinformed item: error
when the lowest aggregated premise score is below 3, correct when it is above 3; items whose lowest score is exactly
3, or without scored premises, are left out. Expected: 216 comments, 46 errors, 39 papers.
"""
import argparse
import json
from pathlib import Path

EXPECT = (216, 46, 39)


def paper_text(p):
    """ReviewScore util.paper_generation(paper) with use_appendix=False."""
    line = "Title:\n%s\n\nAbstract:\n%s" % (p["title"], p["ABSTRACT"])
    for sec, imgs in p["table_figure_positions"].items():
        if sec == "APPENDIX":
            continue
        line += "\n\n"
        for img in imgs:
            if "Figure" in img:
                line += "%s: %s\n\n" % (img, p["figures"][img])
            elif "Table" in img:
                line += "%s: %s\n%s\n\n" % (img, p["tables"][img]["caption"], p["tables"][img]["content"])
        line += "%s:\n%s" % (sec, p["[%s]" % sec])
    return line


def build(root):
    """Comment rows and {paper_id: paper} from a ReviewScore checkout."""
    root = Path(root)
    with open(root / "data/iclr/annotation_iclr.json", encoding="utf-8") as f:
        A = json.load(f)
    with open(root / "data/iclr/papers_iclr.json", encoding="utf-8") as f:
        P = {p["title"]: p for p in json.load(f)}
    rows, papers = [], {}
    for pi, a in enumerate(A):
        for ri, rv in enumerate(a["pairs_by_reviewers"]):
            for vi, r in enumerate(rv["reviews"]):
                if r["type"] != "argument":
                    continue
                ps = [e.get("score_agg") for e in (r.get("eval") or []) if e.get("score_agg") is not None]
                if not ps or min(ps) == 3:
                    continue
                pid = "rs%d" % pi
                rows.append({"id": "rs|%d|%d|%d" % (pi, ri, vi), "source": "rs", "split": "test", "paper_id": pid,
                             "comment": r["review"].strip(), "reviewer_type": "human",
                             "label": "correct" if min(ps) > 3 else "error", "error_type": None,
                             "label_origin": "expert"})
                if pid not in papers:
                    papers[pid] = {"paper_id": pid, "title": a["title"], "text": paper_text(P[a["title"]])}
    return rows, papers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reviewscore", required=True, help="clone of github.com/Hyun-Ryu/ReivewScore")
    ap.add_argument("--out", required=True, help="ReviewFact directory to write rs.jsonl and papers_rs.jsonl into")
    args = ap.parse_args()
    rows, papers = build(args.reviewscore)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "papers_rs.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(papers.values(), key=lambda p: p["paper_id"]):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    got = (len(rows), sum(r["label"] == "error" for r in rows), len(papers))
    status = "ok" if got == EXPECT else "expected %s" % (EXPECT,)
    print("rs: %d comments, %d errors, %d papers  [%s]" % (got + (status,)))
    print("wrote", out)


if __name__ == "__main__":
    main()
