"""Reading ReviewFact, in either layout:
  flat:          <source>.jsonl and papers_<source>.jsonl            (e.g. rc.jsonl; scripts/build_reviewscore.py)
  Hugging Face:  data/<field>/test.jsonl and data/<field>/papers.jsonl (huggingface.co/datasets/OpenLAIR/ReviewFact)
"""
import json
from pathlib import Path

SOURCES = {"rc": "nlp", "rs": "ml", "prb": "science"}  # source -> field (prompt settings of PremiseCheck)


def read_jsonl(path):
    """Rows of a JSONL file; blank lines and a line cut off by an interrupted write are skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def paths(data_dir, source):
    """(comments file, papers file) of `source` in `data_dir`; the flat layout wins if both exist."""
    d = Path(data_dir)
    flat = (d / ("%s.jsonl" % source), d / ("papers_%s.jsonl" % source))
    if flat[0].exists():
        return flat
    expected = [str(flat[0])]
    if source in SOURCES:
        hf = d / "data" / SOURCES[source]
        if (hf / "test.jsonl").exists():
            return hf / "test.jsonl", hf / "papers.jsonl"
        expected.append(str(hf / "test.jsonl"))
    hint = " (the machine-learning part is not distributed; build it with scripts/build_reviewscore.py)" \
        if source == "rs" else ""
    raise FileNotFoundError("no %s comments: expected %s%s" % (source, " or ".join(expected), hint))


def load(data_dir, source):
    """Comments [{id, paper_id, comment, label, ...}] and {paper_id: full text} of one part of ReviewFact."""
    comments_path, papers_path = paths(data_dir, source)
    return read_jsonl(comments_path), {p["paper_id"]: p["text"] for p in read_jsonl(papers_path)}
