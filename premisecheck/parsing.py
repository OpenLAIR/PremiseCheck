"""Parsing of LLM answers: the decomposer's premise list and the judge's probabilities."""
import json
import re


def json_obj(text):
    """The outermost {...} of an LLM answer as a dict, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_premises(text, max_premises=6):
    """Premises of a decomposer answer as [(text, type)].

    The NLP and machine learning prompts return plain strings (type None). The natural-science prompt returns
    {"text", "type"} objects; an unknown type counts as "paper". An unparsable answer gives no premises."""
    out = []
    for p in (json_obj(text) or {}).get("premises") or []:
        if isinstance(p, str) and p.strip():
            out.append((p.strip(), None))
        elif isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
            t = str(p.get("type", "")).strip().lower()
            out.append((p["text"].strip(), t if t in ("paper", "knowledge", "opinion") else "paper"))
    return out[:max_premises]


def parse_probs(p, qids):
    """Map an LLM judge's {"p": ...} answer to question ids, as probabilities in [0, 1].

    LLMs do not always copy the ids: a single-question call often comes back as {"0": 80}, ids may be wrapped in
    brackets, and a few answers are lists. Exact ids win; otherwise a single answer maps to a single question, and
    numeric keys map by position (0- or 1-based) when the counts match."""
    if isinstance(p, list):
        d = {}
        for i, e in enumerate(p):
            if isinstance(e, dict) and "id" in e and "p" in e:
                d[str(e["id"])] = e["p"]
            elif isinstance(e, dict) and len(e) == 1:
                d.update({str(k): v for k, v in e.items()})
            else:
                d[str(i)] = e
        p = d
    if not isinstance(p, dict):
        return {}
    norm = {str(k).strip().strip("[]").strip(): v for k, v in p.items()}

    def val(v):
        try:
            return min(max(float(v), 0.0), 100.0) / 100.0
        except (TypeError, ValueError):
            return None

    out = {q: val(norm[q]) for q in qids if q in norm}
    if len(out) < len(qids):
        if len(qids) == 1 and len(norm) == 1:
            out = {qids[0]: val(next(iter(norm.values())))}
        elif len(norm) == len(qids) and all(k.isdigit() for k in norm):
            ks = sorted(norm, key=int)
            base = int(ks[0])
            if [int(k) - base for k in ks] == list(range(len(qids))):
                out = {q: val(norm[str(base + i)]) for i, q in enumerate(qids)}
    return {q: v for q, v in out.items() if v is not None}
