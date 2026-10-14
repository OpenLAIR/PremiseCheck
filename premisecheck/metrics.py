"""AUROC with factual errors as the positive class, and the paired paper bootstrap of Section 5."""
import random
from collections import defaultdict

import numpy as np


def auroc(scores, positive):
    """AUROC of `scores` for the boolean labels `positive`; ties count one half."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks, ss, i = np.empty(len(s)), s[order], 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paper_bootstrap(rows, fn, B=2000, seed=0):
    """95% interval of fn(rows) when papers are resampled with replacement. rows: dicts with "paper_id"."""
    by = defaultdict(list)
    for r in rows:
        by[r["paper_id"]].append(r)
    ps, rng, vals = sorted(by), random.Random(seed), []
    for _ in range(B):
        v = fn([r for _ in ps for r in by[ps[rng.randrange(len(ps))]]])
        if v == v:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"),) * 2


def paired_difference(rows, B=2000, seed=0):
    """AUROC(a) - AUROC(b) with its paper-bootstrap interval. rows: {"paper_id", "error", "a", "b"} (error scores)."""
    def diff(s):
        y = [r["error"] for r in s]
        return auroc([r["a"] for r in s], y) - auroc([r["b"] for r in s], y)
    return diff(rows), paper_bootstrap(rows, diff, B, seed)
