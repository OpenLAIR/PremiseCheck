import itertools
import math
import random

from premisecheck.metrics import auroc, paired_difference, paper_bootstrap


def brute_auroc(s, y):
    pos = [a for a, l in zip(s, y) if l]
    neg = [a for a, l in zip(s, y) if not l]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)) / (
        len(pos) * len(neg))


def test_auroc_basic():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5] * 4, [True, False, True, False]) == 0.5
    assert math.isnan(auroc([0.1, 0.2], [True, True]))
    assert math.isnan(auroc([0.1, 0.2], [False, False]))


def test_auroc_ties_match_pairwise():
    rng = random.Random(1)
    for _ in range(20):
        s = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(30)]
        y = [rng.random() < 0.3 for _ in range(30)]
        if any(y) and not all(y):
            assert abs(auroc(s, y) - brute_auroc(s, y)) < 1e-12


def rows(n_papers=12, seed=0):
    rng = random.Random(seed)
    out = []
    for p in range(n_papers):
        for _ in range(6):
            e = rng.random() < 0.3
            out.append({"paper_id": "p%d" % p, "error": e, "a": rng.random() + 0.4 * e, "b": rng.random() + 0.2 * e})
    return out


def test_bootstrap_deterministic():
    r = rows()
    fn = lambda s: auroc([x["a"] for x in s], [x["error"] for x in s])
    lo, hi = paper_bootstrap(r, fn, B=300, seed=7)
    assert (lo, hi) == paper_bootstrap(r, fn, B=300, seed=7)
    assert (lo, hi) == paper_bootstrap(list(reversed(r)), fn, B=300, seed=7)  # papers are sorted by id
    assert lo <= fn(r) <= hi


def test_paired_difference():
    r = rows()
    d, (lo, hi) = paired_difference(r, B=300)
    y = [x["error"] for x in r]
    assert abs(d - (auroc([x["a"] for x in r], y) - auroc([x["b"] for x in r], y))) < 1e-12
    assert lo <= hi
    same = [dict(x, b=x["a"]) for x in r]
    assert paired_difference(same, B=100) == (0.0, (0.0, 0.0))
