# PremiseCheck: Efficient Fact-Checking of Human and AI Peer Reviews

Dingjie Song, Zhiling Yan, Ziqian Zhang, Yuxuan Zhang, Lichao Sun

[Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Project page](https://openlair.github.io/premisecheck/) | [ReviewFact dataset](https://huggingface.co/datasets/songdj/ReviewFact)

**TL;DR.** PremiseCheck scores how likely a peer-review comment contains a factual error about the paper it reviews.
An LLM rewrites the comment as explicit premises without reading the paper. A small decision model then checks each
premise against the paper, and the comment is scored by its weakest premise. Only the decision model reads the paper,
so the expensive LLM never processes the long paper text. On ReviewFact, a benchmark of expert-labeled comments from
NLP, machine learning and the natural sciences, PremiseCheck reaches AUROC comparable to a holistic GPT-5.5 judge at
about a fifth to a half of its cost.

## How it works

1. **Decompose.** An LLM (GPT-5.5 in the paper) reads only the comment and lists the premises it relies on.
2. **Route.** For NLP and machine learning reviews, the decision model decides whether each premise can be checked
   from the paper alone (threshold 0.5) or needs general knowledge.
3. **Verify.** The decision model answers whether each premise is true, reading the paper truncated to its first
   24k tokens (o200k_base).
4. **Aggregate.** The comment score is the lowest premise probability. A comment without premises is judged as a
   whole by the decision model.

Per-field settings: NLP and machine learning reviews use routing and the verification question above. For
natural-science reviews the decomposer types each premise (paper, knowledge, opinion), only paper premises are
checked, and the decision model is asked whether the paper contradicts the premise (score = 1 - p). Optionally,
16 phrasings of the verification question are asked in one call and averaged (the last eight are inverted).

Two decision models are supported:

- **Kev-27B** (`jaredpalmer/kev-27b@v1.0`), open weights, served on your own GPU. This is the default; see
  [docs/kev.md](docs/kev.md) for how to serve it.
- **Jev**, via OpenRouter's decisions endpoint (`/api/alpha/decisions`), which requires access to that alpha
  endpoint. Table 1 of the paper uses Jev. With Kev-27B, PremiseCheck reaches similar AUROC on the NLP part.

The decomposer and the holistic baseline use any OpenAI-compatible chat endpoint.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/OpenLAIR/PremiseCheck
cd PremiseCheck
pip install -e .            # or: pip install -e ".[dev]" for the tests
```

The o200k_base tokenizer used to truncate papers is downloaded by `tiktoken` on first use.

Environment variables:

| variable | used for |
|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | decomposer and holistic LLM (any OpenAI-compatible endpoint) |
| `KEV_BASE_URL`, `KEV_API_KEY` | Kev-27B server (default `http://localhost:8000`; the key only if the server requires one) |
| `OPENROUTER_API_KEY` | Jev (optional) |

## Quickstart (Python)

```python
from premisecheck import LLM, JsonlCache, PremiseCheck, kev, truncate

llm = LLM()                       # OPENAI_BASE_URL / OPENAI_API_KEY
pc = PremiseCheck(llm, kev(), field="nlp", decomposer="gpt-5.5", effort="none")

paper = truncate(open("paper.txt").read())     # first 24k tokens, once per paper
r = pc.score("c1", "The paper does not compare against BM25.", paper)

print(r.error_score)              # higher = more likely a factual error
for p in r.premises:
    if p.p is not None:           # premises that were verified
        print(p.kind, round(p.p, 3), p.text)
print(r.usd)                      # {"decompose", "decision", "total"} in USD
```

Use `jev()` instead of `kev()` for Jev, `field="ml"` or `field="science"` for the other settings, and
`phrasings=16` for the 16-phrasing variant. `HolisticLLM(llm, "gpt-5.5", "medium")` is the holistic baseline with
the same `score()` method. Pass `cache=JsonlCache(path)` to `LLM`, `kev` or `jev` to make runs resumable; identical
requests are then answered from the file. For a decomposer without a reasoning setting, pass `effort=None`. A
complete example is in [examples/example.py](examples/example.py).

## Command line

```bash
# PremiseCheck with Kev-27B and the decomposer at reasoning effort "none"
premisecheck run      --data data/ReviewFact --source rc --decision kev --effort none --out runs/pc_rc.jsonl

# holistic GPT-5.5 baseline
premisecheck holistic --data data/ReviewFact --source rc --model gpt-5.5 --effort medium --out runs/hol_rc.jsonl

# AUROC, cost and latency, and the paired paper-bootstrap difference to a reference run
premisecheck eval     --data data/ReviewFact --source rc --scores runs/pc_rc.jsonl --ref runs/hol_rc.jsonl
```

`--source` is `rc` (NLP), `rs` (machine learning) or `prb` (natural sciences); it also selects the per-field
settings (override with `--field nlp|ml|science`). Other options of `run`: `--decision jev|kev`, `--decomposer`,
`--phrasings 1|16`, `--workers`, `--limit`, `--cache-dir`. An existing `--out` file is resumed, and requests are
cached in `.cache/`; comments that fail (for example after an API error) are left out and retried by running the same
command again. Each output line is one comment: `error_score`, `score`, the premises with their probabilities, cost
in USD and latency. Run `premisecheck <command> -h` for all options; `python -m premisecheck` works as well.

To score your own comments, write `<name>.jsonl` (one `{"id", "paper_id", "comment"}` per line; `"label"`, `error`
or `correct`, is needed only for `eval`) and `papers_<name>.jsonl` (`{"paper_id", "text"}`) into a directory and
pass `--data <directory> --source <name> --field nlp`.

## ReviewFact

Download the NLP and natural-science parts from Hugging Face (`pip install -U huggingface_hub` provides `hf`):

```bash
hf download songdj/ReviewFact --repo-type dataset --local-dir data/ReviewFact
```

The machine-learning part comes from ReviewScore, which states no license, so it is not redistributed. Rebuild it
from the ReviewScore repository (the repository name is spelled `ReivewScore`):

```bash
git clone https://github.com/Hyun-Ryu/ReivewScore
python scripts/build_reviewscore.py --reviewscore ReivewScore --out data/ReviewFact
# rs: 216 comments, 46 errors, 39 papers  [ok]
```

`premisecheck` reads both layouts from the same directory: `data/<field>/test.jsonl` and `papers.jsonl` from Hugging
Face, and `rs.jsonl` and `papers_rs.jsonl` from the script.

| part | source | papers | comments | errors |
|---|---|---|---|---|
| NLP (`rc`) | ReviewCritique | 100 | 6,466 | 308 |
| Machine learning (`rs`) | ReviewScore | 39 | 216 | 46 |
| Natural sciences (`prb`) | PeerReview Bench | 78 | 2,585 | 65 |

See the [dataset card](https://huggingface.co/datasets/songdj/ReviewFact) for the label rules. ReviewFact is for
evaluation only: do not use it to train models.

## Results

AUROC (x100) on ReviewFact with each comment checked on its own, and cost as a percentage of holistic GPT-5.5
(reasoning medium). PremiseCheck uses GPT-5.5 as the decomposer at the given reasoning effort and Jev as the decision
model (Table 1 of the paper).

| system | reasoning | NLP AUROC | NLP cost | ML AUROC | ML cost | Science AUROC | Science cost |
|---|---|---|---|---|---|---|---|
| Holistic GPT-5.5 (ref.) | medium | 63.2 | 100.0 | 66.9 | 100.0 | 67.8 | 100.0 |
| PremiseCheck | medium | 63.0 | 51.2 | 68.7 | 46.6 | 67.5 | 47.4 |
| PremiseCheck | low | 64.1 | 34.3 | 63.9 | 29.2 | 65.7 | 26.2 |
| PremiseCheck | none | 63.1 | 24.6 | 63.7 | 19.5 | 66.0 | 22.3 |

PremiseCheck is comparable to holistic GPT-5.5 at a fraction of its cost. On the NLP part all three settings are
non-inferior to the reference (the 95% interval of the paired AUROC difference lies above -3). With 16 phrasings of
the verification question, the NLP AUROC of PremiseCheck (reasoning none) rises from 63.1 to 65.7. The
machine-learning and natural-science parts have only 46 and 65 errors, so their confidence intervals are wide.

## Reproducing the paper

Replaying the original API calls of the paper through this package on the released ReviewFact files gives:

- **NLP and machine learning.** AUROC, cost share and bootstrap intervals of Table 1 reproduce exactly (e.g. NLP
  63.0 / 64.1 / 63.1 for medium / low / none, 65.7 with 16 phrasings on none, holistic GPT-5.5 63.2; ML
  68.7 / 63.9 / 63.7 against 66.9). One accounting difference: the package also counts the cost of the
  whole-comment fallback (comments without premises), so NLP medium is 51.3% of the reference instead of 51.2%.
- **Natural sciences.** The premise path reproduces exactly. For the about 1% of comments with no premise to verify
  (17-22 of 2,585), the paper gave the decision model the PeerReview Bench judgment input, while this package asks
  the generic whole-comment question, so those few scores can differ (AUROC 66.2 with reasoning none on the comments
  covered, against 66.0 in the paper). The holistic GPT-5.5 baseline of this part used the PeerReview Bench judgment
  prompt and is not included.

Fresh API runs are not bit-identical because the models sample, so expect small differences.

## Repository layout

| path | content |
|---|---|
| `premisecheck/pipeline.py` | PremiseCheck: decompose, route, verify, aggregate |
| `premisecheck/prompts.py` | decomposer prompts and decision-model questions, verbatim from the paper |
| `premisecheck/decision.py` | Jev and Kev-27B clients |
| `premisecheck/llm.py`, `holistic.py` | OpenAI-compatible LLM client and the holistic baseline |
| `premisecheck/metrics.py` | AUROC and the paired paper bootstrap |
| `premisecheck/cli.py` | the `premisecheck` command |
| `scripts/build_reviewscore.py` | rebuilds the machine-learning part of ReviewFact |
| `docs/kev.md` | serving Kev-27B |

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests use fake models and a local fake server, so they need no API keys or GPU. The only download is the
o200k_base tokenizer; the test that needs it is skipped if it cannot be loaded.

Questions and bug reports are welcome as [GitHub issues](https://github.com/OpenLAIR/PremiseCheck/issues).

## Citation

```bibtex
@article{song2026premisecheck,
  title   = {{PremiseCheck}: Efficient Fact-Checking of Human and {AI} Peer Reviews},
  author  = {Song, Dingjie and Yan, Zhiling and Zhang, Ziqian and Zhang, Yuxuan and Sun, Lichao},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

Please also cite the source datasets of ReviewFact: ReviewCritique (Du et al., 2024), ReviewScore (Ryu et al., 2025)
and PeerReview Bench (Kim et al., 2026).

## License

The code is released under the [MIT License](LICENSE).

ReviewFact keeps the licenses of its sources:

- **ReviewCritique** (NLP part): Apache NON-AI License 2.0. Evaluation and benchmarking only; it must not be used to
  train models.
- **PeerReview Bench** (natural-science part): CC-BY-4.0, intended for evaluation only, not for training.
- **ReviewScore** (machine-learning part): no license stated; not redistributed. Rebuild it with
  `scripts/build_reviewscore.py`.
