# Serving Kev-27B

Kev-27B is an open-weight decision model. PremiseCheck uses it as a drop-in replacement for Jev: both read a state
(named texts such as `paper` and `proposition`) and answer yes/no questions about it with probabilities. On the NLP
part of ReviewFact, PremiseCheck with Kev-27B reaches AUROC similar to PremiseCheck with Jev.

- Model: https://huggingface.co/jaredpalmer/kev-27b (the paper uses the `v1.0` tag)
- Server code: https://github.com/jaredpalmer/kev (the paper uses the `kev-1.0` tag), which serves a
  TypeSafe-compatible `/v1/systemone` endpoint. Its README has the requirements and the GPU options.

## Start a server

One server needs one 80 GB GPU (for example an A100 or H100). Following the Kev README (Python 3.12 or 3.13 and
[uv](https://docs.astral.sh/uv/)):

```bash
git clone --branch kev-1.0 https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b@v1.0 --host 0.0.0.0 --port 8000
```

The first start downloads the weights (about 51 GB). Set `KEV_API_KEY` on the server to require
`Authorization: Bearer <key>`; without it, the server is open to anyone who can reach the port.

## Point PremiseCheck at it

```bash
export KEV_BASE_URL=http://localhost:8000   # default
export KEV_API_KEY=...                      # only if your server requires one
premisecheck run --data data/ReviewFact --source rc --decision kev --effort none --out runs/pc_rc.jsonl
```

or in Python:

```python
from premisecheck import kev
dm = kev(base_url="http://localhost:8000")
```

The client sends `POST {KEV_BASE_URL}/v1/systemone` with

```json
{"model": "kev-latest", "state": {"paper": "...", "proposition": "..."},
 "questions": {"<qid>": {"type": "noul", "instructions": "The proposition is true according to the paper."}}}
```

and reads `{"answers": {"<qid>": {"noul": 0.83}}}`. Up to 16 questions go in one request. While the server is
unreachable or busy (connection errors, timeouts, HTTP 408, 429 and 5xx), the client logs a warning and retries for up
to an hour. Any server that implements this request and answer format can be used in the same way.
