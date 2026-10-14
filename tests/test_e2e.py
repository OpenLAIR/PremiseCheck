"""The command line end to end against a local fake server that speaks both APIs: OpenAI chat completions
(decomposer, holistic judge) and the TypeSafe-style /v1/systemone (Kev). Only 127.0.0.1 is contacted."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from premisecheck import cli
from premisecheck import prompts as P

PAPERS = {"p0": "We compare BM25 with DPR on Natural Questions.", "p1": "We evaluate on TriviaQA only."}
COMMENTS = [  # an "error" comment claims something the paper contradicts
    ("t0", "p0", "The paper never compares against DPR.", "error"),
    ("t1", "p0", "The paper uses Natural Questions.", "correct"),
    ("t2", "p1", "The paper never evaluates on TriviaQA.", "error"),
    ("t3", "p1", "The paper evaluates on TriviaQA.", "correct"),
]


def chat_answer(system, user):
    if system in (P.DECOMP_NLP, P.DECOMP_ML):
        comment = user.split("\n")[1]
        return json.dumps({"premises": [comment, "Dense retrievers need training data."], "conclusion": "c"})
    assert system == P.LLM_JUDGE_SYSTEM
    qid = user.rsplit("[", 1)[1].split("]")[0]
    return json.dumps({"p": {qid: 20 if "never" in user else 90}})


def noul(state, question):
    if question == P.ROUTE_Q:
        return 0.9 if state["statement"].startswith("The paper") else 0.1
    return 0.1 if "never" in state.get("proposition", "") else 0.85


class Handler(BaseHTTPRequestHandler):
    hits = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.hits.append(self.path)
        if self.path == "/v1/chat/completions":
            msgs = {m["role"]: m["content"] for m in body["messages"]}
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant",
                                            "content": chat_answer(msgs["system"], msgs["user"])}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}}
        elif self.path == "/v1/systemone":
            assert body["model"] == "kev-latest" and len(body["questions"]) <= 16
            out = {"answers": {q: {"type": "noul", "noul": noul(body["state"], v["instructions"])}
                               for q, v in body["questions"].items()},
                   "usage": {"input_tokens": 10}, "latency_ms": 5.0}
        else:
            self.send_error(404)
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("OPENAI_BASE_URL", url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("KEV_BASE_URL", url)
    monkeypatch.setattr(cli, "truncate", lambda text: text)  # no tokenizer download
    Handler.hits = []
    yield url
    srv.shutdown()


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "mine"
    d.mkdir()
    with open(d / "toy.jsonl", "w") as f:
        for cid, pid, comment, label in COMMENTS:
            f.write(json.dumps({"id": cid, "paper_id": pid, "comment": comment, "label": label}) + "\n")
    with open(d / "papers_toy.jsonl", "w") as f:
        for pid, text in PAPERS.items():
            f.write(json.dumps({"paper_id": pid, "text": text}) + "\n")
    return d


def test_cli_end_to_end(server, data_dir, tmp_path, capsys):
    common = ["--data", str(data_dir), "--source", "toy", "--cache-dir", str(tmp_path / "cache"), "--workers", "4"]
    pc, hol = tmp_path / "runs" / "pc.jsonl", tmp_path / "runs" / "hol.jsonl"
    cli.main(["run", *common, "--field", "nlp", "--effort", "none", "--out", str(pc)])
    cli.main(["holistic", *common, "--effort", "medium", "--out", str(hol)])
    rows = {r["id"]: r for r in cli.data.read_jsonl(pc)}
    assert set(rows) == {"t0", "t1", "t2", "t3"}
    assert rows["t0"]["error_score"] == pytest.approx(0.9) and rows["t1"]["error_score"] == pytest.approx(0.15)
    assert [p["kind"] for p in rows["t0"]["premises"]] == ["paper", "knowledge"]
    assert rows["t0"]["usd"]["decompose"] == pytest.approx((1000 * 5.0 + 100 * 30.0) / 1e6)
    n_hits = len(Handler.hits)
    assert Handler.hits.count("/v1/chat/completions") == 8  # 4 decompositions + 4 holistic judgments

    capsys.readouterr()
    cli.main(["eval", "--data", str(data_dir), "--source", "toy", "--scores", str(pc), "--ref", str(hol),
              "--bootstrap", "50"])
    out = capsys.readouterr().out
    assert "toy: 4 comments, 2 factual errors" in out
    assert "scores AUROC 100.0" in out and "ref    AUROC 100.0" in out and "scores - ref: +0.0" in out

    # A new output file with the same settings is answered from the caches: no request reaches the server.
    cli.main(["run", *common, "--field", "nlp", "--effort", "none", "--out", str(tmp_path / "again.jsonl")])
    assert len(Handler.hits) == n_hits
    again = {r["id"]: r["error_score"] for r in cli.data.read_jsonl(tmp_path / "again.jsonl")}
    assert again == {i: r["error_score"] for i, r in rows.items()}


def test_cli_needs_field_for_own_data(data_dir, tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--data", str(data_dir), "--source", "toy", "--out", str(tmp_path / "o.jsonl")])
    assert e.value.code == 2


def test_cli_missing_data(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "--data", str(tmp_path), "--source", "rs", "--scores", str(tmp_path / "x.jsonl")])
    assert "build_reviewscore.py" in str(e.value.code)
