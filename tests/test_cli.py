import json

import pytest

from fakes import FakeDecision, FakeLLM, premises_json
from premisecheck import cli
from premisecheck.pipeline import PremiseCheck


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    labels = ["error", "correct", "correct", "error", "correct", "correct"]
    write_jsonl(d / "toy.jsonl", [{"id": "t%d" % i, "paper_id": "p%d" % (i % 3), "comment": "comment %d" % i,
                                   "label": l} for i, l in enumerate(labels)])
    write_jsonl(d / "papers_toy.jsonl", [{"paper_id": "p%d" % i, "title": "", "text": "paper %d" % i}
                                         for i in range(3)])
    return d


def run_rows(scores):
    return [{"id": "t%d" % i, "error_score": s, "usd": {"total": 0.01 * (i + 1)}, "latency_s": 1.0}
            for i, s in enumerate(scores)]


def test_eval(data_dir, tmp_path, capsys):
    write_jsonl(tmp_path / "a.jsonl", run_rows([0.9, 0.1, 0.2, 0.8, 0.3, 0.4]))
    write_jsonl(tmp_path / "b.jsonl", run_rows([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    cli.main(["eval", "--data", str(data_dir), "--source", "toy", "--scores", str(tmp_path / "a.jsonl"),
              "--ref", str(tmp_path / "b.jsonl"), "--bootstrap", "50"])
    out = capsys.readouterr().out
    assert "toy: 6 comments, 2 factual errors" in out
    assert "scores AUROC 100.0" in out and "ref    AUROC 50.0" in out
    assert "scores - ref: +50.0" in out and "cost 100.0% of ref" in out


def test_eval_partial_run(data_dir, tmp_path, capsys):
    write_jsonl(tmp_path / "a.jsonl", run_rows([0.9, 0.1, 0.2, 0.8])[:4])
    cli.main(["eval", "--data", str(data_dir), "--source", "toy", "--scores", str(tmp_path / "a.jsonl")])
    out = capsys.readouterr().out
    assert "4 of 6 comments" in out and "toy: 4 comments, 2 factual errors" in out


def test_run_and_resume(data_dir, tmp_path, monkeypatch, capsys):
    def decision(state, q):
        return 0.1 if "comment 0" in state.get("proposition", "") or "comment 3" in state.get("proposition", "") \
            else 0.9

    systems = []

    def system(args, cache_dir):
        assert args.decision == "kev" and args.effort == "none"
        llm = FakeLLM(lambda u: premises_json("The paper says %s." % u.split("\n")[1]))
        systems.append(PremiseCheck(llm, FakeDecision(decision), args.field or "nlp", effort=args.effort))
        return systems[-1]

    monkeypatch.setattr(cli, "_system", system)
    monkeypatch.setattr(cli, "truncate", lambda text: text)
    out = tmp_path / "runs" / "pc.jsonl"
    argv = ["run", "--data", str(data_dir), "--source", "toy", "--field", "nlp", "--effort", "none",
            "--out", str(out), "--cache-dir", str(tmp_path / "cache"), "--workers", "2"]
    cli.main(argv[:] + ["--limit", "4"])
    assert len(cli.data.read_jsonl(out)) == 4
    cli.main(argv)
    rows = cli.data.read_jsonl(out)
    assert sorted(r["id"] for r in rows) == ["t%d" % i for i in range(6)]
    assert len(systems[1].llm.calls) == 2  # resumed: only the two comments not yet scored
    by = {r["id"]: r for r in rows}
    assert by["t0"]["paper_id"] == "p0" and by["t0"]["error_score"] == pytest.approx(0.9)
    assert by["t1"]["error_score"] == pytest.approx(0.1)
    capsys.readouterr()
    cli.main(["eval", "--data", str(data_dir), "--source", "toy", "--scores", str(out)])
    assert "scores AUROC 100.0" in capsys.readouterr().out
