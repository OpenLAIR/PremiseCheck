import json

import pytest

from premisecheck import data


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_flat_and_hf_layouts(tmp_path):
    c = [{"id": "rc|1", "paper_id": "p1", "comment": "x", "label": "correct"}]
    p = [{"paper_id": "p1", "title": "t", "text": "paper"}]
    _write(tmp_path / "flat" / "rc.jsonl", c)
    _write(tmp_path / "flat" / "papers_rc.jsonl", p)
    _write(tmp_path / "hf" / "data" / "nlp" / "test.jsonl", c)
    _write(tmp_path / "hf" / "data" / "nlp" / "papers.jsonl", p)
    for d in ("flat", "hf"):
        comments, papers = data.load(tmp_path / d, "rc")
        assert comments == c and papers == {"p1": "paper"}


def test_read_jsonl_skips_partial_line(tmp_path):
    f = tmp_path / "out.jsonl"
    f.write_text('{"id": "a"}\n\n{"id": "b"')
    assert data.read_jsonl(f) == [{"id": "a"}]


def test_missing_part(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_reviewscore"):
        data.load(tmp_path, "rs")
    with pytest.raises(FileNotFoundError, match="mine.jsonl"):
        data.load(tmp_path, "mine")
