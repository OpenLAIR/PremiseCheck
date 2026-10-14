from premisecheck.parsing import json_obj, parse_premises, parse_probs


def test_json_obj():
    assert json_obj('Sure.\n```json\n{"a": 1}\n```') == {"a": 1}
    assert json_obj("no json here") is None
    assert json_obj("{not json}") is None
    assert json_obj(None) is None


def test_parse_premises_plain():
    text = '{"premises": ["  The paper uses BM25. ", "", 3, "BM25 is sparse."], "conclusion": "x"}'
    assert parse_premises(text) == [("The paper uses BM25.", None), ("BM25 is sparse.", None)]


def test_parse_premises_typed():
    text = ('{"premises": [{"text": "A", "type": "Paper"}, {"text": "B", "type": "knowledge"},'
            ' {"text": "C", "type": "opinion"}, {"text": "D", "type": "claim"}, {"text": "E"}, {"text": " "}]}')
    assert parse_premises(text) == [("A", "paper"), ("B", "knowledge"), ("C", "opinion"), ("D", "paper"),
                                    ("E", "paper")]


def test_parse_premises_limit_and_garbage():
    text = '{"premises": [%s]}' % ", ".join('"p%d"' % i for i in range(10))
    assert len(parse_premises(text)) == 6
    assert len(parse_premises(text, max_premises=3)) == 3
    assert parse_premises("I cannot do that.") == []
    assert parse_premises('{"premises": null}') == []
    assert parse_premises('{"conclusion": "x"}') == []


def test_parse_probs_exact_and_clipped():
    assert parse_probs({"a": 80, "b": "20", "c": 150, "d": -5}, ["a", "b", "c", "d"]) == {
        "a": 0.8, "b": 0.2, "c": 1.0, "d": 0.0}


def test_parse_probs_single_and_brackets():
    assert parse_probs({"0": 70}, ["x#H"]) == {"x#H": 0.7}
    assert parse_probs({"[x#H]": 30}, ["x#H"]) == {"x#H": 0.3}


def test_parse_probs_positional():
    assert parse_probs({"1": 10, "2": 90}, ["a", "b"]) == {"a": 0.1, "b": 0.9}
    assert parse_probs({"0": 10, "1": 90}, ["a", "b"]) == {"a": 0.1, "b": 0.9}
    assert parse_probs({"0": 10, "2": 90}, ["a", "b"]) == {}  # not consecutive


def test_parse_probs_lists():
    assert parse_probs([{"id": "a", "p": 40}, {"id": "b", "p": 60}], ["a", "b"]) == {"a": 0.4, "b": 0.6}
    assert parse_probs([{"a": 40}], ["a"]) == {"a": 0.4}
    assert parse_probs([80, 20], ["a", "b"]) == {"a": 0.8, "b": 0.2}


def test_parse_probs_bad_values():
    assert parse_probs({"a": "high", "b": None, "c": 50}, ["a", "b", "c"]) == {"c": 0.5}
    assert parse_probs(None, ["a"]) == {}
    assert parse_probs("80", ["a"]) == {}
    assert parse_probs({"a": 50, "z": 1}, ["a", "b"]) == {"a": 0.5}  # partial answers are kept
