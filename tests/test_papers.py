import pytest

from premisecheck import papers


@pytest.fixture(scope="module")
def encoding():
    pytest.importorskip("tiktoken")
    try:
        return papers._encoding()
    except Exception as e:  # o200k_base is downloaded on first use
        pytest.skip("o200k_base tokenizer unavailable: %s" % e)


def test_truncate(encoding):
    text = "The proposition is true according to the paper. " * 50
    n = papers.count_tokens(text)
    assert papers.truncate(text, n) == text
    short = papers.truncate(text, 20)
    assert papers.count_tokens(short) == 20 and text.startswith(short)
    assert papers.truncate("<|endoftext|> plain", 100) == "<|endoftext|> plain"  # special tokens are plain text
