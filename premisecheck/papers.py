"""Paper input: the systems of the paper read the first 24k tokens (o200k_base) of the paper text."""
from functools import lru_cache

PAPER_TOKENS = 24000


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    return len(_encoding().encode(text, disallowed_special=()))


def truncate(text, max_tokens=PAPER_TOKENS):
    """The first `max_tokens` tokens of `text` (unchanged when it is shorter). Apply once per paper."""
    tokens = _encoding().encode(text, disallowed_special=())
    return text if len(tokens) <= max_tokens else _encoding().decode(tokens[:max_tokens])
