"""Append-only JSONL cache of model calls, keyed by the hash of the request. Makes runs resumable."""
import hashlib
import json
import threading
from pathlib import Path


def request_hash(obj):
    """Cache key of a request: sha256 of its canonical JSON (sorted keys), first 32 hex digits."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:32]


def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            return True
        f.seek(-1, 2)
        return f.read(1) == b"\n"


class JsonlCache:
    """{hash: record} backed by a JSONL file; a later record of the same hash wins. `read_only` keeps new records in
    memory only (for replaying a finished run). Safe to share between threads, not between processes."""

    def __init__(self, path, read_only=False):
        self.path, self.read_only = Path(path), read_only
        self._lock = threading.Lock()
        self._d = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a line cut off by an interrupted run
                    if isinstance(r, dict) and "h" in r:
                        self._d[r["h"]] = r
            if not read_only and not _ends_with_newline(self.path):
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write("\n")  # so the next record does not continue the cut-off line

    def get(self, h):
        return self._d.get(h)

    def put(self, rec):
        """Store a record ({"h": hash, ...}) and append it to the file."""
        with self._lock:
            self._d[rec["h"]] = rec
            if self.read_only:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
