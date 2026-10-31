"""answer_cache.py — shared, thread-safe cache for AI answers.

Why not st.cache_data: it records any Streamlit element drawn while the
function runs (our live progress steps) and tries to replay them on a cache
hit, which fails once that progress box is gone. This cache stores only the
result object.

Behaviour kept from the previous st.cache_data setup:
- answers are reused for `ttl` seconds (no repeat API calls, no extra quota);
- a per-key lock makes a duplicate request (double click, rerun while the first
  call is still running) wait for the first answer instead of calling the AI again;
- failed answers are never cached.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable


class AnswerCache:
    def __init__(self, ttl: float = 600, max_entries: int = 200):
        self.ttl, self.max_entries = ttl, max_entries
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._locks: dict[Hashable, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lookup(self, key: Hashable) -> tuple[bool, Any]:
        with self._guard:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < self.ttl:
                self._data.move_to_end(key)
                return True, hit[1]
            if hit:
                del self._data[key]
        return False, None

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any],
                       should_cache: Callable[[Any], bool] = lambda r: True) -> tuple[Any, bool]:
        """Return (value, from_cache)."""
        found, value = self._lookup(key)
        if found:
            return value, True
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            found, value = self._lookup(key)  # another thread may have just computed it
            if found:
                return value, True
            value = compute()
            if should_cache(value):
                with self._guard:
                    self._data[key] = (time.monotonic(), value)
                    self._data.move_to_end(key)
                    while len(self._data) > self.max_entries:
                        old, _ = self._data.popitem(last=False)
                        self._locks.pop(old, None)
            return value, False

    def __len__(self) -> int:
        return len(self._data)
