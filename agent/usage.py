"""usage.py — lightweight daily usage cap backed by a tiny SQLite counter.

Keeps a global daily question count so a public demo can't run up an unbounded
API bill. Per-session caps are handled in app.py via st.session_state.
"""

from __future__ import annotations

import os
import datetime as dt
import sqlite3

_COUNTER_DB = os.environ.get("USAGE_DB", "data/_usage.db")
MAX_DAILY = int(os.environ.get("MAX_DAILY_QUESTIONS", "200"))


def _conn():
    os.makedirs(os.path.dirname(_COUNTER_DB) or ".", exist_ok=True)
    c = sqlite3.connect(_COUNTER_DB)
    c.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, n INTEGER)")
    return c


def today() -> str:
    return dt.date.today().isoformat()


def get_count() -> int:
    with _conn() as c:
        row = c.execute("SELECT n FROM usage WHERE day = ?", (today(),)).fetchone()
        return int(row[0]) if row else 0


def increment() -> int:
    with _conn() as c:
        c.execute(
            "INSERT INTO usage(day, n) VALUES(?, 1) "
            "ON CONFLICT(day) DO UPDATE SET n = n + 1",
            (today(),),
        )
        row = c.execute("SELECT n FROM usage WHERE day = ?", (today(),)).fetchone()
        return int(row[0])


def daily_limit_reached() -> bool:
    return get_count() >= MAX_DAILY
