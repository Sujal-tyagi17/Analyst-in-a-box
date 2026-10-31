"""Tests for the agent loop: progress events, provider fallback, errors and SQL safety.

No network or API keys are used: providers are the offline mock or monkeypatched.
"""
import os

import pytest

from agent import orchestrator as O
from agent import schema_reader
from agent.tools import SQLValidationError, run_sql

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def sales(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("agent") / "sales.db")
    schema_reader.load_flat_file_to_sqlite(os.path.join(ROOT, "data", "sample_datasets", "sales.csv"),
                                           db, "sales", "csv")
    return db, schema_reader.introspect(db).to_prompt_text()


def test_progress_events_follow_real_steps(sales):
    db, schema = sales
    events = []
    res = O.answer_question("Show total revenue by region as a bar chart", db, schema,
                            provider="mock", progress=lambda e, d: events.append(e))
    assert not res.error
    assert events[:2] == ["sql", "chart"]          # only steps that actually ran
    assert res.charts and res.charts[0]["data"]     # rows kept for the interactive chart
    assert {"x", "y", "chart_type"} <= set(res.charts[0])


def test_fallback_to_second_provider(sales, monkeypatch):
    db, schema = sales
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.setenv("LLM_FALLBACK", "groq,gemini")

    def fake(system, question, runner, history, base_url, model, api_key):
        if "groq" in (base_url or ""):
            raise RuntimeError("Error code: 429 - rate limit exceeded")
        return "**42** answered by the backup"

    monkeypatch.setattr(O, "_run_openai_compatible", fake)
    events = []
    res = O.answer_question("q", db, schema, provider="groq", progress=lambda e, d: events.append(e))
    assert res.answer.startswith("**42**")
    assert res.provider_used == "gemini"
    assert res.fallback_note == "Answered by the backup AI model."
    assert "fallback" in events


def test_all_providers_fail_gives_friendly_message(sales, monkeypatch):
    db, schema = sales
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.setenv("LLM_FALLBACK", "groq,gemini")

    def boom(*a, **k):
        raise RuntimeError("Error code: 429 - {'error': {'message': 'quota', 'secret': 'sk-123'}}")

    monkeypatch.setattr(O, "_run_openai_compatible", boom)
    res = O.answer_question("q", db, schema, provider="groq")
    assert res.error
    assert "rate/quota limit" in res.answer
    assert "sk-123" not in res.answer and "{" not in res.answer   # raw API payload never shown


def test_unknown_error_is_not_exposed(sales, monkeypatch):
    db, schema = sales
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("LLM_FALLBACK", "groq")
    monkeypatch.setattr(O, "_run_openai_compatible",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("internal /path/secret.py line 9")))
    res = O.answer_question("q", db, schema, provider="groq")
    assert "secret.py" not in res.answer


def test_missing_key_message(sales, monkeypatch):
    db, schema = sales
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_FALLBACK", "groq")
    res = O.answer_question("q", db, schema, provider="groq")
    assert res.error and "GROQ_API_KEY" in res.answer


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "update sales set revenue = 0",
    "PRAGMA table_info(sales)",
    "WITH x AS (SELECT 1) DELETE FROM sales",
    "ATTACH DATABASE 'x.db' AS x",
    "",
])
def test_sql_safety_rejects_writes(sales, sql):
    db, _ = sales
    with pytest.raises(SQLValidationError):
        run_sql(sql, db)


def test_sql_auto_limit_and_truncation_warning(sales):
    db, _ = sales
    r = run_sql("SELECT * FROM sales", db, max_rows=10)
    assert r["row_count"] == 10 and r["truncated"] and r["warning"]
