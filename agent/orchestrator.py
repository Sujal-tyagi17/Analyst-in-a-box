"""orchestrator.py — the agent loop (provider-swappable tool-calling).

Supports Anthropic Claude plus any OpenAI-compatible endpoint (OpenAI, Groq,
Google Gemini) via a thin adapter. The LLM is given the schema summary + tool
definitions and autonomously chains run_sql -> run_analysis -> make_chart ->
written insight. Nothing is hardcoded to a dataset.

Provider is selected by LLM_PROVIDER ("groq" | "gemini" | "anthropic" | "openai").
Groq and Gemini have free tiers. Keys come from the matching *_API_KEY env var.
"""

from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import tools as toolmod
from .prompts import build_system_prompt, TOOLS

log = logging.getLogger("aib")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[aib %(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
LLM_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "60"))

# OpenAI-compatible providers (all use the OpenAI SDK with a base_url override).
# Groq and Gemini both offer a *free* tier and support tool/function calling.
OPENAI_COMPATIBLE = {
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "key_env": "OPENAI_API_KEY",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        # gpt-oss-120b does structured tool-calling reliably; llama-3.3 on Groq
        # intermittently emits malformed <function=...> text and gets rejected.
        "key_env": "GROQ_API_KEY",
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
    },
}


@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)   # trace of tool calls
    sql_queries: list[str] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)   # {"image_base64":..., "title":...}
    analyses: list[dict] = field(default_factory=list)
    error: str | None = None
    query_results: list[dict] = field(default_factory=list)  # {"sql", "rows"} for CSV export
    provider_used: str | None = None    # provider that produced this answer
    fallback_note: str | None = None    # set when a fallback provider answered


# ---------------------------------------------------------------------------
# Tool execution (shared across providers)
# ---------------------------------------------------------------------------

class _ToolRunner:
    """Executes tool calls and keeps the last query result for chaining."""

    def __init__(self, db_path: str, result: AgentResult,
                 progress: Callable[[str, str], None] | None = None):
        self.db_path = db_path
        self.result = result
        self.last_query_result: dict | None = None
        self.progress = progress

    def notify(self, event: str, detail: str = "") -> None:
        """Report a real step (model call, SQL, analysis, chart) to the UI, if it listens."""
        if self.progress:
            try:
                self.progress(event, detail)
            except Exception:  # the UI must never break the analysis
                log.debug("progress callback failed", exc_info=True)

    def execute(self, name: str, args: dict) -> dict:
        log.info("tool call: %s args=%s", name, json.dumps(args, default=str)[:300])
        self.notify({"run_sql": "sql", "run_analysis": "analysis", "make_chart": "chart"}.get(name, name),
                    args.get("query") or args.get("method") or args.get("chart_type") or "")
        try:
            if name == "run_sql":
                out = toolmod.run_sql(args["query"], self.db_path)
                self.last_query_result = out
                self.result.sql_queries.append(out["sql"])
                self.result.query_results.append(
                    {"sql": out["sql"], "rows": out["_df"].to_dict(orient="records")})
                # strip the internal DataFrame before returning to the LLM
                payload = {k: v for k, v in out.items() if k != "_df"}
                # cap rows shown to the model to keep tokens sane
                payload["rows"] = payload["rows"][:50]
                self.result.steps.append({"tool": name, "args": args, "ok": True})
                return payload

            if name == "run_analysis":
                if self.last_query_result is None:
                    return {"error": "Call run_sql before run_analysis."}
                method = args.pop("method")
                out = toolmod.run_analysis(self.last_query_result, method, **args)
                self.result.analyses.append(out)
                self.result.steps.append({"tool": name, "args": {"method": method, **args}, "ok": True})
                return out

            if name == "make_chart":
                if self.last_query_result is None:
                    return {"error": "Call run_sql before make_chart."}
                out = toolmod.make_chart(
                    self.last_query_result,
                    chart_type=args.get("chart_type", "bar"),
                    x=args.get("x"), y=args.get("y"), title=args.get("title"),
                )
                if out.get("image_base64"):
                    df = self.last_query_result.get("_df")
                    self.result.charts.append({
                        "image_base64": out["image_base64"],   # matplotlib PNG (used in the PDF report)
                        "title": out.get("title"),
                        "chart_type": out.get("chart_type"), "x": out.get("x"), "y": out.get("y"),
                        # the rows behind the chart, so the UI can redraw it interactively
                        "data": df.head(500).to_dict(orient="records") if df is not None else [],
                    })
                # don't ship the base64 blob back to the model
                echo = {k: v for k, v in out.items() if k != "image_base64"}
                echo["rendered"] = bool(out.get("image_base64"))
                self.result.steps.append({"tool": name, "args": args, "ok": True})
                return echo

            return {"error": f"Unknown tool '{name}'."}
        except toolmod.SQLValidationError as e:
            log.warning("tool %s: SQL rejected: %s", name, e)
            self.result.steps.append({"tool": name, "args": args, "ok": False, "error": str(e)})
            return {"error": f"SQL rejected: {e}"}
        except Exception as e:  # never let a bad tool call crash the loop
            log.exception("tool %s failed", name)
            self.result.steps.append({"tool": name, "args": args, "ok": False, "error": str(e)})
            return {"error": f"Tool error: {e}"}


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _run_anthropic(system: str, question: str, runner: _ToolRunner,
                   history: list[dict] | None) -> str:
    import anthropic
    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_S)
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": question})

    for step in range(MAX_STEPS):
        runner.notify("thinking" if step == 0 else "reviewing", DEFAULT_ANTHROPIC_MODEL)
        resp = client.messages.create(
            model=DEFAULT_ANTHROPIC_MODEL,
            max_tokens=1500,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = runner.execute(block.name, dict(block.input))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, default=str),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Reached the analysis step limit before finishing. Partial results above."


def _run_mock(question: str, schema_text: str, runner: _ToolRunner) -> str:
    """Offline, key-free test double: a generic keyword planner over the schema.

    It knows NOTHING about any specific dataset — it parses the schema summary
    to find the table + numeric/categorical columns, then maps question keywords
    to run_sql / run_analysis / make_chart. Used for CI smoke tests and demos
    without API spend. It is intentionally simpler than a real LLM.
    """
    import re as _re

    q = question.lower()

    # parse schema text: first table name + its columns with declared types
    tbl_match = _re.search(r'TABLE "([^"]+)"', schema_text)
    table = tbl_match.group(1) if tbl_match else "dataset"
    col_rows = _re.findall(r'- "([^"]+)" (\w+)', schema_text)
    numeric, categorical = [], []
    for name, typ in col_rows:
        if any(t in typ.upper() for t in ("INT", "FLOAT", "REAL", "NUM", "DEC")):
            numeric.append(name)
        else:
            categorical.append(name)
    # measures = numeric cols that aren't ids or time/period columns
    _timeish = ("year", "season", "date", "month", "day", "time", "quarter", "week")
    def _is_measure(c):
        cl = c.lower()
        return not cl.endswith("id") and not any(t in cl for t in _timeish)
    measures = [c for c in numeric if _is_measure(c)] or numeric
    dims = [c for c in categorical] or [c for c in numeric if c.lower().endswith("id")]

    def pick(cols, *keywords):
        for kw in keywords:
            for c in cols:
                if kw in c.lower():
                    return c
        return cols[0] if cols else None

    def named(cols):
        """Columns explicitly mentioned in the question (whole-word match)."""
        out = []
        for c in cols:
            token = c.lower().replace("_", " ")
            if _re.search(r"\b" + _re.escape(c.lower()) + r"\b", q) or \
               _re.search(r"\b" + _re.escape(token) + r"\b", q):
                out.append(c)
        return out

    # --- intent routing -------------------------------------------------
    # explicit chart request wins over aggregate keywords
    if any(w in q for w in ("chart", "plot", "graph", "visualize", "distribution")):
        cand_dims = named(categorical) + named(numeric)
        dim = cand_dims[0] if cand_dims else (dims[0] if dims else
              (categorical[0] if categorical else numeric[0]))
        m = named(measures)
        measure = m[0] if m else (pick(measures, "revenue", "points", "score", "value")
                                  or (measures[0] if measures else None))
        if measure == dim and len(measures) > 1:
            measure = next((x for x in measures if x != dim), measure)
        runner.execute("run_sql", {"query":
            f'SELECT "{dim}", SUM("{measure}") AS total FROM "{table}" '
            f'GROUP BY "{dim}" ORDER BY total DESC LIMIT 20'})
        runner.execute("make_chart", {"chart_type": "bar", "x": dim, "y": "total",
                                       "title": f"{measure} by {dim}"})
        return f"Here is a bar chart of total {measure} by {dim}."

    # correlation
    if "correlation" in q or "correlate" in q or "relationship between" in q:
        mentioned = named(numeric)
        x = mentioned[0] if len(mentioned) >= 1 else (measures[0] if measures else None)
        y = mentioned[1] if len(mentioned) >= 2 else (measures[-1] if len(measures) > 1 else None)
        runner.execute("run_sql", {"query": f'SELECT "{x}", "{y}" FROM "{table}"'})
        a = runner.execute("run_analysis", {"method": "correlation", "x": x, "y": y})
        return f"Correlation analysis between {x} and {y}: {a.get('note', '')}"

    # outliers / anomalies
    if any(w in q for w in ("outlier", "anomal", "unusual", "abnormal")):
        col = pick(measures, "revenue", "amount", "score", "value") or (measures[0] if measures else None)
        runner.execute("run_sql", {"query": f'SELECT "{col}" FROM "{table}"'})
        a = runner.execute("run_analysis", {"method": "outliers", "column": col})
        return f"Outlier scan on {col}: {a.get('note', '')}"

    # group comparison ("differ", "compare", "between X and Y")
    if any(w in q for w in ("differ", "compare", "difference between")):
        ng = named(categorical)
        grp = ng[0] if ng else (dims[0] if dims else None)
        nv = named(measures)
        val = nv[0] if nv else (pick(measures, "score", "revenue", "value")
                                or (measures[0] if measures else None))
        runner.execute("run_sql", {"query": f'SELECT "{grp}", "{val}" FROM "{table}"'})
        a = runner.execute("run_analysis", {"method": "group_compare", "group": grp, "value": val})
        return f"Group comparison of {val} across {grp}: {a.get('note', '')}"

    # count / how many  (but "highest number of X" is a max, handled below)
    _superlative = any(w in q for w in ("highest", "most", "maximum", "max", "largest", "top", "greatest"))
    if (("how many" in q or "number of" in q or q.strip().startswith("count"))
            and not _superlative):
        where = ""
        # detect "<col> <value>" numeric filters like "grade level 12"
        for c in numeric:
            m = _re.search(_re.escape(c.lower().replace("_", " ")) + r"\D{0,4}(\d+)", q)
            if m:
                where = f' WHERE "{c}" = {m.group(1)}'
                break
        # entity count e.g. "how many teams" -> distinct of a dim
        dim = None
        for c in categorical:
            if c.lower().rstrip("s") in q:
                dim = c
                break
        if dim and not where:
            r = runner.execute("run_sql", {"query": f'SELECT COUNT(DISTINCT "{dim}") AS n FROM "{table}"'})
        else:
            r = runner.execute("run_sql", {"query": f'SELECT COUNT(*) AS n FROM "{table}"{where}'})
        n = r["rows"][0]["n"] if r.get("rows") else "?"
        return f"There are {n} matching records."

    # average / mean
    if "average" in q or "mean" in q:
        col = None
        for c in measures:
            if c.lower() in q or c.lower().replace("_", " ") in q:
                col = c
                break
        col = col or pick(measures, "score", "revenue", "value")
        r = runner.execute("run_sql", {"query": f'SELECT ROUND(AVG("{col}"), 2) AS avg_val FROM "{table}"'})
        v = r["rows"][0]["avg_val"] if r.get("rows") else "?"
        return f"The average {col} is {v}."

    # highest / maximum single value vs. ranking by group
    if any(w in q for w in ("highest", "most", "top", "maximum", "max", "best", "largest")):
        nm = named(measures)
        measure = nm[0] if nm else None
        # if a dimension entity is named -> ranking; else scalar max
        nd = named(categorical)
        dim = nd[0] if nd else None
        # a "per single X" phrasing means row-level max, not a group total
        single = any(p in q for p in ("single", "in a ", "per game", "any "))
        # "units"/"points"/"revenue" default measure
        measure = measure or pick(measures, "revenue", "points", "score", "units", "value")
        if dim and not single:
            agg = "SUM"
            r = runner.execute("run_sql", {"query":
                f'SELECT "{dim}", {agg}("{measure}") AS total FROM "{table}" '
                f'GROUP BY "{dim}" ORDER BY total DESC LIMIT 1'})
            if r.get("rows"):
                row = r["rows"][0]
                return f"{row[dim]} has the highest total {measure} ({row['total']})."
        r = runner.execute("run_sql", {"query": f'SELECT MAX("{measure}") AS mx FROM "{table}"'})
        v = r["rows"][0]["mx"] if r.get("rows") else "?"
        return f"The highest {measure} is {v}."

    # fallback: describe
    r = runner.execute("run_sql", {"query": f'SELECT * FROM "{table}" LIMIT 5'})
    return f"Loaded {table}. Columns: {', '.join(numeric + categorical)}."


def _openai_tools() -> list[dict]:
    return [{
        "type": "function",
        "function": {"name": t["name"], "description": t["description"],
                     "parameters": t["input_schema"]},
    } for t in TOOLS]


def _run_openai_compatible(system: str, question: str, runner: _ToolRunner,
                           history: list[dict] | None,
                           base_url: str | None, model: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(timeout=LLM_TIMEOUT_S, api_key=api_key,
                    **({"base_url": base_url} if base_url else {}))
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})
    oa_tools = _openai_tools()

    for step in range(MAX_STEPS):
        log.info("%s step %d: calling model", model, step + 1)
        runner.notify("thinking" if step == 0 else "reviewing", model)
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=oa_tools,
        )
        msg = resp.choices[0].message
        log.info("%s step %d: finish=%s tool_calls=%d text_len=%d", model, step + 1,
                 resp.choices[0].finish_reason, len(msg.tool_calls or []),
                 len(msg.content or ""))
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            if answer:
                return answer
            log.warning("%s returned empty final text; asking again without tools", model)
            runner.notify("writing", model)
            # Some providers (notably Gemini) occasionally end with an empty
            # message after tool calls. Ask once more for the written answer.
            messages.append({
                "role": "user",
                "content": "Using the tool results above, write the final answer "
                           "for the user now, in plain language. Do not call any tools.",
            })
            resp = client.chat.completions.create(model=model, messages=messages)
            answer = (resp.choices[0].message.content or "").strip()
            if answer:
                return answer
            raise RuntimeError(f"{model} returned an empty answer")

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = runner.execute(tc.function.name, args)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(out, default=str),
            })

    return "Reached the analysis step limit before finishing. Partial results above."


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def _key_env_for(provider: str) -> str | None:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider in OPENAI_COMPATIBLE:
        return OPENAI_COMPATIBLE[provider]["key_env"]
    return None


def _short_error(err: str | None) -> str:
    """One-line, human-readable version of a provider error for the UI."""
    e = err or ""
    low = e.lower()
    if "429" in e or "ratelimit" in low or "resource_exhausted" in low or "quota" in low:
        return "rate/quota limit reached"
    if "401" in e or "403" in e or "api key" in low or "authentication" in low:
        return "invalid or missing API key"
    if "timeout" in low or "timed out" in low:
        return "request timed out"
    if "empty answer" in low:
        return "returned an empty answer"
    if "connection" in low or "network" in low or "resolve" in low:
        return "the AI service couldn't be reached"
    if e.startswith("Missing ") and "_API_KEY" in e:
        return "an API key is missing"
    return "the AI service returned an error"  # raw details are logged, never shown


def _provider_chain(primary: str) -> list[str]:
    """Primary provider first, then fallbacks from LLM_FALLBACK (default: groq,gemini).

    Fallback providers without an API key configured are skipped.
    """
    raw = os.environ.get("LLM_FALLBACK", "groq,gemini")
    chain = [primary]
    for p in (x.strip().lower() for x in raw.split(",")):
        if not p or p in chain or p == "mock":
            continue
        key_env = _key_env_for(p)
        if key_env and os.environ.get(key_env):
            chain.append(p)
    return chain


def answer_question(
    question: str,
    db_path: str,
    schema_text: str,
    provider: str | None = None,
    history: list[dict] | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> AgentResult:
    """Answer with the primary provider; if it fails, retry on each fallback provider.

    `progress(event, detail)` is called for each real step: thinking, sql,
    analysis, chart, reviewing, writing and fallback.

    Never raises — errors are captured on the returned AgentResult.
    """
    primary = (provider or os.environ.get("LLM_PROVIDER", "groq")).lower()
    failures: list[str] = []
    result: AgentResult | None = None
    chain = _provider_chain(primary)
    log.info("question=%r providers=%s", question[:120], chain)
    for i, p in enumerate(chain):
        if i and progress:
            progress("fallback", p)
        t0 = time.time()
        result = _answer_with_provider(question, db_path, schema_text, p, history, progress)
        log.info("provider=%s done in %.1fs error=%s answer_len=%d", p, time.time() - t0,
                 result.error, len(result.answer or ""))
        if not result.error:
            if failures:
                log.info("answered by fallback %s after: %s", p, "; ".join(failures))
                result.fallback_note = "Answered by the backup AI model."
            return result
        failures.append(_short_error(result.error))
    if result is not None:
        if all(f == "an API key is missing" for f in failures):
            keys = " or ".join(k for k in (_key_env_for(p) for p in chain) if k)
            result.answer = f"The AI isn't configured: add {keys} to the app's Secrets."
        else:
            reasons = "; ".join(dict.fromkeys(failures))  # unique, in order
            result.answer = (f"The AI couldn't answer right now ({reasons}). "
                             "Please try again in a minute.")
    return result


def _answer_with_provider(
    question: str,
    db_path: str,
    schema_text: str,
    provider: str,
    history: list[dict] | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> AgentResult:
    """Run the full agent loop for one question on one provider. Never raises."""
    result = AgentResult(answer="", provider_used=provider)
    runner = _ToolRunner(db_path, result, progress)
    system = build_system_prompt(schema_text)

    # resolve which API key env var this provider needs
    if provider == "anthropic":
        key_env = "ANTHROPIC_API_KEY"
    elif provider in OPENAI_COMPATIBLE:
        key_env = OPENAI_COMPATIBLE[provider]["key_env"]
    else:
        key_env = None

    if provider != "mock" and key_env and not os.environ.get(key_env):
        result.error = f"Missing {key_env}."
        result.answer = f"The AI isn't configured: add {key_env} to the app's Secrets."
        return result

    try:
        start = time.time()
        if provider == "anthropic":
            result.answer = _run_anthropic(system, question, runner, history)
        elif provider in OPENAI_COMPATIBLE:
            cfg = OPENAI_COMPATIBLE[provider]
            result.answer = _run_openai_compatible(
                system, question, runner, history,
                base_url=cfg["base_url"], model=cfg["model"],
                api_key=os.environ.get(cfg["key_env"], ""),
            )
        elif provider == "mock":
            result.answer = _run_mock(question, schema_text, runner)
        else:
            result.error = f"Unknown provider '{provider}'."
            result.answer = result.error
        result.steps.append({"elapsed_s": round(time.time() - start, 2)})
    except Exception as e:
        log.exception("provider %s failed", provider)
        result.error = f"{type(e).__name__}: {e}"   # full detail stays in the logs
        result.answer = f"The AI couldn't answer right now ({_short_error(result.error)})."
    return result
