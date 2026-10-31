"""app.py — Streamlit UI for Analyst-in-a-Box.

Flow: load a dataset -> automatic profile and dashboard -> ask questions in
plain language -> see the generated SQL, statistics, chart and written insight.
Dataset-agnostic end to end.
"""

from __future__ import annotations

import os
import re
import json
import base64
import tempfile

import streamlit as st

# Local development: pick up keys from a .env file (gitignored) if one exists.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

# Streamlit Cloud: keys / config come from Secrets (never committed). Only read
# st.secrets when a secrets.toml exists; otherwise Streamlit prints an error box.
_CONFIG_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "LLM_PROVIDER", "LLM_FALLBACK", "ANTHROPIC_MODEL", "OPENAI_MODEL",
                "GROQ_MODEL", "GEMINI_MODEL")
try:
    if st.secrets.load_if_toml_exists():
        for _k in _CONFIG_KEYS:
            if _k in st.secrets and not os.environ.get(_k):
                os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

import pandas as pd
from agent import schema_reader, usage, ui
from agent.orchestrator import answer_question, AgentResult, log
from analytics import profiling

MAX_SESSION_QUESTIONS = int(os.environ.get("MAX_SESSION_QUESTIONS", "15"))
MAX_UPLOAD_MB = 200

SAMPLES = {
    "sales": ("Sales", "Orders, regions, revenue"),
    "sports": ("Sports", "Teams, seasons, points"),
    "education": ("Education", "Students, scores, study hours"),
}

# Generic starter questions that work on any dataset.
EXAMPLE_QS = [
    "Give me a quick summary of this dataset.",
    "What are the top categories by the main metric?",
    "Is there a correlation between the numeric columns?",
    "Are there any outliers I should know about?",
]

SECTIONS = {
    "Overview": ":material/dashboard:",
    "AI Analyst": ":material/forum:",
    "Data Explorer": ":material/table_view:",
}

st.set_page_config(page_title="Analyst-in-a-Box", page_icon="📊", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    if "db_path" not in ss:
        log.info("new browser session started (empty state)")
    ss.setdefault("db_path", None)
    ss.setdefault("schema", None)
    ss.setdefault("schema_text", None)
    ss.setdefault("dataset_name", None)
    ss.setdefault("df", None)            # DataFrame read back from SQLite (for profiling)
    ss.setdefault("profile", None)       # analytics.profiling.Profile
    ss.setdefault("loaded_file_id", None)
    ss.setdefault("upload_error", None)    # shown in the main area, not the sidebar
    ss.setdefault("chat", [])            # list of {"q":..., "result": AgentResult}
    ss.setdefault("history", [])         # provider-neutral msg history for follow-ups
    ss.setdefault("n_questions", 0)
    ss.setdefault("pending_q", None)     # queued question from example chips
    ss.setdefault("nav", "Overview")


_init_state()
ss = st.session_state


def _load(file_obj_or_path, name: str, ftype: str | None, size: int | None = None):
    db_dir = tempfile.mkdtemp(prefix="aib_")
    db_path = os.path.join(db_dir, "dataset.db")
    table = schema_reader.load_flat_file_to_sqlite(file_obj_or_path, db_path,
                                                   table_name=name, file_type=ftype)
    schema = schema_reader.introspect(db_path)
    t = schema.tables[0]
    if t.row_count == 0:
        raise ValueError("The file has no data rows.")
    date_cols = [c.name for c in t.columns if any(k in c.dtype.upper() for k in ("DATE", "TIME"))]
    df = profiling.load_table(db_path, table, date_cols)
    ss.db_path = db_path
    ss.schema = schema
    ss.schema_text = schema.to_prompt_text()
    ss.dataset_name = name
    ss.df = df
    ss.profile = profiling.build_profile(df, size)
    ss.chat = []
    ss.history = []
    ss.n_questions = 0
    ss.nav = "Overview"


def _load_sample(key: str):
    path = os.path.join("data", "sample_datasets", f"{key}.csv")
    try:
        _load(path, key, "csv", os.path.getsize(path))
        ss.upload_error = None
        st.toast(f"Loaded the {SAMPLES[key][0]} sample", icon=":material/check_circle:")
        return True
    except Exception:
        log.exception("sample load failed")
        st.error("Couldn't load the sample dataset.")
        return False


def _load_upload(uploaded):
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
        return
    try:
        ftype = uploaded.name.rsplit(".", 1)[-1].lower()
        _load(uploaded, uploaded.name.rsplit(".", 1)[0], ftype, uploaded.size)
        ss.loaded_file_id = uploaded.file_id
        ss.upload_error = None
        st.toast(f"Loaded {uploaded.name}", icon=":material/check_circle:")
    except schema_reader.DatasetError as e:
        ss.loaded_file_id = uploaded.file_id  # don't retry the same broken file every rerun
        ss.upload_error = f"Couldn't load **{uploaded.name}**: {e}"
    except Exception:
        log.exception("upload load failed")
        ss.loaded_file_id = uploaded.file_id
        ss.upload_error = (f"Couldn't read **{uploaded.name}**. Check that it is a valid CSV or "
                           "Excel file with column names in the first row.")


def _sample_buttons(key_prefix: str, stacked: bool = False):
    cols = [st.container() for _ in SAMPLES] if stacked else st.columns(len(SAMPLES))
    for (key, (label, desc)), col in zip(SAMPLES.items(), cols):
        with col:
            if st.button(label, key=f"{key_prefix}_{key}", use_container_width=True,
                         help=desc, icon=":material/dataset:"):
                if _load_sample(key) and key_prefix == "main":
                    st.rerun()  # the empty state was already drawn this run


# ---------------------------------------------------------------------------
# Sidebar: data source + usage
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Analyst-in-a-Box</div>', unsafe_allow_html=True)
    st.caption("AI data analyst for any CSV or Excel file.")

    st.markdown('<div class="side-label">Sample datasets</div>', unsafe_allow_html=True)
    _sample_buttons("side", stacked=True)

    st.markdown('<div class="side-label">Upload your own</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Dataset (CSV / XLSX)", type=["csv", "xlsx", "xls"],
                                label_visibility="collapsed", key="uploader")
    if uploaded is None:
        ss.upload_error = None
    elif uploaded.file_id != ss.loaded_file_id:
        _load_upload(uploaded)  # loads as soon as a new file is chosen

    # Provider + fallback order come from Secrets (LLM_PROVIDER / LLM_FALLBACK).
    if not any(os.environ.get(k) for k in
               ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")) \
            and os.environ.get("LLM_PROVIDER") != "mock":
        st.warning("No AI API key configured — add GROQ_API_KEY in Secrets.", icon=":material/key:")

    st.divider()
    st.markdown(ui.meter_html(ss.n_questions, MAX_SESSION_QUESTIONS, "Questions this session"),
                unsafe_allow_html=True)
    st.markdown(ui.meter_html(usage.get_count(), usage.MAX_DAILY, "Questions today (all users)"),
                unsafe_allow_html=True)


if ss.upload_error:
    st.error(ss.upload_error, icon=":material/error:")

# ---------------------------------------------------------------------------
# Empty state: no dataset yet (data loading also lives here for mobile users,
# where the sidebar is hidden by default)
# ---------------------------------------------------------------------------
if ss.schema is None:
    st.markdown(ui.hero_html(), unsafe_allow_html=True)
    st.markdown(ui.empty_state_html(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        _sample_buttons("main")
        st.caption("Or use **Upload your own** in the sidebar (tap › at the top left on mobile).")
    st.stop()


prof: profiling.Profile = ss.profile
df: pd.DataFrame = ss.df
table = ss.schema.tables[0]

st.markdown(ui.topbar_html(ss.dataset_name, prof.rows, prof.columns,
                           profiling.fmt_bytes(prof.file_bytes or prof.memory_bytes)),
            unsafe_allow_html=True)
nav = st.segmented_control("Section", list(SECTIONS), key="nav",
                           format_func=lambda s: f"{SECTIONS[s]}\u2002{s}",
                           label_visibility="collapsed")
if nav is None:  # clicking the selected segment deselects it; fall back to Overview
    nav = "Overview"


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def render_overview():
    other = f"{prof.n_other} other" if prof.n_other else ""
    st.markdown(ui.kpi_cards([
        {"value": f"{prof.rows:,}", "label": "Rows", "tone": "accent"},
        {"value": prof.columns, "label": "Columns",
         "note": ", ".join(x for x in [f"{prof.n_datetime} date" if prof.n_datetime else "", other] if x)},
        {"value": prof.n_numeric, "label": "Numeric columns"},
        {"value": prof.n_categorical, "label": "Categorical columns"},
        {"value": f"{prof.missing_cells:,}", "label": "Missing values",
         "note": f"{prof.missing_pct:.1f}% of cells", "tone": "warn" if prof.missing_cells else ""},
        {"value": f"{prof.duplicate_rows:,}", "label": "Duplicate rows",
         "tone": "warn" if prof.duplicate_rows else ""},
    ]), unsafe_allow_html=True)

    if prof.metrics:
        st.markdown(ui.section_html("Key metrics",
                                    "Detected automatically from the numeric columns"),
                    unsafe_allow_html=True)
        st.markdown(ui.kpi_cards([{"value": m["value"], "label": m["label"], "note": m["sub"]}
                                  for m in prof.metrics]), unsafe_allow_html=True)

    st.markdown(ui.section_html("Key insights", "Calculated directly from the data, not by the AI"),
                unsafe_allow_html=True)
    st.markdown(ui.insights_html(prof.insights), unsafe_allow_html=True)

    st.markdown(ui.section_html("Columns by type"), unsafe_allow_html=True)
    by_kind: dict[str, list[str]] = {}
    for c, k in prof.kinds.items():
        by_kind.setdefault(profiling.KIND_LABEL[k], []).append(c)
    cols = st.columns(min(len(by_kind), 3) or 1)
    for i, (label, names) in enumerate(by_kind.items()):
        with cols[i % len(cols)]:
            st.markdown(f"**{label}** ({len(names)})")
            st.caption(", ".join(names))


def _is_date_only(s: pd.Series) -> bool:
    d = s.dropna()
    return bool(len(d)) and bool((d.dt.normalize() == d).all())


def _date_column_config(frame: pd.DataFrame) -> dict:
    """Show date-only columns as YYYY-MM-DD instead of '2023-01-02 00:00:00'."""
    return {c: st.column_config.DatetimeColumn(c, format="YYYY-MM-DD")
            for c in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[c]) and _is_date_only(frame[c])}


def _fmt_value(v) -> str:
    if isinstance(v, pd.Timestamp) or hasattr(v, "strftime"):
        ts = pd.Timestamp(v)
        return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float):
        return profiling.fmt_num(v)
    return str(v)


# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------
def render_explorer():
    st.markdown(ui.section_html("Data preview", f"First {min(len(df), 1000):,} of {len(df):,} rows"),
                unsafe_allow_html=True)
    st.dataframe(df.head(1000), use_container_width=True, hide_index=True, height=360,
                 column_config=_date_column_config(df))

    st.markdown(ui.section_html("Column details",
                                f"Table `{table.name}` as the AI sees it, with the detected role of each column"),
                unsafe_allow_html=True)
    n = max(len(df), 1)
    info = pd.DataFrame([{
        "Column": c.name,
        "Role": profiling.KIND_LABEL[prof.kinds.get(c.name, profiling.TEXT)],
        "SQL type": c.dtype,
        "Missing": c.null_count,
        "Missing %": round(100 * c.null_count / n, 1),
        "Unique": c.distinct_count,
        "Examples": ", ".join(_fmt_value(v) for v in df[c.name].dropna().unique()[:4])
                    if c.name in df.columns else "",
    } for c in table.columns])
    st.dataframe(info, use_container_width=True, hide_index=True,
                 column_config={"Missing %": st.column_config.ProgressColumn(
                     "Missing %", min_value=0, max_value=100, format="%.1f%%")})


# ---------------------------------------------------------------------------
# AI Analyst (chat)
# ---------------------------------------------------------------------------
class _AnswerFailed(Exception):
    """Raised inside the cached call so failed answers are never cached."""
    def __init__(self, result):
        super().__init__(result.error)
        self.result = result


@st.cache_data(ttl=600, max_entries=200, show_spinner=False)
def _cached_answer(question, db_path, schema_text, provider, history_json):
    # If the same question is submitted twice (double click / a rerun while the
    # first call is still running), Streamlit holds a per-key lock here, so the
    # second run waits for and reuses the first answer instead of calling the
    # LLM again and burning free-tier quota.
    res = answer_question(question, db_path=db_path, schema_text=schema_text,
                          provider=provider, history=json.loads(history_json))
    if res.error:
        raise _AnswerFailed(res)
    return res


def _clean_answer(answer: str) -> str:
    # charts are rendered separately; drop any image markdown the model adds
    answer = re.sub(r"!\[[^\]]*\](\([^)]*\))?", "", answer).strip()
    # The model sometimes guesses a currency for unlabeled money columns; drop
    # the symbol unless the dataset's schema actually mentions that currency.
    schema_txt = (ss.get("schema_text") or "").lower()
    for sym, hints in (("$", ("usd", "dollar", "$")),
                       ("₹", ("inr", "rupee", "₹")),
                       ("€", ("eur", "euro", "€")),
                       ("£", ("gbp", "pound", "£"))):
        if not any(k in schema_txt for k in hints):
            answer = re.sub(re.escape(sym) + r"\s?(?=\d)", "", answer)
    # escape "$" so Streamlit doesn't treat "$...$" as a LaTeX math formula
    return answer.replace("$", "\\$")


def _render_result(res):
    if res.sql_queries:
        with st.expander("Generated SQL", icon=":material/code:", expanded=False):
            for q in res.sql_queries:
                st.code(q, language="sql")
    for ch in res.charts:
        st.image(base64.b64decode(ch["image_base64"]), use_container_width=True)
    if res.analyses:
        with st.expander("Statistical analysis", icon=":material/query_stats:", expanded=False):
            for a in res.analyses:
                st.json(a)
    if getattr(res, "fallback_note", None):
        st.caption(f"↪ {res.fallback_note}")
    if res.error:
        st.error(res.answer.replace("$", "\\$"))
    elif not (res.answer or "").strip():
        st.warning("The model returned an empty answer. Please try asking again.", icon=":material/warning:")
    else:
        # blank lines let Markdown (bold, lists, tables) render inside the styled box
        st.markdown(f'<div class="insight">\n\n{_clean_answer(res.answer)}\n\n</div>',
                    unsafe_allow_html=True)


def render_analyst():
    for turn in ss.chat:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            _render_result(turn["result"])

    if not ss.chat:
        st.markdown(ui.section_html("Ask a question",
                                    "In English or Hindi. Follow-up questions remember the conversation."),
                    unsafe_allow_html=True)
        ex_cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QS):
            with ex_cols[i % 2]:
                if st.button(eq, key=f"eq_{i}", use_container_width=True):
                    ss.pending_q = eq
                    st.rerun()

    typed = st.chat_input("Ask about your data…")
    question = typed or ss.pending_q
    ss.pending_q = None
    if not question:
        return
    log.info("new question: %r (chat turns=%d)", question[:120], len(ss.chat))

    if ss.n_questions >= MAX_SESSION_QUESTIONS:
        st.warning("Question limit reached for this session. Reload the page to start a new session.",
                   icon=":material/block:")
        return
    if usage.daily_limit_reached():
        st.warning("Daily question limit reached — please try again tomorrow.", icon=":material/block:")
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data…"):
            try:
                res = _cached_answer(
                    question, ss.db_path, ss.schema_text,
                    os.environ.get("LLM_PROVIDER"),
                    json.dumps(ss.history, default=str),
                )
            except _AnswerFailed as f:
                res = f.result
            except Exception as e:  # last-resort guard so the question never vanishes
                log.exception("answer_question crashed")
                res = AgentResult(answer=f"Something went wrong: {type(e).__name__}: {e}",
                                  error=str(e))
        log.info("rendering answer: error=%s answer_len=%d charts=%d", res.error,
                 len(res.answer or ""), len(res.charts))
        _render_result(res)

    ss.n_questions += 1
    usage.increment()
    ss.chat.append({"q": question, "result": res})
    ss.history.append({"role": "user", "content": question})
    ss.history.append({"role": "assistant", "content": res.answer})
    ss.history = ss.history[-12:]
    log.info("saved to chat (now %d turns); rerunning", len(ss.chat))
    st.rerun()   # refresh sidebar meters + collapse example chips


if nav == "Overview":
    render_overview()
elif nav == "AI Analyst":
    render_analyst()
elif nav == "Data Explorer":
    render_explorer()
