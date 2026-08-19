"""app.py — Streamlit UI for Analyst-in-a-Box.

Flow: upload dataset -> auto schema preview -> chat questions -> see generated
SQL + chart + written insight. Dataset-agnostic end to end.
"""

from __future__ import annotations

import os
import base64
import tempfile

import streamlit as st

# Allow API keys / config to come from Streamlit Secrets (never committed).
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
           "LLM_PROVIDER", "ANTHROPIC_MODEL", "OPENAI_MODEL", "GROQ_MODEL", "GEMINI_MODEL"):
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

import pandas as pd
from agent import schema_reader, usage, ui
from agent.orchestrator import answer_question

MAX_SESSION_QUESTIONS = int(os.environ.get("MAX_SESSION_QUESTIONS", "15"))

SAMPLES = {
    "sales": ("🛒", "Sales", "Orders, regions, revenue"),
    "sports": ("⚽", "Sports", "Teams, seasons, points"),
    "education": ("🎓", "Education", "Students, scores, study hours"),
}

# Generic starter questions that work on any dataset.
EXAMPLE_QS = [
    "Give me a quick summary of this dataset.",
    "What are the top categories by the main metric?",
    "Is there a correlation between the numeric columns?",
    "Are there any outliers I should know about?",
]

st.set_page_config(page_title="Analyst-in-a-Box", page_icon="📊", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("db_path", None)
    ss.setdefault("schema", None)
    ss.setdefault("schema_text", None)
    ss.setdefault("dataset_name", None)
    ss.setdefault("chat", [])            # list of {"q":..., "result": AgentResult}
    ss.setdefault("history", [])         # provider-neutral msg history for follow-ups
    ss.setdefault("n_questions", 0)
    ss.setdefault("pending_q", None)     # queued question from example chips


_init_state()


def _load(file_obj_or_path, name: str, ftype: str | None):
    db_dir = tempfile.mkdtemp(prefix="aib_")
    db_path = os.path.join(db_dir, "dataset.db")
    schema_reader.load_flat_file_to_sqlite(file_obj_or_path, db_path,
                                           table_name=name, file_type=ftype)
    schema = schema_reader.introspect(db_path)
    st.session_state.db_path = db_path
    st.session_state.schema = schema
    st.session_state.schema_text = schema.to_prompt_text()
    st.session_state.dataset_name = name
    st.session_state.chat = []
    st.session_state.history = []
    st.session_state.n_questions = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">📊 Analyst-in-a-Box</div>', unsafe_allow_html=True)
    st.caption("Your dataset-agnostic AI data analyst.")

    st.markdown('<div class="side-label">1 · Try a sample</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for (key, (emoji, label, _desc)), col in zip(SAMPLES.items(), cols):
        with col:
            if st.button(f"{emoji}\n{label}", key=f"s_{key}", use_container_width=True):
                try:
                    _load(os.path.join("data", "sample_datasets", f"{key}.csv"), key, "csv")
                    st.toast(f"Loaded {label} sample", icon="✅")
                except Exception as e:
                    st.error(f"Load failed: {e}")

    st.markdown('<div class="side-label">or upload your own</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Dataset (CSV / XLSX)", type=["csv", "xlsx", "xls"],
                                label_visibility="collapsed")
    if uploaded is not None:
        if st.button("📥 Load uploaded file", use_container_width=True, type="primary"):
            try:
                ftype = uploaded.name.rsplit(".", 1)[-1].lower()
                _load(uploaded, uploaded.name.rsplit(".", 1)[0], ftype)
                st.toast(f"Loaded {uploaded.name}", icon="✅")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.divider()
    st.markdown('<div class="side-label">2 · Model</div>', unsafe_allow_html=True)
    # groq & gemini have generous FREE tiers (no credit card) and support tool calling
    PROVIDER_KEY = {
        "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
    }
    PROVIDER_LABEL = {
        "groq": "Groq · Llama 3.3 (free)", "gemini": "Google Gemini (free)",
        "anthropic": "Anthropic Claude (paid)", "openai": "OpenAI GPT (paid)",
    }
    _opts = list(PROVIDER_KEY)
    _default = os.environ.get("LLM_PROVIDER", "groq")
    provider = st.selectbox(
        "LLM provider", _opts,
        index=_opts.index(_default) if _default in _opts else 0,
        format_func=lambda p: PROVIDER_LABEL[p],
        label_visibility="collapsed",
    )
    os.environ["LLM_PROVIDER"] = provider
    key_env = PROVIDER_KEY[provider]
    if not os.environ.get(key_env):
        st.warning(f"{key_env} not set — add it in Secrets.", icon="🔑")
    else:
        st.caption(f"🟢 {provider} key detected")
    if provider in ("groq", "gemini"):
        _url = "console.groq.com/keys" if provider == "groq" else "aistudio.google.com/apikey"
        st.caption(f"💡 Free key → {_url}")

    st.divider()
    st.markdown(ui.meter_html(st.session_state.n_questions, MAX_SESSION_QUESTIONS,
                              "Session usage"), unsafe_allow_html=True)
    st.markdown(ui.meter_html(usage.get_count(), usage.MAX_DAILY,
                              "Daily usage (all users)"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(ui.hero_html(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Schema section
# ---------------------------------------------------------------------------
schema = st.session_state.schema
if schema is None:
    st.markdown(ui.welcome_html(), unsafe_allow_html=True)
else:
    t = schema.tables[0]
    n_num = sum(1 for c in t.columns if any(k in c.dtype.upper()
                for k in ("INT", "FLOAT", "REAL", "NUM", "DEC")))
    st.markdown(ui.metric_badges([
        (f"{t.row_count:,}", "Rows", True),
        (str(len(t.columns)), "Columns", False),
        (str(n_num), "Numeric fields", False),
        (str(sum(c.null_count for c in t.columns)), "Null cells", False),
    ]), unsafe_allow_html=True)

    with st.expander(f"🔎  Auto-detected schema · table `{t.name}`", expanded=False):
        cols_df = pd.DataFrame([{
            "column": c.name, "type": c.dtype, "nulls": c.null_count,
            "distinct": c.distinct_count,
            "examples": ", ".join(str(v) for v in c.sample_values[:4]),
        } for c in t.columns])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)
        if t.sample_rows:
            st.caption("Sample rows")
            st.dataframe(pd.DataFrame(t.sample_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------------
def _render_result(res):
    if res.sql_queries:
        with st.expander("🧾 Generated SQL", expanded=False):
            for q in res.sql_queries:
                st.code(q, language="sql")
    for ch in res.charts:
        st.image(base64.b64decode(ch["image_base64"]), caption=ch.get("title"),
                 use_container_width=True)
    if res.analyses:
        with st.expander("📈 Statistical analysis", expanded=False):
            for a in res.analyses:
                st.json(a)
    if res.error:
        st.error(res.answer)
    else:
        st.markdown(f'<div class="insight">{res.answer}</div>', unsafe_allow_html=True)


for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        _render_result(turn["result"])


# ---------------------------------------------------------------------------
# Example-question chips (only before the first question)
# ---------------------------------------------------------------------------
if schema is not None and not st.session_state.chat:
    st.caption("Try one of these:")
    ex_cols = st.columns(2)
    for i, eq in enumerate(EXAMPLE_QS):
        with ex_cols[i % 2]:
            if st.button(eq, key=f"eq_{i}", use_container_width=True):
                st.session_state.pending_q = eq
                st.rerun()


# ---------------------------------------------------------------------------
# Question handling (chat input + queued example) with abuse caps
# ---------------------------------------------------------------------------
typed = st.chat_input("Ask about your data…" if schema is not None else "Load a dataset first")
question = typed or st.session_state.pending_q
st.session_state.pending_q = None

if question:
    if schema is None:
        st.warning("Load a dataset first.")
    elif st.session_state.n_questions >= MAX_SESSION_QUESTIONS:
        st.warning("Demo limit reached for this session — check the video walkthrough "
                   "or README for a full example run.", icon="🛑")
    elif usage.daily_limit_reached():
        st.warning("Daily demo limit reached — please try again tomorrow.", icon="🛑")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data…"):
                res = answer_question(
                    question,
                    db_path=st.session_state.db_path,
                    schema_text=st.session_state.schema_text,
                    provider=os.environ.get("LLM_PROVIDER"),
                    history=st.session_state.history,
                )
            _render_result(res)

        st.session_state.n_questions += 1
        usage.increment()
        st.session_state.chat.append({"q": question, "result": res})
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": res.answer})
        st.session_state.history = st.session_state.history[-12:]
        st.rerun()   # refresh sidebar meters + collapse example chips
