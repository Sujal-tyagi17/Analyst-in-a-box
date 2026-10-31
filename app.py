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
from agent.answer_cache import AnswerCache
from agent.orchestrator import answer_question, AgentResult, log
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from analytics import cleaning, eda, profiling, report
from analytics.charts import CONFIG as CHART_CONFIG, agent_chart

MAX_SESSION_QUESTIONS = int(os.environ.get("MAX_SESSION_QUESTIONS", "15"))
MAX_UPLOAD_MB = 200

SAMPLES = {
    "sales": ("Sales", "Orders, regions, revenue"),
    "sports": ("Sports", "Teams, seasons, points"),
    "education": ("Education", "Students, scores, study hours"),
}


SECTIONS = {
    "Overview": ":material/dashboard:",
    "AI Analyst": ":material/forum:",
    "Data Explorer": ":material/table_view:",
    "Analytics": ":material/insights:",
    "Reports": ":material/description:",
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
    ss.setdefault("original", None)        # snapshot of the uploaded data while cleaned data is active
    ss.setdefault("clean_preview", None)   # cleaning.CleaningResult awaiting the user's decision
    ss.setdefault("clean_log", None)       # change log of the cleaning currently in use
    ss.setdefault("query_log", [])         # every question this session (kept when the chat is cleared)
    ss.setdefault("pdf", None)             # (key, bytes) of the last generated report
    ss.setdefault("inflight", None)        # question being answered in the background


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
    ss.original = None
    ss.clean_preview = None
    ss.clean_log = None
    ss.query_log = []
    ss.pdf = None
    ss.inflight = None


def _activate_dataframe(frame: pd.DataFrame):
    """Make `frame` the dataset the AI and dashboard use (a new SQLite DB; the old one is kept)."""
    db_path = os.path.join(tempfile.mkdtemp(prefix="aib_"), "dataset.db")
    table = ss.schema.tables[0].name
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    frame.to_sql(table, engine, if_exists="replace", index=False)
    engine.dispose()
    schema = schema_reader.introspect(db_path)
    ss.db_path, ss.schema, ss.schema_text = db_path, schema, schema.to_prompt_text()
    ss.df = frame
    ss.profile = profiling.build_profile(frame, ss.profile.file_bytes if ss.profile else None)
    # answers about the old data would be misleading now
    ss.chat, ss.history, ss.inflight = [], [], None


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

    if ss.query_log:
        st.markdown('<div class="side-label">Recent questions</div>', unsafe_allow_html=True)
        recent = list(dict.fromkeys(e["question"] for e in reversed(ss.query_log)))[:5]
        for i, q in enumerate(recent):
            if st.button(q if len(q) <= 60 else q[:57] + "…", key=f"recent_{i}", help="Ask again",
                         use_container_width=True, icon=":material/history:"):
                ss.pending_q = q
                ss.nav = "AI Analyst"

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
    if ss.original is not None:
        st.info("You are analysing the **cleaned** data. The original upload is unchanged; "
                "restore it under **Data quality & cleaning**.", icon=":material/cleaning_services:")
    t_prev, t_cols, t_clean = st.tabs(["Preview", "Columns", "Data quality & cleaning"])
    with t_prev:
        render_preview()
    with t_cols:
        render_columns()
    with t_clean:
        render_cleaning()


def render_preview():
    st.markdown(ui.section_html("Data preview", f"First {min(len(df), 1000):,} of {len(df):,} rows"),
                unsafe_allow_html=True)
    st.dataframe(df.head(1000), use_container_width=True, hide_index=True, height=420,
                 column_config=_date_column_config(df))


def render_columns():
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


def render_cleaning():
    base = ss.original["df"] if ss.original is not None else df
    report = cleaning.quality_report(base)
    dups = int(base.duplicated().sum())
    n_issue_cols = int((report["Issues"] != "None found").sum())
    st.markdown(ui.kpi_cards([
        {"value": f"{int(base.isna().sum().sum()):,}", "label": "Missing values",
         "tone": "warn" if base.isna().any().any() else ""},
        {"value": f"{dups:,}", "label": "Duplicate rows", "tone": "warn" if dups else ""},
        {"value": n_issue_cols, "label": "Columns with issues", "tone": "warn" if n_issue_cols else ""},
    ]), unsafe_allow_html=True)
    st.markdown(ui.section_html("Column-wise data quality",
                                "Checks for missing values, wrong types, inconsistent text and outliers (IQR rule)"),
                unsafe_allow_html=True)
    st.dataframe(report, use_container_width=True, hide_index=True,
                 column_config={"Missing %": st.column_config.ProgressColumn(
                     "Missing %", min_value=0, max_value=100, format="%.1f%%"),
                     "Issues": st.column_config.TextColumn("Issues", width="large")})

    if ss.original is not None:
        st.success("Cleaned data is in use for the dashboard, analytics and AI answers.",
                   icon=":material/check_circle:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Restore original data", icon=":material/undo:", use_container_width=True):
                orig = ss.original
                ss.db_path, ss.schema, ss.schema_text = orig["db_path"], orig["schema"], orig["schema_text"]
                ss.df, ss.profile = orig["df"], orig["profile"]
                ss.original, ss.clean_preview, ss.clean_log = None, None, None
                ss.chat, ss.history = [], []
                st.rerun()
        with c2:
            _download_csv(df, f"{ss.dataset_name}_cleaned.csv", "Download cleaned CSV")
        return

    st.markdown(ui.section_html("Clean the data",
                                "Choose the operations, preview the result, then decide whether to use it. "
                                "The original upload is never changed."),
                unsafe_allow_html=True)
    with st.form("clean_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            o_dup = st.checkbox("Remove duplicate rows", value=True)
            o_ws = st.checkbox("Trim extra spaces in text", value=True)
            o_case = st.checkbox("Unify inconsistent spellings (e.g. north / North)", value=True)
            o_types = st.checkbox("Convert numbers and dates stored as text", value=True)
            o_empty = st.checkbox("Remove completely empty columns", value=True)
        with c2:
            o_miss = st.radio("Missing values", ["keep", "fill", "drop_rows"], horizontal=False,
                              format_func={"keep": "Keep as missing",
                                           "fill": "Fill (median for numbers, most common value for text)",
                                           "drop_rows": "Remove rows with any missing value"}.get)
            o_out = st.radio("Outliers (IQR rule)", ["keep", "cap", "remove"],
                             format_func={"keep": "Keep", "cap": "Cap to the IQR bounds",
                                          "remove": "Remove those rows"}.get)
        preview = st.form_submit_button("Preview cleaning", icon=":material/visibility:", type="primary")
    if preview:
        ss.clean_preview = cleaning.apply_cleaning(base, cleaning.CleaningOptions(
            remove_duplicates=o_dup, trim_whitespace=o_ws, fix_case_variants=o_case,
            convert_types=o_types, drop_empty_columns=o_empty, missing=o_miss, outliers=o_out))

    res: cleaning.CleaningResult | None = ss.clean_preview
    if res is None:
        return
    if not res.changed:
        st.info("With these options there is nothing to change: the data is already clean.",
                icon=":material/check_circle:")
        return
    st.markdown(ui.section_html("Preview of the changes"), unsafe_allow_html=True)
    st.markdown(ui.kpi_cards([
        {"value": f"{res.rows_before:,} → {res.rows_after:,}", "label": "Rows"},
        {"value": f"{res.missing_before:,} → {res.missing_after:,}", "label": "Missing values"},
        {"value": f"{base.shape[1]} → {res.data.shape[1]}", "label": "Columns"},
    ]), unsafe_allow_html=True)
    st.markdown("\n".join(f"- {c}" for c in res.changes))
    st.dataframe(res.data.head(200), use_container_width=True, hide_index=True, height=260,
                 column_config=_date_column_config(res.data))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Use cleaned data for analysis", icon=":material/check:", type="primary",
                     use_container_width=True):
            ss.original = {"db_path": ss.db_path, "schema": ss.schema, "schema_text": ss.schema_text,
                           "df": df, "profile": ss.profile}
            _activate_dataframe(res.data)
            ss.clean_log = list(res.changes)
            ss.clean_preview = None
            st.toast("Now using the cleaned data. The chat was cleared.", icon=":material/check_circle:")
            st.rerun()
    with c2:
        _download_csv(res.data, f"{ss.dataset_name}_cleaned.csv", "Download cleaned CSV")


def _download_csv(frame: pd.DataFrame, filename: str, label: str):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", icon=":material/download:", use_container_width=True)


# ---------------------------------------------------------------------------
# Analytics (automated EDA)
# ---------------------------------------------------------------------------
def _plot(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CHART_CONFIG)


def render_analytics():
    kinds = prof.kinds
    nums = eda.numeric_columns(df, kinds)
    cats = eda.categorical_columns(df, kinds)
    dates = eda.datetime_columns(kinds)
    if not nums and not cats:
        st.info("This dataset has no numeric or categorical columns to analyse.", icon=":material/info:")
        return

    if nums:
        st.markdown(ui.section_html("Descriptive statistics",
                                    "Mean, median, spread and quartiles of every numeric column"),
                    unsafe_allow_html=True)
        desc = eda.describe_numeric(df, nums)
        fmt = {c: st.column_config.NumberColumn(c, format="%.2f")
               for c in ["Mean", "Median", "Std dev", "Min", "Q1 (25%)", "Q3 (75%)", "Max", "Skew"]}
        st.dataframe(desc, use_container_width=True, hide_index=True, column_config=fmt)

        st.markdown(ui.section_html("Distributions", "How the values of one numeric column are spread"),
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            col = st.selectbox("Numeric column", nums, key="eda_num",
                               index=nums.index(profiling.primary_measure(df, kinds))
                               if profiling.primary_measure(df, kinds) in nums else 0)
        with c2:
            by = st.selectbox("Compare across", ["(none)"] + [c for c in cats if df[c].nunique() <= 20],
                              key="eda_by")
        g1, g2 = st.columns(2)
        with g1:
            _plot(eda.histogram(df, col))
        with g2:
            _plot(eda.box_plot(df, col, None if by == "(none)" else by))

    if cats:
        st.markdown(ui.section_html("Categorical columns", "How often each value appears"),
                    unsafe_allow_html=True)
        default_cat = profiling.primary_category(df, kinds)
        ccol = st.selectbox("Categorical column", cats, key="eda_cat",
                            index=cats.index(default_cat) if default_cat in cats else 0)
        freq = eda.category_frequencies(df, ccol)
        g1, g2 = st.columns([3, 2])
        with g1:
            _plot(eda.frequency_bar(freq, ccol))
        with g2:
            st.dataframe(freq.assign(Share=freq["Share"] * 100), use_container_width=True,
                         hide_index=True, column_config={"Share": st.column_config.ProgressColumn(
                             "Share", min_value=0, max_value=100, format="%.1f%%")})

    if len(nums) >= 2:
        st.markdown(ui.section_html("Relationships between numeric columns",
                                    "Pearson correlation: +1 moves together, −1 moves opposite, 0 no linear link"),
                    unsafe_allow_html=True)
        g1, g2 = st.columns([3, 2])
        with g1:
            _plot(eda.correlation_heatmap(df, nums))
        with g2:
            pairs = eda.strongest_pairs(df, nums)
            st.caption("Strongest pairs")
            st.dataframe(pairs, use_container_width=True, hide_index=True,
                         column_config={"r": st.column_config.NumberColumn("r", format="%.2f")})
        top = pairs.iloc[0] if len(pairs) else None
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X axis", nums, key="eda_x",
                             index=nums.index(top["Column A"]) if top is not None else 0)
        with c2:
            y_opts = [c for c in nums if c != x]
            y_default = top["Column B"] if top is not None and top["Column B"] in y_opts else y_opts[0]
            y = st.selectbox("Y axis", y_opts, key="eda_y", index=y_opts.index(y_default)
                             if y_default in y_opts else 0)
        _plot(eda.scatter(df, x, y))

    if dates and nums:
        st.markdown(ui.section_html("Trend over time"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            dcol = st.selectbox("Date column", dates, key="eda_date")
        with c2:
            pm = profiling.primary_measure(df, kinds)
            vcol = st.selectbox("Value", nums, key="eda_val", index=nums.index(pm) if pm in nums else 0)
        _plot(eda.time_trend(df, dcol, vcol))

    if nums:
        st.markdown(ui.section_html("Outliers", "Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR"),
                    unsafe_allow_html=True)
        outl = eda.outlier_table(df, nums)
        st.dataframe(outl.assign(Share=outl["Share"] * 100), use_container_width=True, hide_index=True,
                     column_config={"Share": st.column_config.NumberColumn("Share", format="%.1f%%"),
                                    "Lower bound": st.column_config.NumberColumn(format="%.2f"),
                                    "Upper bound": st.column_config.NumberColumn(format="%.2f")})


# ---------------------------------------------------------------------------
# Reports / exports
# ---------------------------------------------------------------------------
def render_reports():
    st.markdown(ui.section_html("PDF analysis report",
                                "Dataset overview, key metrics, findings, charts and the AI's answers "
                                "with the SQL behind them"), unsafe_allow_html=True)
    answered = [t for t in ss.chat if not t["result"].error]
    c1, c2 = st.columns(2)
    with c1:
        inc_ai = st.checkbox(f"Include AI questions and answers ({len(answered)})", value=bool(answered),
                             disabled=not answered)
    with c2:
        inc_charts = st.checkbox("Include charts", value=True)
    pdf_key = (ss.db_path, len(ss.chat), inc_ai, inc_charts)
    if st.button("Generate PDF report", icon=":material/picture_as_pdf:", type="primary"):
        with st.spinner("Building the report…"):
            try:
                ss.pdf = (pdf_key, report.build_pdf(ss.dataset_name, prof, df, ss.chat, ss.clean_log,
                                                    include_ai=inc_ai, include_charts=inc_charts))
            except Exception:
                log.exception("PDF build failed")
                ss.pdf = None
                st.error("Couldn't build the PDF report.", icon=":material/error:")
    if ss.pdf and ss.pdf[0] == pdf_key:
        st.download_button("Download PDF report", ss.pdf[1], file_name=f"{ss.dataset_name}_report.pdf",
                           mime="application/pdf", icon=":material/download:")
    if any(re.search(r"[\u0900-\u097F]", t["q"] + (t["result"].answer or "")) for t in answered):
        st.caption("Hindi text can't be displayed in the PDF yet; those answers are marked in the report.")

    st.markdown(ui.section_html("Data exports"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _download_csv(df, f"{ss.dataset_name}{'_cleaned' if ss.original is not None else ''}.csv",
                      "Current data (CSV)" + (" · cleaned" if ss.original is not None else ""))
    with c2:
        if ss.original is not None:
            _download_csv(ss.original["df"], f"{ss.dataset_name}_original.csv", "Original data (CSV)")
    st.caption("Each AI answer also has its own result CSV and chart PNG, and every interactive chart "
               "has a PNG download in its toolbar.")

    st.markdown(ui.section_html("Query history", "Every question asked this session, including cleared chats"),
                unsafe_allow_html=True)
    if not ss.query_log:
        st.caption("No questions asked yet.")
        return
    hist = pd.DataFrame([{"Time (UTC)": e["time"], "Question": e["question"], "Status": e["status"],
                          "SQL queries": len(e["sql"]), "Charts": e["charts"],
                          "SQL": " | ".join(" ".join(q.split()) for q in e["sql"])} for e in ss.query_log])
    st.dataframe(hist.drop(columns="SQL"), use_container_width=True, hide_index=True)
    st.download_button("Query history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                       file_name=f"{ss.dataset_name}_query_history.csv", mime="text/csv",
                       icon=":material/download:")


# ---------------------------------------------------------------------------
# AI Analyst (chat)
# ---------------------------------------------------------------------------
@st.cache_resource
def _answer_cache() -> AnswerCache:
    """One cache shared by all sessions, like st.cache_data was (see agent/answer_cache.py)."""
    return AnswerCache(ttl=600, max_entries=200)


@st.cache_resource
def _workers() -> ThreadPoolExecutor:
    """Background threads for AI answers. An answer keeps running even if the page
    reruns (a click or a new question), which would otherwise kill it mid-way."""
    return ThreadPoolExecutor(max_workers=8, thread_name_prefix="aib-answer")


def _submit_question(question: str):
    """Start answering in the background; progress events go into a queue."""
    events: queue.Queue = queue.Queue()
    # capture plain values now: worker threads can't read st.session_state
    db_path, schema_text, history = ss.db_path, ss.schema_text, list(ss.history)
    provider = os.environ.get("LLM_PROVIDER")
    key = (question, db_path, schema_text, provider, json.dumps(history, default=str))

    def work():
        try:
            res, cached = _answer_cache().get_or_compute(
                key,
                lambda: answer_question(question, db_path=db_path, schema_text=schema_text,
                                        provider=provider, history=history,
                                        progress=lambda e, d: events.put((e, d))),
                should_cache=lambda r: not r.error,   # failed answers are never cached
            )
            if cached:
                log.info("answer served from cache")
            return res
        except Exception as e:  # last-resort guard so the question never vanishes
            log.exception("answer_question crashed")
            return AgentResult(answer="Something went wrong while analysing. Please try again.",
                               error=str(e))

    ss.inflight = {"q": question, "future": _workers().submit(work), "events": events}
    log.info("new question: %r (chat turns=%d)", question[:120], len(ss.chat))


def _finish_inflight():
    """Move a finished background answer into the chat, history and query log."""
    job = ss.inflight
    ss.inflight = None
    question, res = job["q"], job["future"].result()
    log.info("saved answer: error=%s answer_len=%d charts=%d", res.error, len(res.answer or ""), len(res.charts))
    ss.chat.append({"q": question, "result": res})
    ss.query_log.append({"time": time.strftime("%H:%M:%S", time.gmtime()), "question": question,
                         "status": "error" if res.error else "answered",
                         "sql": res.sql_queries, "charts": len(res.charts),
                         "answer": (res.answer or "")[:300]})
    ss.history.append({"role": "user", "content": question})
    ss.history.append({"role": "assistant", "content": _history_entry(res)})
    ss.history = ss.history[-12:]


def _wait_for_inflight():
    """Show the running question with live progress until its answer is ready."""
    job = ss.inflight
    with st.chat_message("user"):
        st.markdown(job["q"])
    with st.chat_message("assistant"):
        status = st.status("Understanding the question…", expanded=True)
        show = _Progress(status)
        while True:
            try:
                while True:
                    show(*job["events"].get_nowait())
            except queue.Empty:
                pass
            if job["future"].done():
                break
            time.sleep(0.25)
        failed = bool(job["future"].result().error)
        status.update(label="Couldn't complete the analysis" if failed else "Analysis complete",
                      state="error" if failed else "complete", expanded=False)


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


def _render_analysis(a: dict):
    """Statistical results as plain language plus the key numbers, instead of raw JSON."""
    m = a.get("method", "")
    p = a.get("p_value")
    p_txt = "" if p is None else ("p < 0.001" if p < 0.001 else f"p = {p:.3g}")
    sig = None if p is None else ("statistically significant (p < 0.05)" if p < 0.05
                                  else "not statistically significant — could be chance")
    if m == "correlation" and "pearson_r" in a:
        r = a["pearson_r"]
        strength = ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4
                    else "weak" if abs(r) >= 0.2 else "essentially no")
        st.markdown(f"**Correlation · {a['x']} vs {a['y']}**  \n"
                    f"r = **{r:.2f}** ({strength} {'positive' if r > 0 else 'negative'} relationship) · "
                    f"{p_txt}, {sig}.")
    elif m == "trend" and "slope" in a:
        direction = "rising" if a["slope"] > 0 else "falling" if a["slope"] < 0 else "flat"
        st.markdown(f"**Trend · {a['y']}**  \n{direction.capitalize()} by {a['slope']:,.3g} per step · "
                    f"R² = {a['r_squared']:.2f} (share of variation the trend explains) · {sig}.")
    elif m == "outliers" and "n_outliers" in a:
        lo, hi = a["bounds"]
        st.markdown(f"**Outliers · {a['column']}**  \n**{a['n_outliers']}** value(s) outside the normal "
                    f"range {lo:,.2f} – {hi:,.2f} (IQR rule).")
    elif m == "group_compare" and "group_means" in a:
        test = {"welch_t": "Welch t-test", "anova": "ANOVA"}.get(a.get("test"), "comparison")
        st.markdown(f"**Group comparison · {a['value']} by {a['group']}**"
                    + (f"  \n{test}: {p_txt}, {sig}." if p is not None else ""))
        st.dataframe(pd.DataFrame({a["group"]: list(a["group_means"]),
                                   f"average {a['value']}": list(a["group_means"].values())}),
                     hide_index=True, use_container_width=True)
    elif m == "describe" and "summary" in a:
        st.markdown("**Summary statistics**")
        st.dataframe(pd.DataFrame(a["summary"]), use_container_width=True)
    else:
        st.markdown(a.get("note", ""))


def _render_result(res, key: str = "new"):
    if res.error:
        st.error(_clean_answer(res.answer), icon=":material/error:")
    elif not (res.answer or "").strip():
        st.warning("The model returned an empty answer. Please try asking again.", icon=":material/warning:")
    else:
        # blank lines let Markdown (bold, lists, tables) render inside the styled box
        st.markdown(f'<div class="insight">\n\n{_clean_answer(res.answer)}\n\n</div>',
                    unsafe_allow_html=True)
    for ch in res.charts:
        fig = agent_chart(ch)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, theme=None, config=CHART_CONFIG)
        else:
            st.image(base64.b64decode(ch["image_base64"]), use_container_width=True)
    if res.analyses:
        with st.expander("Statistical analysis", icon=":material/query_stats:", expanded=False):
            for a in res.analyses:
                _render_analysis(a)
    if res.sql_queries:
        with st.expander("Generated SQL", icon=":material/code:", expanded=False):
            for q in res.sql_queries:
                st.code(q, language="sql")
    if getattr(res, "fallback_note", None):
        st.caption(f"↪ {res.fallback_note}")
    _result_downloads(res, key)


def _result_downloads(res, key: str):
    results = [r for r in getattr(res, "query_results", []) if r.get("rows")]
    pngs = [c for c in res.charts if c.get("image_base64")]
    if res.error or not (results or pngs):
        return
    cols = st.columns(4)
    if results:
        last = results[-1]  # the query the answer is based on
        with cols[0]:
            st.download_button("Result (CSV)", pd.DataFrame(last["rows"]).to_csv(index=False).encode("utf-8"),
                               file_name=f"query_result_{key}.csv", mime="text/csv",
                               icon=":material/download:", key=f"dl_csv_{key}", use_container_width=True)
    for i, c in enumerate(pngs[:3]):
        with cols[1 + i]:
            st.download_button("Chart (PNG)", base64.b64decode(c["image_base64"]),
                               file_name=f"chart_{key}_{i + 1}.png", mime="image/png",
                               icon=":material/image:", key=f"dl_png_{key}_{i}", use_container_width=True)


_STEP_LABELS = {
    "thinking": "Understanding the question",
    "reviewing": "Reviewing the results",
    "writing": "Writing the answer",
    "analysis": "Performing statistical analysis",
    "chart": "Preparing the visualization",
    "fallback": "Main AI model unavailable, switching to the backup model",
}


class _Progress:
    """Turns the agent's real steps into lines inside an st.status box."""

    def __init__(self, status):
        self.status = status

    def __call__(self, event: str, detail: str = ""):
        if event == "sql":
            self.status.write(":material/check: Generated SQL")
            self.status.update(label="Executing the query…")
            self.status.write(":material/database: Executing the query")
            return
        text = _STEP_LABELS.get(event)
        if not text:
            return
        if event == "analysis" and detail:
            text += f" ({detail.replace('_', ' ')})"
        if event == "chart" and detail:
            text += f" ({detail} chart)"
        self.status.update(label=f"{text}…")
        self.status.write(f":material/arrow_right: {text}")


def _history_entry(res) -> str:
    """What the model sees of its previous answer: the text plus the SQL it ran,
    so follow-ups ('now break that down') can reuse the same filters."""
    text = res.answer or ""
    if res.sql_queries:
        sql = " | ".join(q.replace("\n", " ")[:300] for q in res.sql_queries[-2:])
        text += f"\n[SQL used: {sql}]"
    return text


def render_analyst():
    typed = st.chat_input("Ask about your data…")
    if typed:
        # queue it; if an answer is still running, this one starts right after
        ss.pending_q = typed
    if ss.chat or ss.inflight is not None:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(f"{len(ss.chat)} question(s) in this conversation · "
                       f"{MAX_SESSION_QUESTIONS - ss.n_questions} left this session")
        with c2:
            if st.button("Clear chat", icon=":material/delete_sweep:", use_container_width=True,
                         help="Start a new conversation. The dataset stays loaded."):
                ss.chat, ss.history = [], []
                st.rerun()
    for i, turn in enumerate(ss.chat):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            _render_result(turn["result"], key=str(i))

    if ss.inflight is not None and ss.pending_q:
        st.caption(f"Next in line: “{ss.pending_q}”")
    if not ss.chat and ss.inflight is None:
        st.markdown(ui.section_html("Ask a question",
                                    "In English or Hindi. Follow-up questions remember the conversation."),
                    unsafe_allow_html=True)
        ex_cols = st.columns(2)
        for i, eq in enumerate(profiling.suggest_questions(df, prof.kinds)):
            with ex_cols[i % 2]:
                if st.button(eq, key=f"eq_{i}", use_container_width=True):
                    ss.pending_q = eq
                    st.rerun()

    if ss.inflight is not None:
        _wait_for_inflight()          # blocks (with live progress) until the answer is ready
        _finish_inflight()
        st.rerun()                    # redraw: answer in the chat, meters updated

    question = ss.pending_q
    ss.pending_q = None
    if not question:
        return
    if ss.n_questions >= MAX_SESSION_QUESTIONS:
        st.warning("Question limit reached for this session. Reload the page to start a new session.",
                   icon=":material/block:")
        return
    if usage.daily_limit_reached():
        st.warning("Daily question limit reached — please try again tomorrow.", icon=":material/block:")
        return
    ss.n_questions += 1
    usage.increment()
    _submit_question(question)
    st.rerun()

# An answer that finished while the user was on another section joins the chat now.
if ss.inflight is not None and ss.inflight["future"].done() and nav != "AI Analyst":
    _finish_inflight()

if nav == "Overview":
    render_overview()
elif nav == "AI Analyst":
    render_analyst()
elif nav == "Data Explorer":
    render_explorer()
elif nav == "Analytics":
    render_analytics()
elif nav == "Reports":
    render_reports()
