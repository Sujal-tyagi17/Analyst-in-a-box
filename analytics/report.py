"""report.py — build a PDF analysis report from the session.

Contents: dataset overview, key metrics, data-computed findings, charts drawn
from the data, the AI questions and answers (with the SQL they ran) and a short
summary. Charts use a light Matplotlib style so they print well.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .profiling import (DATETIME, Profile, fmt_bytes, is_additive, measures,  # noqa: E402
                        primary_category, primary_measure)

ACCENT, INK, MUTED, GRID = "#5B3FD9", "#1F2430", "#6B7280", "#E5E7EB"
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ---------------------------------------------------------------------------
# Charts (light theme for print)
# ---------------------------------------------------------------------------

def _fig(title: str):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    return fig, ax


def _png(fig) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _thousands(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2g}"))


def chart_png(kind: str, df: pd.DataFrame, x: str | None, y: str | None, title: str) -> bytes | None:
    """Render one chart (bar, line, scatter, histogram, box) as a print-friendly PNG."""
    try:
        fig, ax = _fig(title)
        if kind == "histogram":
            col = x if x in df.columns else y
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.hist(s, bins=min(40, max(10, int(np.sqrt(len(s))))), color=ACCENT, edgecolor="white")
            ax.set_xlabel(col.replace("_", " "), color=MUTED, fontsize=9)
            ax.set_ylabel("rows", color=MUTED, fontsize=9)
        elif kind == "box":
            if x and x in df.columns:
                groups = list(df.dropna(subset=[x, y]).groupby(x)[y])
                ax.boxplot([g for _, g in groups], labels=[str(k) for k, _ in groups])
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            else:
                ax.boxplot(pd.to_numeric(df[y], errors="coerce").dropna())
            ax.set_ylabel(y.replace("_", " "), color=MUTED, fontsize=9)
            _thousands(ax)
        else:
            if x not in df.columns or y not in df.columns:
                plt.close(fig)
                return None
            d = df[[x, y]].dropna().head(50)
            if kind == "line":
                ax.plot(d[x], d[y], color=ACCENT, marker="o", markersize=3.5, linewidth=1.8)
            elif kind == "scatter":
                ax.scatter(d[x], d[y], color=ACCENT, alpha=.6, s=12)
            else:
                ax.bar(d[x].astype(str), d[y], color=ACCENT, width=0.6)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_xlabel(x.replace("_", " "), color=MUTED, fontsize=9)
            ax.set_ylabel(y.replace("_", " "), color=MUTED, fontsize=9)
            _thousands(ax)
        return _png(fig)
    except Exception:
        plt.close("all")
        return None


def overview_charts(df: pd.DataFrame, kinds: dict[str, str]) -> list[tuple[str, bytes]]:
    """The few charts that summarise any dataset; each only if the data supports it."""
    out: list[tuple[str, bytes]] = []
    m, g = primary_measure(df, kinds), primary_category(df, kinds)
    if m and g:
        agg = "sum" if is_additive(m) else "mean"
        by = df.groupby(g)[m].agg(agg).sort_values(ascending=False).head(15).reset_index()
        word = "Total" if agg == "sum" else "Average"
        png = chart_png("bar", by, g, m, f"{word} {m.replace('_', ' ')} by {g.replace('_', ' ')}")
        if png:
            out.append(("comparison", png))
    dates = [c for c, k in kinds.items() if k == DATETIME]
    if m and dates:
        d = dates[0]
        span = (df[d].max() - df[d].min()).days if df[d].notna().any() else 0
        freq = "D" if span <= 62 else "W" if span <= 240 else "MS"
        agg = "sum" if is_additive(m) else "mean"
        ts = df[[d, m]].dropna().set_index(d)[m].resample(freq).agg(agg).reset_index()
        png = chart_png("line", ts, d, m, f"{'Total' if agg == 'sum' else 'Average'} {m.replace('_', ' ')} over time")
        if png:
            out.append(("trend", png))
    if m:
        png = chart_png("histogram", df, m, None, f"Distribution of {m.replace('_', ' ')}")
        if png:
            out.append(("distribution", png))
    ms = measures(df, kinds)
    if len(ms) >= 2:
        corr = df[ms].apply(pd.to_numeric, errors="coerce").corr()
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=150)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = [c.replace("_", " ") for c in ms]
        ax.set_xticks(range(len(ms)), labels, rotation=35, ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(ms)), labels, fontsize=8, color=MUTED)
        for i in range(len(ms)):
            for j in range(len(ms)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.values[i, j]) > .5 else INK)
        ax.set_title("Correlation between numeric columns", loc="left", fontsize=11, color=INK,
                     fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out.append(("correlation", _png(fig)))
    return out


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class _PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "", 8)
            self.set_text_color(130, 136, 150)
            self.cell(0, 6, "Analyst-in-a-Box · analysis report", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(130, 136, 150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def _md_to_pdf_text(text: str) -> str:
    """Make the AI's Markdown readable in the PDF (bold kept; tables/bullets flattened)."""
    lines = []
    for line in (text or "").splitlines():
        s = line.rstrip()
        if re.fullmatch(r"\s*\|?\s*:?-{2,}.*", s):   # table separator row
            continue
        if s.strip().startswith("|"):
            s = "   ".join(c.strip() for c in s.strip().strip("|").split("|"))
        s = re.sub(r"^\s*[-*]\s+", "• ", s)
        s = re.sub(r"^#+\s*", "", s)
        s = s.replace("\\$", "$")
        lines.append(s)
    return "\n".join(lines).strip()


def build_pdf(dataset: str, prof: Profile, df: pd.DataFrame, chat: list[dict],
              cleaning_log: list[str] | None = None, include_ai: bool = True,
              include_charts: bool = True) -> bytes:
    pdf = _PDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("DejaVu", "", os.path.join(_FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVuMono", "", os.path.join(_FONT_DIR, "DejaVuSansMono.ttf"))
    pdf.set_margins(16, 16, 16)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def h1(t):
        pdf.set_font("DejaVu", "B", 20)
        pdf.set_text_color(31, 36, 48)
        pdf.multi_cell(W, 10, t, new_x="LMARGIN", new_y="NEXT")

    def h2(t):
        pdf.ln(4)
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(91, 63, 217)
        pdf.multi_cell(W, 8, t, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def body(t, size=10, color=(55, 60, 72), md=True):
        pdf.set_font("DejaVu", "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(W, 5.4, t, markdown=md, align="L", new_x="LMARGIN", new_y="NEXT")

    def image(png: bytes, width: float | None = None):
        width = width or W
        if pdf.get_y() + width * 0.48 > pdf.h - 20:
            pdf.add_page()
        pdf.image(io.BytesIO(png), w=width)
        pdf.ln(2)

    # ---- title
    h1(f"Analysis report: {dataset}")
    body(f"Generated {datetime.now(timezone.utc):%d %b %Y, %H:%M} UTC with Analyst-in-a-Box.", 9, (120, 126, 140), md=False)
    if cleaning_log:
        body("Analysis uses the **cleaned** version of the data (the original upload was not changed).", 9)

    # ---- overview
    h2("Dataset overview")
    rows = [("Rows", f"{prof.rows:,}"), ("Columns", str(prof.columns)),
            ("Numeric columns", str(prof.n_numeric)), ("Categorical columns", str(prof.n_categorical)),
            ("Date columns", str(prof.n_datetime)),
            ("Missing values", f"{prof.missing_cells:,} ({prof.missing_pct:.1f}% of cells)"),
            ("Duplicate rows", f"{prof.duplicate_rows:,}"),
            ("Size", fmt_bytes(prof.file_bytes or prof.memory_bytes))]
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(55, 60, 72)
    with pdf.table(col_widths=(45, 75), width=120, align="LEFT", line_height=6.4,
                   borders_layout="HORIZONTAL_LINES", first_row_as_headings=False) as t:
        for k, v in rows:
            r = t.row()
            r.cell(k)
            r.cell(v)

    if cleaning_log:
        h2("Cleaning applied")
        body("\n".join(f"• {c}" for c in cleaning_log), 9.5)

    # ---- metrics
    if prof.metrics:
        h2("Key metrics")
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(55, 60, 72)
        with pdf.table(col_widths=(60, 40, 70), line_height=6.4, borders_layout="HORIZONTAL_LINES") as t:
            h = t.row()
            for c in ("Metric", "Value", "Detail"):
                h.cell(c)
            for m in prof.metrics:
                r = t.row()
                r.cell(m["label"])
                r.cell(m["value"])
                r.cell(m["sub"])

    # ---- findings
    h2("Main findings")
    body("Calculated directly from the data:", 9, (120, 126, 140), md=False)
    body("\n".join(f"• {i}" for i in prof.insights))

    # ---- charts
    if include_charts:
        charts = overview_charts(df, prof.kinds)
        if charts:
            if pdf.get_y() > pdf.h - 110:
                pdf.add_page()
            h2("Charts")
            for _, png in charts:
                image(png, W if _ != "correlation" else W * 0.7)

    # ---- AI analysis
    answered = [t for t in chat if not t["result"].error and (t["result"].answer or "").strip()]
    if include_ai and answered:
        pdf.add_page()
        h2("Questions answered by the AI analyst")
        body("Each answer was produced from the SQL queries listed under it.", 9, (120, 126, 140), md=False)
        for n, turn in enumerate(answered, 1):
            res = turn["result"]
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", 11)
            pdf.set_text_color(31, 36, 48)
            q = turn["q"]
            pdf.multi_cell(W, 6, f"{n}. " + (q if not _DEVANAGARI.search(q) else "(question asked in Hindi)"), new_x="LMARGIN", new_y="NEXT")
            ans = _md_to_pdf_text(res.answer)
            if _DEVANAGARI.search(ans):
                body("This answer is in Hindi, which the PDF font can't display yet. "
                     "See the answer in the app.", 9.5, (120, 126, 140), md=False)
            else:
                body(ans)
            if include_charts:
                for ch in res.charts:
                    png = chart_png((ch.get("chart_type") or "bar").lower(), pd.DataFrame(ch.get("data") or []),
                                    ch.get("x"), ch.get("y"), ch.get("title") or "")
                    if png:
                        image(png, W * 0.85)
            if res.sql_queries:
                pdf.set_font("DejaVuMono", "", 7.5)
                pdf.set_text_color(100, 106, 120)
                for sql in res.sql_queries:
                    pdf.multi_cell(W, 4, "SQL: " + " ".join(sql.split()), new_x="LMARGIN", new_y="NEXT")

    # ---- summary
    h2("Summary")
    lines = [f"{prof.rows:,} rows and {prof.columns} columns were analysed."]
    if answered:
        lines.append(f"{len(answered)} question(s) were answered by the AI analyst, "
                     f"using {sum(len(t['result'].sql_queries) for t in answered)} SQL queries.")
    lines.append("All figures in this report come from the data itself or from the SQL queries shown; "
                 "no numbers were estimated.")
    body(" ".join(lines))
    return bytes(pdf.output())
