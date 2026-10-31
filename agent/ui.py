"""ui.py — presentation layer (CSS + small HTML components) for the app.

Kept separate from app logic so app.py stays readable. Nothing here is
dataset-specific; every value passed in is escaped before it is rendered.
"""

from __future__ import annotations

import html
import re

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --violet:#7C5CFF; --blue:#4F8BFF; --cyan:#22D3EE;
  --bg:#0B0E14; --card:#121722; --card-2:#171E2B; --card-3:#1C2433;
  --border:#232B3B; --border-2:#2C3548;
  --text:#E6E9EF; --muted:#8B94A7; --faint:#5F6880;
  --good:#34D399; --warn:#FBBF24; --bad:#F87171;
  --grad:linear-gradient(120deg,#7C5CFF 0%,#4F8BFF 55%,#22D3EE 100%);
  --radius:14px;
}

html, body, [class*="css"], .stMarkdown, button, input, textarea{ font-family:'Inter',sans-serif; }
.stApp{ background:
    radial-gradient(1000px 420px at 10% -10%, rgba(124,92,255,.13), transparent 60%),
    radial-gradient(800px 420px at 100% -5%, rgba(79,139,255,.10), transparent 55%),
    var(--bg); }

/* hide default chrome */
#MainMenu, footer{ visibility:hidden; height:0; }
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"]{ display:none; }
.block-container{ padding-top:1.6rem; padding-bottom:6rem; max-width:1200px; }

h1,h2,h3,h4{ letter-spacing:-.01em; }

/* ---------- header / hero ---------- */
.hero{ margin:.2rem 0 1.2rem; }
.hero .title{
  font-size:2.3rem; font-weight:800; letter-spacing:-.025em; margin:0; line-height:1.1;
  background:var(--grad); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.hero p{ color:var(--muted); font-size:1.02rem; margin:.55rem 0 0; max-width:700px; line-height:1.55; }
.chiprow{ display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.9rem; }
.chip{ font-size:.76rem; font-weight:600; color:#C9D1E0; background:rgba(124,92,255,.09);
  border:1px solid var(--border); padding:.3rem .65rem; border-radius:999px; }
.topbar{ display:flex; align-items:center; justify-content:space-between; gap:1rem;
  flex-wrap:wrap; margin:.1rem 0 .9rem; }
.topbar .title{ font-size:1.35rem; font-weight:800; letter-spacing:-.02em;
  background:var(--grad); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
.dsbadge{ font-size:.8rem; color:var(--muted); background:var(--card);
  border:1px solid var(--border); padding:.35rem .75rem; border-radius:999px; }
.dsbadge b{ color:var(--text); font-weight:600; }

/* ---------- section headers ---------- */
.sec{ margin:1.3rem 0 .6rem; }
.sec .h{ font-size:1.02rem; font-weight:700; color:var(--text); }
.sec .s{ font-size:.84rem; color:var(--muted); margin-top:.15rem; }

/* ---------- KPI + metric cards ---------- */
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.7rem; margin:.2rem 0 .4rem; }
.kpi{ background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
  padding:.85rem 1rem; }
.kpi .v{ font-size:1.45rem; font-weight:700; color:var(--text); font-family:'JetBrains Mono',monospace;
  letter-spacing:-.02em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .k{ font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; margin-top:.2rem; }
.kpi .n{ font-size:.74rem; color:var(--faint); margin-top:.15rem; }
.kpi.accent{ border-color:rgba(124,92,255,.45);
  background:linear-gradient(160deg, rgba(124,92,255,.14), var(--card) 70%); }
.kpi.warn .v{ color:var(--warn); }

/* ---------- insight list ---------- */
.insights{ background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
  padding:.4rem 1.1rem; }
.insights .row{ display:flex; gap:.75rem; padding:.7rem 0; border-bottom:1px solid var(--border);
  color:#CDD3DF; font-size:.92rem; line-height:1.5; }
.insights .row:last-child{ border-bottom:none; }
.insights .dot{ flex:0 0 8px; height:8px; margin-top:.5rem; border-radius:50%; background:var(--grad); }
.insights b{ color:var(--text); }

/* ---------- empty state ---------- */
.empty{ text-align:center; padding:2.2rem 1.5rem 1rem; }
.empty .big{ font-size:1.25rem; font-weight:700; color:var(--text); }
.empty .sub{ color:var(--muted); margin-top:.4rem; font-size:.95rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0E1320,#0B0E14);
  border-right:1px solid var(--border); }
[data-testid="stSidebar"] .brand{ font-weight:800; font-size:1.12rem;
  background:var(--grad); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
.side-label{ font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin:.5rem 0 .35rem; }
.meter{ height:6px; background:#1E2636; border-radius:999px; overflow:hidden; margin:.3rem 0 .1rem; }
.meter > span{ display:block; height:100%; background:var(--grad); border-radius:999px; }
.meter-txt{ font-size:.7rem; color:var(--muted); display:flex; justify-content:space-between; }

/* ---------- widgets ---------- */
.stButton > button, [data-testid="stDownloadButton"] > button{
  border-radius:11px; border:1px solid var(--border-2); font-weight:600;
  background:var(--card-2); color:var(--text); transition:border-color .15s ease, box-shadow .15s ease; }
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover{
  border-color:var(--violet); box-shadow:0 0 0 3px rgba(124,92,255,.14); color:var(--text); }
.stButton > button[kind="primary"]{ background:var(--grad); color:#0B0E14; border:none; }
[data-testid="stButtonGroup"]{ margin-bottom:.4rem; }
[data-testid="stButtonGroup"] button{ font-weight:600; padding-left:.9rem; padding-right:.9rem; }
[data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] p{ display:flex; align-items:center; gap:.4rem; }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:var(--radius);
  background:rgba(18,23,34,.6); }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; }
[data-testid="stFileUploaderDropzone"]{ background:var(--card); border:1px dashed var(--border-2); border-radius:12px; }
code, pre, .stCode{ font-family:'JetBrains Mono',monospace!important; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{ background:var(--card); border:1px solid var(--border);
  border-radius:16px; padding:.5rem .4rem; }
[data-testid="stChatMessageAvatarUser"]{ background:var(--violet)!important; }
[data-testid="stChatMessageAvatarAssistant"]{ background:var(--blue)!important; }
[data-testid="stChatInput"]{ border-radius:14px; }
.insight{ border-left:3px solid var(--violet);
  background:linear-gradient(90deg, rgba(124,92,255,.09), transparent);
  padding:.7rem 1rem; border-radius:6px; margin-top:.3rem; }

/* ---------- mobile ---------- */
@media (max-width: 640px){
  .block-container{ padding-left:.9rem; padding-right:.9rem; padding-top:3.2rem; }
  .hero .title{ font-size:1.75rem; }
  .hero p{ font-size:.95rem; }
  .kpis{ grid-template-columns:repeat(2,minmax(0,1fr)); gap:.5rem; }
  .kpi{ padding:.7rem .8rem; }
  .kpi .v{ font-size:1.15rem; }
  .insights{ padding:.2rem .8rem; }
  .topbar .title{ font-size:1.15rem; }
}
</style>
"""

_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _md_bold(text: str) -> str:
    """Escape text, then turn **x** into <b>x</b> (the only markup insights use)."""
    return _BOLD.sub(r"<b>\1</b>", html.escape(text))


def hero_html() -> str:
    return """
<div class="hero">
  <div class="title">Analyst-in-a-Box</div>
  <p>Upload a CSV or Excel file and ask questions in plain English or Hindi. It writes and
     runs its own SQL, calculates the statistics, draws the chart and explains the result.</p>
  <div class="chiprow">
    <span class="chip">Automatic schema</span><span class="chip">Natural language to SQL</span>
    <span class="chip">Statistics</span><span class="chip">Charts</span>
    <span class="chip">Plain-language insights</span>
  </div>
</div>
"""


def topbar_html(dataset: str, rows: int, cols: int, size: str = "") -> str:
    extra = f" · {html.escape(size)}" if size else ""
    return (
        '<div class="topbar"><div class="title">Analyst-in-a-Box</div>'
        f'<div class="dsbadge">Dataset <b>{html.escape(dataset)}</b> · {rows:,} rows · {cols} columns{extra}</div></div>'
    )


def empty_state_html() -> str:
    return """
<div class="empty">
  <div class="big">Load a dataset to begin</div>
  <div class="sub">Try a bundled sample below or upload your own CSV / Excel file.
    The schema is detected automatically.</div>
</div>
"""


def section_html(title: str, subtitle: str = "") -> str:
    sub = f'<div class="s">{html.escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="sec"><div class="h">{html.escape(title)}</div>{sub}</div>'


def kpi_cards(items: list[dict]) -> str:
    """items: dicts with value, label, optional note, optional tone ('accent' | 'warn')."""
    cells = "".join(
        f'<div class="kpi {html.escape(i.get("tone", ""))}">'
        f'<div class="v" title="{html.escape(str(i["value"]))}">{html.escape(str(i["value"]))}</div>'
        f'<div class="k">{html.escape(i["label"])}</div>'
        + (f'<div class="n">{html.escape(i["note"])}</div>' if i.get("note") else "")
        + "</div>"
        for i in items
    )
    return f'<div class="kpis">{cells}</div>'


def insights_html(lines: list[str]) -> str:
    rows = "".join(f'<div class="row"><span class="dot"></span><span>{_md_bold(t)}</span></div>'
                   for t in lines)
    return f'<div class="insights">{rows}</div>'


def meter_html(used: int, total: int, label: str) -> str:
    pct = min(100, int(100 * used / total)) if total else 0
    return (
        f'<div class="side-label">{html.escape(label)}</div>'
        f'<div class="meter"><span style="width:{pct}%"></span></div>'
        f'<div class="meter-txt"><span>{used} used</span><span>{total} max</span></div>'
    )


def fix_bold_markers(text: str) -> str:
    """Repair the model's broken bold Markdown so no stray '**' shows up.

    Handles, per line: nested bold ('**A **B** C**' -> '**A B C**') and an
    unmatched marker ('**Total peaked at **42** in June.' -> 'Total peaked at **42** in June.').
    """
    out = []
    for line in text.split("\n"):
        n = line.count("**")
        stripped = line.strip()
        if n >= 4 and stripped.startswith("**") and stripped.endswith("**"):
            inner = stripped[2:-2].replace("**", "")
            line = line[: len(line) - len(line.lstrip())] + f"**{inner}**"
        elif n % 2 == 1:
            i = line.find("**")
            # drop the marker that has no partner: a leading one, otherwise the last one
            if stripped.startswith("**"):
                line = line[:i] + line[i + 2:]
            else:
                j = line.rfind("**")
                line = line[:j] + line[j + 2:]
        out.append(line)
    return "\n".join(out)
