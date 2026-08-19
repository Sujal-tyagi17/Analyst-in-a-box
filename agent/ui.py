"""ui.py — presentation layer (CSS + reusable HTML components) for the app.

Kept separate from app logic so app.py stays readable. Nothing here is
dataset-specific.
"""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --violet:#7C5CFF; --cyan:#22D3EE; --pink:#F472B6;
  --bg:#0B0E14; --card:#141922; --card-2:#1A2130;
  --border:#232A3A; --text:#E6E9EF; --muted:#8B94A7;
  --grad:linear-gradient(120deg,#7C5CFF 0%,#22D3EE 100%);
}

html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{ background:
    radial-gradient(1100px 500px at 12% -8%, rgba(124,92,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.12), transparent 55%),
    var(--bg); }

/* hide default chrome */
#MainMenu, header[data-testid="stHeader"], footer{ visibility:hidden; height:0; }
[data-testid="stToolbar"]{ display:none; }
.block-container{ padding-top:2.2rem; padding-bottom:6rem; max-width:1150px; }

/* ---------- hero ---------- */
.hero{ margin:.2rem 0 1.4rem; }
.hero h1{
  font-size:2.55rem; font-weight:800; letter-spacing:-.02em; margin:0;
  background:var(--grad); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; line-height:1.1;
}
.hero p{ color:var(--muted); font-size:1.05rem; margin:.5rem 0 0; max-width:680px; }
.chiprow{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1rem; }
.chip{
  font-size:.78rem; font-weight:600; color:#cbd3e1;
  background:rgba(124,92,255,.10); border:1px solid var(--border);
  padding:.34rem .7rem; border-radius:999px; backdrop-filter:blur(6px);
}
.chip b{ color:var(--cyan); }

/* ---------- cards ---------- */
.card{
  background:linear-gradient(180deg, rgba(26,33,48,.85), rgba(20,25,34,.85));
  border:1px solid var(--border); border-radius:18px; padding:1.15rem 1.25rem;
  box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.welcome{ text-align:center; padding:2.6rem 1.5rem; }
.welcome .big{ font-size:1.35rem; font-weight:700; color:var(--text); }
.welcome .sub{ color:var(--muted); margin-top:.4rem; }

/* metric badges */
.metricrow{ display:flex; gap:.8rem; flex-wrap:wrap; margin:.2rem 0 1rem; }
.metric{
  flex:1; min-width:120px; background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:.8rem 1rem;
}
.metric .v{ font-size:1.5rem; font-weight:800; color:var(--text);
  font-family:'JetBrains Mono',monospace; }
.metric .k{ font-size:.72rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:.06em; margin-top:.15rem; }
.metric.accent .v{ background:var(--grad); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0E1320, #0B0E14);
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .brand{
  font-weight:800; font-size:1.15rem; display:flex; align-items:center; gap:.5rem;
  background:var(--grad); -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.side-label{ font-size:.72rem; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); margin:.4rem 0 .3rem; }

/* usage meter */
.meter{ height:7px; background:#1e2636; border-radius:999px; overflow:hidden; margin:.35rem 0 .1rem; }
.meter > span{ display:block; height:100%; background:var(--grad); border-radius:999px; }
.meter-txt{ font-size:.72rem; color:var(--muted); display:flex; justify-content:space-between; }

/* ---------- buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button{
  border-radius:12px; border:1px solid var(--border); font-weight:600;
  background:var(--card-2); color:var(--text); transition:.15s ease;
}
.stButton > button:hover{ border-color:var(--violet);
  box-shadow:0 0 0 3px rgba(124,92,255,.15); transform:translateY(-1px); }
.stButton > button[kind="primary"]{
  background:var(--grad); color:#0b0e14; border:none; font-weight:700;
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{
  background:var(--card); border:1px solid var(--border);
  border-radius:16px; padding:.4rem .3rem;
}
[data-testid="stChatMessageAvatarUser"]{ background:var(--violet)!important; }
[data-testid="stChatMessageAvatarAssistant"]{ background:var(--cyan)!important; }
[data-testid="stChatInput"] textarea{ font-size:1rem; }
[data-testid="stChatInput"]{ border-radius:14px; }

/* code / sql */
code, pre, .stCode{ font-family:'JetBrains Mono',monospace!important; }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:14px;
  background:rgba(20,25,34,.6); }

/* dataframe polish */
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; }

/* insight callout */
.insight{
  border-left:3px solid transparent; border-image:var(--grad) 1;
  background:linear-gradient(90deg, rgba(124,92,255,.10), transparent);
  padding:.7rem 1rem; border-radius:8px; margin-top:.3rem;
}
</style>
"""


def hero_html() -> str:
    return """
<div class="hero">
  <h1>📊 Analyst-in-a-Box</h1>
  <p>Upload any dataset and ask questions in plain English or Hindi. It writes its
     own SQL, runs the stats, draws the chart, and explains what it means — like a
     junior data analyst that never sleeps.</p>
  <div class="chiprow">
    <span class="chip">🔎 Auto <b>schema</b></span>
    <span class="chip">🧾 Writes <b>SQL</b></span>
    <span class="chip">📈 <b>Stats</b> &amp; trends</span>
    <span class="chip">📊 <b>Charts</b></span>
    <span class="chip">💡 Plain-language <b>insight</b></span>
  </div>
</div>
"""


def welcome_html() -> str:
    return """
<div class="card welcome">
  <div class="big">👈 Load a dataset to begin</div>
  <div class="sub">Pick a bundled sample or upload your own CSV / Excel from the sidebar.
    The schema is detected automatically — no setup, no config.</div>
</div>
"""


def metric_badges(items: list[tuple[str, str, bool]]) -> str:
    """items: list of (value, label, accent)."""
    cells = "".join(
        f'<div class="metric{" accent" if acc else ""}">'
        f'<div class="v">{v}</div><div class="k">{k}</div></div>'
        for v, k, acc in items
    )
    return f'<div class="metricrow">{cells}</div>'


def meter_html(used: int, total: int, label: str) -> str:
    pct = min(100, int(100 * used / total)) if total else 0
    return (
        f'<div class="side-label">{label}</div>'
        f'<div class="meter"><span style="width:{pct}%"></span></div>'
        f'<div class="meter-txt"><span>{used} used</span><span>{total} max</span></div>'
    )
