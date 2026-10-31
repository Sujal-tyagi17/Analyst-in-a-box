"""eda.py — automated exploratory data analysis (tables + Plotly figures).

Each function returns either a DataFrame or a Plotly figure, computed directly
from the data. Functions return None when a chart doesn't apply (e.g. no
numeric columns), so the UI never shows an irrelevant or empty chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .charts import CORR_SCALE, PALETTE, label, style
from .profiling import CATEGORY, DATETIME, MEASURE, PERIOD, iqr_outliers, is_additive


def numeric_columns(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    return [c for c, k in kinds.items() if k == MEASURE and df[c].nunique(dropna=True) > 1]


def categorical_columns(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    return [c for c, k in kinds.items() if k in (CATEGORY, PERIOD) and df[c].nunique(dropna=True) > 1]


def datetime_columns(kinds: dict[str, str]) -> list[str]:
    return [c for c, k in kinds.items() if k == DATETIME]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean, median, std, quartiles, skew and IQR outliers for each numeric column."""
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        n_out, _, _ = iqr_outliers(s)
        rows.append({
            "Column": c, "Count": int(s.count()), "Mean": s.mean(), "Median": s.median(),
            "Std dev": s.std(), "Min": s.min(), "Q1 (25%)": s.quantile(0.25),
            "Q3 (75%)": s.quantile(0.75), "Max": s.max(), "Skew": s.skew(), "Outliers": n_out,
        })
    return pd.DataFrame(rows)


def outlier_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        n_out, lo, hi = iqr_outliers(df[c])
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({"Column": c, "Outliers": n_out,
                     "Share": n_out / len(s) if len(s) else 0.0,
                     "Lower bound": lo, "Upper bound": hi,
                     "Most extreme": f"{s[(s < lo) | (s > hi)].abs().max():,.2f}" if n_out else "–"})
    return pd.DataFrame(rows).sort_values("Outliers", ascending=False)


def category_frequencies(df: pd.DataFrame, col: str, top: int = 15) -> pd.DataFrame:
    vc = df[col].astype("string").fillna("(missing)").value_counts()
    out = vc.head(top).rename_axis(col).reset_index(name="Count")
    if len(vc) > top:
        out.loc[len(out)] = [f"Other ({len(vc) - top} values)", int(vc.iloc[top:].sum())]
    out["Share"] = out["Count"] / out["Count"].sum()
    return out


def strongest_pairs(df: pd.DataFrame, cols: list[str], top: int = 5) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame()
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append({"Column A": a, "Column B": b, "r": r,
                              "Strength": ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4
                                           else "weak" if abs(r) >= 0.2 else "none"),
                              "Direction": "positive" if r > 0 else "negative"})
    out = pd.DataFrame(pairs)
    if out.empty:
        return out
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def histogram(df: pd.DataFrame, col: str) -> go.Figure:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = px.histogram(s, x=col, nbins=min(40, max(10, int(np.sqrt(len(s))))),
                       color_discrete_sequence=[PALETTE[0]])
    fig.add_vline(x=s.median(), line_dash="dash", line_color=PALETTE[2],
                  annotation_text=f"median {s.median():,.2f}", annotation_position="top right",
                  annotation_font_color=PALETTE[2], annotation_bgcolor="rgba(11,14,20,.75)")
    fig.update_traces(hovertemplate=f"{label(col)}: %{{x}}<br>rows: %{{y:,}}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=label(col))
    fig.update_yaxes(title="Rows")
    return style(fig, f"Distribution of {label(col)}", 340)


def box_plot(df: pd.DataFrame, col: str, by: str | None = None) -> go.Figure:
    if by:
        sub = df[[col, by]].dropna()
        order = sub.groupby(by)[col].median().sort_values(ascending=False).index.tolist()
        fig = px.box(sub, x=by, y=col, points="outliers", category_orders={by: order},
                     color_discrete_sequence=[PALETTE[1]])
        fig.update_xaxes(title=label(by))
        title = f"{label(col)} by {label(by)}"
    else:
        fig = px.box(df, y=col, points="outliers", color_discrete_sequence=[PALETTE[1]])
        title = f"Spread and outliers of {label(col)}"
    fig.update_yaxes(title=label(col))
    return style(fig, title, 340)


def frequency_bar(freq: pd.DataFrame, col: str) -> go.Figure:
    f = freq.iloc[::-1]  # largest at the top of a horizontal bar chart
    fig = go.Figure(go.Bar(x=f["Count"], y=f[col].astype(str), orientation="h",
                           marker_color=PALETTE[0],
                           customdata=f["Share"],
                           hovertemplate="%{y}<br>%{x:,} rows (%{customdata:.1%})<extra></extra>",
                           text=[f"{v:.0%}" for v in f["Share"]], textposition="outside",
                           textfont=dict(color="#9AA3B2")))
    fig.update_xaxes(title="Rows")
    fig.update_yaxes(title=None, type="category")
    return style(fig, f"Most common values of {label(col)}", max(260, 28 * len(f) + 90))


def correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure | None:
    if len(cols) < 2:
        return None
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
    labels = [label(c) for c in cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=CORR_SCALE,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(color="#E6E9EF"),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color="#9AA3B2")),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style(fig, "Correlation between numeric columns (Pearson r)", max(320, 60 * len(cols) + 120))


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    sub = df[[c for c in {x, y, color} if c]].dropna()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=0)
    fig = px.scatter(sub, x=x, y=y, color=color, opacity=0.75, trendline="ols",
                     trendline_scope="overall", trendline_color_override=PALETTE[2],
                     color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=7, line=dict(width=0)), selector=dict(mode="markers"))
    fig.update_layout(showlegend=bool(color))
    fig.update_xaxes(title=label(x))
    fig.update_yaxes(title=label(y))
    r = sub[x].corr(sub[y])
    return style(fig, f"{label(y)} vs {label(x)} (r = {r:.2f}, line = linear fit)", 380)


def time_trend(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure | None:
    sub = df[[date_col, value_col]].dropna()
    if sub.empty:
        return None
    span_days = (sub[date_col].max() - sub[date_col].min()).days
    freq, word = ("D", "day") if span_days <= 62 else ("W", "week") if span_days <= 365 * 2 // 3 else ("MS", "month")
    agg = "sum" if is_additive(value_col) else "mean"
    ts = sub.set_index(date_col)[value_col].resample(freq).agg(agg).reset_index()
    fig = go.Figure(go.Scatter(x=ts[date_col], y=ts[value_col], mode="lines+markers",
                               line=dict(color=PALETTE[0], width=2.4), marker=dict(size=6),
                               hovertemplate=f"%{{x|%d %b %Y}}<br>{label(value_col)}: %{{y:,.2f}}<extra></extra>"))
    fig.update_yaxes(title=f"{'Total' if agg == 'sum' else 'Average'} {label(value_col)}")
    fig.update_xaxes(title=None)
    return style(fig, f"{'Total' if agg == 'sum' else 'Average'} {label(value_col)} per {word}", 340)
