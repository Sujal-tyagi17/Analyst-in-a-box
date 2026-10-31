"""charts.py — the shared dark Plotly theme and small figure helpers.

Every interactive chart in the app goes through `style()` so they all share
the same colours, fonts and hover format.
"""

from __future__ import annotations

import plotly.graph_objects as go

PALETTE = ["#7C5CFF", "#4F8BFF", "#22D3EE", "#34D399", "#FBBF24", "#F472B6", "#A78BFA", "#94A3B8"]
TEXT, MUTED, GRID, BG = "#E6E9EF", "#9AA3B2", "#232B3B", "rgba(0,0,0,0)"

# diverging scale for correlations: blue (negative) -> neutral -> purple (positive)
CORR_SCALE = [[0.0, "#22D3EE"], [0.5, "#1C2433"], [1.0, "#7C5CFF"]]
SEQ_SCALE = [[0.0, "#1C2433"], [1.0, "#7C5CFF"]]


def style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)) if title else None,
        height=height,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", size=12, color=MUTED),
        colorway=PALETTE,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        hoverlabel=dict(bgcolor="#171E2B", bordercolor="#2C3548", font=dict(color=TEXT, size=12)),
        legend=dict(bgcolor=BG, font=dict(color=MUTED)),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED),
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED),
                     separatethousands=True, automargin=True)
    return fig


def label(col: str) -> str:
    return str(col).replace("_", " ")


# Shown on every chart's toolbar; "Download plot as PNG" comes from here.
CONFIG = {"displaylogo": False,
          "toImageButtonOptions": {"format": "png", "scale": 2},
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def agent_chart(spec: dict) -> go.Figure | None:
    """Rebuild a chart the AI requested (make_chart) as an interactive Plotly figure.

    `spec` carries chart_type, x, y, title and the query rows behind the chart.
    Returns None if the rows don't contain the columns (the PNG is used instead).
    """
    import pandas as pd
    import plotly.express as px

    rows = spec.get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    kind, x, y = (spec.get("chart_type") or "bar").lower(), spec.get("x"), spec.get("y")
    title = spec.get("title")
    try:
        if kind == "histogram":
            col = x if x in df.columns else y
            fig = px.histogram(df, x=col, color_discrete_sequence=[PALETTE[0]])
            fig.update_yaxes(title="Rows")
            fig.update_xaxes(title=label(col))
        elif kind == "box":
            if x and x in df.columns:
                fig = px.box(df, x=x, y=y, points="outliers", color_discrete_sequence=[PALETTE[1]])
                fig.update_xaxes(title=label(x))
            else:
                fig = px.box(df, y=y, points="outliers", color_discrete_sequence=[PALETTE[1]])
            fig.update_yaxes(title=label(y))
        elif x in df.columns and y in df.columns:
            if kind == "line":
                fig = px.line(df, x=x, y=y, markers=True, color_discrete_sequence=[PALETTE[0]])
            elif kind == "scatter":
                fig = px.scatter(df, x=x, y=y, opacity=.8, color_discrete_sequence=[PALETTE[0]])
            else:
                d = df.copy()
                d[x] = d[x].astype(str)
                fig = px.bar(d, x=x, y=y, color_discrete_sequence=[PALETTE[0]], text_auto=".3s")
                fig.update_traces(textposition="outside", cliponaxis=False,
                                  textfont=dict(color=MUTED))
                fig.update_xaxes(type="category")
            fig.update_xaxes(title=label(x))
            fig.update_yaxes(title=label(y))
        else:
            return None
    except Exception:
        return None
    fig.update_traces(hovertemplate=None)
    return style(fig, title, 380)
