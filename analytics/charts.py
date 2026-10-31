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
