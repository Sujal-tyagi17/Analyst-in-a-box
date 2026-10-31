"""profiling.py — dataset-agnostic profile, key metrics and automatic insights.

Nothing here knows about any specific dataset: column roles (measure, category,
date, identifier) are inferred from the data itself, and every insight is a
plain computation over the DataFrame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Column-name hints. Only used to pick sensible defaults (e.g. average vs total),
# never to assume what a dataset contains.
_ID_NAME = re.compile(r"(^id$|_id$|^id_|_no$|^no$|code$|_key$|uuid)", re.IGNORECASE)
_NON_ADDITIVE = re.compile(
    r"(pct|percent|rate|ratio|score|rating|avg|mean|price|age|hours|year|season|"
    r"lat|lon|temp|grade|level|rank|index|days|duration|margin)",
    re.IGNORECASE,
)

_CODE_NAME = re.compile(r"(grade|level|class|tier|group|type|category|segment|cluster|flag|gender)",
                        re.IGNORECASE)

MEASURE, CATEGORY, DATETIME, IDENTIFIER, PERIOD, TEXT = (
    "measure", "category", "datetime", "identifier", "period", "text")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_table(db_path: str, table: str, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table back from the SQLite DB the app created, restoring dates."""
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(f'SELECT * FROM "{table}"'), conn)
    finally:
        engine.dispose()
    for c in date_columns or []:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------

def column_kind(s: pd.Series) -> str:
    """Infer the analytical role of one column."""
    name = str(s.name)
    non_null = s.dropna()
    n = len(non_null)
    if pd.api.types.is_bool_dtype(s):
        return CATEGORY
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    if pd.api.types.is_numeric_dtype(s):
        if n == 0:
            return MEASURE
        nunique = non_null.nunique()
        is_int = bool(np.all(np.mod(non_null, 1) == 0))
        if _ID_NAME.search(name) and is_int:
            return IDENTIFIER
        if is_int and nunique == n and n > 20:
            # every value unique and integer: a row identifier, not a quantity
            return IDENTIFIER
        if is_int and 1900 <= non_null.min() <= non_null.max() <= 2100 and nunique <= 60:
            return PERIOD
        if is_int and nunique <= 12 and _CODE_NAME.search(name):
            return CATEGORY  # small integer codes named like one, e.g. grade_level 9-12
        return MEASURE
    # text
    nunique = non_null.nunique()
    if _ID_NAME.search(name) and nunique > 0.9 * max(n, 1):
        return IDENTIFIER
    if n and nunique <= max(30, 0.05 * n):
        return CATEGORY
    return TEXT


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    return {c: column_kind(df[c]) for c in df.columns}


def is_additive(col: str) -> bool:
    """True if summing the column is meaningful (revenue, units), not for scores or prices."""
    return not _NON_ADDITIVE.search(col)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_num(v: float | int | None, decimals: int | None = None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    v = float(v)
    if decimals is None:
        decimals = 0 if abs(v) >= 100 or v == int(v) else 2
    return f"{v:,.{decimals}f}"


def fmt_bytes(n: int | None) -> str:
    if not n:
        return "–"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    rows: int
    columns: int
    kinds: dict[str, str]
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_other: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    memory_bytes: int
    file_bytes: int | None = None
    metrics: list[dict] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)


def measures(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    """Numeric measure columns, skipping constants (a column that is 38 in every row says nothing)."""
    return [c for c, k in kinds.items() if k == MEASURE and df[c].nunique(dropna=True) > 1]


def categories(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    return [c for c, k in kinds.items() if k in (CATEGORY, PERIOD)]


def primary_measure(df: pd.DataFrame, kinds: dict[str, str]) -> str | None:
    ms = measures(df, kinds)
    additive = [c for c in ms if is_additive(c)]
    if additive:
        # the additive measure with the largest total is usually the headline one
        return max(additive, key=lambda c: abs(float(df[c].sum(skipna=True))))
    return ms[0] if ms else None


def primary_category(df: pd.DataFrame, kinds: dict[str, str]) -> str | None:
    cats = [c for c in categories(df, kinds) if 2 <= df[c].nunique() <= 20]
    if not cats:
        return None
    # prefer a real breakdown (3-12 groups, e.g. region) over a yes/no flag;
    # among those, the fewest groups = the broadest split (region before city)
    rich = [c for c in cats if 3 <= df[c].nunique() <= 12]
    return min(rich or cats, key=lambda c: df[c].nunique())


def key_metrics(df: pd.DataFrame, kinds: dict[str, str], limit: int = 4) -> list[dict]:
    """Headline numbers for the measure columns: total for additive ones, average otherwise."""
    out: list[dict] = []
    ms = measures(df, kinds)
    order = {c: i for i, c in enumerate(ms)}
    ms = sorted(ms, key=lambda c: (not is_additive(c), order[c]))
    for c in ms[:limit]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        label = c.replace("_", " ")
        if is_additive(c) and (s >= 0).all():
            out.append({"label": f"Total {label}", "value": fmt_num(s.sum()),
                        "sub": f"avg {fmt_num(s.mean())} · median {fmt_num(s.median())}"})
        else:
            out.append({"label": f"Average {label}", "value": fmt_num(s.mean(), 2 if abs(s.mean()) < 100 else 0),
                        "sub": f"range {fmt_num(s.min())} – {fmt_num(s.max())}"})
    return out


def iqr_outliers(s: pd.Series) -> tuple[int, float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 4:
        return 0, float("nan"), float("nan")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), float(lo), float(hi)


def auto_insights(df: pd.DataFrame, kinds: dict[str, str], limit: int = 5) -> list[str]:
    """Plain-language observations, each one a direct computation on the data."""
    out: list[str] = []
    if df.empty:
        return ["The dataset has no rows."]

    # 1. time span
    for c, k in kinds.items():
        if k == DATETIME:
            d = df[c].dropna()
            if len(d):
                days = (d.max() - d.min()).days
                out.append(f"**{c}** covers {d.min():%d %b %Y} to {d.max():%d %b %Y} ({days:,} days).")
            break

    m = primary_measure(df, kinds)
    g = primary_category(df, kinds)

    # 2. biggest group
    if g:
        if m and is_additive(m):
            by = df.groupby(g, dropna=True)[m].sum().sort_values(ascending=False)
            total = by.sum()
            if len(by) and total:
                out.append(f"**{by.index[0]}** has the highest total {m.replace('_', ' ')} "
                           f"({fmt_num(by.iloc[0])}, {by.iloc[0] / total:.0%} of the total) "
                           f"across {len(by)} {g.replace('_', ' ')} groups.")
        else:
            vc = df[g].value_counts(dropna=True)
            if len(vc):
                out.append(f"**{vc.index[0]}** is the most common {g.replace('_', ' ')} "
                           f"({vc.iloc[0] / vc.sum():.0%} of rows, {len(vc)} groups).")

    # 3. distribution of the main measure
    if m:
        s = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(s) >= 4:
            n_out, _, hi = iqr_outliers(s)
            skew = "right-skewed: a few large values pull the average up" if s.mean() > 1.2 * s.median() else \
                   "left-skewed" if s.mean() < 0.8 * s.median() else "fairly symmetric"
            out.append(f"**{m}** ranges from {fmt_num(s.min())} to {fmt_num(s.max())} "
                       f"(median {fmt_num(s.median())}, mean {fmt_num(s.mean())}); the distribution is {skew}"
                       + (f", with {n_out} outliers above {fmt_num(hi)}." if n_out and s.max() > hi else "."))

    # 4. strongest correlation between measures
    ms = measures(df, kinds)
    if len(ms) >= 2:
        corr = df[ms].apply(pd.to_numeric, errors="coerce").corr().abs()
        np.fill_diagonal(corr.values, np.nan)
        if corr.notna().any().any():
            a, b = corr.stack().idxmax()
            r = df[[a, b]].apply(pd.to_numeric, errors="coerce").corr().loc[a, b]
            if abs(r) >= 0.3:
                strength = "strongly" if abs(r) >= 0.7 else "moderately"
                direction = "positively" if r > 0 else "negatively"
                out.append(f"**{a}** and **{b}** are {strength} {direction} correlated (r = {r:.2f}).")
            else:
                out.append(f"No pair of numeric columns is strongly correlated (largest |r| = {abs(r):.2f}).")

    # 5. data quality
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    dups = int(df.duplicated().sum())
    if miss.empty and not dups:
        out.append("No missing values and no duplicate rows.")
    else:
        parts = []
        if not miss.empty:
            parts.append(f"{len(miss)} column(s) have missing values (most in **{miss.index[0]}**: "
                         f"{miss.iloc[0] / len(df):.0%})")
        if dups:
            parts.append(f"{dups} duplicate row(s)")
        out.append("Data quality: " + "; ".join(parts) + ". See Data Explorer → Data quality & cleaning.")
    return out[:limit]


def build_profile(df: pd.DataFrame, file_bytes: int | None = None) -> Profile:
    kinds = column_kinds(df)
    n_num = sum(k == MEASURE for k in kinds.values())
    n_cat = sum(k in (CATEGORY, PERIOD) for k in kinds.values())
    n_dt = sum(k == DATETIME for k in kinds.values())
    cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    return Profile(
        rows=len(df), columns=df.shape[1], kinds=kinds,
        n_numeric=n_num, n_categorical=n_cat, n_datetime=n_dt,
        n_other=df.shape[1] - n_num - n_cat - n_dt,
        missing_cells=missing, missing_pct=(missing / cells * 100) if cells else 0.0,
        duplicate_rows=int(df.duplicated().sum()) if len(df) else 0,
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        file_bytes=file_bytes,
        metrics=key_metrics(df, kinds),
        insights=auto_insights(df, kinds),
    )


KIND_LABEL = {MEASURE: "Numeric", CATEGORY: "Category", PERIOD: "Period",
              DATETIME: "Date/time", IDENTIFIER: "Identifier", TEXT: "Text"}
