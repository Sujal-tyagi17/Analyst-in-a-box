"""cleaning.py — data quality checks and opt-in cleaning operations.

Rules:
- Nothing here modifies the DataFrame it is given; cleaning returns a new copy.
- Every operation reports exactly what it changed, so the UI can show it before
  the user decides to use the cleaned data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .profiling import DATETIME, IDENTIFIER, MEASURE, column_kinds, iqr_outliers, KIND_LABEL

_NUMERIC_TEXT = re.compile(r"^\s*[-+]?[₹$€£]?\s*[-+]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s*%?\s*$")


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def _text(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str)


def numeric_text_share(s: pd.Series) -> float:
    """Share of non-empty text values that are really numbers ('1,234', '₹500', '12%')."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return 0.0
    t = _text(s)
    t = t[t.str.strip() != ""]
    return float(t.str.match(_NUMERIC_TEXT).mean()) if len(t) else 0.0


def date_text_share(s: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return 0.0
    t = _text(s)
    if not len(t) or not t.str.contains(r"\d").mean() > 0.9:
        return 0.0
    parsed = pd.to_datetime(t, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def whitespace_count(s: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return 0
    t = _text(s)
    return int((t != t.str.strip().str.replace(r"\s+", " ", regex=True)).sum())


def case_variant_groups(s: pd.Series) -> int:
    """Number of values written in more than one way, e.g. 'North' / 'north' / 'NORTH '."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return 0
    t = _text(s)
    if t.nunique() > 500:
        return 0
    norm = t.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    spellings = pd.DataFrame({"raw": t.str.strip(), "norm": norm}).drop_duplicates()
    return int((spellings.groupby("norm").size() > 1).sum())


def quality_report(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per column: role, missing, unique values and the issues found."""
    kinds = kinds or column_kinds(df)
    n = max(len(df), 1)
    rows = []
    for c in df.columns:
        s = df[c]
        issues: list[str] = []
        miss = int(s.isna().sum())
        if miss == len(df):
            issues.append("completely empty")
        elif miss / n > 0.5:
            issues.append("more than half missing")
        if s.nunique(dropna=True) == 1 and len(df) > 1:
            issues.append("same value in every row")
        if numeric_text_share(s) >= 0.8:
            issues.append("numbers stored as text")
        elif date_text_share(s) >= 0.8:
            issues.append("dates stored as text")
        ws = whitespace_count(s)
        if ws:
            issues.append(f"{ws} value(s) with extra spaces")
        cv = case_variant_groups(s)
        if cv:
            issues.append(f"{cv} value(s) spelled inconsistently")
        if kinds.get(c) == MEASURE:
            n_out, lo, hi = iqr_outliers(s)
            if n_out:
                issues.append(f"{n_out} outlier(s) outside {lo:,.2f} – {hi:,.2f}")
        rows.append({
            "Column": c,
            "Role": KIND_LABEL.get(kinds.get(c, ""), "Text"),
            "Missing": miss,
            "Missing %": round(100 * miss / n, 1),
            "Unique": int(s.nunique(dropna=True)),
            "Issues": "; ".join(issues) if issues else "None found",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

@dataclass
class CleaningOptions:
    remove_duplicates: bool = True
    trim_whitespace: bool = True
    fix_case_variants: bool = True
    convert_types: bool = True
    drop_empty_columns: bool = True
    missing: str = "keep"     # keep | drop_rows | fill
    outliers: str = "keep"    # keep | cap | remove


@dataclass
class CleaningResult:
    data: pd.DataFrame
    changes: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    missing_before: int = 0
    missing_after: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _to_number(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.str.replace(r"[₹$€£,%\s]", "", regex=True)
    return pd.to_numeric(t, errors="coerce").astype("float64")  # "12%" -> 12, not 0.12


def apply_cleaning(df: pd.DataFrame, opts: CleaningOptions) -> CleaningResult:
    """Return a cleaned COPY of df and a plain-language log of every change."""
    out = df.copy()
    log: list[str] = []
    res = CleaningResult(out, log, rows_before=len(df), missing_before=int(df.isna().sum().sum()))

    if opts.drop_empty_columns:
        empty = [c for c in out.columns if out[c].isna().all()]
        if empty:
            out = out.drop(columns=empty)
            log.append(f"Removed {len(empty)} completely empty column(s): {', '.join(empty)}.")

    text_cols = [c for c in out.columns
                 if not pd.api.types.is_numeric_dtype(out[c])
                 and not pd.api.types.is_datetime64_any_dtype(out[c])]

    if opts.trim_whitespace:
        total = 0
        for c in text_cols:
            n = whitespace_count(out[c])
            if n:
                mask = out[c].notna()
                out.loc[mask, c] = (out.loc[mask, c].astype(str).str.strip()
                                    .str.replace(r"\s+", " ", regex=True))
                total += n
        # blank strings are missing values in disguise
        blanks = 0
        for c in text_cols:
            m = out[c].astype("string").str.strip().eq("").fillna(False)
            if m.any():
                blanks += int(m.sum())
                out.loc[m, c] = pd.NA
        if total:
            log.append(f"Trimmed extra spaces in {total} text value(s).")
        if blanks:
            log.append(f"Turned {blanks} blank text value(s) into missing values.")

    if opts.fix_case_variants:
        fixed_cols = []
        for c in text_cols:
            if not case_variant_groups(out[c]):
                continue
            s = out[c]
            mask = s.notna()
            key = s[mask].astype(str).str.strip().str.lower()
            # use the most common spelling of each value
            canon = s[mask].astype(str).str.strip().groupby(key).agg(lambda v: v.value_counts().index[0])
            out.loc[mask, c] = key.map(canon)
            fixed_cols.append(c)
        if fixed_cols:
            log.append("Unified values spelled in different ways (e.g. 'north' / 'North') in: "
                       + ", ".join(fixed_cols) + ".")

    if opts.convert_types:
        for c in text_cols:
            if c not in out.columns:
                continue
            if numeric_text_share(out[c]) >= 0.8:
                before = int(out[c].notna().sum())
                out[c] = _to_number(out[c])
                lost = before - int(out[c].notna().sum())
                log.append(f"Converted '{c}' from text to numbers"
                           + (f" ({lost} value(s) that weren't numbers became missing)." if lost else "."))
            elif date_text_share(out[c]) >= 0.8:
                before = int(out[c].notna().sum())
                out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
                lost = before - int(out[c].notna().sum())
                log.append(f"Converted '{c}' from text to dates"
                           + (f" ({lost} value(s) that weren't dates became missing)." if lost else "."))

    if opts.remove_duplicates:
        d = int(out.duplicated().sum())
        if d:
            out = out.drop_duplicates().reset_index(drop=True)
            log.append(f"Removed {d} duplicate row(s).")

    kinds = column_kinds(out)
    if opts.outliers in ("cap", "remove"):
        measures = [c for c, k in kinds.items() if k == MEASURE]
        for c in measures:
            n_out, lo, hi = iqr_outliers(out[c])
            if not n_out:
                continue
            if opts.outliers == "cap":
                out[c] = out[c].clip(lower=lo, upper=hi)
                log.append(f"Capped {n_out} outlier(s) in '{c}' to the range {lo:,.2f} – {hi:,.2f}.")
            else:
                keep = out[c].isna() | out[c].between(lo, hi)
                out = out[keep].reset_index(drop=True)
                log.append(f"Removed {n_out} row(s) with outliers in '{c}' (outside {lo:,.2f} – {hi:,.2f}).")

    if opts.missing == "drop_rows":
        n0 = len(out)
        out = out.dropna().reset_index(drop=True)
        if n0 - len(out):
            log.append(f"Removed {n0 - len(out)} row(s) that had at least one missing value.")
    elif opts.missing == "fill":
        kinds = column_kinds(out)
        for c in out.columns:
            m = int(out[c].isna().sum())
            if not m:
                continue
            if kinds.get(c) == IDENTIFIER:
                continue  # never invent an ID
            if pd.api.types.is_numeric_dtype(out[c]):
                v = out[c].median()
                out[c] = out[c].fillna(v)
                log.append(f"Filled {m} missing value(s) in '{c}' with the median ({v:,.2f}).")
            elif kinds.get(c) == DATETIME:
                continue  # a made-up date would be misleading; leave missing
            else:
                mode = out[c].mode(dropna=True)
                v = mode.iloc[0] if len(mode) else "Unknown"
                out[c] = out[c].fillna(v)
                log.append(f"Filled {m} missing value(s) in '{c}' with the most common value ('{v}').")

    res.data = out
    res.changes = log
    res.rows_after = len(out)
    res.missing_after = int(out.isna().sum().sum())
    return res
