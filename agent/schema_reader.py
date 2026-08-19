"""schema_reader.py — dataset-agnostic ingestion + schema introspection.

Loads any flat file (CSV/XLSX) or an existing SQLite/Postgres URL into a local
SQLite database and produces a compact, LLM-friendly schema summary. Nothing here
knows about any specific dataset or domain — everything is derived at runtime.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_identifier(name: str) -> str:
    """Make an arbitrary column/table name safe to use as a SQL identifier."""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]", "", name)
    if not name:
        name = "col"
    if re.match(r"^\d", name):
        name = "c_" + name
    return name


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Schema data model
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_count: int
    distinct_count: int
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    row_count: int
    columns: list[ColumnInfo]
    sample_rows: list[dict] = field(default_factory=list)


@dataclass
class DatasetSchema:
    db_path: str
    tables: list[TableInfo]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_text(self) -> str:
        """Compact text form fed to the LLM as context on every question."""
        lines: list[str] = []
        for t in self.tables:
            lines.append(f'TABLE "{t.name}" ({t.row_count} rows)')
            for c in t.columns:
                samples = ", ".join(str(v) for v in c.sample_values[:5])
                lines.append(
                    f'  - "{c.name}" {c.dtype} '
                    f"(nulls={c.null_count}, distinct={c.distinct_count}) "
                    f"e.g. [{samples}]"
                )
            if t.sample_rows:
                lines.append("  sample rows:")
                for row in t.sample_rows[:3]:
                    lines.append("    " + json.dumps(row, default=str, ensure_ascii=False))
            lines.append("")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _infer_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort type inference: try numeric, then datetime, keep as text otherwise."""
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # only accept numeric conversion if it doesn't destroy most values
            non_null = df[col].notna().sum()
            if non_null and converted.notna().sum() >= 0.9 * non_null:
                df[col] = converted
                continue
            # try datetime
            try:
                dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if non_null and dt.notna().sum() >= 0.9 * non_null:
                    df[col] = dt
            except Exception:
                pass
    return df


def load_flat_file_to_sqlite(
    file_path_or_buffer: Any,
    db_path: str,
    table_name: str = "dataset",
    file_type: str | None = None,
) -> str:
    """Load a CSV/XLSX into a SQLite DB. Returns the table name used."""
    if file_type is None and isinstance(file_path_or_buffer, str):
        file_type = os.path.splitext(file_path_or_buffer)[1].lower().lstrip(".")

    if file_type in ("xlsx", "xls"):
        df = pd.read_excel(file_path_or_buffer)
    else:  # default csv/tsv
        sep = "\t" if file_type == "tsv" else ","
        df = pd.read_csv(file_path_or_buffer, sep=sep)

    df.columns = _dedupe([_sanitize_identifier(c) for c in df.columns])
    df = _infer_and_convert(df)

    table_name = _sanitize_identifier(table_name)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    engine.dispose()
    return table_name


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def introspect(db_path: str, sample_rows: int = 5) -> DatasetSchema:
    """Introspect all tables in a SQLite DB and build a DatasetSchema."""
    engine = create_engine(f"sqlite:///{db_path}")
    return _introspect_engine(engine, db_path, sample_rows)


def introspect_url(db_url: str, sample_rows: int = 5) -> DatasetSchema:
    """Introspect an arbitrary SQLAlchemy URL (e.g. postgres)."""
    engine = create_engine(db_url)
    return _introspect_engine(engine, db_url, sample_rows)


def _introspect_engine(engine: Engine, db_ref: str, sample_rows: int) -> DatasetSchema:
    insp = inspect(engine)
    tables: list[TableInfo] = []

    with engine.connect() as conn:
        for tname in insp.get_table_names():
            # row count
            try:
                row_count = conn.execute(text(f'SELECT COUNT(*) FROM "{tname}"')).scalar() or 0
            except Exception:
                row_count = 0

            # sample rows via pandas for clean typing
            try:
                sample_df = pd.read_sql(text(f'SELECT * FROM "{tname}" LIMIT :n'),
                                        conn, params={"n": sample_rows})
            except Exception:
                sample_df = pd.DataFrame()

            columns: list[ColumnInfo] = []
            for col in insp.get_columns(tname):
                cname = col["name"]
                dtype = str(col.get("type", "UNKNOWN"))
                null_count = 0
                distinct_count = 0
                samples: list[Any] = []
                try:
                    null_count = conn.execute(
                        text(f'SELECT COUNT(*) FROM "{tname}" WHERE "{cname}" IS NULL')
                    ).scalar() or 0
                    distinct_count = conn.execute(
                        text(f'SELECT COUNT(DISTINCT "{cname}") FROM "{tname}"')
                    ).scalar() or 0
                    rows = conn.execute(
                        text(f'SELECT DISTINCT "{cname}" FROM "{tname}" '
                             f'WHERE "{cname}" IS NOT NULL LIMIT 5')
                    ).fetchall()
                    samples = [r[0] for r in rows]
                except Exception:
                    pass
                columns.append(ColumnInfo(cname, dtype, null_count, distinct_count, samples))

            sample_records = sample_df.to_dict(orient="records") if not sample_df.empty else []
            tables.append(TableInfo(tname, row_count, columns, sample_records))

    engine.dispose()
    return DatasetSchema(db_ref, tables)


# ---------------------------------------------------------------------------
# Convenience: file -> (db, schema)
# ---------------------------------------------------------------------------

def ingest(
    file_path_or_buffer: Any,
    db_path: str,
    table_name: str = "dataset",
    file_type: str | None = None,
) -> DatasetSchema:
    """One-shot: load a flat file into SQLite and return its schema."""
    load_flat_file_to_sqlite(file_path_or_buffer, db_path, table_name, file_type)
    return introspect(db_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        out_db = sys.argv[2] if len(sys.argv) > 2 else "data/_tmp.db"
        os.makedirs(os.path.dirname(out_db) or ".", exist_ok=True)
        schema = ingest(path, out_db, table_name=os.path.splitext(os.path.basename(path))[0])
        print(schema.to_prompt_text())
