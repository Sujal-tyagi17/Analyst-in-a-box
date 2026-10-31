"""Tests for the deterministic analytics package and file loading."""
import os

import numpy as np
import pandas as pd
import pytest

from agent import schema_reader
from analytics import profiling as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "data", "sample_datasets")


def _load_sample(name, tmp_path):
    db = str(tmp_path / f"{name}.db")
    table = schema_reader.load_flat_file_to_sqlite(os.path.join(SAMPLES, f"{name}.csv"), db,
                                                   table_name=name, file_type="csv")
    t = schema_reader.introspect(db).tables[0]
    dates = [c.name for c in t.columns if "DATE" in c.dtype.upper() or "TIME" in c.dtype.upper()]
    return P.load_table(db, table, dates)


# ---------------------------------------------------------------- column roles

def test_column_kinds_sales(tmp_path):
    kinds = P.column_kinds(_load_sample("sales", tmp_path))
    assert kinds["order_id"] == P.IDENTIFIER
    assert kinds["order_date"] == P.DATETIME
    assert kinds["region"] == P.CATEGORY
    assert kinds["revenue"] == P.MEASURE


def test_column_kinds_other_samples(tmp_path):
    sports = P.column_kinds(_load_sample("sports", tmp_path))
    assert sports["season"] == P.PERIOD
    edu = P.column_kinds(_load_sample("education", tmp_path))
    assert edu["student_id"] == P.IDENTIFIER
    assert edu["grade_level"] == P.CATEGORY
    assert edu["exam_score"] == P.MEASURE


def test_small_integer_measure_stays_numeric():
    df = pd.DataFrame({"quantity": np.tile([1, 2, 3, 4], 30), "rating": np.tile([1, 2, 3, 4, 5], 24)})
    kinds = P.column_kinds(df)
    assert kinds["quantity"] == P.MEASURE
    assert kinds["rating"] == P.MEASURE


# ---------------------------------------------------------------- metrics & insights

def test_metrics_match_pandas(tmp_path):
    df = _load_sample("sales", tmp_path)
    prof = P.build_profile(df)
    total = next(m for m in prof.metrics if m["label"] == "Total revenue")
    assert total["value"] == f"{df['revenue'].sum():,.0f}"
    # unit price is not additive, so it must be averaged, never summed
    assert any(m["label"] == "Average unit price" for m in prof.metrics)
    assert not any(m["label"] == "Total unit price" for m in prof.metrics)


def test_constant_column_not_a_metric(tmp_path):
    prof = P.build_profile(_load_sample("sports", tmp_path))  # games = 38 in every row
    assert not any("games" in m["label"] for m in prof.metrics)


def test_insight_numbers_are_real(tmp_path):
    df = _load_sample("sales", tmp_path)
    prof = P.build_profile(df)
    top = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    line = next(i for i in prof.insights if "highest total revenue" in i)
    assert str(top.index[0]) in line and f"{top.iloc[0]:,.0f}" in line


def test_profile_counts_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 1, None, 4], "b": ["x", "x", "y", None]})
    prof = P.build_profile(df)
    assert prof.missing_cells == 2
    assert prof.duplicate_rows == 1
    assert any("Data quality" in i for i in prof.insights)


def test_profile_handles_tiny_and_text_only_frames():
    P.build_profile(pd.DataFrame({"name": ["a", "b"]}))  # no numeric columns
    P.build_profile(pd.DataFrame({"v": [1.0]}))          # single row


# ---------------------------------------------------------------- file loading

def test_cp1252_csv_loads(tmp_path):
    f = tmp_path / "win.csv"
    f.write_bytes("name,city,amount\nJosé,München,10\nRenée,Zürich,20\n".encode("cp1252"))
    schema_reader.load_flat_file_to_sqlite(str(f), str(tmp_path / "w.db"), "win", "csv")
    df = P.load_table(str(tmp_path / "w.db"), "win")
    assert df["city"].tolist() == ["München", "Zürich"]


@pytest.mark.parametrize("content, msg", [
    (b"a,b,c\n", "no data rows"),
    (b"", "empty"),
    (bytes(range(256)) * 4, "binary"),
])
def test_bad_files_give_clear_errors(tmp_path, content, msg):
    f = tmp_path / "bad.csv"
    f.write_bytes(content)
    with pytest.raises(schema_reader.DatasetError, match=msg):
        schema_reader.load_flat_file_to_sqlite(str(f), str(tmp_path / "b.db"), "bad", "csv")


def test_excel_upload_matches_csv(tmp_path):
    src = pd.read_csv(os.path.join(SAMPLES, "sales.csv"))
    x = tmp_path / "sales.xlsx"
    src.to_excel(x, index=False)
    schema_reader.load_flat_file_to_sqlite(str(x), str(tmp_path / "x.db"), "sales", "xlsx")
    df = P.load_table(str(tmp_path / "x.db"), "sales")
    assert len(df) == len(src)
    assert round(df["revenue"].sum(), 2) == round(src["revenue"].sum(), 2)
