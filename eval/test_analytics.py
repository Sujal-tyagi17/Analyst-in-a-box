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


# ---------------------------------------------------------------- cleaning

from analytics import cleaning as C, eda as E  # noqa: E402


@pytest.fixture
def messy():
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 4],
        "region": ["North", "north", "  South", "  South", "South"],
        "spend": ["1,200", "₹300", "450", "450", "2,000"],
        "age": [30.0, None, 41.0, 41.0, 25.0],
        "empty": [None] * 5,
    })


def test_quality_report_flags_issues(messy):
    rep = C.quality_report(messy).set_index("Column")["Issues"]
    assert "numbers stored as text" in rep["spend"]
    assert "spelled inconsistently" in rep["region"]
    assert "extra spaces" in rep["region"]
    assert "completely empty" in rep["empty"]


def test_cleaning_never_modifies_original(messy):
    before = messy.copy()
    C.apply_cleaning(messy, C.CleaningOptions(missing="fill", outliers="cap"))
    pd.testing.assert_frame_equal(messy, before)


def test_cleaning_results_and_log(messy):
    res = C.apply_cleaning(messy, C.CleaningOptions(missing="fill"))
    out = res.data
    assert "empty" not in out.columns
    assert sorted(out["region"].unique()) == ["North", "South"]
    assert out["spend"].tolist() == [1200.0, 300.0, 450.0, 2000.0]  # duplicate removed, text -> numbers
    assert out["age"].isna().sum() == 0
    assert res.rows_before == 5 and res.rows_after == 4
    assert any("duplicate" in c for c in res.changes)
    assert any("median" in c for c in res.changes)


def test_cleaning_options_off_changes_nothing(messy):
    res = C.apply_cleaning(messy, C.CleaningOptions(
        remove_duplicates=False, trim_whitespace=False, fix_case_variants=False,
        convert_types=False, drop_empty_columns=False))
    assert not res.changed
    pd.testing.assert_frame_equal(res.data, messy)


def test_outlier_cap_and_remove():
    df = pd.DataFrame({"v": [10, 11, 12, 13, 12, 11, 500.0]})
    capped = C.apply_cleaning(df, C.CleaningOptions(outliers="cap", remove_duplicates=False)).data
    assert capped["v"].max() < 500 and len(capped) == 7
    removed = C.apply_cleaning(df, C.CleaningOptions(outliers="remove", remove_duplicates=False)).data
    assert 500 not in removed["v"].tolist() and len(removed) == 6


def test_fill_never_invents_ids_or_dates():
    df = pd.DataFrame({"order_id": [1, None, 3] * 10,
                       "d": pd.to_datetime(["2024-01-01", None, "2024-01-03"] * 10),
                       "x": [1.0, None, 3.0] * 10})
    out = C.apply_cleaning(df, C.CleaningOptions(missing="fill", remove_duplicates=False)).data
    assert out["order_id"].isna().sum() == 10
    assert out["d"].isna().sum() == 10
    assert out["x"].isna().sum() == 0


# ---------------------------------------------------------------- EDA

def test_describe_numeric_matches_pandas(tmp_path):
    df = _load_sample("sales", tmp_path)
    d = E.describe_numeric(df, ["revenue"]).iloc[0]
    assert d["Mean"] == pytest.approx(df["revenue"].mean())
    assert d["Median"] == pytest.approx(df["revenue"].median())
    assert d["Q3 (75%)"] == pytest.approx(df["revenue"].quantile(0.75))


def test_eda_figures_build_for_every_sample(tmp_path):
    for name in ("sales", "sports", "education"):
        df = _load_sample(name, tmp_path)
        kinds = P.column_kinds(df)
        nums, cats = E.numeric_columns(df, kinds), E.categorical_columns(df, kinds)
        for c in nums:
            assert E.histogram(df, c).data
            assert E.box_plot(df, c).data
        for c in cats:
            freq = E.category_frequencies(df, c)
            assert freq["Count"].sum() == len(df)
            assert E.frequency_bar(freq, c).data
        if len(nums) >= 2:
            assert E.correlation_heatmap(df, nums).data
            top = E.strongest_pairs(df, nums).iloc[0]
            assert E.scatter(df, top["Column A"], top["Column B"]).data
        for d in E.datetime_columns(kinds):
            assert E.time_trend(df, d, nums[0]).data


def test_heatmap_skipped_with_one_numeric_column():
    assert E.correlation_heatmap(pd.DataFrame({"a": [1, 2, 3]}), ["a"]) is None
