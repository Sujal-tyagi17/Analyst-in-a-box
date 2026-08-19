import os
import pandas as pd
import pytest
from agent.schema_reader import ingest
from agent.tools import run_sql, run_analysis, make_chart, SQLValidationError

@pytest.fixture(scope="module")
def sample_db(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data")
    db_file = str(tmp_dir / "test.db")
    csv_file = str(tmp_dir / "test.csv")
    df = pd.DataFrame({
        "category": ["A", "B", "A", "B", "C"],
        "sales": [100.0, 200.0, 150.0, 300.0, 250.0],
        "quantity": [10, 20, 15, 30, 25]
    })
    df.to_csv(csv_file, index=False)
    ingest(csv_file, db_file, table_name="sales")
    return db_file

def test_run_sql_success(sample_db):
    res = run_sql("SELECT * FROM sales", sample_db)
    assert res["row_count"] == 5
    assert len(res["columns"]) == 3
    assert res["warning"] is None

def test_run_sql_forbidden(sample_db):
    with pytest.raises(SQLValidationError):
        run_sql("DROP TABLE sales", sample_db)

def test_run_analysis_correlation(sample_db):
    res = run_sql("SELECT * FROM sales", sample_db)
    analysis = run_analysis(res, "correlation", x="sales", y="quantity")
    assert analysis["method"] == "correlation"
    assert "pearson_r" in analysis

def test_run_analysis_describe(sample_db):
    res = run_sql("SELECT * FROM sales", sample_db)
    analysis = run_analysis(res, "describe")
    assert analysis["method"] == "describe"
    assert "summary" in analysis

def test_run_analysis_group_compare(sample_db):
    res = run_sql("SELECT * FROM sales", sample_db)
    analysis = run_analysis(res, "group_compare", group="category", value="sales")
    assert analysis["method"] == "group_compare"
    assert "group_means" in analysis

def test_make_chart(sample_db):
    res = run_sql("SELECT * FROM sales", sample_db)
    chart = make_chart(res, chart_type="bar", x="category", y="sales")
    assert chart["chart_type"] == "bar"
    assert chart["image_base64"] is not None
