"""
Phase 7 — App Tests
Tests that all app modules import correctly, data loads,
and all utility functions work without launching Streamlit.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED
from src.utils.logger import get_logger

logger = get_logger("test.app")


def test_master_parquet_accessible():
    path = DATA_PROCESSED / "reviews_topics.parquet"
    assert path.exists(), f"Master parquet not found: {path}"
    df = pd.read_parquet(path)
    assert len(df) > 1000
    print(f"✓ Master parquet accessible: {len(df):,} rows")


def test_streamlit_utils_imports():
    from src.streamlit_utils import (
        compute_kpis, filter_dataframe,
    )
    print("✓ streamlit_utils imports cleanly")


def test_compute_kpis():
    from src.streamlit_utils import compute_kpis
    df   = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    kpis = compute_kpis(df)

    required_keys = [
        "total", "pct_positive", "avg_rating",
        "num_topics", "model_agree", "vader_agree",
        "bert_agree", "coherence",
    ]
    for key in required_keys:
        assert key in kpis, f"Missing KPI: {key}"

    assert kpis["total"] > 1000
    assert 0 <= kpis["pct_positive"] <= 100
    assert 1 <= kpis["avg_rating"] <= 5
    assert kpis["num_topics"] == 8
    assert 0 <= kpis["model_agree"] <= 100

    print(f"✓ compute_kpis: total={kpis['total']:,}, "
          f"pct_positive={kpis['pct_positive']}%, "
          f"avg_rating={kpis['avg_rating']}★")


def test_filter_dataframe_no_filters():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, [], [], [], "")
    assert len(filtered) == len(df)
    print(f"✓ filter_dataframe (no filters): {len(filtered):,} rows returned")


def test_filter_dataframe_by_topic():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, ["battery_value"], [], [], "")
    assert len(filtered) > 0
    assert (filtered["topic_label"] == "battery_value").all()
    print(f"✓ filter_dataframe (topic=battery_value): {len(filtered):,} rows")


def test_filter_dataframe_by_sentiment():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, [], ["negative"], [], "")
    assert len(filtered) > 0
    assert (filtered["bert_label"] == "negative").all()
    print(f"✓ filter_dataframe (sentiment=negative): {len(filtered):,} rows")


def test_filter_dataframe_by_rating():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, [], [], [1, 2], "")
    assert len(filtered) > 0
    assert filtered["rating"].isin([1, 2]).all()
    print(f"✓ filter_dataframe (rating=1,2): {len(filtered):,} rows")


def test_filter_dataframe_by_keyword():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, [], [], [], "battery")
    assert len(filtered) > 0
    assert filtered["review_text"].str.contains("battery", case=False, na=False).all()
    print(f"✓ filter_dataframe (keyword='battery'): {len(filtered):,} rows")


def test_filter_dataframe_combined():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, ["battery_value"], ["negative"], [1, 2], "")
    assert len(filtered) >= 0   # may be 0 with tight filters — just shouldn't crash
    print(f"✓ filter_dataframe (combined filters): {len(filtered):,} rows")


def test_filter_dataframe_no_results():
    from src.streamlit_utils import filter_dataframe
    df       = pd.read_parquet(DATA_PROCESSED / "reviews_topics.parquet")
    filtered = filter_dataframe(df, [], [], [], "xyzthisdoesnotexist99999")
    assert len(filtered) == 0
    print("✓ filter_dataframe (no results): correctly returns 0 rows")


def test_all_charts_importable():
    from src.visualization.charts import (
        chart_sentiment_donut,
        chart_rating_bar,
        chart_sentiment_trend,
        chart_topic_bar,
        chart_topic_sentiment_heatmap,
        chart_avg_rating_topic,
        chart_vader_histogram,
    )
    print("✓ All chart functions importable")


def test_app_file_exists():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    assert app_path.exists(), f"app.py not found at {app_path}"
    content = app_path.read_text()
    assert "st.set_page_config" in content
    assert "Overview" in content
    assert "Sentiment Analysis" in content
    assert "Topic Explorer" in content
    assert "Review Search" in content
    print("✓ app.py exists and contains all 4 page definitions")


def test_streamlit_config_exists():
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert config_path.exists(), f".streamlit/config.toml not found"
    content = config_path.read_text()
    assert "dark" in content
    print("✓ .streamlit/config.toml exists with dark theme")


if __name__ == "__main__":
    print("\nRunning Phase 7 app tests...\n")

    test_master_parquet_accessible()
    test_streamlit_utils_imports()
    test_compute_kpis()
    test_filter_dataframe_no_filters()
    test_filter_dataframe_by_topic()
    test_filter_dataframe_by_sentiment()
    test_filter_dataframe_by_rating()
    test_filter_dataframe_by_keyword()
    test_filter_dataframe_combined()
    test_filter_dataframe_no_results()
    test_all_charts_importable()
    test_app_file_exists()
    test_streamlit_config_exists()

    print("\n✅ All Phase 7 app tests passed.")
