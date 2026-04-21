"""
Phase 6 — Visualization Tests
Tests data preparation functions and chart rendering.
"""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUTS_FIGURES
from src.utils.logger import get_logger
from src.visualization.dashboard_data import (
    load_master,
    prep_sentiment_distribution,
    prep_rating_distribution,
    prep_sentiment_over_time,
    prep_topic_distribution,
    prep_topic_sentiment_heatmap,
    prep_avg_rating_per_topic,
    prep_vader_score_distribution,
    prep_topic_wordcloud_tokens,
)
from src.visualization.charts import (
    chart_sentiment_donut,
    chart_rating_bar,
    chart_sentiment_trend,
    chart_topic_bar,
    chart_topic_sentiment_heatmap,
    chart_avg_rating_topic,
    chart_vader_histogram,
    chart_wordcloud,
)

logger = get_logger("test.visualization")

# Load once — shared across all tests
_DF = None

def get_df():
    global _DF
    if _DF is None:
        _DF = load_master()
    return _DF


# ── Data preparation tests ────────────────────────────────────────────────────

def test_load_master():
    df = get_df()
    assert len(df) > 1000
    assert "dominant_topic" in df.columns
    assert "vader_score" in df.columns
    assert "bert_label" in df.columns
    print(f"✓ load_master: {len(df):,} rows")


def test_prep_sentiment_distribution():
    df  = get_df()
    out = prep_sentiment_distribution(df)
    assert "model" in out.columns
    assert "label" in out.columns
    assert "count" in out.columns
    assert set(out["model"].unique()) == {"VADER", "DistilBERT"}
    assert out["count"].sum() > 0
    print(f"✓ prep_sentiment_distribution: {len(out)} rows")


def test_prep_rating_distribution():
    df  = get_df()
    out = prep_rating_distribution(df)
    assert "rating" in out.columns
    assert "count" in out.columns
    assert len(out) >= 3        # at least 3 distinct ratings
    assert out["count"].sum() > 0
    print(f"✓ prep_rating_distribution: {len(out)} rating levels")


def test_prep_sentiment_over_time():
    df  = get_df()
    out = prep_sentiment_over_time(df)
    assert "month" in out.columns
    assert "count" in out.columns
    assert "bert_label" in out.columns
    assert len(out) > 0
    print(f"✓ prep_sentiment_over_time: {len(out)} monthly data points")


def test_prep_topic_distribution():
    df  = get_df()
    out = prep_topic_distribution(df)
    assert "topic_label" in out.columns
    assert "count" in out.columns
    assert "avg_rating" in out.columns
    assert len(out) == 8        # 8 topics
    print(f"✓ prep_topic_distribution: {len(out)} topics")


def test_prep_topic_sentiment_heatmap():
    df  = get_df()
    out = prep_topic_sentiment_heatmap(df)
    assert "topic_label" in out.columns
    assert "pct_negative" in out.columns
    assert out["pct_negative"].between(0, 100).all()
    print(f"✓ prep_topic_sentiment_heatmap: {len(out)} topics")


def test_prep_avg_rating_per_topic():
    df  = get_df()
    out = prep_avg_rating_per_topic(df)
    assert "topic_label" in out.columns
    assert "avg_rating" in out.columns
    assert "ci" in out.columns
    assert out["avg_rating"].between(1, 5).all()
    print(f"✓ prep_avg_rating_per_topic: avg range [{out['avg_rating'].min():.2f}, {out['avg_rating'].max():.2f}]")


def test_prep_vader_score_distribution():
    df  = get_df()
    out = prep_vader_score_distribution(df)
    assert "vader_score" in out.columns
    assert out["vader_score"].between(-1, 1).all()
    print(f"✓ prep_vader_score_distribution: {len(out):,} rows")


def test_prep_topic_wordcloud_tokens():
    df  = get_df()
    out = prep_topic_wordcloud_tokens(df)
    assert isinstance(out, dict)
    assert len(out) == 8
    for label, tokens in out.items():
        assert len(tokens) > 0, f"Empty tokens for topic: {label}"
    print(f"✓ prep_topic_wordcloud_tokens: {len(out)} topics with tokens")


# ── Chart rendering tests ─────────────────────────────────────────────────────

def _assert_figure(fig: go.Figure, name: str, min_traces: int = 1):
    assert isinstance(fig, go.Figure), f"{name} did not return a Figure"
    assert len(fig.data) >= min_traces, f"{name} has fewer than {min_traces} traces"
    assert fig.layout.height is not None
    print(f"✓ {name}: {len(fig.data)} trace(s), height={fig.layout.height}")


def test_chart_sentiment_donut():
    df  = get_df()
    fig = chart_sentiment_donut(prep_sentiment_distribution(df))
    _assert_figure(fig, "chart_sentiment_donut", min_traces=2)


def test_chart_rating_bar():
    df  = get_df()
    fig = chart_rating_bar(prep_rating_distribution(df))
    _assert_figure(fig, "chart_rating_bar")


def test_chart_sentiment_trend():
    df  = get_df()
    fig = chart_sentiment_trend(prep_sentiment_over_time(df))
    _assert_figure(fig, "chart_sentiment_trend")


def test_chart_topic_bar():
    df  = get_df()
    fig = chart_topic_bar(prep_topic_distribution(df))
    _assert_figure(fig, "chart_topic_bar")


def test_chart_topic_sentiment_heatmap():
    df  = get_df()
    fig = chart_topic_sentiment_heatmap(prep_topic_sentiment_heatmap(df))
    _assert_figure(fig, "chart_topic_sentiment_heatmap")


def test_chart_avg_rating_topic():
    df  = get_df()
    fig = chart_avg_rating_topic(prep_avg_rating_per_topic(df))
    _assert_figure(fig, "chart_avg_rating_topic")


def test_chart_vader_histogram():
    df  = get_df()
    fig = chart_vader_histogram(prep_vader_score_distribution(df))
    _assert_figure(fig, "chart_vader_histogram", min_traces=2)


def test_chart_wordcloud():
    df  = get_df()
    wc_tokens = prep_topic_wordcloud_tokens(df)
    first_label  = list(wc_tokens.keys())[0]
    first_tokens = wc_tokens[first_label]
    b64 = chart_wordcloud(first_label, first_tokens)
    assert isinstance(b64, str)
    assert len(b64) > 100, "Word cloud base64 string too short"
    print(f"✓ chart_wordcloud: generated for '{first_label}' ({len(b64)} chars)")


# ── Output file tests ─────────────────────────────────────────────────────────

def test_exported_figures_exist():
    expected = [
        "01_sentiment_donut.html",
        "02_rating_bar.html",
        "03_sentiment_trend.html",
        "04_topic_bar.html",
        "05_topic_heatmap.html",
        "06_avg_rating_topic.html",
        "07_vader_histogram.html",
        "wordclouds.html",
    ]
    missing = []
    for fname in expected:
        path = OUTPUTS_FIGURES / fname
        if not path.exists():
            missing.append(fname)
        else:
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {fname} ({size_kb:.1f} KB)")

    if missing:
        print(f"\n  ⚠ Missing figures (run export_figures.py first): {missing}")
    else:
        print(f"✓ All {len(expected)} figure files exist")


if __name__ == "__main__":
    print("\nRunning Phase 6 visualization tests...\n")

    print("--- Data Preparation Tests ---")
    test_load_master()
    test_prep_sentiment_distribution()
    test_prep_rating_distribution()
    test_prep_sentiment_over_time()
    test_prep_topic_distribution()
    test_prep_topic_sentiment_heatmap()
    test_prep_avg_rating_per_topic()
    test_prep_vader_score_distribution()
    test_prep_topic_wordcloud_tokens()

    print("\n--- Chart Rendering Tests ---")
    test_chart_sentiment_donut()
    test_chart_rating_bar()
    test_chart_sentiment_trend()
    test_chart_topic_bar()
    test_chart_topic_sentiment_heatmap()
    test_chart_avg_rating_topic()
    test_chart_vader_histogram()
    test_chart_wordcloud()

    print("\n--- Exported Figure File Tests ---")
    test_exported_figures_exist()

    print("\n✅ All Phase 6 visualization tests passed.")
