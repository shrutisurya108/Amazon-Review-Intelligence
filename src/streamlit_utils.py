"""
Phase 7 — Streamlit App: Utility Functions
Cached data loaders and KPI helpers shared across all pages.
@st.cache_data ensures the master parquet is loaded only once per session.
"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED
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
from src.visualization.charts import chart_all_wordclouds
from src.utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_data(show_spinner="Loading dataset...")
def get_master() -> pd.DataFrame:
    """Loads and caches the master parquet. Called once per session."""
    return load_master()


@st.cache_data(show_spinner=False)
def get_all_prep_data(_df: pd.DataFrame) -> dict:
    """
    Pre-computes all aggregated DataFrames for chart rendering.
    Cached so switching pages doesn't recompute.
    Prefix _ on arg name tells Streamlit not to hash the DataFrame.
    """
    return {
        "sentiment_dist"  : prep_sentiment_distribution(_df),
        "rating_dist"     : prep_rating_distribution(_df),
        "monthly"         : prep_sentiment_over_time(_df),
        "topic_dist"      : prep_topic_distribution(_df),
        "topic_heatmap"   : prep_topic_sentiment_heatmap(_df),
        "avg_rating"      : prep_avg_rating_per_topic(_df),
        "vader_scores"    : prep_vader_score_distribution(_df),
        "wc_tokens"       : prep_topic_wordcloud_tokens(_df),
    }


@st.cache_data(show_spinner="Generating word clouds...")
def get_wordclouds(_wc_tokens: dict) -> dict:
    """Generates and caches all word cloud base64 images."""
    return chart_all_wordclouds(_wc_tokens)


def compute_kpis(df: pd.DataFrame) -> dict:
    """Computes top-level KPI values for the Overview page."""
    total         = len(df)
    pct_positive  = (df["bert_label"] == "positive").mean() * 100
    avg_rating    = df["rating"].mean()
    num_topics    = df["dominant_topic"].nunique()
    model_agree   = df["sentiment_agree"].mean() * 100
    vader_agree   = (
        df[df["rating"] != 3]
        .assign(star=lambda d: d["rating"].apply(lambda r: "positive" if r >= 4 else "negative"))
        .pipe(lambda d: (d["vader_label"] == d["star"]).mean() * 100)
    )
    bert_agree    = (
        df[df["rating"] != 3]
        .assign(star=lambda d: d["rating"].apply(lambda r: "positive" if r >= 4 else "negative"))
        .pipe(lambda d: (d["bert_label"] == d["star"]).mean() * 100)
    )

    return {
        "total"        : total,
        "pct_positive" : round(pct_positive, 1),
        "avg_rating"   : round(avg_rating, 2),
        "num_topics"   : num_topics,
        "model_agree"  : round(model_agree, 1),
        "vader_agree"  : round(vader_agree, 1),
        "bert_agree"   : round(bert_agree, 1),
        "coherence"    : 0.4121,
    }


def render_kpi_row(kpis: dict) -> None:
    """Renders the top KPI metric cards."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Reviews",       f"{kpis['total']:,}")
    c2.metric("Positive Sentiment",  f"{kpis['pct_positive']}%",
              delta=f"BERT model")
    c3.metric("Avg Star Rating",     f"{kpis['avg_rating']}★")
    c4.metric("Topics Discovered",   str(kpis["num_topics"]))
    c5.metric("BERT vs Stars",       f"{kpis['bert_agree']}%",
              delta=f"VADER: {kpis['vader_agree']}%")
    c6.metric("LDA Coherence",       f"{kpis['coherence']}")


def filter_dataframe(
    df: pd.DataFrame,
    topics: list[str],
    sentiments: list[str],
    ratings: list[int],
    search_term: str,
) -> pd.DataFrame:
    """
    Applies sidebar filters to the master DataFrame for the Review Search page.
    Returns filtered DataFrame.
    """
    filtered = df.copy()

    if topics:
        filtered = filtered[filtered["topic_label"].isin(topics)]

    if sentiments:
        filtered = filtered[filtered["bert_label"].isin(sentiments)]

    if ratings:
        filtered = filtered[filtered["rating"].isin(ratings)]

    if search_term.strip():
        mask = filtered["review_text"].str.contains(
            search_term.strip(), case=False, na=False
        )
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)
