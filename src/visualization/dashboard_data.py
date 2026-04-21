"""
Phase 6 — Visualization: Dashboard Data Preparation
Loads the master parquet and produces all aggregated DataFrames
that the chart functions in charts.py consume.

All functions return clean pandas DataFrames — no Plotly logic here.
This separation makes the data layer independently testable.
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_PROCESSED
from src.utils.logger import get_logger

logger = get_logger(__name__)

MASTER_PATH = DATA_PROCESSED / "reviews_topics.parquet"


def load_master() -> pd.DataFrame:
    """Loads the Phase 5 master parquet. Raises if not found."""
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master parquet not found at {MASTER_PATH}. "
            "Run Phase 5 first: python src/modeling/topic_pipeline.py"
        )
    df = pd.read_parquet(MASTER_PATH)
    logger.info(f"Loaded master dataset: {len(df):,} rows × {df.shape[1]} columns")
    return df


def prep_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns counts for VADER and BERT sentiment labels.
    Used by: donut chart (chart_sentiment_donut).
    """
    vader = (
        df["vader_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"vader_label": "label", "count": "count"})
    )
    vader["model"] = "VADER"

    bert = (
        df["bert_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"bert_label": "label", "count": "count"})
    )
    bert["model"] = "DistilBERT"

    return pd.concat([vader, bert], ignore_index=True)


def prep_rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rating counts with sentiment majority label per star.
    Used by: bar chart (chart_rating_bar).
    """
    agg = (
        df.groupby("rating")
        .agg(
            count=("rating", "count"),
            avg_vader=("vader_score", "mean"),
            pct_positive=("bert_label", lambda x: (x == "positive").mean()),
        )
        .reset_index()
    )
    agg["sentiment"] = agg["pct_positive"].apply(
        lambda p: "positive" if p >= 0.6 else ("negative" if p <= 0.4 else "mixed")
    )
    return agg


def prep_sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns monthly sentiment counts for trend line chart.
    Requires review_date column. Drops rows with NaT dates.
    Used by: line chart (chart_sentiment_trend).
    """
    df_dated = df.dropna(subset=["review_date"]).copy()
    df_dated["review_date"] = df_dated["review_date"].dt.tz_localize(None) if df_dated["review_date"].dt.tz is not None else df_dated["review_date"]
    df_dated["month"] = df_dated["review_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df_dated.groupby(["month", "bert_label"])
        .size()
        .reset_index(name="count")
    )

    # Only keep months with at least 5 reviews to avoid noise
    month_totals = monthly.groupby("month")["count"].sum()
    valid_months = month_totals[month_totals >= 5].index
    monthly = monthly[monthly["month"].isin(valid_months)]

    return monthly.sort_values("month")


def prep_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns review count and avg rating per topic, sorted by count.
    Used by: horizontal bar chart (chart_topic_bar).
    """
    agg = (
        df.groupby(["dominant_topic", "topic_label"])
        .agg(
            count=("dominant_topic", "count"),
            avg_rating=("rating", "mean"),
            pct_positive=("bert_label", lambda x: (x == "positive").mean()),
            pct_negative=("bert_label", lambda x: (x == "negative").mean()),
        )
        .reset_index()
        .sort_values("count", ascending=True)
    )
    agg["avg_rating"] = agg["avg_rating"].round(2)
    agg["pct_positive"] = (agg["pct_positive"] * 100).round(1)
    agg["pct_negative"] = (agg["pct_negative"] * 100).round(1)
    return agg


def prep_topic_sentiment_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table: topics × sentiment labels → review counts.
    Used by: heatmap (chart_topic_sentiment_heatmap).
    """
    pivot = (
        df.groupby(["topic_label", "bert_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure both columns exist even if one sentiment is missing
    for col in ["positive", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total"] = pivot["positive"] + pivot["negative"]
    pivot["pct_negative"] = (pivot["negative"] / pivot["total"] * 100).round(1)
    pivot["pct_positive"] = (pivot["positive"] / pivot["total"] * 100).round(1)
    return pivot.sort_values("pct_negative", ascending=False)


def prep_avg_rating_per_topic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns average star rating per topic with confidence intervals.
    Used by: bar chart (chart_avg_rating_topic).
    """
    agg = (
        df.groupby("topic_label")["rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "std": "std_rating", "count": "n"})
    )
    # 95% CI: 1.96 * std / sqrt(n)
    agg["ci"] = (1.96 * agg["std_rating"] / np.sqrt(agg["n"])).round(3)
    agg["avg_rating"] = agg["avg_rating"].round(3)
    return agg.sort_values("avg_rating", ascending=False)


def prep_vader_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns raw vader_score column for histogram.
    Used by: histogram (chart_vader_histogram).
    """
    return df[["vader_score", "bert_label", "rating"]].copy()


def prep_topic_wordcloud_tokens(df: pd.DataFrame) -> dict[str, str]:
    """
    Returns a dict of {topic_label: space-joined token string} for word clouds.
    Handles cases where tokens column contains lists, numpy arrays, or strings.
    Used by: chart_wordclouds().
    """
    import ast

    result = {}
    for topic_label, group in df.groupby("topic_label"):
        all_tokens = []
        for token_list in group["tokens"]:
            # Handle all possible types after parquet round-trips
            if isinstance(token_list, list):
                all_tokens.extend([str(t) for t in token_list if t])
            elif isinstance(token_list, str) and token_list.strip():
                try:
                    parsed = ast.literal_eval(token_list)
                    if isinstance(parsed, list):
                        all_tokens.extend([str(t) for t in parsed if t])
                    else:
                        all_tokens.extend(token_list.split())
                except (ValueError, SyntaxError):
                    all_tokens.extend(token_list.split())
            elif hasattr(token_list, "__iter__"):
                all_tokens.extend([str(t) for t in token_list if t])

        # Fallback: use review_normalized text if tokens empty
        if not all_tokens and "review_normalized" in group.columns:
            for text in group["review_normalized"].dropna():
                if isinstance(text, str) and text.strip():
                    all_tokens.extend(text.split())

        result[topic_label] = " ".join(all_tokens)

    return result


if __name__ == "__main__":
    df = load_master()
    print(f"✓ load_master: {df.shape}")
    print(f"✓ sentiment_distribution:\n{prep_sentiment_distribution(df)}")
    print(f"✓ rating_distribution:\n{prep_rating_distribution(df)}")
    print(f"✓ topic_distribution:\n{prep_topic_distribution(df)[['topic_label','count','avg_rating']]}")
    print(f"✓ topic_sentiment_heatmap:\n{prep_topic_sentiment_heatmap(df)[['topic_label','positive','negative','pct_negative']]}")
    print(f"\n✅ dashboard_data.py OK")
