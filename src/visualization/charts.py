"""
Phase 6 — Visualization: Plotly Chart Functions
Each function takes a prepared DataFrame and returns a Plotly figure.
All charts use a consistent dark theme and color palette.

Import pattern in Streamlit:
    from src.visualization.charts import chart_sentiment_donut, ...
"""
from pathlib import Path
import sys
import io
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for server/Streamlit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Design tokens — consistent across all charts ───────────────────────────────
COLORS = {
    "positive"  : "#2ecc71",   # green
    "negative"  : "#e74c3c",   # red
    "neutral"   : "#95a5a6",   # grey
    "mixed"     : "#f39c12",   # amber
    "primary"   : "#3498db",   # blue
    "background": "#0f1117",   # dark bg
    "surface"   : "#1a1d2e",   # card bg
    "text"      : "#ffffff",
    "subtext"   : "#a0aec0",
}

TOPIC_COLORS = px.colors.qualitative.Set2  # 8 distinct colors for 8 topics

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["background"],
    plot_bgcolor =COLORS["surface"],
    font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
    margin=dict(l=40, r=40, t=60, b=40),
)


def _apply_base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Applies consistent layout to every figure."""
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=title,
            font=dict(size=16, color=COLORS["text"]),
            x=0.03,
        ),
        height=height,
    )
    return fig


# ── Chart 1 — Sentiment Distribution Donut ────────────────────────────────────

def chart_sentiment_donut(df_sentiment: pd.DataFrame) -> go.Figure:
    """
    Side-by-side donut charts: VADER labels vs DistilBERT labels.
    Input: output of prep_sentiment_distribution().
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["VADER (Rule-based)", "DistilBERT (Transformer)"],
    )

    color_map = {
        "positive": COLORS["positive"],
        "negative": COLORS["negative"],
        "neutral" : COLORS["neutral"],
    }

    for col_idx, model in enumerate(["VADER", "DistilBERT"], start=1):
        subset = df_sentiment[df_sentiment["model"] == model]
        colors = [color_map.get(l, COLORS["primary"]) for l in subset["label"]]
        fig.add_trace(
            go.Pie(
                labels=subset["label"],
                values=subset["count"],
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=COLORS["background"], width=2)),
                textinfo="label+percent",
                textfont=dict(size=12),
                name=model,
            ),
            row=1, col=col_idx,
        )

    fig = _apply_base(fig, "Sentiment Distribution — VADER vs DistilBERT", height=400)
    fig.update_layout(showlegend=False)
    return fig


# ── Chart 2 — Rating Distribution Bar ─────────────────────────────────────────

def chart_rating_bar(df_ratings: pd.DataFrame) -> go.Figure:
    """
    Bar chart of review counts per star rating, colored by sentiment majority.
    Input: output of prep_rating_distribution().
    """
    sentiment_colors = {
        "positive": COLORS["positive"],
        "negative": COLORS["negative"],
        "mixed"   : COLORS["mixed"],
    }
    bar_colors = [sentiment_colors.get(s, COLORS["primary"]) for s in df_ratings["sentiment"]]

    fig = go.Figure(
        go.Bar(
            x=[f"{r}★" for r in df_ratings["rating"]],
            y=df_ratings["count"],
            marker=dict(color=bar_colors, line=dict(color=COLORS["background"], width=1)),
            text=df_ratings["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Reviews: %{y:,}<br>"
                "Avg VADER: %{customdata[0]:.2f}<br>"
                "% Positive (BERT): %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
            customdata=df_ratings[["avg_vader", "pct_positive"]].values,
        )
    )

    fig = _apply_base(fig, "Review Count by Star Rating")
    fig.update_xaxes(
        title_text="Star Rating",
        gridcolor=COLORS["surface"],
        color=COLORS["subtext"],
    )
    fig.update_yaxes(
        title_text="Number of Reviews",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    return fig


# ── Chart 3 — Sentiment Trend Over Time ───────────────────────────────────────

def chart_sentiment_trend(df_monthly: pd.DataFrame) -> go.Figure:
    """
    Line chart of monthly positive/negative review counts.
    Input: output of prep_sentiment_over_time().
    """
    fig = go.Figure()

    label_styles = {
        "positive": dict(color=COLORS["positive"], dash="solid"),
        "negative": dict(color=COLORS["negative"], dash="dot"),
    }

    for label, style in label_styles.items():
        subset = df_monthly[df_monthly["bert_label"] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["month"],
                y=subset["count"],
                mode="lines+markers",
                name=label.capitalize(),
                line=dict(color=style["color"], dash=style["dash"], width=2),
                marker=dict(size=5),
                hovertemplate="%{x|%b %Y}: %{y:,} reviews<extra></extra>",
            )
        )

    fig = _apply_base(fig, "Sentiment Trend Over Time (Monthly)", height=400)
    fig.update_xaxes(
        title_text="Month",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    fig.update_yaxes(
        title_text="Number of Reviews",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    fig.update_layout(
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor="#2d3250",
            font=dict(color=COLORS["text"]),
        )
    )
    return fig


# ── Chart 4 — Topic Distribution Horizontal Bar ───────────────────────────────

def chart_topic_bar(df_topics: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of review counts per topic.
    Input: output of prep_topic_distribution().
    """
    colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(df_topics))]

    fig = go.Figure(
        go.Bar(
            y=df_topics["topic_label"].str.replace("_", " ").str.title(),
            x=df_topics["count"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=COLORS["background"], width=1)),
            text=df_topics["count"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Reviews: %{x:,}<br>"
                "Avg Rating: %{customdata[0]:.2f}★<br>"
                "% Positive: %{customdata[1]:.1f}%<br>"
                "% Negative: %{customdata[2]:.1f}%"
                "<extra></extra>"
            ),
            customdata=df_topics[["avg_rating", "pct_positive", "pct_negative"]].values,
        )
    )

    fig = _apply_base(fig, "Review Volume by Topic", height=460)
    fig.update_xaxes(
        title_text="Number of Reviews",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    fig.update_yaxes(color=COLORS["subtext"])
    return fig


# ── Chart 5 — Topic × Sentiment Heatmap ──────────────────────────────────────

def chart_topic_sentiment_heatmap(df_heatmap: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing % negative reviews per topic — surfaces complaint hotspots.
    Input: output of prep_topic_sentiment_heatmap().
    """
    topic_labels = df_heatmap["topic_label"].str.replace("_", " ").str.title().tolist()

    fig = go.Figure(
        go.Heatmap(
            z=[[row] for row in df_heatmap["pct_negative"]],
            y=topic_labels,
            x=["% Negative Reviews"],
            colorscale=[
                [0.0,  COLORS["positive"]],
                [0.5,  COLORS["mixed"]],
                [1.0,  COLORS["negative"]],
            ],
            text=[[f"{v:.1f}%"] for v in df_heatmap["pct_negative"]],
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            showscale=True,
            colorbar=dict(
                title="% Negative",
                tickfont=dict(color=COLORS["subtext"]),
                titlefont=dict(color=COLORS["subtext"]),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Negative: %{z:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig = _apply_base(fig, "Complaint Hotspots — % Negative Reviews per Topic", height=460)
    fig.update_xaxes(color=COLORS["subtext"])
    fig.update_yaxes(color=COLORS["subtext"])
    return fig


# ── Chart 6 — Average Rating per Topic ───────────────────────────────────────

def chart_avg_rating_topic(df_avg: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of avg star rating per topic with error bars (95% CI).
    Input: output of prep_avg_rating_per_topic().
    """
    colors = [
        COLORS["positive"] if r >= 4.0
        else COLORS["mixed"] if r >= 3.0
        else COLORS["negative"]
        for r in df_avg["avg_rating"]
    ]

    fig = go.Figure(
        go.Bar(
            y=df_avg["topic_label"].str.replace("_", " ").str.title(),
            x=df_avg["avg_rating"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=COLORS["background"], width=1)),
            error_x=dict(
                type="data",
                array=df_avg["ci"].tolist(),
                color=COLORS["subtext"],
                thickness=1.5,
                width=4,
            ),
            text=df_avg["avg_rating"].apply(lambda x: f"{x:.2f}★"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Avg Rating: %{x:.3f}★<br>"
                "Reviews: %{customdata[0]:,}<br>"
                "95% CI: ±%{customdata[1]:.3f}"
                "<extra></extra>"
            ),
            customdata=df_avg[["n", "ci"]].values,
        )
    )

    fig = _apply_base(fig, "Average Star Rating by Topic (with 95% CI)", height=460)
    fig.update_xaxes(
        title_text="Average Rating",
        range=[0, 6],
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    fig.update_yaxes(color=COLORS["subtext"])
    return fig


# ── Chart 7 — VADER Score Histogram ──────────────────────────────────────────

def chart_vader_histogram(df_scores: pd.DataFrame) -> go.Figure:
    """
    Histogram of VADER compound scores, colored by BERT label.
    Input: output of prep_vader_score_distribution().
    """
    fig = go.Figure()

    label_colors = {
        "positive": COLORS["positive"],
        "negative": COLORS["negative"],
    }

    for label, color in label_colors.items():
        subset = df_scores[df_scores["bert_label"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset["vader_score"],
                name=label.capitalize(),
                marker=dict(color=color, opacity=0.75),
                nbinsx=50,
                hovertemplate=f"{label.capitalize()}<br>VADER Score: %{{x:.2f}}<br>Count: %{{y:,}}<extra></extra>",
            )
        )

    fig = _apply_base(fig, "VADER Sentiment Score Distribution", height=400)
    fig.update_layout(
        barmode="overlay",
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor="#2d3250",
            font=dict(color=COLORS["text"]),
        ),
    )
    fig.update_xaxes(
        title_text="VADER Compound Score (-1 = most negative, +1 = most positive)",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    fig.update_yaxes(
        title_text="Number of Reviews",
        gridcolor="#2d3250",
        color=COLORS["subtext"],
    )
    return fig


# ── Chart 8 — Word Clouds per Topic ──────────────────────────────────────────

def chart_wordcloud(topic_label: str, token_string: str) -> str:
    """
    Generates a word cloud for a single topic.
    Returns base64-encoded PNG string for embedding in Streamlit/HTML.
    Input: topic_label string, space-joined token string.
    """
    if not token_string.strip():
        return ""

    wc = WordCloud(
        width=700,
        height=350,
        background_color="#1a1d2e",
        colormap="Blues",
        max_words=80,
        prefer_horizontal=0.85,
        collocations=False,
    ).generate(token_string)

    fig_mpl, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a1d2e")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        topic_label.replace("_", " ").title(),
        color="white",
        fontsize=13,
        pad=8,
    )

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1d2e")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig_mpl)

    return img_b64


def chart_all_wordclouds(token_dict: dict[str, str]) -> dict[str, str]:
    """
    Generates word clouds for all topics.
    Returns dict of {topic_label: base64_png_string}.
    """
    result = {}
    for topic_label, token_string in token_dict.items():
        logger.info(f"Generating word cloud: {topic_label}")
        result[topic_label] = chart_wordcloud(topic_label, token_string)
    return result


if __name__ == "__main__":
    from src.visualization.dashboard_data import (
        load_master, prep_sentiment_distribution, prep_rating_distribution,
        prep_sentiment_over_time, prep_topic_distribution,
        prep_topic_sentiment_heatmap, prep_avg_rating_per_topic,
        prep_vader_score_distribution, prep_topic_wordcloud_tokens,
    )

    df = load_master()
    print("Testing all chart functions on master dataset...\n")

    fig1 = chart_sentiment_donut(prep_sentiment_distribution(df))
    print(f"✓ chart_sentiment_donut     : {len(fig1.data)} traces")

    fig2 = chart_rating_bar(prep_rating_distribution(df))
    print(f"✓ chart_rating_bar          : {len(fig2.data)} traces")

    fig3 = chart_sentiment_trend(prep_sentiment_over_time(df))
    print(f"✓ chart_sentiment_trend     : {len(fig3.data)} traces")

    fig4 = chart_topic_bar(prep_topic_distribution(df))
    print(f"✓ chart_topic_bar           : {len(fig4.data)} traces")

    fig5 = chart_topic_sentiment_heatmap(prep_topic_sentiment_heatmap(df))
    print(f"✓ chart_topic_sentiment_heatmap : {len(fig5.data)} traces")

    fig6 = chart_avg_rating_topic(prep_avg_rating_per_topic(df))
    print(f"✓ chart_avg_rating_topic    : {len(fig6.data)} traces")

    fig7 = chart_vader_histogram(prep_vader_score_distribution(df))
    print(f"✓ chart_vader_histogram     : {len(fig7.data)} traces")

    wc_data = prep_topic_wordcloud_tokens(df)
    wc_b64  = chart_wordcloud(
        list(wc_data.keys())[0],
        list(wc_data.values())[0],
    )
    print(f"✓ chart_wordcloud           : {len(wc_b64)} chars (base64 PNG)")

    print("\n✅ All chart functions OK.")
