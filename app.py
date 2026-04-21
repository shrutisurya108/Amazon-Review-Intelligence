"""
Phase 7 — Streamlit App: Main Entry Point
Amazon Electronics Review Intelligence Dashboard

Run locally:
    streamlit run app.py

Pages:
    Overview         — KPI cards, sentiment donut, rating distribution
    Sentiment        — Trend over time, VADER histogram, model comparison
    Topic Explorer   — Topic bar, heatmap, avg rating, word clouds
    Review Search    — Filter and search individual reviews
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────────────────────
from src.streamlit_utils import (
    get_master, get_all_prep_data, get_wordclouds,
    compute_kpis, render_kpi_row, filter_dataframe,
)
from src.visualization.charts import (
    chart_sentiment_donut,
    chart_rating_bar,
    chart_sentiment_trend,
    chart_topic_bar,
    chart_topic_sentiment_heatmap,
    chart_avg_rating_topic,
    chart_vader_histogram,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #1a1d2e; }

  /* Metric cards */
  [data-testid="stMetric"] {
      background-color: #1a1d2e;
      border: 1px solid #2d3250;
      border-radius: 8px;
      padding: 16px 12px;
  }
  [data-testid="stMetricLabel"]  { color: #a0aec0 !important; font-size: 12px; }
  [data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 22px; }
  [data-testid="stMetricDelta"]  { color: #3498db !important; font-size: 11px; }

  /* Section headers */
  .section-header {
      color: #a0aec0;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin: 24px 0 8px 0;
      padding-bottom: 6px;
      border-bottom: 1px solid #2d3250;
  }

  /* Info box */
  .insight-box {
      background: #1a1d2e;
      border-left: 3px solid #3498db;
      border-radius: 4px;
      padding: 10px 14px;
      margin: 8px 0;
      font-size: 13px;
      color: #a0aec0;
  }

  /* Word cloud image */
  .wc-card {
      background: #1a1d2e;
      border-radius: 8px;
      padding: 8px;
      margin-bottom: 12px;
  }

  /* Hide Streamlit default footer */
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
df   = get_master()
prep = get_all_prep_data(df)
kpis = compute_kpis(df)


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Amazon Review\nIntelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=["Overview", "Sentiment Analysis", "Topic Explorer", "Review Search"],
        format_func=lambda p: {
            "Overview"           : "📊  Overview",
            "Sentiment Analysis" : "📈  Sentiment Analysis",
            "Topic Explorer"     : "🏷️   Topic Explorer",
            "Review Search"      : "🔍  Review Search",
        }[p],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<div style='color:#a0aec0;font-size:12px;'>"
        f"<b>{kpis['total']:,}</b> reviews<br>"
        f"<b>Electronics</b> category<br>"
        f"2009 – 2019<br><br>"
        f"Pipeline: spaCy · VADER<br>DistilBERT · LDA"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Amazon Electronics Review Intelligence")
    st.markdown(
        "<div style='color:#a0aec0;margin-bottom:24px;'>"
        "NLP pipeline processing 21,737 Amazon Electronics reviews through "
        "sentiment modeling (VADER + DistilBERT) and LDA topic modeling."
        "</div>",
        unsafe_allow_html=True,
    )

    # KPI row
    render_kpi_row(kpis)

    st.markdown("<div class='section-header'>Sentiment Distribution</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_sentiment_donut(prep["sentiment_dist"]),
        use_container_width=True,
    )

    st.markdown("<div class='section-header'>Rating Distribution</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_rating_bar(prep["rating_dist"]),
        use_container_width=True,
    )

    st.markdown("<div class='insight-box'>💡 <b>Key Finding:</b> "
                "78% of reviews are 5★ or 4★, yet DistilBERT classifies "
                "~28% as negative — indicating mixed or nuanced language "
                "even in high-star reviews. VADER achieved 85.7% agreement "
                "with star ratings vs DistilBERT's 79.3%, showing rule-based "
                "models can outperform transformers on domain-specific text."
                "</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.markdown(
        "<div style='color:#a0aec0;margin-bottom:24px;'>"
        "Two-stage sentiment pipeline: VADER rule-based baseline vs "
        "DistilBERT transformer inference, evaluated against ground-truth star ratings."
        "</div>",
        unsafe_allow_html=True,
    )

    # Model comparison cards
    c1, c2, c3 = st.columns(3)
    c1.metric("VADER Accuracy",    f"{kpis['vader_agree']}%",
              delta="vs star ratings")
    c2.metric("DistilBERT Accuracy", f"{kpis['bert_agree']}%",
              delta="vs star ratings (excl. 3★)")
    c3.metric("Model Agreement",   f"{kpis['model_agree']}%",
              delta="VADER vs BERT labels")

    st.markdown("<div class='section-header'>Sentiment Trend Over Time</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_sentiment_trend(prep["monthly"]),
        use_container_width=True,
    )

    st.markdown("<div class='section-header'>VADER Score Distribution</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_vader_histogram(prep["vader_scores"]),
        use_container_width=True,
    )

    st.markdown("<div class='section-header'>Model Comparison</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_sentiment_donut(prep["sentiment_dist"]),
        use_container_width=True,
    )

    st.markdown("<div class='insight-box'>💡 <b>Modeling Insight:</b> "
                "VADER outperforms DistilBERT on this corpus because Amazon "
                "reviews contain direct sentiment language (\"great\", \"terrible\") "
                "that VADER's lexicon handles well. DistilBERT was fine-tuned on "
                "movie reviews (SST-2), a different domain — demonstrating the "
                "importance of domain-matched training data for transformer models."
                "</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TOPIC EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Topic Explorer":
    st.title("Topic Explorer")
    st.markdown(
        "<div style='color:#a0aec0;margin-bottom:24px;'>"
        f"LDA topic modeling discovered <b>8 recurring themes</b> across 21,737 reviews "
        f"with a coherence score of <b>0.4121</b> (good range: 0.4–0.7)."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Review Volume by Topic</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        chart_topic_bar(prep["topic_dist"]),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Complaint Hotspots</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            chart_topic_sentiment_heatmap(prep["topic_heatmap"]),
            use_container_width=True,
        )

    with col2:
        st.markdown("<div class='section-header'>Avg Rating by Topic</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            chart_avg_rating_topic(prep["avg_rating"]),
            use_container_width=True,
        )

    st.markdown("<div class='section-header'>Topic Word Clouds</div>",
                unsafe_allow_html=True)

    # Topic selector
    all_topics     = sorted(df["topic_label"].unique().tolist())
    selected_topic = st.selectbox(
        "Select a topic to view its word cloud:",
        options=all_topics,
        format_func=lambda t: t.replace("_", " ").title(),
    )

    with st.spinner(f"Generating word cloud for {selected_topic}..."):
        wc_dict = get_wordclouds(prep["wc_tokens"])

    if selected_topic in wc_dict and wc_dict[selected_topic]:
        st.markdown(
            f"<div class='wc-card'>"
            f"<img src='data:image/png;base64,{wc_dict[selected_topic]}' "
            f"style='width:100%;border-radius:4px;'>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show top keywords for selected topic
        topic_row = prep["topic_dist"][
            prep["topic_dist"]["topic_label"] == selected_topic
        ]
        if not topic_row.empty:
            row = topic_row.iloc[0]
            st.markdown(
                f"<div class='insight-box'>"
                f"<b>{selected_topic.replace('_', ' ').title()}</b> — "
                f"{int(row['count']):,} reviews · "
                f"Avg rating: {row['avg_rating']:.2f}★ · "
                f"{row['pct_positive']:.1f}% positive · "
                f"{row['pct_negative']:.1f}% negative"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning(f"Word cloud not available for {selected_topic}")

    st.markdown("<div class='insight-box'>💡 <b>Topic Insight:</b> "
                "'Battery Value' dominates at 34.7% of reviews — the most "
                "common concern for Electronics buyers. 'Gift Recommendations' "
                "at 23.6% shows Amazon Electronics is heavily purchased as gifts. "
                "'Kindle E-readers' scores the highest avg rating (4.70★) while "
                "'Customer Service Issues' is the smallest but most negative cluster."
                "</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REVIEW SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Review Search":
    st.title("Review Search")
    st.markdown(
        "<div style='color:#a0aec0;margin-bottom:24px;'>"
        "Filter and search individual reviews by topic, sentiment, rating, or keyword."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            all_topics = sorted(df["topic_label"].unique().tolist())
            sel_topics = st.multiselect(
                "Topic",
                options=all_topics,
                default=[],
                format_func=lambda t: t.replace("_", " ").title(),
            )

        with fc2:
            sel_sentiments = st.multiselect(
                "Sentiment (BERT)",
                options=["positive", "negative"],
                default=[],
            )

        with fc3:
            sel_ratings = st.multiselect(
                "Star Rating",
                options=[1, 2, 3, 4, 5],
                default=[],
                format_func=lambda r: f"{r}★",
            )

        with fc4:
            search_term = st.text_input(
                "Keyword search",
                placeholder="e.g. battery, screen, broken...",
            )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered_df = filter_dataframe(df, sel_topics, sel_sentiments, sel_ratings, search_term)

    # ── Results summary ───────────────────────────────────────────────────────
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Matching Reviews", f"{len(filtered_df):,}")
    if len(filtered_df):
        rc2.metric("Avg Rating",  f"{filtered_df['rating'].mean():.2f}★")
        rc3.metric("% Positive",
                   f"{(filtered_df['bert_label'] == 'positive').mean() * 100:.1f}%")
        rc4.metric("Avg VADER Score",
                   f"{filtered_df['vader_score'].mean():.3f}")
    else:
        rc2.metric("Avg Rating", "—")
        rc3.metric("% Positive", "—")
        rc4.metric("Avg VADER Score", "—")

    st.markdown("<div class='section-header'>Reviews</div>",
                unsafe_allow_html=True)

    if len(filtered_df) == 0:
        st.info("No reviews match the current filters. Try adjusting your selection.")
    else:
        # Display columns
        display_cols = [
            "rating", "bert_label", "vader_score",
            "topic_label", "product_name", "review_text",
        ]
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        # Rename for readability
        rename_map = {
            "rating"      : "⭐ Rating",
            "bert_label"  : "Sentiment",
            "vader_score" : "VADER",
            "topic_label" : "Topic",
            "product_name": "Product",
            "review_text" : "Review",
        }

        show_df = (
            filtered_df[display_cols]
            .rename(columns=rename_map)
            .head(500)   # cap at 500 for performance
        )

        st.dataframe(
            show_df,
            use_container_width=True,
            height=500,
            column_config={
                "⭐ Rating"  : st.column_config.NumberColumn(format="%d ★"),
                "VADER"      : st.column_config.NumberColumn(format="%.3f"),
                "Sentiment"  : st.column_config.TextColumn(),
                "Topic"      : st.column_config.TextColumn(),
                "Product"    : st.column_config.TextColumn(),
                "Review"     : st.column_config.TextColumn(),
            },
        )

        if len(filtered_df) > 500:
            st.caption(f"Showing 500 of {len(filtered_df):,} matching reviews.")

    st.markdown(
        "<div class='insight-box'>💡 Try filtering by topic 'customer_service_issues' "
        "with sentiment 'negative' to surface the most actionable complaints, "
        "or search 'battery' with 1★ to find common failure patterns.</div>",
        unsafe_allow_html=True,
    )
