"""
Phase 6 — Visualization: Figure Exporter
Renders all 8 charts and saves them as standalone interactive HTML files
to outputs/figures/. These files work in any browser with no server needed.

Run: python src/visualization/export_figures.py
"""
from pathlib import Path
import sys
import base64

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    chart_all_wordclouds,
)

logger = get_logger(__name__)


def save_figure(fig: go.Figure, filename: str) -> Path:
    """Saves a Plotly figure as a standalone HTML file."""
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / filename
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",   # loads Plotly from CDN — keeps file small
        full_html=True,
    )
    size_kb = path.stat().st_size / 1024
    logger.info(f"Saved: {filename} ({size_kb:.1f} KB)")
    return path


def save_wordcloud_html(wc_dict: dict, filename: str = "wordclouds.html") -> Path:
    """Saves all word clouds as a single HTML page with a grid layout."""
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / filename

    cards = ""
    for topic_label, b64 in wc_dict.items():
        if not b64:
            continue
        label_title = topic_label.replace("_", " ").title()
        cards += f"""
        <div style="background:#1a1d2e;border-radius:8px;padding:12px;margin:8px;">
            <img src="data:image/png;base64,{b64}"
                 style="width:100%;border-radius:4px;"
                 alt="{label_title}">
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Topic Word Clouds — Amazon Electronics Reviews</title>
  <style>
    body {{ background:#0f1117; color:white; font-family:Inter,Arial,sans-serif;
            padding:24px; margin:0; }}
    h1   {{ font-size:20px; color:#a0aec0; margin-bottom:20px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
    @media(max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <h1>Topic Word Clouds — Amazon Electronics Reviews</h1>
  <div class="grid">{cards}</div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    logger.info(f"Saved: {filename} ({size_kb:.1f} KB)")
    return path


def run() -> None:
    """Exports all charts to outputs/figures/."""
    logger.info("=" * 60)
    logger.info("Phase 6 — Figure Export START")
    logger.info("=" * 60)

    df = load_master()

    charts_to_export = [
        ("01_sentiment_donut.html",    chart_sentiment_donut,          prep_sentiment_distribution),
        ("02_rating_bar.html",         chart_rating_bar,               prep_rating_distribution),
        ("03_sentiment_trend.html",    chart_sentiment_trend,          prep_sentiment_over_time),
        ("04_topic_bar.html",          chart_topic_bar,                prep_topic_distribution),
        ("05_topic_heatmap.html",      chart_topic_sentiment_heatmap,  prep_topic_sentiment_heatmap),
        ("06_avg_rating_topic.html",   chart_avg_rating_topic,         prep_avg_rating_per_topic),
        ("07_vader_histogram.html",    chart_vader_histogram,          prep_vader_score_distribution),
    ]

    saved = []
    for filename, chart_fn, data_fn in charts_to_export:
        logger.info(f"Generating {filename}...")
        data = data_fn(df)
        fig  = chart_fn(data)
        path = save_figure(fig, filename)
        saved.append(path)

    # Word clouds
    logger.info("Generating word clouds...")
    wc_tokens = prep_topic_wordcloud_tokens(df)
    wc_dict   = chart_all_wordclouds(wc_tokens)
    wc_path   = save_wordcloud_html(wc_dict)
    saved.append(wc_path)

    print("\n" + "=" * 55)
    print("  FIGURES EXPORTED")
    print("=" * 55)
    for p in saved:
        size_kb = p.stat().st_size / 1024
        print(f"  ✓ {p.name:<35} {size_kb:>7.1f} KB")
    print(f"\n  All saved to: {OUTPUTS_FIGURES}")
    print("=" * 55)

    logger.info("Phase 6 — Figure Export COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
    print("\n✅ All figures exported successfully.")
