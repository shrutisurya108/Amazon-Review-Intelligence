"""
Phase 5 — Topic Modeling: Pipeline Orchestrator
Loads the trained LDA model, assigns each review its dominant topic,
merges with the sentiment parquet, and saves the final master dataset.

Run: python src/modeling/topic_pipeline.py
"""
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np
from gensim import corpora

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_PROCESSED, OUTPUTS_REPORTS, LDA_NUM_TOPICS
from src.utils.logger import get_logger
from src.modeling.lda_model import (
    load_tokens, build_corpus, run as train_run,
    load_artifacts, print_topics, TOPIC_LABELS,
)

logger = get_logger(__name__)

OUTPUT_PATH = DATA_PROCESSED / "reviews_topics.parquet"


def assign_topics(
    df: pd.DataFrame,
    token_lists: list,
    model,
    dictionary,
) -> pd.DataFrame:
    """
    For each review, infers the dominant topic and its probability score.
    Adds columns: dominant_topic, topic_label, topic_score, topic_keywords.
    Returns updated DataFrame.
    """
    logger.info(f"Assigning dominant topics to {len(df):,} reviews...")

    corpus = [dictionary.doc2bow(doc) for doc in token_lists]

    dominant_topics  = []
    topic_scores     = []
    topic_labels_col = []
    topic_keywords_col = []

    for bow in corpus:
        # Get topic distribution for this document
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)

        if not topic_dist:
            dominant_topics.append(0)
            topic_scores.append(0.0)
        else:
            # Find topic with highest probability
            best_topic, best_score = max(topic_dist, key=lambda x: x[1])
            dominant_topics.append(int(best_topic))
            topic_scores.append(round(float(best_score), 4))

    # Map topic IDs to labels and keywords
    for topic_id in dominant_topics:
        label    = TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")
        words    = model.show_topic(topic_id, topn=5)
        keywords = " ".join([w for w, _ in words])
        topic_labels_col.append(label)
        topic_keywords_col.append(keywords)

    df = df.copy()
    df["dominant_topic"]  = dominant_topics
    df["topic_label"]     = topic_labels_col
    df["topic_score"]     = topic_scores
    df["topic_keywords"]  = topic_keywords_col

    dist = df["dominant_topic"].value_counts().sort_index().to_dict()
    logger.info(f"Topic distribution: {dist}")
    logger.info("Topic assignment complete.")

    return df


def write_topic_report(df: pd.DataFrame, model, coherence: float) -> Path:
    """
    Writes a detailed topic analysis report.
    Shows top keywords, review counts, and example reviews per topic.
    """
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_REPORTS / "topic_report.txt"

    lines = [
        "=" * 65,
        "  LDA TOPIC MODELING REPORT",
        "  Amazon Electronics Reviews — NLP Pipeline",
        "=" * 65,
        "",
        f"  Total reviews analyzed : {len(df):,}",
        f"  Number of topics       : {LDA_NUM_TOPICS}",
        f"  Coherence score (c_v)  : {coherence:.4f}",
        "  (c_v range: 0–1, good models score 0.4–0.7)",
        "",
    ]

    for topic_id in range(LDA_NUM_TOPICS):
        label   = TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")
        words   = model.show_topic(topic_id, topn=12)
        kws     = " | ".join([f"{w}({s:.3f})" for w, s in words])
        subset  = df[df["dominant_topic"] == topic_id]
        count   = len(subset)
        pct     = count / len(df) * 100

        # Rating distribution within this topic
        rating_dist = subset["rating"].value_counts().sort_index().to_dict()

        # Top 3 example reviews for this topic
        top_reviews = (
            subset.nlargest(3, "topic_score")[["review_text", "rating", "topic_score"]]
            if len(subset) >= 3
            else subset[["review_text", "rating", "topic_score"]]
        )

        lines += [
            "-" * 65,
            f"  Topic {topic_id:02d} — {label.upper()}",
            f"  Reviews : {count:,} ({pct:.1f}% of corpus)",
            f"  Ratings : {rating_dist}",
            f"  Keywords: {kws}",
            "",
            "  Top example reviews:",
        ]

        for _, row in top_reviews.iterrows():
            preview = str(row["review_text"])[:120].replace("\n", " ")
            lines.append(f"    [{int(row['rating'])}★ | score={row['topic_score']:.3f}] {preview}...")

        lines.append("")

    lines.append("=" * 65)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Topic report saved: {report_path}")
    return report_path


def print_summary(df: pd.DataFrame, coherence: float, elapsed: float) -> None:
    """Prints topic assignment summary to terminal."""
    print("\n" + "=" * 60)
    print("  TOPIC MODELING SUMMARY")
    print("=" * 60)
    print(f"  Reviews analyzed   : {len(df):,}")
    print(f"  Topics discovered  : {LDA_NUM_TOPICS}")
    print(f"  Coherence (c_v)    : {coherence:.4f}")
    print(f"  Processing time    : {elapsed:.1f}s")
    print(f"\n  Topic distribution:")

    topic_counts = df["dominant_topic"].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        label = TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")
        bar   = "█" * (count * 25 // len(df))
        pct   = count / len(df) * 100
        print(f"    {topic_id:02d} [{label:<22}] {count:>5,} ({pct:4.1f}%) {bar}")

    print(f"\n  Sentiment × Topic breakdown (avg rating per topic):")
    avg_ratings = df.groupby("dominant_topic")["rating"].mean()
    for topic_id, avg in avg_ratings.items():
        label = TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")
        stars = "★" * round(avg)
        print(f"    {topic_id:02d} [{label:<22}] avg {avg:.2f} {stars}")

    print("=" * 60 + "\n")


def save_master(df: pd.DataFrame) -> Path:
    """Saves the final master dataset with all features."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Master dataset saved: {OUTPUT_PATH.name} ({size_mb:.2f} MB)")
    return OUTPUT_PATH


def run() -> pd.DataFrame:
    """Full topic pipeline. Returns master DataFrame."""
    logger.info("=" * 60)
    logger.info("Phase 5 — Topic Pipeline START")
    logger.info("=" * 60)

    start = time.time()

    # Check if model already trained — skip retraining if artifacts exist
    from src.modeling.lda_model import MODEL_PATH, DICT_PATH

    if MODEL_PATH.exists() and DICT_PATH.exists():
        logger.info("LDA artifacts found — loading from disk (skipping retraining).")
        model, dictionary   = load_artifacts()
        df, token_lists     = load_tokens()
        # Recompute coherence from saved model
        from src.modeling.lda_model import compute_coherence
        coherence = compute_coherence(model, token_lists, dictionary)
        print_topics(model)
    else:
        logger.info("No saved artifacts found — training LDA from scratch.")
        model, dictionary, token_lists, df, coherence = train_run()

    df = assign_topics(df, token_lists, model, dictionary)

    elapsed = time.time() - start

    save_master(df)
    write_topic_report(df, model, coherence)
    print_summary(df, coherence, elapsed)

    logger.info("Phase 5 — Topic Pipeline COMPLETE")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run()
    print(f"✅ Master dataset ready. Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Saved : {OUTPUT_PATH}")
