"""
Phase 4 — Sentiment Modeling: Pipeline Orchestrator
Runs both VADER and DistilBERT, merges results, computes agreement,
saves reviews_sentiment.parquet, and writes a sentiment_report.txt.

Run: python src/modeling/sentiment_pipeline.py
"""
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_PROCESSED, OUTPUTS_REPORTS
from src.utils.logger import get_logger
from src.modeling.vader_sentiment import run_vader, vader_vs_stars
from src.modeling.distilbert_sentiment import run_distilbert, bert_vs_stars

logger = get_logger(__name__)


def load_processed() -> pd.DataFrame:
    """Loads the Phase 3 processed parquet."""
    path = DATA_PROCESSED / "reviews_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found at {path}. "
            "Run Phase 3 first: python src/preprocessing/pipeline.py"
        )
    df = pd.read_parquet(path)
    logger.info(f"Loaded processed parquet: {len(df):,} rows × {df.shape[1]} columns")
    return df


def add_star_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds star_sentiment column as ground truth for agreement comparison.
    4–5★ = positive, 1–2★ = negative, 3★ = neutral.
    """
    def label(r):
        if r >= 4:   return "positive"
        elif r <= 2: return "negative"
        else:        return "neutral"

    df = df.copy()
    df["star_sentiment"] = df["rating"].apply(label)
    return df


def add_agreement_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment_agree column: True if VADER and BERT labels match.
    Note: BERT only outputs positive/negative — neutral VADER rows are excluded.
    """
    df = df.copy()
    df["sentiment_agree"] = df["vader_label"] == df["bert_label"]
    return df


def save_sentiment(df: pd.DataFrame) -> Path:
    """Saves final sentiment-enriched DataFrame."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "reviews_sentiment.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved sentiment parquet: {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def write_report(df: pd.DataFrame, vader_agree: float, bert_agree: float, elapsed: float) -> Path:
    """Writes a plain-text sentiment analysis report."""
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_REPORTS / "sentiment_report.txt"

    vader_dist = df["vader_label"].value_counts().to_dict()
    bert_dist  = df["bert_label"].value_counts().to_dict()
    model_agree = df["sentiment_agree"].mean()
    avg_conf    = df["bert_confidence"].mean()

    lines = [
        "=" * 60,
        "  SENTIMENT ANALYSIS REPORT",
        "  Amazon Electronics Reviews — NLP Pipeline",
        "=" * 60,
        "",
        f"  Total reviews analyzed   : {len(df):,}",
        f"  Processing time          : {elapsed:.1f}s",
        "",
        "  VADER (Baseline) Results",
        "  ------------------------",
        f"  Positive  : {vader_dist.get('positive', 0):>6,}  ({vader_dist.get('positive', 0)/len(df):.1%})",
        f"  Neutral   : {vader_dist.get('neutral',  0):>6,}  ({vader_dist.get('neutral',  0)/len(df):.1%})",
        f"  Negative  : {vader_dist.get('negative', 0):>6,}  ({vader_dist.get('negative', 0)/len(df):.1%})",
        f"  Agreement with star ratings : {vader_agree:.1%}",
        "",
        "  DistilBERT Results",
        "  ------------------",
        f"  Positive  : {bert_dist.get('positive', 0):>6,}  ({bert_dist.get('positive', 0)/len(df):.1%})",
        f"  Negative  : {bert_dist.get('negative', 0):>6,}  ({bert_dist.get('negative', 0)/len(df):.1%})",
        f"  Avg confidence            : {avg_conf:.3f}",
        f"  Agreement with star ratings : {bert_agree:.1%}",
        "",
        "  Model Comparison",
        "  ----------------",
        f"  VADER vs DistilBERT agreement : {model_agree:.1%}",
        f"  DistilBERT improvement over VADER: +{(bert_agree - vader_agree):.1%}",
        "",
        "=" * 60,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Sentiment report saved: {report_path}")
    return report_path


def print_summary(df: pd.DataFrame, vader_agree: float, bert_agree: float) -> None:
    """Prints report to terminal."""
    print("\n" + "=" * 55)
    print("  SENTIMENT ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Reviews analyzed         : {len(df):,}")
    print(f"  VADER accuracy (vs stars): {vader_agree:.1%}")
    print(f"  BERT  accuracy (vs stars): {bert_agree:.1%}")
    print(f"  BERT improvement         : +{(bert_agree - vader_agree):.1%}")
    print(f"  Model agreement          : {df['sentiment_agree'].mean():.1%}")
    print(f"\n  Sample results:")
    cols = ["rating", "vader_label", "bert_label", "bert_confidence", "sentiment_agree"]
    print(df[cols].sample(5, random_state=42).to_string(index=False))
    print("=" * 55 + "\n")


def run() -> pd.DataFrame:
    """Full sentiment pipeline. Returns enriched DataFrame."""
    logger.info("=" * 60)
    logger.info("Phase 4 — Sentiment Pipeline START")
    logger.info("=" * 60)

    start = time.time()

    df = load_processed()
    df = add_star_sentiment(df)

    logger.info("Stage 1/2: VADER baseline...")
    df = run_vader(df)
    vader_agree = vader_vs_stars(df)

    logger.info("Stage 2/2: DistilBERT inference...")
    df = run_distilbert(df)
    bert_agree = bert_vs_stars(df)

    df = add_agreement_flag(df)

    elapsed = time.time() - start

    save_sentiment(df)
    write_report(df, vader_agree, bert_agree, elapsed)
    print_summary(df, vader_agree, bert_agree)

    logger.info("Phase 4 — Sentiment Pipeline COMPLETE")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run()
    print(f"✅ Sentiment parquet ready. Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
