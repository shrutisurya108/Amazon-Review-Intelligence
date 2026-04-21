"""
Phase 4 — Sentiment Modeling: VADER Baseline
Applies VADER (Valence Aware Dictionary and sEntiment Reasoner) to
the review_clean column. VADER is rule-based, fast, and requires no GPU.

Output columns added:
  vader_score  → compound score in [-1.0, +1.0]
  vader_label  → 'positive' | 'neutral' | 'negative'
"""
from pathlib import Path
import sys

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# VADER thresholds (standard from the original paper)
_POS_THRESHOLD =  0.05
_NEG_THRESHOLD = -0.05

_ANALYZER = SentimentIntensityAnalyzer()


def score_text(text: str) -> tuple[float, str]:
    """
    Scores a single text string with VADER.
    Returns (compound_score, label).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, "neutral"

    scores = _ANALYZER.polarity_scores(text)
    compound = scores["compound"]

    if compound >= _POS_THRESHOLD:
        label = "positive"
    elif compound <= _NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return round(compound, 4), label


def run_vader(df: pd.DataFrame, text_col: str = "review_clean") -> pd.DataFrame:
    """
    Applies VADER to every row in the DataFrame.
    Adds vader_score and vader_label columns.
    Returns updated DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Run preprocessing pipeline first.")

    logger.info(f"Running VADER on {len(df):,} reviews...")

    scores  = []
    labels  = []

    for text in tqdm(df[text_col], desc="VADER scoring", unit="reviews"):
        s, l = score_text(text)
        scores.append(s)
        labels.append(l)

    df = df.copy()
    df["vader_score"] = scores
    df["vader_label"] = labels

    dist = df["vader_label"].value_counts().to_dict()
    logger.info(f"VADER label distribution: {dist}")
    logger.info("VADER scoring complete.")

    return df


def vader_vs_stars(df: pd.DataFrame) -> float:
    """
    Computes agreement rate between VADER label and star rating.
    Stars 4–5 = positive, 1–2 = negative, 3 = neutral.
    Returns agreement rate as float 0–1.
    """
    def star_to_sentiment(r):
        if r >= 4:
            return "positive"
        elif r <= 2:
            return "negative"
        else:
            return "neutral"

    df = df.copy()
    df["star_sentiment"] = df["rating"].apply(star_to_sentiment)
    agree = (df["vader_label"] == df["star_sentiment"]).mean()
    logger.info(f"VADER vs star-rating agreement: {agree:.1%}")
    return agree


if __name__ == "__main__":
    from config import DATA_PROCESSED
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    sample = df.head(500).copy()
    sample = run_vader(sample)
    rate = vader_vs_stars(sample)
    print(f"\n✅ VADER OK on 500 rows.")
    print(f"   Agreement with star ratings: {rate:.1%}")
    print(f"   Sample:\n{sample[['review_clean','rating','vader_score','vader_label']].head(5).to_string()}")
