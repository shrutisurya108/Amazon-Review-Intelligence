"""
Phase 3 — NLP Preprocessing: Cleaner
Removes noise from raw review text:
  - HTML tags and entities
  - URLs and email addresses
  - Currency symbols and prices
  - Non-ASCII / emoji characters
  - Excessive punctuation and whitespace
  - Standalone numbers

Output is clean English prose, ready for NLP normalization.
"""
import re
import html
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Compiled regex patterns (compiled once at import for performance) ──────────
_HTML_TAG       = re.compile(r"<[^>]+>")
_HTML_ENTITY    = re.compile(r"&[a-z]+;|&#\d+;")
_URL            = re.compile(r"https?://\S+|www\.\S+")
_EMAIL          = re.compile(r"\S+@\S+\.\S+")
_PRICE          = re.compile(r"\$[\d,.]+|\d+\s*dollars?", re.IGNORECASE)
_NON_ASCII      = re.compile(r"[^\x00-\x7F]+")
_PUNCTUATION    = re.compile(r"[^\w\s]")
_STANDALONE_NUM = re.compile(r"\b\d+\b")
_WHITESPACE     = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Applies full noise-removal pipeline to a single review string.
    Returns cleaned text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Decode HTML entities first (e.g. &amp; → &)
    text = html.unescape(text)

    # 2. Strip HTML tags
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)

    # 3. Remove URLs and emails
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)

    # 4. Remove prices (before general number removal)
    text = _PRICE.sub(" ", text)

    # 5. Remove non-ASCII characters (emoji, unicode symbols)
    text = _NON_ASCII.sub(" ", text)

    # 6. Remove punctuation (keep word characters and spaces)
    text = _PUNCTUATION.sub(" ", text)

    # 7. Remove standalone numbers (not part of words like "4K", "MP3")
    text = _STANDALONE_NUM.sub(" ", text)

    # 8. Normalize whitespace — collapse multiple spaces to one
    text = _WHITESPACE.sub(" ", text).strip()

    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Applies clean_text() to every row in the DataFrame.
    Adds 'review_clean' column and drops rows where cleaning left empty strings.
    Returns updated DataFrame.
    """
    logger.info(f"Cleaning {len(df):,} reviews...")

    df = df.copy()
    df["review_clean"] = df[text_col].apply(clean_text)

    before = len(df)
    df = df[df["review_clean"].str.len() > 10].reset_index(drop=True)
    dropped = before - len(df)

    if dropped:
        logger.warning(f"Dropped {dropped:,} reviews that became empty after cleaning")

    logger.info(f"Cleaning complete. {len(df):,} reviews retained.")
    return df


def log_cleaning_sample(df: pd.DataFrame, n: int = 3) -> None:
    """Logs before/after examples so you can visually verify cleaning."""
    logger.info("=== Cleaning Sample (before → after) ===")
    samples = df.sample(n=min(n, len(df)), random_state=42)
    for _, row in samples.iterrows():
        logger.info(f"  BEFORE: {str(row['review_text'])[:120]}")
        logger.info(f"  AFTER : {str(row['review_clean'])[:120]}")
        logger.info("  ---")


if __name__ == "__main__":
    from config import DATA_INTERIM
    df = pd.read_parquet(DATA_INTERIM / "reviews_raw.parquet")
    df = clean_dataframe(df)
    log_cleaning_sample(df)
    print(f"\n✅ Cleaner OK. Sample output:\n{df[['review_text','review_clean']].head(2).to_string()}")
