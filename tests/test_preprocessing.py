"""
Phase 3 — Preprocessing Tests
Tests cleaner, normalizer, and pipeline output independently.
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED, DATA_INTERIM
from src.utils.logger import get_logger
from src.preprocessing.cleaner import clean_text, clean_dataframe
from src.preprocessing.normalizer import normalize_text

logger = get_logger("test.preprocessing")


# ── Cleaner unit tests ────────────────────────────────────────────────────────

def test_clean_removes_html():
    raw = "Great product! <br><b>Very</b> happy with it."
    out = clean_text(raw)
    assert "<" not in out and ">" not in out, f"HTML tags remain: {out}"
    assert "Great" in out and "happy" in out
    print("✓ Cleaner removes HTML tags")


def test_clean_removes_urls():
    raw = "Check this https://amazon.com/product?id=123 for more info"
    out = clean_text(raw)
    assert "http" not in out and "amazon.com" not in out, f"URL remains: {out}"
    print("✓ Cleaner removes URLs")


def test_clean_removes_prices():
    raw = "Paid $49.99 for this and it broke in 3 days"
    out = clean_text(raw)
    assert "$" not in out, f"Price symbol remains: {out}"
    print("✓ Cleaner removes prices")


def test_clean_removes_emoji():
    raw = "Love it so much 😍🔥 best buy ever!!!"
    out = clean_text(raw)
    assert "😍" not in out and "🔥" not in out, f"Emoji remains: {out}"
    print("✓ Cleaner removes emoji/non-ASCII")


def test_clean_empty_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    print("✓ Cleaner handles empty/null input safely")


def test_clean_preserves_meaning():
    raw = "The battery life is excellent and screen resolution is amazing"
    out = clean_text(raw)
    assert "battery" in out.lower()
    assert "excellent" in out.lower()
    print("✓ Cleaner preserves meaningful words")


# ── Normalizer unit tests ─────────────────────────────────────────────────────

def test_normalize_lowercases():
    text = "Great PRODUCT Amazing Quality"
    norm, tokens = normalize_text(text)
    assert norm == norm.lower(), f"Uppercase remains: {norm}"
    print("✓ Normalizer lowercases text")


def test_normalize_lemmatizes():
    text = "The batteries were running very quickly and stopped working"
    norm, tokens = normalize_text(text)
    # "batteries" → "battery", "running" → "run", "stopped" → "stop"
    assert "battery" in tokens or "batteries" not in tokens
    print("✓ Normalizer lemmatizes tokens")


def test_normalize_removes_stopwords():
    text = "This is a very good product and I really like it a lot"
    norm, tokens = normalize_text(text)
    stopwords_present = [t for t in tokens if t in {"this", "is", "a", "and", "it"}]
    assert len(stopwords_present) == 0, f"Stopwords remain: {stopwords_present}"
    print("✓ Normalizer removes stopwords")


def test_normalize_filters_short_tokens():
    text = "I go to a big red box to buy it"
    norm, tokens = normalize_text(text)
    short = [t for t in tokens if len(t) < 3]
    assert len(short) == 0, f"Short tokens remain: {short}"
    print("✓ Normalizer filters tokens shorter than 3 chars")


def test_normalize_empty_input():
    norm, tokens = normalize_text("")
    assert norm == "" and tokens == []
    norm2, tokens2 = normalize_text(None)
    assert norm2 == "" and tokens2 == []
    print("✓ Normalizer handles empty/null input safely")


# ── Pipeline output tests ─────────────────────────────────────────────────────

def test_processed_parquet_exists():
    path = DATA_PROCESSED / "reviews_processed.parquet"
    assert path.exists(), (
        f"Processed parquet not found at {path}. "
        "Run: python src/preprocessing/pipeline.py"
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    assert size_mb > 0.1, f"File too small: {size_mb:.3f} MB"
    print(f"✓ Processed parquet exists ({size_mb:.2f} MB)")


def test_processed_columns():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    required = ["review_text", "rating", "review_clean", "review_normalized", "tokens", "token_count"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print(f"✓ All required columns present: {required}")


def test_processed_no_empty_tokens():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    empty = (df["token_count"] < 3).sum()
    assert empty == 0, f"{empty} rows have fewer than 3 tokens"
    print(f"✓ No rows with empty token lists")


def test_processed_token_count_stats():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    avg = df["token_count"].mean()
    assert avg >= 5, f"Average token count too low: {avg:.1f} (expected ≥ 5)"
    print(f"✓ Avg token count: {avg:.1f}")


def test_processed_ratings_intact():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    assert df["rating"].between(1, 5).all(), "Ratings outside 1–5 range"
    assert len(df["rating"].unique()) >= 3, "Too few unique rating values"
    print(f"✓ Ratings intact: {sorted(df['rating'].unique().tolist())}")


def test_processed_row_count():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    assert len(df) >= 1000, f"Too few rows after processing: {len(df)}"
    print(f"✓ Row count: {len(df):,}")


if __name__ == "__main__":
    print("\nRunning Phase 3 preprocessing tests...\n")

    print("--- Cleaner Unit Tests ---")
    test_clean_removes_html()
    test_clean_removes_urls()
    test_clean_removes_prices()
    test_clean_removes_emoji()
    test_clean_empty_input()
    test_clean_preserves_meaning()

    print("\n--- Normalizer Unit Tests ---")
    test_normalize_lowercases()
    test_normalize_lemmatizes()
    test_normalize_removes_stopwords()
    test_normalize_filters_short_tokens()
    test_normalize_empty_input()

    print("\n--- Pipeline Output Tests ---")
    test_processed_parquet_exists()
    test_processed_columns()
    test_processed_no_empty_tokens()
    test_processed_token_count_stats()
    test_processed_ratings_intact()
    test_processed_row_count()

    print("\n✅ All Phase 3 preprocessing tests passed.")
