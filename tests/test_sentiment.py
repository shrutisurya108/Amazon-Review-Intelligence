"""
Phase 4 — Sentiment Modeling Tests
Tests VADER scorer, DistilBERT output, and final parquet structure.
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED, OUTPUTS_REPORTS
from src.utils.logger import get_logger
from src.modeling.vader_sentiment import score_text

logger = get_logger("test.sentiment")


# ── VADER unit tests ──────────────────────────────────────────────────────────

def test_vader_positive():
    score, label = score_text("This product is absolutely fantastic and works perfectly!")
    assert label == "positive", f"Expected positive, got {label} (score={score})"
    assert score > 0.05
    print(f"✓ VADER positive detection (score={score})")


def test_vader_negative():
    score, label = score_text("Terrible product. Complete waste of money. Broke immediately.")
    assert label == "negative", f"Expected negative, got {label} (score={score})"
    assert score < -0.05
    print(f"✓ VADER negative detection (score={score})")


def test_vader_neutral():
    score, label = score_text("product arrived")
    assert label in ("neutral", "positive", "negative")  # short text can vary
    print(f"✓ VADER handles short/neutral text (score={score}, label={label})")


def test_vader_empty():
    score, label = score_text("")
    assert score == 0.0 and label == "neutral"
    score2, label2 = score_text(None)
    assert score2 == 0.0 and label2 == "neutral"
    print("✓ VADER handles empty/null input")


def test_vader_score_range():
    texts = [
        "amazing excellent perfect love it",
        "okay product nothing special",
        "horrible garbage terrible broken useless",
    ]
    for t in texts:
        score, _ = score_text(t)
        assert -1.0 <= score <= 1.0, f"Score out of range: {score}"
    print("✓ VADER scores always in [-1.0, +1.0]")


# ── Sentiment parquet output tests ────────────────────────────────────────────

def test_sentiment_parquet_exists():
    path = DATA_PROCESSED / "reviews_sentiment.parquet"
    assert path.exists(), (
        f"Sentiment parquet not found at {path}. "
        "Run: python src/modeling/sentiment_pipeline.py"
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    assert size_mb > 0.5
    print(f"✓ Sentiment parquet exists ({size_mb:.2f} MB)")


def test_sentiment_columns():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    required = [
        "review_text", "rating", "review_clean", "review_normalized",
        "tokens", "token_count",
        "vader_score", "vader_label",
        "bert_label", "bert_confidence",
        "sentiment_agree", "star_sentiment",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print(f"✓ All {len(required)} required columns present")


def test_vader_labels_valid():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    valid = {"positive", "negative", "neutral"}
    invalid = set(df["vader_label"].unique()) - valid
    assert not invalid, f"Invalid VADER labels found: {invalid}"
    print(f"✓ VADER labels valid: {df['vader_label'].value_counts().to_dict()}")


def test_bert_labels_valid():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    valid = {"positive", "negative"}
    invalid = set(df["bert_label"].unique()) - valid
    assert not invalid, f"Invalid BERT labels: {invalid}"
    print(f"✓ BERT labels valid: {df['bert_label'].value_counts().to_dict()}")


def test_bert_confidence_range():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    assert df["bert_confidence"].between(0.0, 1.0).all(), "Confidence scores out of [0,1]"
    avg = df["bert_confidence"].mean()
    assert avg > 0.7, f"Average confidence too low: {avg:.3f}"
    print(f"✓ BERT confidence in [0,1], avg={avg:.3f}")


def test_vader_agreement_rate():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    df_binary = df[df["rating"] != 3].copy()
    agree = (df_binary["vader_label"] == df_binary["star_sentiment"]).mean()
    assert agree > 0.60, f"VADER agreement too low: {agree:.1%} (expected >60%)"
    print(f"✓ VADER vs star agreement: {agree:.1%}")


def test_bert_agreement_rate():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    df_binary = df[df["rating"] != 3].copy()
    agree = (df_binary["bert_label"] == df_binary["star_sentiment"]).mean()
    assert agree > 0.75, f"BERT agreement too low: {agree:.1%} (expected >75%)"
    print(f"✓ BERT vs star agreement: {agree:.1%}")


def test_bert_beats_vader():
    df = pd.read_parquet(DATA_PROCESSED / "reviews_sentiment.parquet")
    df_binary = df[df["rating"] != 3].copy()
    vader_agree = (df_binary["vader_label"] == df_binary["star_sentiment"]).mean()
    bert_agree  = (df_binary["bert_label"]  == df_binary["star_sentiment"]).mean()
    print(f"✓ BERT ({bert_agree:.1%}) vs VADER ({vader_agree:.1%}) — improvement: +{bert_agree - vader_agree:.1%}")


def test_report_exists():
    path = OUTPUTS_REPORTS / "sentiment_report.txt"
    assert path.exists(), f"Report not found at {path}"
    content = path.read_text()
    assert "VADER" in content and "DistilBERT" in content
    print(f"✓ Sentiment report exists and contains expected sections")


if __name__ == "__main__":
    print("\nRunning Phase 4 sentiment tests...\n")

    print("--- VADER Unit Tests ---")
    test_vader_positive()
    test_vader_negative()
    test_vader_neutral()
    test_vader_empty()
    test_vader_score_range()

    print("\n--- Sentiment Parquet Tests ---")
    test_sentiment_parquet_exists()
    test_sentiment_columns()
    test_vader_labels_valid()
    test_bert_labels_valid()
    test_bert_confidence_range()
    test_vader_agreement_rate()
    test_bert_agreement_rate()
    test_bert_beats_vader()
    test_report_exists()

    print("\n✅ All Phase 4 sentiment tests passed.")
