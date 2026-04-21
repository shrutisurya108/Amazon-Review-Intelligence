"""
Phase 2 — Ingestion Tests
Validates downloader output and loader output independently.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_RAW, DATA_INTERIM
from src.utils.logger import get_logger

logger = get_logger("test.ingestion")


def test_raw_csv_exists():
    csv_files = list(DATA_RAW.glob("*.csv"))
    assert len(csv_files) > 0, (
        f"No CSV files in {DATA_RAW}. Run: python src/ingestion/downloader.py"
    )
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        assert size_mb > 0.1, f"{f.name} is too small ({size_mb:.2f} MB) — may be corrupt"
        logger.info(f"✓ Raw file OK: {f.name} ({size_mb:.1f} MB)")
    print(f"✓ Raw CSV exists: {[f.name for f in csv_files]}")


def test_interim_parquet_exists():
    parquet = DATA_INTERIM / "reviews_raw.parquet"
    assert parquet.exists(), (
        f"Interim parquet not found at {parquet}. Run: python src/ingestion/loader.py"
    )
    size_mb = parquet.stat().st_size / (1024 * 1024)
    assert size_mb > 0.01, f"Parquet file is suspiciously small: {size_mb:.3f} MB"
    logger.info(f"✓ Interim parquet: {size_mb:.2f} MB")
    print(f"✓ Interim parquet exists ({size_mb:.2f} MB)")


def test_dataframe_structure():
    parquet = DATA_INTERIM / "reviews_raw.parquet"
    df = pd.read_parquet(parquet)

    assert len(df) > 100, f"Too few rows: {len(df)}"
    print(f"✓ Row count: {len(df):,}")

    required = ["review_text", "rating"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"✓ Required columns present: {required}")

    assert df["review_text"].isna().sum() == 0, "Null values found in review_text"
    assert df["rating"].isna().sum() == 0, "Null values found in rating"
    print("✓ No nulls in review_text or rating")

    assert df["rating"].between(1, 5).all(), "Ratings outside 1–5 range detected"
    print("✓ All ratings within 1–5 range")

    short = (df["review_text"].str.len() <= 10).sum()
    assert short == 0, f"{short} reviews are too short (≤10 chars)"
    print("✓ All reviews longer than 10 characters")

    logger.info(f"DataFrame structure test passed: {df.shape}")


def test_rating_distribution():
    df = pd.read_parquet(DATA_INTERIM / "reviews_raw.parquet")
    dist = df["rating"].value_counts().sort_index()
    print(f"✓ Rating distribution: { {int(k): int(v) for k, v in dist.items()} }")
    assert len(dist) >= 3, "Expected at least 3 distinct rating values"
    logger.info("Rating distribution test passed")


if __name__ == "__main__":
    print("\nRunning Phase 2 ingestion tests...\n")
    test_raw_csv_exists()
    test_interim_parquet_exists()
    test_dataframe_structure()
    test_rating_distribution()
    print("\n✅ All Phase 2 ingestion tests passed.")
