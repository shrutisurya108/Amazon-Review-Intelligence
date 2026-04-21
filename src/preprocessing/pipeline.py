"""
Phase 3 — NLP Preprocessing: Pipeline Orchestrator
Runs the full preprocessing sequence:
  1. Load interim parquet (from Phase 2)
  2. Clean  (cleaner.py)
  3. Normalize (normalizer.py)
  4. Save processed parquet to data/processed/

This is the single entry point for Phase 3.
Run: python src/preprocessing/pipeline.py
"""
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_INTERIM, DATA_PROCESSED, SAMPLE_SIZE, RANDOM_SEED
from src.utils.logger import get_logger
from src.preprocessing.cleaner import clean_dataframe, log_cleaning_sample
from src.preprocessing.normalizer import normalize_dataframe, log_normalization_sample

logger = get_logger(__name__)


def load_interim() -> pd.DataFrame:
    """Loads the Phase 2 interim parquet."""
    path = DATA_INTERIM / "reviews_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Interim parquet not found at {path}. "
            "Run Phase 2 first: python src/ingestion/loader.py"
        )
    df = pd.read_parquet(path)
    logger.info(f"Loaded interim parquet: {len(df):,} rows × {df.shape[1]} columns")
    return df


def maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataset is larger than SAMPLE_SIZE, take a stratified sample
    that preserves the rating distribution. Otherwise use full dataset.
    """
    if len(df) <= SAMPLE_SIZE:
        logger.info(f"Dataset size ({len(df):,}) ≤ SAMPLE_SIZE ({SAMPLE_SIZE:,}). Using full dataset.")
        return df

    logger.info(f"Sampling {SAMPLE_SIZE:,} rows from {len(df):,} (stratified by rating)...")
    df_sampled = (
        df.groupby("rating", group_keys=False)
        .apply(lambda g: g.sample(
            n=max(1, int(SAMPLE_SIZE * len(g) / len(df))),
            random_state=RANDOM_SEED,
        ))
        .reset_index(drop=True)
    )
    logger.info(f"Sample shape: {df_sampled.shape}")
    return df_sampled


def save_processed(df: pd.DataFrame) -> Path:
    """Saves final processed DataFrame to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "reviews_processed.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved processed parquet: {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def print_summary(df: pd.DataFrame, elapsed: float) -> None:
    """Prints a processing summary to terminal."""
    print("\n" + "=" * 55)
    print("  PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Final row count    : {len(df):,}")
    print(f"  Columns            : {df.columns.tolist()}")
    print(f"  Avg token count    : {df['token_count'].mean():.1f}")
    print(f"  Min token count    : {df['token_count'].min()}")
    print(f"  Max token count    : {df['token_count'].max()}")
    print(f"  Processing time    : {elapsed:.1f}s")
    print(f"\n  Sample normalized reviews:")
    for _, row in df.sample(3, random_state=42).iterrows():
        print(f"    [{row['rating']}★] {row['review_normalized'][:80]}...")
    print("=" * 55 + "\n")


def run() -> pd.DataFrame:
    """Full preprocessing pipeline. Returns processed DataFrame."""
    logger.info("=" * 60)
    logger.info("Phase 3 — Preprocessing Pipeline START")
    logger.info("=" * 60)

    start = time.time()

    df = load_interim()
    df = maybe_sample(df)

    logger.info("Step 1/2: Cleaning...")
    df = clean_dataframe(df)
    log_cleaning_sample(df)

    logger.info("Step 2/2: Normalizing...")
    df = normalize_dataframe(df)
    log_normalization_sample(df)

    out_path = save_processed(df)
    elapsed = time.time() - start

    print_summary(df, elapsed)

    logger.info("Phase 3 — Preprocessing Pipeline COMPLETE")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run()
    print(f"✅ Processed parquet ready. Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
