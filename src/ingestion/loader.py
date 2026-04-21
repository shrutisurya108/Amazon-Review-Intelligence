"""
Phase 2 — Data Ingestion: Loader
Loads raw CSV into a pandas DataFrame, validates structure,
standardizes column names, and saves a clean interim parquet snapshot.
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_RAW, DATA_INTERIM, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns we need — mapped from Datafiniti schema → our standard names
COLUMN_MAP = {
    "reviews.text":   "review_text",
    "reviews.rating": "rating",
    "reviews.date":   "review_date",
    "name":           "product_name",
    "reviews.title":  "review_title",
    "reviews.username": "reviewer",
}

REQUIRED_SOURCE_COLS = ["reviews.text", "reviews.rating"]


def find_primary_csv() -> Path:
    """Finds the largest CSV in data/raw/ — handles multi-file datasets."""
    csv_files = list(DATA_RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {DATA_RAW}. Run downloader.py first.")

    primary = max(csv_files, key=lambda f: f.stat().st_size)
    logger.info(f"Loading: {primary.name} ({primary.stat().st_size / 1e6:.1f} MB)")
    return primary


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Loads raw CSV with encoding fallback."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed — retrying with latin-1 encoding.")
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    logger.info(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.debug(f"Raw columns: {df.columns.tolist()}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Confirms required source columns are present."""
    missing = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    logger.info("✓ Required columns present")


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Renames columns to our standard schema
    - Drops rows with missing review text or rating
    - Parses dates
    - Coerces rating to numeric
    - Resets index
    """
    # Keep only columns that exist in this file
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(rename.keys())].rename(columns=rename)

    logger.info(f"Kept {len(rename)} columns: {list(rename.values())}")

    before = len(df)
    df = df.dropna(subset=["review_text", "rating"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped:,} rows with missing text or rating")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        invalid_dates = df["review_date"].isna().sum()
        if invalid_dates:
            logger.warning(f"{invalid_dates:,} rows had unparseable dates — set to NaT")

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 10]

    df = df.reset_index(drop=True)
    logger.info(f"Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Prints a human-readable summary of the loaded dataset."""
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total reviews      : {len(df):,}")
    print(f"  Columns            : {df.columns.tolist()}")
    print(f"  Rating distribution:")
    rating_counts = df["rating"].value_counts().sort_index()
    for star, count in rating_counts.items():
        bar = "█" * (count * 30 // len(df))
        print(f"    {star}★  {count:>6,}  {bar}")
    if "review_date" in df.columns:
        valid_dates = df["review_date"].dropna()
        if len(valid_dates):
            print(f"  Date range         : {valid_dates.min().date()} → {valid_dates.max().date()}")
    print(f"  Missing values     :")
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            print(f"    {col}: {n:,}")
    print("=" * 55 + "\n")


def save_interim(df: pd.DataFrame) -> Path:
    """Saves standardized DataFrame as parquet to data/interim/."""
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERIM / "reviews_raw.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved interim parquet: {out_path.name} ({size_mb:.2f} MB)")
    return out_path


def run() -> pd.DataFrame:
    """Full load pipeline. Returns clean DataFrame."""
    logger.info("=" * 60)
    logger.info("Phase 2 — Loader START")
    logger.info("=" * 60)

    csv_path = find_primary_csv()
    df_raw   = load_raw(csv_path)
    validate_columns(df_raw)
    df_clean = standardize(df_raw)
    summarize(df_clean)
    save_interim(df_clean)

    logger.info("Phase 2 — Loader COMPLETE")
    logger.info("=" * 60)

    return df_clean


if __name__ == "__main__":
    df = run()
    print(f"✅ Interim parquet saved. DataFrame shape: {df.shape}")
