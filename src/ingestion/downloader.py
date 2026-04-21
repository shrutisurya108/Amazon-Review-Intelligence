"""
Phase 2 — Data Ingestion: Downloader
Downloads the Datafiniti Amazon Electronics reviews dataset from Kaggle,
extracts it into data/raw/, and validates the download.
Compatible with kaggle>=1.6.0 which uses KaggleApi (not KaggleApiExtended).
"""
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DATA_RAW
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_SLUG = "datafiniti/consumer-reviews-of-amazon-products"


def authenticate_kaggle():
    """
    Authenticates with Kaggle API using ~/.kaggle/kaggle.json.
    Returns authenticated API client.
    """
    logger.info("Authenticating with Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
    except ImportError:
        import kaggle
        api = kaggle.api
    api.authenticate()
    logger.info("Kaggle authentication successful.")
    return api


def download_dataset(api) -> Path:
    """
    Downloads the dataset zip from Kaggle into data/raw/.
    Skips download if already present.
    Returns path to the raw directory.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    existing_csvs = list(DATA_RAW.glob("*.csv"))
    if existing_csvs:
        logger.info(f"Dataset already downloaded: {[f.name for f in existing_csvs]}")
        return DATA_RAW

    logger.info(f"Downloading dataset: {DATASET_SLUG}")
    logger.info(f"Destination: {DATA_RAW}")

    api.dataset_download_files(
        dataset=DATASET_SLUG,
        path=str(DATA_RAW),
        unzip=False,
        quiet=False,
    )

    zip_files = list(DATA_RAW.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            "Download completed but no zip file found in data/raw/. "
            "Check your Kaggle credentials and that you accepted the dataset terms."
        )

    logger.info(f"Download complete. Extracting {zip_files[0].name}...")

    with zipfile.ZipFile(zip_files[0], "r") as z:
        z.extractall(DATA_RAW)
        extracted = z.namelist()
        logger.info(f"Extracted {len(extracted)} file(s): {extracted}")

    zip_files[0].unlink()
    logger.info("Zip file removed after extraction.")

    return DATA_RAW


def validate_download() -> Path:
    """
    Confirms at least one CSV exists in data/raw/ and is non-empty.
    Returns path to the primary CSV file.
    """
    csv_files = list(DATA_RAW.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_RAW}. "
            "Download may have failed — check your Kaggle credentials and dataset terms acceptance."
        )

    primary = max(csv_files, key=lambda f: f.stat().st_size)
    size_mb = primary.stat().st_size / (1024 * 1024)

    if size_mb < 0.1:
        raise ValueError(
            f"File {primary.name} is suspiciously small ({size_mb:.2f} MB). May be corrupt."
        )

    logger.info(f"Validated: {primary.name} ({size_mb:.1f} MB)")

    if len(csv_files) > 1:
        logger.info(f"Additional files found: {[f.name for f in csv_files[1:]]}")

    return primary


def run() -> Path:
    """Full download pipeline. Returns path to primary CSV."""
    logger.info("=" * 60)
    logger.info("Phase 2 — Downloader START")
    logger.info("=" * 60)

    api = authenticate_kaggle()
    download_dataset(api)
    primary_csv = validate_download()

    logger.info("Phase 2 — Downloader COMPLETE")
    logger.info("=" * 60)

    return primary_csv


if __name__ == "__main__":
    csv_path = run()
    print(f"\n✅ Dataset ready at: {csv_path}")
