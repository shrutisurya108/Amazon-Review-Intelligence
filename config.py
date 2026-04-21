"""
Central configuration for amazon-review-intelligence.
All paths and parameters are defined here — import this in every module.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_INTERIM    = ROOT_DIR / "data" / "interim"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_FIGURES = ROOT_DIR / "outputs" / "figures"
OUTPUTS_REPORTS = ROOT_DIR / "outputs" / "reports"
OUTPUTS_MODELS  = ROOT_DIR / "outputs" / "models"

# ── Logs ──────────────────────────────────────────────────────────────────────
LOGS_DIR = ROOT_DIR / "logs"

# ── Dataset ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET      = "snap/amazon-fine-food-reviews"   # overridden below for Electronics
RAW_FILE            = DATA_RAW / "reviews.json.gz"
PROCESSED_FILE      = DATA_PROCESSED / "reviews_clean.parquet"
SAMPLE_SIZE         = 50_000   # rows to work with (manageable on 8GB RAM)
RANDOM_SEED         = 42
CATEGORY            = "Electronics"

# ── NLP ───────────────────────────────────────────────────────────────────────
SPACY_MODEL         = "en_core_web_sm"
MAX_REVIEW_TOKENS   = 512
LDA_NUM_TOPICS      = 8
LDA_PASSES          = 10

# ── Sentiment ─────────────────────────────────────────────────────────────────
DISTILBERT_MODEL    = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_BATCH     = 32

# ── Kaggle credentials (loaded from .env) ─────────────────────────────────────
KAGGLE_USERNAME     = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY          = os.getenv("KAGGLE_KEY")
