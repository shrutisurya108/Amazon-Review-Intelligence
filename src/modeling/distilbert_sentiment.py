"""
Phase 4 — Sentiment Modeling: DistilBERT Inference
Applies distilbert-base-uncased-finetuned-sst-2-english to review_clean.
Runs on CPU (device=-1). Truncates reviews to MAX_REVIEW_TOKENS.

Output columns added:
  bert_label       → 'positive' | 'negative'
  bert_confidence  → model confidence score [0.0, 1.0]
"""
import warnings
import os
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DISTILBERT_MODEL, SENTIMENT_BATCH, MAX_REVIEW_TOKENS
from src.utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model is loaded lazily — only when run_distilbert() is called
# This prevents heavy loading during Streamlit app startup
_clf = None


def load_model():
    global _clf
    if _clf is not None:
        return _clf
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    logger.info(f"Loading DistilBERT model: {DISTILBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL)
    _clf = hf_pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        max_length=MAX_REVIEW_TOKENS,
    )
    logger.info("DistilBERT model loaded successfully.")
    return _clf



def run_distilbert(df: pd.DataFrame, text_col: str = "review_clean") -> pd.DataFrame:
    """
    Runs DistilBERT inference in batches on the DataFrame.
    Adds bert_label and bert_confidence columns.
    Returns updated DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Run preprocessing pipeline first.")

    clf = load_model()
    texts = df[text_col].fillna("").tolist()

    logger.info(f"Running DistilBERT on {len(texts):,} reviews (batch_size={SENTIMENT_BATCH})...")
    logger.info("This takes ~5–15 min on CPU for 24K reviews. Progress shown below.")

    labels      = []
    confidences = []

    # Process in batches with tqdm progress
    for i in tqdm(
        range(0, len(texts), SENTIMENT_BATCH),
        desc="DistilBERT",
        unit="batch",
    ):
        batch = texts[i : i + SENTIMENT_BATCH]

        # Replace empty strings with a neutral placeholder
        batch = [t if t.strip() else "average product" for t in batch]

        results = clf(batch)

        for res in results:
            labels.append(res["label"].lower())          # 'positive' or 'negative'
            confidences.append(round(res["score"], 4))

    df = df.copy()
    df["bert_label"]      = labels
    df["bert_confidence"] = confidences

    dist = df["bert_label"].value_counts().to_dict()
    avg_conf = df["bert_confidence"].mean()
    logger.info(f"DistilBERT label distribution: {dist}")
    logger.info(f"DistilBERT avg confidence: {avg_conf:.3f}")
    logger.info("DistilBERT inference complete.")

    return df


def bert_vs_stars(df: pd.DataFrame) -> float:
    """
    Computes agreement between DistilBERT label and star rating.
    Stars 4–5 = positive, 1–2 = negative, 3 = neutral (excluded from binary comparison).
    Returns agreement rate as float 0–1.
    """
    # Only compare on clear positive/negative reviews (exclude 3★ neutral)
    df_binary = df[df["rating"] != 3].copy()
    df_binary["star_sentiment"] = df_binary["rating"].apply(
        lambda r: "positive" if r >= 4 else "negative"
    )
    agree = (df_binary["bert_label"] == df_binary["star_sentiment"]).mean()
    logger.info(
        f"DistilBERT vs star-rating agreement (excl. 3★): {agree:.1%} "
        f"on {len(df_binary):,} reviews"
    )
    return agree


if __name__ == "__main__":
    from config import DATA_PROCESSED
    df = pd.read_parquet(DATA_PROCESSED / "reviews_processed.parquet")
    sample = df.head(200).copy()
    sample = run_distilbert(sample)
    rate = bert_vs_stars(sample)
    print(f"\n✅ DistilBERT OK on 200 rows.")
    print(f"   Agreement with star ratings: {rate:.1%}")
    print(
        f"   Sample:\n"
        f"{sample[['review_clean','rating','bert_label','bert_confidence']].head(5).to_string()}"
    )
