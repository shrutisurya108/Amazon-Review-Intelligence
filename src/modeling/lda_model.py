"""
Phase 5 — Topic Modeling: LDA Model
Trains a Latent Dirichlet Allocation model on the token lists from
the preprocessed reviews. Finds recurring themes/topics across the corpus.

Steps:
  1. Build dictionary and bag-of-words corpus from tokens
  2. Train LDA with LDA_NUM_TOPICS topics
  3. Compute coherence score (c_v) as quality metric
  4. Save trained model and dictionary to outputs/models/
  5. Print topic keywords for human interpretation
"""
import pickle
from pathlib import Path
import sys

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATA_PROCESSED, OUTPUTS_MODELS,
    LDA_NUM_TOPICS, LDA_PASSES, RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Paths for saving model artifacts
DICT_PATH  = OUTPUTS_MODELS / "lda_dictionary.pkl"
MODEL_PATH = OUTPUTS_MODELS / "lda_model.pkl"

# Words to filter from topics (too generic to be meaningful)
_EXTRA_FILTER = {
    "product", "amazon", "item", "buy", "purchase", "order",
    "get", "got", "use", "used", "good", "great", "nice", "bad",
    "really", "just", "like", "love", "hate", "work", "make",
    "thing", "come", "going", "say", "know", "think", "want",
    "need", "look", "give", "take", "little", "big", "new", "old",
}

# Human-readable labels for each topic — assigned after inspecting keywords
# These will be updated after training when you see the actual keywords
TOPIC_LABELS = {
    0: "kids_tablets",
    1: "customer_service_issues",
    2: "apps_entertainment",
    3: "device_features",
    4: "battery_value",
    5: "gift_recommendations",
    6: "media_camera",
    7: "kindle_ereaders",
}


def load_tokens(path: Path = None) -> tuple[pd.DataFrame, list[list[str]]]:
    """
    Loads the sentiment parquet and extracts token lists.
    Falls back to processed parquet if sentiment parquet not found.
    Returns (df, token_lists).
    """
    sentiment_path  = DATA_PROCESSED / "reviews_sentiment.parquet"
    processed_path  = DATA_PROCESSED / "reviews_processed.parquet"

    if sentiment_path.exists():
        df = pd.read_parquet(sentiment_path)
        logger.info(f"Loaded sentiment parquet: {len(df):,} rows")
    elif processed_path.exists():
        df = pd.read_parquet(processed_path)
        logger.info(f"Loaded processed parquet: {len(df):,} rows")
    else:
        raise FileNotFoundError("No processed parquet found. Run Phase 3 pipeline first.")

    # tokens column is stored as list — ensure correct type
    token_lists = df["tokens"].tolist()

    # Filter extra generic words from each document
    token_lists = [
        [t for t in doc if t not in _EXTRA_FILTER and len(t) >= 3]
        for doc in token_lists
    ]

    # Remove documents with fewer than 3 tokens after filtering
    valid = [(df_idx, tl) for df_idx, tl in enumerate(token_lists) if len(tl) >= 3]
    valid_indices  = [v[0] for v in valid]
    token_lists    = [v[1] for v in valid]

    df = df.iloc[valid_indices].reset_index(drop=True)
    logger.info(f"Valid documents for LDA: {len(token_lists):,}")

    return df, token_lists


def build_corpus(token_lists: list[list[str]]) -> tuple:
    """
    Builds gensim dictionary and bag-of-words corpus.
    Filters extremes: removes very rare and very common words.
    Returns (dictionary, corpus).
    """
    logger.info("Building gensim dictionary...")
    dictionary = corpora.Dictionary(token_lists)

    before = len(dictionary)
    dictionary.filter_extremes(
        no_below=10,     # word must appear in at least 10 documents
        no_above=0.85,   # word must appear in at most 85% of documents
        keep_n=10_000,   # keep top 10K words by frequency
    )
    after = len(dictionary)
    logger.info(f"Dictionary: {before:,} → {after:,} tokens after filtering extremes")

    corpus = [dictionary.doc2bow(doc) for doc in token_lists]
    logger.info(f"Corpus built: {len(corpus):,} documents")

    return dictionary, corpus


def train_lda(corpus, dictionary) -> LdaModel:
    """
    Trains the LDA model with configured hyperparameters.
    Returns trained LdaModel.
    """
    logger.info(
        f"Training LDA: {LDA_NUM_TOPICS} topics, "
        f"{LDA_PASSES} passes — this takes ~2–4 min..."
    )

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=LDA_NUM_TOPICS,
        passes=LDA_PASSES,
        random_state=RANDOM_SEED,
        alpha="auto",          # asymmetric prior — learns per-topic weights
        eta="auto",            # learns per-word weights
        minimum_probability=0.01,
        per_word_topics=False,
    )

    logger.info("LDA training complete.")
    return model


def compute_coherence(model: LdaModel, token_lists: list, dictionary) -> float:
    """
    Computes c_v coherence score — standard metric for LDA quality.
    Higher is better. Good models score 0.4–0.7.
    Returns coherence score as float.
    """
    logger.info("Computing coherence score (c_v)...")

    coherence_model = CoherenceModel(
        model=model,
        texts=token_lists,
        dictionary=dictionary,
        coherence="c_v",
    )
    score = coherence_model.get_coherence()
    logger.info(f"Coherence score (c_v): {score:.4f}")
    return score


def print_topics(model: LdaModel, num_words: int = 10) -> dict:
    """
    Prints top keywords per topic for human inspection.
    Returns dict of {topic_id: [keywords]}.
    """
    topics = {}
    print("\n" + "=" * 60)
    print("  LDA TOPICS — Top Keywords")
    print("=" * 60)

    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=num_words)
        keywords = [w for w, _ in words]
        label = TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")
        topics[topic_id] = keywords
        print(f"\n  Topic {topic_id:02d} [{label}]")
        print(f"  {' | '.join(keywords)}")

    print("\n" + "=" * 60)
    return topics


def save_artifacts(model: LdaModel, dictionary) -> None:
    """Saves LDA model and dictionary to outputs/models/."""
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(DICT_PATH, "wb") as f:
        pickle.dump(dictionary, f)

    logger.info(f"LDA model saved: {MODEL_PATH}")
    logger.info(f"Dictionary saved: {DICT_PATH}")


def load_artifacts() -> tuple[LdaModel, corpora.Dictionary]:
    """Loads saved LDA model and dictionary. Used by topic_pipeline.py."""
    if not MODEL_PATH.exists() or not DICT_PATH.exists():
        raise FileNotFoundError(
            "LDA artifacts not found. Run lda_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(DICT_PATH, "rb") as f:
        dictionary = pickle.load(f)
    logger.info("LDA model and dictionary loaded from disk.")
    return model, dictionary


def run() -> tuple[LdaModel, corpora.Dictionary, list, pd.DataFrame, float]:
    """
    Full LDA training pipeline.
    Returns (model, dictionary, token_lists, df, coherence_score).
    """
    logger.info("=" * 60)
    logger.info("Phase 5 — LDA Model Training START")
    logger.info("=" * 60)

    df, token_lists       = load_tokens()
    dictionary, corpus    = build_corpus(token_lists)
    model                 = train_lda(corpus, dictionary)
    coherence             = compute_coherence(model, token_lists, dictionary)
    topics                = print_topics(model)
    save_artifacts(model, dictionary)

    logger.info("Phase 5 — LDA Model Training COMPLETE")
    logger.info("=" * 60)

    return model, dictionary, token_lists, df, coherence


if __name__ == "__main__":
    model, dictionary, token_lists, df, coherence = run()
    print(f"\n✅ LDA training complete.")
    print(f"   Topics       : {LDA_NUM_TOPICS}")
    print(f"   Coherence(cv): {coherence:.4f}")
    print(f"   Model saved  : {MODEL_PATH}")
