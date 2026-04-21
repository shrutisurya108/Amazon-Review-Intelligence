"""
Phase 3 — NLP Preprocessing: Normalizer
Applies linguistic normalization to cleaned review text:
  - Lowercasing
  - Tokenization (spaCy)
  - Stopword removal (spaCy + NLTK combined list)
  - Lemmatization (spaCy)
  - Short token filtering (removes tokens < 3 chars)

Output columns:
  review_normalized  ← normalized text string (for VADER / DistilBERT)
  tokens             ← list of clean tokens (for LDA topic modeling)
  token_count        ← number of tokens (for quality filtering)
"""
from pathlib import Path
import sys

import pandas as pd
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import SPACY_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Load spaCy model once at module level (expensive — do not load in loops) ──
logger.info(f"Loading spaCy model: {SPACY_MODEL}")
try:
    _NLP = spacy.load(SPACY_MODEL, disable=["parser", "ner"])  # only need tagger
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' not found. "
        f"Run: python -m spacy download {SPACY_MODEL}"
    )

# ── Combined stopword list: spaCy + NLTK ──────────────────────────────────────
_SPACY_STOPS = _NLP.Defaults.stop_words
_NLTK_STOPS  = set(stopwords.words("english"))
_ALL_STOPS   = _SPACY_STOPS | _NLTK_STOPS

# ── Additional domain-specific stopwords for Amazon reviews ───────────────────
_DOMAIN_STOPS = {
    "product", "amazon", "item", "buy", "purchase", "order", "ship",
    "shipping", "seller", "price", "review", "star", "rating", "got",
    "get", "use", "used", "using", "would", "could", "also", "one",
    "two", "three", "four", "five", "really", "just", "even", "still",
}
_ALL_STOPS = _ALL_STOPS | _DOMAIN_STOPS

MIN_TOKEN_LEN = 3  # discard tokens shorter than this


def normalize_text(text: str) -> tuple[str, list[str]]:
    """
    Normalizes a single cleaned review string.

    Returns:
        (normalized_string, token_list)
        normalized_string  → space-joined lemmas (for sentiment models)
        token_list         → list of lemma strings (for LDA)
    """
    if not isinstance(text, str) or not text.strip():
        return "", []

    doc = _NLP(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_ not in _ALL_STOPS
            and len(token.lemma_) >= MIN_TOKEN_LEN
            and token.is_alpha          # letters only — no leftover numbers/symbols
        )
    ]

    return " ".join(tokens), tokens


def normalize_dataframe(df: pd.DataFrame, text_col: str = "review_clean") -> pd.DataFrame:
    """
    Applies normalize_text() to every row using spaCy's pipe() for speed.
    Adds columns: review_normalized, tokens, token_count.
    Drops rows where normalization produced empty token lists.
    Returns updated DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Run cleaner.py first."
        )

    logger.info(f"Normalizing {len(df):,} reviews (spaCy pipe — this takes ~2–3 min)...")

    df = df.copy()
    texts = df[text_col].fillna("").tolist()

    normalized_strings = []
    token_lists        = []

    # spaCy pipe() processes in batches — much faster than row-by-row
    batch_size = 500
    for doc in tqdm(
        _NLP.pipe(
            (t.lower() for t in texts),
            batch_size=batch_size,
        ),
        total=len(texts),
        desc="Normalizing",
        unit="reviews",
    ):
        tokens = [
            token.lemma_
            for token in doc
            if (
                not token.is_stop
                and not token.is_punct
                and not token.is_space
                and token.lemma_ not in _ALL_STOPS
                and len(token.lemma_) >= MIN_TOKEN_LEN
                and token.is_alpha
            )
        ]
        normalized_strings.append(" ".join(tokens))
        token_lists.append(tokens)

    df["review_normalized"] = normalized_strings
    df["tokens"]            = token_lists
    df["token_count"]       = df["tokens"].apply(len)

    before = len(df)
    df = df[df["token_count"] >= 3].reset_index(drop=True)
    dropped = before - len(df)

    if dropped:
        logger.warning(f"Dropped {dropped:,} reviews with fewer than 3 tokens after normalization")

    logger.info(
        f"Normalization complete. {len(df):,} reviews retained. "
        f"Avg token count: {df['token_count'].mean():.1f}"
    )
    return df


def log_normalization_sample(df: pd.DataFrame, n: int = 3) -> None:
    """Logs before/after examples for visual verification."""
    logger.info("=== Normalization Sample (clean → normalized) ===")
    samples = df.sample(n=min(n, len(df)), random_state=42)
    for _, row in samples.iterrows():
        logger.info(f"  CLEAN     : {str(row['review_clean'])[:100]}")
        logger.info(f"  NORMALIZED: {str(row['review_normalized'])[:100]}")
        logger.info(f"  TOKENS    : {row['tokens'][:10]}")
        logger.info("  ---")


if __name__ == "__main__":
    from config import DATA_INTERIM
    df = pd.read_parquet(DATA_INTERIM / "reviews_raw.parquet")

    # Quick test on 100 rows
    sample = df.head(100).copy()
    from src.preprocessing.cleaner import clean_dataframe
    sample = clean_dataframe(sample)
    sample = normalize_dataframe(sample)
    log_normalization_sample(sample)
    print(f"\n✅ Normalizer OK on 100 rows. Columns: {sample.columns.tolist()}")
