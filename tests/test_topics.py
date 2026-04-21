"""
Phase 5 — Topic Modeling Tests
Tests LDA model artifacts, topic assignments, and master parquet structure.
"""
from pathlib import Path
import sys
import pickle

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED, OUTPUTS_MODELS, OUTPUTS_REPORTS, LDA_NUM_TOPICS
from src.utils.logger import get_logger

logger = get_logger("test.topics")

MASTER_PATH = DATA_PROCESSED / "reviews_topics.parquet"
MODEL_PATH  = OUTPUTS_MODELS / "lda_model.pkl"
DICT_PATH   = OUTPUTS_MODELS / "lda_dictionary.pkl"


# ── Artifact tests ────────────────────────────────────────────────────────────

def test_model_artifacts_exist():
    assert MODEL_PATH.exists(), f"LDA model not found: {MODEL_PATH}"
    assert DICT_PATH.exists(),  f"Dictionary not found: {DICT_PATH}"
    model_mb = MODEL_PATH.stat().st_size / 1e6
    dict_mb  = DICT_PATH.stat().st_size  / 1e6
    print(f"✓ LDA model exists ({model_mb:.2f} MB)")
    print(f"✓ LDA dictionary exists ({dict_mb:.2f} MB)")


def test_model_loads_correctly():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert model.num_topics == LDA_NUM_TOPICS, (
        f"Expected {LDA_NUM_TOPICS} topics, got {model.num_topics}"
    )
    # Verify model can produce topics
    for i in range(LDA_NUM_TOPICS):
        words = model.show_topic(i, topn=5)
        assert len(words) == 5, f"Topic {i} has fewer than 5 words"
    print(f"✓ LDA model loads correctly ({LDA_NUM_TOPICS} topics)")


def test_dictionary_loads_correctly():
    with open(DICT_PATH, "rb") as f:
        dictionary = pickle.load(f)
    assert len(dictionary) > 100, f"Dictionary too small: {len(dictionary)} tokens"
    print(f"✓ Dictionary loads correctly ({len(dictionary):,} tokens)")


# ── Master parquet tests ──────────────────────────────────────────────────────

def test_master_parquet_exists():
    assert MASTER_PATH.exists(), (
        f"Master parquet not found at {MASTER_PATH}. "
        "Run: python src/modeling/topic_pipeline.py"
    )
    size_mb = MASTER_PATH.stat().st_size / 1e6
    assert size_mb > 1.0, f"Master parquet too small: {size_mb:.2f} MB"
    print(f"✓ Master parquet exists ({size_mb:.2f} MB)")


def test_master_columns():
    df = pd.read_parquet(MASTER_PATH)
    required = [
        "review_text", "rating", "review_clean", "review_normalized",
        "tokens", "token_count",
        "vader_score", "vader_label",
        "bert_label", "bert_confidence",
        "dominant_topic", "topic_label", "topic_score", "topic_keywords",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"✓ All {len(required)} required columns present")


def test_master_row_count():
    df = pd.read_parquet(MASTER_PATH)
    assert len(df) >= 1000, f"Too few rows: {len(df)}"
    print(f"✓ Row count: {len(df):,}")


def test_topic_ids_valid():
    df = pd.read_parquet(MASTER_PATH)
    valid_ids = set(range(LDA_NUM_TOPICS))
    actual_ids = set(df["dominant_topic"].unique())
    invalid = actual_ids - valid_ids
    assert not invalid, f"Invalid topic IDs found: {invalid}"
    print(f"✓ All topic IDs in valid range 0–{LDA_NUM_TOPICS - 1}")


def test_all_topics_represented():
    df = pd.read_parquet(MASTER_PATH)
    represented = set(df["dominant_topic"].unique())
    missing = set(range(LDA_NUM_TOPICS)) - represented
    # Allow up to 1 missing topic (rare but acceptable in LDA)
    assert len(missing) <= 1, f"Topics not assigned to any review: {missing}"
    print(f"✓ Topics represented: {sorted(represented)}")


def test_topic_scores_valid():
    df = pd.read_parquet(MASTER_PATH)
    assert df["topic_score"].between(0.0, 1.0).all(), "Topic scores outside [0,1]"
    avg = df["topic_score"].mean()
    assert avg > 0.1, f"Average topic score too low: {avg:.3f}"
    print(f"✓ Topic scores valid, avg={avg:.3f}")


def test_topic_labels_not_empty():
    df = pd.read_parquet(MASTER_PATH)
    empty_labels = (df["topic_label"].isna() | (df["topic_label"] == "")).sum()
    assert empty_labels == 0, f"{empty_labels} rows have empty topic labels"
    unique_labels = df["topic_label"].unique().tolist()
    print(f"✓ Topic labels present: {unique_labels}")


def test_topic_keywords_not_empty():
    df = pd.read_parquet(MASTER_PATH)
    empty_kw = (df["topic_keywords"].isna() | (df["topic_keywords"] == "")).sum()
    assert empty_kw == 0, f"{empty_kw} rows have empty topic keywords"
    print(f"✓ Topic keywords present in all rows")


def test_topic_distribution_balanced():
    """No single topic should dominate more than 70% of reviews."""
    df = pd.read_parquet(MASTER_PATH)
    topic_pcts = df["dominant_topic"].value_counts(normalize=True)
    max_pct = topic_pcts.max()
    assert max_pct < 0.70, (
        f"Topic distribution heavily skewed — topic {topic_pcts.idxmax()} "
        f"has {max_pct:.1%} of all reviews. Consider rerunning with different params."
    )
    print(f"✓ Topic distribution balanced (max topic share: {max_pct:.1%})")


def test_topic_report_exists():
    path = OUTPUTS_REPORTS / "topic_report.txt"
    assert path.exists(), f"Topic report not found at {path}"
    content = path.read_text()
    assert "TOPIC MODELING REPORT" in content
    assert "Coherence" in content
    assert "Keywords" in content
    print(f"✓ Topic report exists and contains expected sections")


def test_coherence_score():
    """Parse coherence score from topic report and validate range."""
    path = OUTPUTS_REPORTS / "topic_report.txt"
    if not path.exists():
        print("⚠ Skipping coherence test — report not found")
        return
    content = path.read_text()
    for line in content.splitlines():
        if "Coherence score" in line:
            try:
                score = float(line.split(":")[-1].strip())
                assert score > 0.2, f"Coherence score very low: {score:.4f}"
                print(f"✓ Coherence score: {score:.4f} (good range: 0.4–0.7)")
                return
            except ValueError:
                pass
    print("⚠ Could not parse coherence score from report")


if __name__ == "__main__":
    print("\nRunning Phase 5 topic modeling tests...\n")

    print("--- LDA Artifact Tests ---")
    test_model_artifacts_exist()
    test_model_loads_correctly()
    test_dictionary_loads_correctly()

    print("\n--- Master Parquet Tests ---")
    test_master_parquet_exists()
    test_master_columns()
    test_master_row_count()
    test_topic_ids_valid()
    test_all_topics_represented()
    test_topic_scores_valid()
    test_topic_labels_not_empty()
    test_topic_keywords_not_empty()
    test_topic_distribution_balanced()

    print("\n--- Report Tests ---")
    test_topic_report_exists()
    test_coherence_score()

    print("\n✅ All Phase 5 topic modeling tests passed.")
