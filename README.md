# Amazon Review Intelligence 📊

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://amazon-review-intelligence-nlp.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

An end-to-end NLP data science project that processes 21,737 Amazon Electronics reviews through
a complete 8-phase machine learning pipeline — from raw Kaggle data to a live interactive dashboard.

**→ [View the Live Dashboard](https://amazon-review-intelligence-nlp.streamlit.app)**

---

## What This Project Does

This project answers three practical questions about Amazon Electronics reviews:

- **What do customers feel?** — Dual sentiment modeling with VADER and DistilBERT, both evaluated against star ratings
- **What do customers talk about?** — LDA topic modeling discovers 8 recurring themes in the corpus
- **Where are the problems?** — A topic × sentiment heatmap surfaces the clusters with the most negative reviews

All model inference runs locally. The pre-computed results are stored in a Parquet file and served through
a Streamlit dashboard deployed on Streamlit Cloud, with no inference at runtime.

---

## Key Results

| Metric | Value |
|---|---|
| Reviews analyzed | 21,737 |
| Dataset date range | 2009 – 2019 |
| VADER accuracy vs. star ratings | **85.7%** |
| DistilBERT accuracy vs. star ratings | **79.3%** |
| VADER–DistilBERT agreement | **74.8%** |
| LDA topics discovered | **8** |
| LDA coherence score | **0.4121** |
| Dominant topic | Battery & Value (34.7% of reviews) |
| Highest-rated topic | Kindle E-Readers (avg 4.70★) |

**Notable finding:** VADER outperforms DistilBERT by 6.4 percentage points on this corpus.
Amazon reviews use direct, unambiguous sentiment language that a lexicon handles efficiently.
DistilBERT was fine-tuned on SST-2 (movie reviews) — a different domain — which is exactly
why domain-matched training data matters in real-world NLP.

---

## LDA Topics

| Label | Share of Reviews | Avg Rating |
|---|---|---|
| Battery & Value | 34.7% | 4.36★ |
| Gift Recommendations | 23.6% | 4.66★ |
| Kids Tablets | 10.2% | 4.56★ |
| Device Features | 10.1% | 4.13★ |
| Kindle E-Readers | 9.9% | **4.70★** |
| Apps & Entertainment | 6.1% | 4.33★ |
| Media & Camera | 3.4% | 4.54★ |
| Customer Service Issues | 2.0% | 4.12★ |

---

## Pipeline Architecture
Raw CSV (27,226 rows — Kaggle)
│
▼
Phase 1 — Project Scaffold
config.py, logger, directory structure
│
▼
Phase 2 — Data Ingestion
Kaggle API → reviews_raw.parquet (24,191 rows after deduplication)
│
▼
Phase 3 — NLP Preprocessing
spaCy en_core_web_sm — lemmatize, tokenize, normalize
→ reviews_processed.parquet
│
▼
Phase 4 — Sentiment Modeling
VADER (rule-based) + DistilBERT (transformer, lazy-loaded)
→ reviews_sentiment.parquet
│
▼
Phase 5 — LDA Topic Modeling
Gensim LDA, 8 topics, coherence 0.4121
→ reviews_topics.parquet  ← committed to repo
│
▼
Phase 6 — Plotly Visualizations
8 chart types exported as standalone HTML + base64 word clouds
│
▼
Phase 7 — Streamlit Dashboard
4 pages, dark theme, @st.cache_data, word cloud selector
│
▼
Phase 8 — Deployment
Streamlit Cloud, Python 3.11 pinned via runtime.txt
Lean requirements — no torch/transformers at serving time

---

## Dashboard Pages

**Overview** — Six KPI cards summarizing the full dataset, a sentiment donut chart, and
a rating distribution bar chart with a key findings callout.

**Sentiment Analysis** — Monthly sentiment trend over the 10-year dataset, VADER score
histogram, and a side-by-side model comparison (VADER vs. DistilBERT accuracy and agreement rate).

**Topic Explorer** — Review volume by topic, a topic × sentiment heatmap to identify complaint
hotspots, average rating per topic with 95% confidence intervals, and an interactive word cloud
for any of the 8 LDA topics.

**Review Search** — Filter 21,737 reviews by any combination of topic, sentiment label, star
rating, and keyword. Displays up to 500 results with live summary metrics (avg rating, % positive,
avg VADER score).

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data handling | pandas, pyarrow, numpy |
| NLP | spaCy (en_core_web_sm), NLTK |
| Sentiment modeling | vaderSentiment, DistilBERT (Hugging Face transformers + PyTorch) |
| Topic modeling | Gensim LDA |
| Visualization | Plotly, matplotlib, wordcloud |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Data source | Datafiniti Consumer Reviews of Amazon Products (Kaggle) |

---

## Running This Project Locally

### Prerequisites

- Python 3.11
- Git
- A Kaggle account with an API key at `~/.kaggle/kaggle.json`

### Setup

```bash
# Clone the repository
git clone https://github.com/shrutisurya108/Amazon-Review-Intelligence.git
cd Amazon-Review-Intelligence

# Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install pandas pyarrow numpy nltk spacy gensim vaderSentiment \
    transformers torch scikit-learn plotly streamlit \
    matplotlib wordcloud tqdm python-dotenv

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

### Option A — Skip the pipeline, run the dashboard now

The master dataset is committed to the repo. You can launch the dashboard immediately:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. All four pages will work out of the box.

### Option B — Run the full pipeline from scratch

```bash
# Phase 2 — Download data from Kaggle
python src/ingestion/downloader.py
python src/ingestion/loader.py

# Phase 3 — NLP preprocessing
python src/preprocessing/pipeline.py

# Phase 4 — Sentiment modeling (downloads DistilBERT ~268MB on first run)
python src/modeling/sentiment_pipeline.py

# Phase 5 — LDA topic modeling
python src/modeling/topic_pipeline.py

# Phase 6 — Export charts
python src/visualization/export_figures.py

# Phase 7 — Launch the dashboard
python -m streamlit run app.py
```

> Use `python -m streamlit run` rather than `streamlit run` to ensure the correct
> Python version is used, especially on systems where multiple Python versions are installed.

---

## Project Structure
Amazon-Review-Intelligence/
├── app.py                           # Streamlit entry point — 4-page dashboard
├── config.py                        # Centralized config: all paths and hyperparameters
├── requirements.txt                 # Cloud-optimized dependencies (no heavy ML)
├── runtime.txt                      # Python 3.11 pin for Streamlit Cloud
├── .python-version                  # Python 3.11 pin for pyenv
├── .streamlit/
│   └── config.toml                  # Dark theme configuration
├── data/
│   ├── interim/                     # reviews_raw.parquet (gitignored)
│   └── processed/
│       └── reviews_topics.parquet   # Master dataset — committed to repo
├── src/
│   ├── ingestion/                   # Phase 2: Kaggle download + data loader
│   ├── preprocessing/               # Phase 3: spaCy cleaner + normalizer + pipeline
│   ├── modeling/                    # Phase 4–5: VADER, DistilBERT, LDA
│   ├── visualization/               # Phase 6: Plotly charts + dashboard data prep
│   └── streamlit_utils.py           # Cached loaders, KPI helpers, filter logic
├── outputs/
│   └── figures/                     # Exported HTML charts + word cloud images
├── notebooks/                       # Exploratory analysis notebooks
├── logs/                            # Pipeline execution logs
└── tests/                           # Unit tests

---

## Deployment Notes

The Streamlit Cloud deployment uses a lean `requirements.txt` that intentionally excludes
`torch`, `transformers`, `spaCy`, and `scikit-learn`. All sentiment scores and topic labels
are pre-computed locally and stored in the committed Parquet file. The cloud app reads from
this file directly — no model inference runs at serving time. This keeps the deployment
straightforward and cold start under 15 seconds.

Python version is pinned via `runtime.txt` for Streamlit Cloud and `.python-version` for
local pyenv users.

---

## Collaboration and Acknowledgement
This project was built and developed in collaboration with [Harshith Bhattaram](https://github.com/maniharshith68).


## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)
