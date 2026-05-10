# MatchMyJob

**Automatically map free-text job titles to standardized O\*NET occupation codes.**

MatchMyJob takes a plain-English job title (and optional description) and instantly finds the best matching occupation from the U.S. [O\*NET database](https://www.onetonline.org/) — a government-maintained list of 1,016 standardized job categories with official SOC codes.

---

## Features

- **Single lookup** — type a job title and get the top 3 occupation matches with confidence scores
- **Batch processing** — upload a CSV or Excel file and match thousands of rows in seconds
- **Confidence scoring** — each match comes with a percentage score (High ≥65% / Moderate 50–64% / Low <50%)
- **Interactive charts** — confidence distribution and occupation group breakdown
- **Downloadable results** — export matched data as CSV

---

## How It Works

1. Your job title (and optional description) is converted into a numerical representation using a fine-tuned language model
2. The model compares it against all 1,016 O\*NET occupations across 6 fields — title, alternate titles, description, tasks, skills, and tools
3. Results are ranked by weighted similarity and returned instantly with the top 3 candidates

---

## Accuracy

| Metric | Score |
|---|---|
| Top-1 accuracy | 43% |
| Top-3 accuracy | 62% |
| Knowledge base | 1,016 O\*NET occupations |

> Human inter-rater reliability on the same task is ~56–61%, making these results competitive with manual coding.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Input Format

For batch uploads, your CSV or Excel file needs at minimum a **Job Title** column. A **Job Description** column is optional but improves accuracy.

| Job Title | Job Description |
|---|---|
| Software Developer | Builds backend APIs, manages databases, deploys to cloud |
| Financial Analyst | Analyzes investment portfolios, prepares financial reports |
| Registered Nurse | Provides patient care, administers medications |

Column names are flexible — you map them after uploading.

---

## Project Structure

```
matchmyjob/
├── app.py                          # Streamlit web app
├── requirements.txt
├── src/
│   ├── config.py                   # Paths and constants
│   ├── master_data_creation.py     # Builds O*NET knowledge base
│   ├── finetune_model.py           # Fine-tunes the embedding model
│   ├── vectorize_kb.py             # Pre-computes KB embeddings
│   ├── optimize_weights.py         # Optimizes field weights
│   └── match_engine_eval.py        # Evaluation script
├── data/
│   ├── raw/                        # Source O*NET files and training data
│   ├── processed/                  # onet_knowledge_base.csv, optimal_weights.csv
│   └── output/                     # Evaluation results
├── models/
│   └── matchmyjob-finetuned/       # Fine-tuned sentence-transformer model
└── vectors/
    └── kb_vectors_*.pt             # Pre-computed KB embedding cache
```

---

## Tech Stack

| Component | Library |
|---|---|
| Web app | Streamlit |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Deep learning | PyTorch |
| Data | pandas, numpy |
| Charts | Plotly |

---

## License

© 2026 Ankit Katre. All rights reserved.
