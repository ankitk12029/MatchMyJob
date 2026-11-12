# MatchMyJob

[![CI](https://github.com/ankitk12029/MatchMyJob/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitk12029/MatchMyJob/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24111664.svg)](https://doi.org/10.5281/zenodo.24111664)

**Automatically map free-text job titles to standardized O\*NET occupation codes.**

MatchMyJob takes a plain-English job title (and optional description) and instantly finds the best matching occupation from the U.S. [O\*NET database](https://www.onetonline.org/) — a government-maintained list of 1,016 standardized job categories with official SOC codes.

---

## Features

- **Single lookup** — type a job title and get the top 3 occupation matches with confidence scores
- **Batch processing** — upload a CSV or Excel file and match thousands of rows in seconds
- **Confidence scoring** — each match comes with a percentage score (High ≥65% / Moderate 50–64.9% / Low <50%)
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
| Top-1 accuracy | 41.8% (276 / 661) |
| Top-3 accuracy | 60.1% (397 / 661) |
| Major-group (2-digit SOC) accuracy | 73.4% (485 / 661) |
| Knowledge base | 1,016 O\*NET occupations |

> Measured on a held-out test split of 661 human-coded job records; see `data/output/eval_summary.md` for the protocol and `data/output/eval_machine_vs_human.csv` for per-row predictions.

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
│   ├── scoring.py                  # Shared scoring: BM25 + title boost, confidence tiers
│   ├── match_engine.py             # Batch matching of a CSV of job titles
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

## Data Availability

The six O*NET source files, the knowledge base, the learned field weights, the fine-tuned model, the vector caches, the per-row evaluation output and a test dataset for running the batch workflow (`data/test/heldout_test_input.csv`: 292 held-out records whose job titles never appear in the training data, with `Expected_SOC` and `Expected_Title` labels) are in this repository. The de-identified training data (`data/train/training_data.csv`) are also included; see [Training data](#training-data) below. The original spreadsheets with the full, unredacted records are not redistributed. The held-out test split is the 20% of the primary dataset that fine-tuning set aside (661 records; the 292-record subset above is published), so the training and test records do not overlap.

---

## Training data

`data/train/training_data.csv` contains the 5,259 records used to fine-tune the embedding model: 2,640 survey responses with a free-text job title and job description, and 2,619 title-only records (empty description). Each record has a human-assigned O*NET-SOC code.

| Column | Content |
|---|---|
| `Job Title` | Respondent's free-text job title |
| `Job Description` | Respondent's free-text description of their duties (empty for title-only records) |
| `O*NET-SOC Code` | O*NET-SOC 2019 code assigned by a human coder |

**Source.** Free-text job titles and descriptions from survey respondents, matched to O\*NET-SOC codes by human raters. This is the training set that `src/finetune_model.py` builds from `master_training_data.xlsx` (primary records, 80% after the seeded split) and `training_data_without_descrption.xlsx` (title-only records, two or more words). The held-out 20% split is excluded.

**De-identification steps.**
1. Kept only the three columns above. The source sheets hold nothing else per record except the O\*NET occupation title, which was dropped as redundant with the code.
2. Replaced e-mail addresses, web addresses and domain names, phone numbers, SSN-like numbers and `@` handles with `[REDACTED]`. Only two domain names were found, both in one record.
3. Replaced employer, brand and institution names that a manual review of the text found with `[ORG]`, and one personal first name with `[NAME]`. This covered 46 records, for example fast-food chains, retailers, a city government, and a few named agencies and research groups. Widely known regulators and software products named as part of the duties (for example FDA, IRS, Microsoft Word) were kept.
4. Removed one record whose description was a political statement rather than a job description.
5. Spot-checked a random sample of rows and a scan of every unusual capitalised word in the text.

This is a best-effort process, not a guarantee. The text is free-form, so a missed identifier is possible; please report one by opening an issue.

**Data notes.**
- Text is otherwise unchanged, including the respondents' spelling errors.
- About 146 title-only records have the letters "na" missing from a word (for example "Mager" for "Manager", "Alyst" for "Analyst"). This is damage in the source spreadsheet and is left as-is, because it is what the model trained on.
- Two O\*NET-SOC codes carried a stray `+I544` suffix in the source and were cleaned to `31-1014.00`.

---

## Citation

If you use MatchMyJob, please cite it using [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).

---

## License

Released under the [MIT License](LICENSE).
