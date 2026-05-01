"""
match_engine_eval.py — MatchMyJob Evaluation Pipeline
======================================================
Matching strategy (no cross-encoder, no LLM):

  Stage 1 — Weighted cosine similarity across 6 KB fields
            Weights auto-loaded from optimal_weights.csv if it exists.

  Stage 2 — BM25 hybrid re-scoring
            Pure keyword matching over O*NET title + alt-titles + description.
            Fixes the top-3 → top-1 gap: when two SOC codes are semantically
            close, exact keyword overlap (e.g. "Retail" → "Retail Salespersons")
            breaks the tie.
            Final score = cosine_score + BM25_WEIGHT * bm25_score (normalized).

  Stage 3 — Title word boost
            +0.03 per overlapping word between survey title and O*NET title.

Metrics reported:
  Top-1: machine's #1 prediction exactly matches human SOC code
  Top-3: correct SOC appears anywhere in the top-3 predictions
"""

import os
import re
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from config import (
    KB_PATH, FINETUNED_MODEL_PATH,
    GT_TEST_SPLIT_S3_PATH, EVAL_OUTPUT_PATH,
    vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option('display.max_colwidth', None)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME       = str(FINETUNED_MODEL_PATH)
BM25_WEIGHT      = 0.0    # BM25 hurts on this dataset — cosine sim alone is stronger
# Must match BGE_QUERY_PREFIX used in finetune_model.py — query side only, not KB profiles.
BGE_QUERY_PREFIX = "Represent this job for retrieval: "

# Default cosine weights (overridden by optimal_weights.csv if present)
# WEIGHT_TASKS       = 0.25
# WEIGHT_DESCRIPTION = 0.20
# WEIGHT_SKILLS      = 0.15
# WEIGHT_OFC_TITLE   = 0.25
# WEIGHT_ALT_TITLES  = 0.10
# WEIGHT_TOOLS       = 0.05

# WEIGHT_TASKS       = 0.1979
# WEIGHT_DESCRIPTION = 0.0869
# WEIGHT_SKILLS      = 0.0087
# WEIGHT_OFC_TITLE   = 0.3110
# WEIGHT_ALT_TITLES  = 0.3474
# WEIGHT_TOOLS       = 0.0480

# WEIGHT_TASKS       = 0.2702
# WEIGHT_DESCRIPTION = 0.1121
# WEIGHT_SKILLS      = 0.0052
# WEIGHT_OFC_TITLE   = 0.3206
# WEIGHT_ALT_TITLES  = 0.2707
# WEIGHT_TOOLS       = 0.0212

# WEIGHT_TASKS       = 0.2017
# WEIGHT_DESCRIPTION = 0.1021
# WEIGHT_SKILLS      = 0.0063
# WEIGHT_OFC_TITLE   = 0.3941
# WEIGHT_ALT_TITLES  = 0.2898
# WEIGHT_TOOLS       = 0.0061

WEIGHT_TASKS       = 0.0051
WEIGHT_DESCRIPTION = 0.0700
WEIGHT_SKILLS      = 0.0405
WEIGHT_OFC_TITLE   = 0.3212
WEIGHT_ALT_TITLES  = 0.5055
WEIGHT_TOOLS       = 0.0578
# Auto-load optimized weights
_weights_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'optimal_weights.csv'
if _weights_path.exists():
    _w = pd.read_csv(_weights_path).set_index('field')['weight']
    WEIGHT_TASKS       = float(_w.get('Tasks',       WEIGHT_TASKS))
    WEIGHT_DESCRIPTION = float(_w.get('Description', WEIGHT_DESCRIPTION))
    WEIGHT_SKILLS      = float(_w.get('Skills',      WEIGHT_SKILLS))
    WEIGHT_OFC_TITLE   = float(_w.get('Ofc_Title',   WEIGHT_OFC_TITLE))
    WEIGHT_ALT_TITLES  = float(_w.get('Alt_Titles',  WEIGHT_ALT_TITLES))
    WEIGHT_TOOLS       = float(_w.get('Tools',       WEIGHT_TOOLS))
    print(f"[INFO] Loaded optimized weights from {_weights_path.name}")


# ─── LOADERS ──────────────────────────────────────────────────────────────────

def load_vector_cache(model_name: str) -> dict:
    cache_path = vector_cache_path(model_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Vector cache not found: {cache_path}\n"
            "Run:  python src/vectorize_kb.py\n"
        )
    print(f"   Cache  : {cache_path.name}")
    bundle = torch.load(cache_path, map_location='cpu', weights_only=False)
    meta   = bundle.get('meta', {})
    print(f"   Created: {meta.get('created_at', 'unknown')}")
    print(f"   KB rows: {meta.get('kb_rows', '?')}")
    return bundle


def load_test_split() -> pd.DataFrame:
    if not GT_TEST_SPLIT_S3_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found: {GT_TEST_SPLIT_S3_PATH}\n"
            "Run:  python src/finetune_model.py\n"
        )
    df = pd.read_excel(GT_TEST_SPLIT_S3_PATH, dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Normalise column names — master_training_data uses different names
    if 'Job title' in df.columns:
        df = df.rename(columns={
            'Job title'       : 'occupation',
            'Job description' : 'Job description',
            'O*NET_code'      : 'onet_soc',
            'O*NET Job title' : 'onet_title',
        })

    # Fix swapped onet_soc / onet_title columns (affects ~25% of rows in source data)
    SOC_PATTERN = re.compile(r'^\d{2}-\d{4}\.\d{2}')
    is_swapped = ~df['onet_soc'].apply(lambda x: bool(SOC_PATTERN.match(str(x).strip())))
    if is_swapped.sum() > 0:
        df.loc[is_swapped, ['onet_soc', 'onet_title']] = (
            df.loc[is_swapped, ['onet_title', 'onet_soc']].values
        )
        print(f"   Fixed  : {is_swapped.sum()} rows with swapped onet_soc/onet_title.")

    # Drop rows still without a valid SOC after fix
    still_invalid = ~df['onet_soc'].apply(lambda x: bool(SOC_PATTERN.match(str(x).strip())))
    if still_invalid.sum():
        df = df[~still_invalid].reset_index(drop=True)
        print(f"   Dropped: {still_invalid.sum()} rows with no valid SOC code.")

    print(f"   Test split: {len(df):,} rows evaluable")
    return df


# ─── BM25 INDEX ───────────────────────────────────────────────────────────────

def build_bm25_index(kb_df: pd.DataFrame) -> BM25Okapi:
    """
    Tokenises O*NET title + alt-titles + description for each occupation.
    BM25 is a pure keyword scorer — fast, no model required.
    """
    corpus = []
    for _, row in kb_df.iterrows():
        doc = " ".join(filter(None, [
            str(row.get('Title', '')),
            str(row.get('All_Alt_Titles', ''))[:200],
            str(row.get('Description', ''))[:300],
        ]))
        corpus.append(doc.lower().split())
    return BM25Okapi(corpus)


def bm25_scores_for_query(bm25: BM25Okapi, query: str) -> np.ndarray:
    tokens = query.lower().split()
    scores = np.array(bm25.get_scores(tokens), dtype=np.float32)
    # Normalize to [0, 1] so it's on the same scale as cosine sim
    max_s  = scores.max()
    if max_s > 0:
        scores = scores / max_s
    return scores


# ─── TITLE BOOST ──────────────────────────────────────────────────────────────

def apply_title_boost(survey_title: str, kb_df: pd.DataFrame,
                      scores: np.ndarray, boost: float = 0.03) -> np.ndarray:
    survey_words = set(survey_title.lower().split())
    boosted = scores.copy()
    for idx, kb_title in enumerate(kb_df['Title']):
        overlap = len(survey_words & set(str(kb_title).lower().split()))
        if overlap >= 1:
            boosted[idx] += boost * overlap
    return boosted


# ─── MATCHING ─────────────────────────────────────────────────────────────────

def run_matching(model, bm25: BM25Okapi, bundle: dict,
                 kb_df: pd.DataFrame,
                 texts: list, survey_titles: list) -> tuple:

    prefixed = [BGE_QUERY_PREFIX + t for t in texts]
    query_embs = model.encode(prefixed, convert_to_tensor=True, show_progress_bar=True)

    # Stage 1: weighted cosine similarity (all 6 fields)
    cosine_scores = (
        util.cos_sim(query_embs, bundle['task_embeddings'])      * WEIGHT_TASKS       +
        util.cos_sim(query_embs, bundle['desc_embeddings'])      * WEIGHT_DESCRIPTION +
        util.cos_sim(query_embs, bundle['skill_embeddings'])     * WEIGHT_SKILLS      +
        util.cos_sim(query_embs, bundle['ofc_title_embeddings']) * WEIGHT_OFC_TITLE   +
        util.cos_sim(query_embs, bundle['alt_title_embeddings']) * WEIGHT_ALT_TITLES  +
        util.cos_sim(query_embs, bundle['tool_embeddings'])      * WEIGHT_TOOLS
    ).cpu().numpy()   # shape: (n_queries, n_kb)

    soc_codes, titles, confidences, top3_socs_list = [], [], [], []

    for i in range(len(texts)):
        # Stage 2: add BM25 hybrid score
        bm25_s  = bm25_scores_for_query(bm25, texts[i])
        final_s = cosine_scores[i] + BM25_WEIGHT * bm25_s

        # Stage 3: title word boost
        final_s = apply_title_boost(survey_titles[i], kb_df, final_s)

        top3_idx = np.argsort(final_s)[::-1][:3].tolist()
        top3_socs_list.append([kb_df.iloc[idx]['O*NET-SOC Code'] for idx in top3_idx])

        best_idx   = top3_idx[0]
        confidence = round(float(cosine_scores[i][best_idx]) * 100, 2)
        soc_codes.append(kb_df.iloc[best_idx]['O*NET-SOC Code'])
        titles.append(kb_df.iloc[best_idx]['Title'])
        confidences.append(confidence)

    return soc_codes, titles, confidences, top3_socs_list


# ─── BUILD RESULTS ────────────────────────────────────────────────────────────

def build_results(df: pd.DataFrame, kb_df: pd.DataFrame,
                  model, bm25: BM25Okapi, bundle: dict) -> pd.DataFrame:
    user_texts    = (df['occupation'] + " " + df['Job description']).tolist()
    survey_titles = df['occupation'].tolist()

    machine_socs, machine_titles, confs, top3_socs_list = run_matching(
        model, bm25, bundle, kb_df, user_texts, survey_titles
    )

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        machine_soc  = machine_socs[i]
        human_soc    = str(row.get('onet_soc', '')).strip()
        human_title  = str(row.get('onet_title', '')).strip()

        correct      = (1 if machine_soc == human_soc else 0) if human_soc else ''
        correct_top3 = (1 if human_soc in top3_socs_list[i] else 0) if human_soc else ''

        rows.append({
            'User_Occupation'   : row.get('occupation', ''),
            'User_Description'  : row.get('Job description', ''),
            'Machine_SOC_Code'  : machine_soc,
            'Machine_Title'     : machine_titles[i],
            'Machine_Confidence': confs[i],
            'Human_SOC_Code'    : human_soc,
            'Human_Title'       : human_title,
            'Correct'           : correct,
            'Correct_Top3'      : correct_top3,
        })

    return pd.DataFrame(rows)


# ─── SUMMARY ──────────────────────────────────────────────────────────────────

def print_summary(out_df: pd.DataFrame):
    sub = out_df[out_df['Correct'].isin([0, 1])].copy()
    sub['Correct']      = sub['Correct'].astype(int)
    sub['Correct_Top3'] = sub['Correct_Top3'].astype(int)

    n  = len(sub)
    c1 = sub['Correct'].sum()
    c3 = sub['Correct_Top3'].sum()

    conf_c = sub[sub['Correct'] == 1]['Machine_Confidence'].mean() if c1 > 0 else 0
    conf_w = sub[sub['Correct'] == 0]['Machine_Confidence'].mean() if (n - c1) > 0 else 0

    print(f"\n{'─' * 62}")
    print(f"  EVALUATION RESULTS")
    print(f"{'─' * 62}")
    print(f"  Rows evaluated  : {n:,}")
    print(f"  Top-1 correct   : {c1}  ({c1/n*100:.1f}%)")
    print(f"  Top-3 correct   : {c3}  ({c3/n*100:.1f}%)  ← correct SOC in top-3")
    print(f"  Top-3 gap       : {c3 - c1}  rows correct in top-3 but not top-1")
    print(f"  Avg confidence  : {sub['Machine_Confidence'].mean():.1f}%")
    print(f"  Conf gap (✓/✗)  : {abs(conf_c - conf_w):.1f}%")

    no_soc = out_df[out_df['Correct'] == '']
    if len(no_soc):
        print(f"  Rows w/o SOC    : {len(no_soc)}  (ran but no comparison)")
    print(f"{'─' * 62}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_eval_pipeline():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Evaluation")
    print("  cosine sim (6 fields) + BM25 hybrid + title boost")
    print(f"  BM25 weight: {BM25_WEIGHT}")
    print("=" * 62)

    print(f"\n[1/5] Loading Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    print(f"      {len(kb_df):,} occupations")

    print(f"\n[2/5] Loading test split...")
    test_df = load_test_split()

    print(f"\n[3/5] Loading vector cache...")
    bundle = load_vector_cache(MODEL_NAME)

    print(f"\n[4/5] Building BM25 index over KB titles + descriptions...")
    bm25 = build_bm25_index(kb_df)
    print(f"      BM25 index built over {len(kb_df):,} occupations")

    print(f"\n[5/5] Loading model and running matching on {len(test_df):,} rows...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    out_df = build_results(test_df, kb_df, model, bm25, bundle)

    os.makedirs(EVAL_OUTPUT_PATH.parent, exist_ok=True)
    out_df.to_csv(EVAL_OUTPUT_PATH, index=False)
    print(f"      Output → {EVAL_OUTPUT_PATH}")

    print_summary(out_df)

    cols = ['User_Occupation', 'Machine_Title', 'Machine_Confidence',
            'Human_Title', 'Correct', 'Correct_Top3']
    print(f"\n  SAMPLE (first 8 rows):")
    print(out_df[cols].head(8).to_string(index=False))
    print(f"\n  Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_eval_pipeline()
