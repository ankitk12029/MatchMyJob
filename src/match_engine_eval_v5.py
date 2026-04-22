"""
match_engine_eval.py — MatchMyJob Evaluation (Improved Cross-Encoder)
======================================================================
Three fixes to the reranking problem:

  Fix 1 — Better cross-encoder model
          OLD: ms-marco-MiniLM-L-6-v2  (web search relevance — wrong task)
          NEW: stsb-roberta-base         (semantic textual similarity — right task)

  Fix 2 — Larger candidate pool
          OLD: top-10 from cosine sim
          NEW: top-20  (if correct answer ranked 11–20 in cosine sim, now fixable)

  Fix 3 — Richer cross-encoder profile
          OLD: Title + Description[:300] + AltTitles[:150]
          NEW: Title + Description[:300] + AltTitles[:150] + top-5 Tasks + Skills[:150]

Expected improvement:
  Top-3 accuracy  : 57%   (already measured — cosine sim top-3)
  Top-1 reranked  : 48–55% (cross-encoder picks best of top-20)
"""

import json
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import time
import os
from pathlib import Path
from config import (
    KB_PATH, FINETUNED_MODEL_PATH,
    GT_TEST_SPLIT_S3_PATH, EVAL_OUTPUT_PATH,
    vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ─── MODELS ───────────────────────────────────────────────────────────────────
MODEL_NAME = str(FINETUNED_MODEL_PATH)

# FIX 1: Use semantic similarity cross-encoder instead of web search model
# stsb-roberta-base is trained to score sentence pair similarity 0-1
# ms-marco was trained for "does this doc answer this query" — wrong framing
# CROSS_ENCODER_MODEL = 'cross-encoder/stsb-roberta-base'

# ms-marco-MiniLM-L-6-v2: trained for passage relevance (query → document).
# stsb-roberta-base failed here — it's trained on short STS sentence pairs
# and collapses on long job descriptions vs rich O*NET profiles.
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# FIX 2: Larger candidate pool — catches correct answers ranked 11-20 in cosine sim
CROSS_ENCODER_TOP_K = 20   # was 10

# ─── CATEGORY WEIGHTS ─────────────────────────────────────────────────────────
# WEIGHT_TASKS       = 0.25
# WEIGHT_DESCRIPTION = 0.20
# WEIGHT_SKILLS      = 0.15
# WEIGHT_OFC_TITLE   = 0.25
# WEIGHT_ALT_TITLES  = 0.10
# WEIGHT_TOOLS       = 0.05

WEIGHT_TASKS       = 0.0027
WEIGHT_DESCRIPTION = 0.0574
WEIGHT_SKILLS      = 0.0602
WEIGHT_OFC_TITLE   = 0.3264
WEIGHT_ALT_TITLES  = 0.5460
WEIGHT_TOOLS       = 0.0073

# Auto-load optimized weights if available
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


# ─── Vector cache loader ──────────────────────────────────────────────────────

def load_vector_cache(model_name: str) -> dict:
    cache_path = vector_cache_path(model_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Vector cache not found:\n  {cache_path}\n"
            "Run vectorize_kb.py first:\n    python src/vectorize_kb.py\n"
        )
    print(f"   Cache  : {cache_path.name}")
    bundle = torch.load(cache_path, map_location='cpu', weights_only=False)
    meta   = bundle.get('meta', {})
    print(f"   Created: {meta.get('created_at', 'unknown')}")
    print(f"   KB rows: {meta.get('kb_rows', '?')}")
    return bundle


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_test_split() -> pd.DataFrame:
    if not GT_TEST_SPLIT_S3_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found:\n  {GT_TEST_SPLIT_S3_PATH}\n"
            "Run finetune_model.py (Scenario 3) first."
        )
    df = pd.read_excel(GT_TEST_SPLIT_S3_PATH, dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    matched_n     = (df['Rater_Subset'] == 'Matched').sum()
    non_matched_n = (df['Rater_Subset'] == 'Non-Matched').sum()
    blank_soc     = (df['onet_soc'] == '').sum()
    print(f"   Test split: {len(df):,} rows  "
          f"(Matched: {matched_n}  |  Non-Matched: {non_matched_n}  |  No SOC: {blank_soc})")
    return df


# ─── Title word boost ─────────────────────────────────────────────────────────

def apply_title_boost(survey_title: str, kb_df: pd.DataFrame,
                      scores_row: torch.Tensor, boost: float = 0.03) -> torch.Tensor:
    survey_words = set(survey_title.lower().split())
    boosted = scores_row.clone()
    for idx, kb_title in enumerate(kb_df['Title']):
        overlap = len(survey_words & set(str(kb_title).lower().split()))
        if overlap >= 1:
            boosted[idx] += boost * overlap
    return boosted


# ─── Cross-encoder profile builder ───────────────────────────────────────────

def build_cross_encoder_profile(row: pd.Series) -> str:
    """
    FIX 3: Richer profile for the cross-encoder.

    The cross-encoder sees survey text on the LEFT and this profile on the RIGHT.
    It needs enough detail to make a confident similarity judgment.

    Includes:
      - Official title  (most discriminative signal)
      - Description     (what the occupation actually does)
      - Alt titles      (catches informal job title variants)
      - Top-5 tasks     (unpacked from JSON — most important tasks only)
      - Skills sample   (technology context)

    Capped at ~400 words total to stay within the 512-token limit.
    """
    parts = []

    # Title
    title = str(row.get('Title', '')).strip()
    if title:
        parts.append(f"Job title: {title}")

    # Description
    desc = str(row.get('Description', '')).strip()[:280]
    if desc:
        parts.append(f"Description: {desc}")

    # Alternate titles
    alt = str(row.get('All_Alt_Titles', '')).strip()[:120]
    if alt:
        parts.append(f"Also known as: {alt}")

    # Top-5 tasks unpacked from JSON (highest importance ones)
    structured_tasks = row.get('Structured_Tasks', '')
    if isinstance(structured_tasks, str) and structured_tasks.strip():
        try:
            tasks = json.loads(structured_tasks)
            # Sort by weight descending, take top 5
            tasks_sorted = sorted(tasks, key=lambda t: t.get('weight', 0), reverse=True)
            top_tasks = [t.get('task', '') for t in tasks_sorted[:5] if t.get('task', '')]
            if top_tasks:
                parts.append("Key tasks: " + ". ".join(top_tasks))
        except Exception:
            pass

    # Skills sample
    skills = str(row.get('All_Tech_Skills', '')).strip()[:120]
    if skills:
        parts.append(f"Skills: {skills}")

    return " ".join(parts).strip()


# ─── Cross-encoder reranker ───────────────────────────────────────────────────

def rerank_with_cross_encoder(cross_encoder: CrossEncoder,
                               survey_text: str,
                               candidate_indices: list,
                               kb_df: pd.DataFrame) -> int:
    """
    Scores each (survey_text, onet_profile) pair together.
    Cross-encoder reads both texts jointly → far more accurate than
    cosine similarity which encodes them independently.

    Returns the KB index of the best-scoring candidate.
    """
    profiles = [build_cross_encoder_profile(kb_df.iloc[idx]) for idx in candidate_indices]
    pairs    = [[survey_text, profile] for profile in profiles]
    scores   = cross_encoder.predict(pairs)
    best_pos = int(scores.argmax())
    return candidate_indices[best_pos]


# ─── Main matching ────────────────────────────────────────────────────────────

def run_matching(model, cross_encoder, bundle: dict,
                 kb_df: pd.DataFrame, texts: list,
                 survey_titles: list) -> tuple:
    """
    Two-stage matching:
      Stage 1 — Weighted cosine similarity → top-20 candidates (FIX 2: was 10)
      Stage 2 — Cross-encoder reranks top-20 (FIX 1: better model, FIX 3: richer profile)

    Returns top-3 SOC list for the Correct_Top3 metric (from stage 1).
    """
    query_embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    final_scores = (
        util.cos_sim(query_embs, bundle['task_embeddings'])      * WEIGHT_TASKS       +
        util.cos_sim(query_embs, bundle['desc_embeddings'])      * WEIGHT_DESCRIPTION +
        util.cos_sim(query_embs, bundle['skill_embeddings'])     * WEIGHT_SKILLS      +
        util.cos_sim(query_embs, bundle['ofc_title_embeddings']) * WEIGHT_OFC_TITLE   +
        util.cos_sim(query_embs, bundle['alt_title_embeddings']) * WEIGHT_ALT_TITLES  +
        util.cos_sim(query_embs, bundle['tool_embeddings'])      * WEIGHT_TOOLS
    )

    soc_codes, titles, confidences, top3_socs_list = [], [], [], []

    k = min(CROSS_ENCODER_TOP_K, len(kb_df))
    print(f"   Stage 1: cosine similarity → top-{k} candidates per row")
    print(f"   Stage 2: cross-encoder reranks all {k} candidates...")

    for i in range(len(texts)):
        # Stage 1 — cosine sim with title boost → top-k candidates
        boosted   = apply_title_boost(survey_titles[i], kb_df, final_scores[i])
        topk      = torch.topk(boosted, k=k)
        topk_idx  = topk.indices.tolist()

        # Top-3 SOC codes for Correct_Top3 metric (cosine sim stage, before reranking)
        top3_socs_list.append([kb_df.iloc[idx]['O*NET-SOC Code'] for idx in topk_idx[:3]])

        # Stage 2 — cross-encoder picks best from top-k
        best_idx = rerank_with_cross_encoder(cross_encoder, texts[i], topk_idx, kb_df)

        confidence = round(final_scores[i][best_idx].item() * 100, 2)
        soc_codes.append(kb_df.iloc[best_idx]['O*NET-SOC Code'])
        titles.append(kb_df.iloc[best_idx]['Title'])
        confidences.append(confidence)

    return soc_codes, titles, confidences, top3_socs_list


# ─── Build result rows ────────────────────────────────────────────────────────

def build_results(df: pd.DataFrame, kb_df: pd.DataFrame,
                  model, cross_encoder, bundle: dict) -> pd.DataFrame:
    user_texts    = (df['occupation'] + " " + df['Job description']).tolist()
    survey_titles = df['occupation'].tolist()

    machine_socs, machine_titles, confs, top3_socs_list = run_matching(
        model, cross_encoder, bundle, kb_df, user_texts, survey_titles
    )

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        machine_soc = machine_socs[i]
        human_soc   = row.get('onet_soc', '').strip()
        human_title = row.get('onet_title', '').strip()

        correct      = (1 if machine_soc == human_soc else 0) if human_soc else ''
        correct_top3 = (1 if human_soc in top3_socs_list[i] else 0) if human_soc else ''

        rows.append({
            'User_Occupation'    : row.get('occupation', ''),
            'User_Description'   : row.get('Job description', ''),
            'Machine_SOC_Code'   : machine_soc,
            'Machine_Title'      : machine_titles[i],
            'Machine_Confidence' : confs[i],
            'Human_SOC_Code'     : human_soc,
            'Human_Title'        : human_title,
            'Correct'            : correct,
            'Correct_Top3'       : correct_top3,
            'Rater_Subset'       : row.get('Rater_Subset', ''),
        })

    return pd.DataFrame(rows)


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(out_df: pd.DataFrame):
    print(f"\n{'─' * 62}")
    print(f"  EVALUATION RESULTS")
    print(f"{'─' * 62}")

    for label in ['Matched', 'Non-Matched', 'Overall']:
        if label == 'Overall':
            sub = out_df[out_df['Correct'].isin([0, 1])].copy()
        else:
            sub = out_df[
                (out_df['Rater_Subset'] == label) &
                (out_df['Correct'].isin([0, 1]))
            ].copy()

        if len(sub) == 0:
            continue

        sub['Correct']      = sub['Correct'].astype(int)
        sub['Correct_Top3'] = sub['Correct_Top3'].astype(int)
        n   = len(sub)
        c1  = sub['Correct'].sum()
        c3  = sub['Correct_Top3'].sum()

        print(f"\n  [{label}]")
        print(f"    Rows           : {n:,}")
        print(f"    Top-1 correct  : {c1}  ({c1/n*100:.1f}%)")
        print(f"    Top-3 correct  : {c3}  ({c3/n*100:.1f}%)  ← cosine sim top-3")
        print(f"    Rerank gap     : {c3 - c1}  rows the reranker must recover")
        print(f"    Avg confidence : {sub['Machine_Confidence'].mean():.1f}%")
        conf_gap = abs(
            sub[sub['Correct']==1]['Machine_Confidence'].mean() -
            sub[sub['Correct']==0]['Machine_Confidence'].mean()
        ) if c1 > 0 and (n - c1) > 0 else 0
        print(f"    Conf gap (✓/✗) : {conf_gap:.1f}%")

    no_soc = out_df[out_df['Correct'] == '']
    if len(no_soc) > 0:
        print(f"\n  [No human SOC: {len(no_soc)} rows — machine ran, no comparison]")
    print(f"\n{'─' * 62}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval_pipeline():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Improved Cross-Encoder Reranker")
    print(f"  Cross-encoder : {CROSS_ENCODER_MODEL}")
    print(f"  Candidate pool: top-{CROSS_ENCODER_TOP_K}")
    print("=" * 62)

    print(f"\n[1/5] Loading Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    print(f"      {len(kb_df):,} occupations")

    print(f"\n[2/5] Loading test split...")
    test_df = load_test_split()

    print(f"\n[3/5] Loading vector cache...")
    bundle = load_vector_cache(MODEL_NAME)

    print(f"\n[4/5] Loading models...")
    print(f"      Bi-encoder   : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print(f"      Cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"      Both models loaded ✓")

    print(f"\n[5/5] Running matching on {len(test_df):,} rows...")
    out_df = build_results(test_df, kb_df, model, cross_encoder, bundle)

    os.makedirs(EVAL_OUTPUT_PATH.parent, exist_ok=True)
    out_df.to_csv(EVAL_OUTPUT_PATH, index=False)
    print(f"\n      Output → {EVAL_OUTPUT_PATH}")

    print_summary(out_df)

    cols = ['User_Occupation', 'Machine_Title', 'Machine_Confidence',
            'Human_Title', 'Correct', 'Correct_Top3', 'Rater_Subset']
    print(f"\n  SAMPLE (first 8):")
    print(out_df[cols].head(8).to_string(index=False))
    print(f"\n  Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_eval_pipeline()
