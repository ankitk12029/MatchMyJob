"""
match_engine_eval.py — MatchMyJob Evaluation Pipeline
=======================================================
Loads pre-computed KB vectors from cache (built by vectorize_kb.py)
and runs the matching pipeline against the 15% held-out test split.

Output columns:
    Respondent_ID      → original survey ID
    User_Occupation    → what the respondent typed
    User_Description   → respondent's job description
    Machine_SOC_Code   → top SOC code predicted by the model
    Machine_Title      → corresponding O*NET title
    Machine_Confidence → weighted cosine similarity score (0–100%)
    Human_SOC_Code     → gold standard SOC code from human raters
    Human_Title        → corresponding O*NET title
    Correct            → 1 if exact SOC match, 0 otherwise

Run order:
    1. master_data_creation.py  → KB
    2. finetune_model.py        → saves ground_truth_test_split.xlsx
    3. vectorize_kb.py          → saves vector cache
    4. THIS FILE                → eval on test split
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import time
import os
from pathlib import Path
from config import (
    KB_PATH, FINETUNED_MODEL_PATH, GT_TEST_SPLIT_PATH,
    EVAL_OUTPUT_PATH, vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ─── MODEL ────────────────────────────────────────────────────────────────────
MODEL_NAME = str(FINETUNED_MODEL_PATH)   # switch to 'all-MiniLM-L6-v2' for baseline

# ─── CATEGORY WEIGHTS (must match match_engine.py exactly) ────────────────────
WEIGHT_TASKS       = 0.45
WEIGHT_DESCRIPTION = 0.20
WEIGHT_SKILLS      = 0.15
WEIGHT_OFC_TITLE   = 0.10
WEIGHT_ALT_TITLES  = 0.05
WEIGHT_TOOLS       = 0.05


# ─── Vector cache loader (identical to match_engine.py) ──────────────────────

def load_vector_cache(model_name: str) -> dict:
    cache_path = vector_cache_path(model_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Vector cache not found at:\n  {cache_path}\n\n"
            "Run vectorize_kb.py first:\n"
            "    python src/vectorize_kb.py\n"
        )
    print(f"   Loading vector cache: {cache_path.name}")
    bundle = torch.load(cache_path, map_location='cpu', weights_only=False)
    meta   = bundle.get('meta', {})
    print(f"   Cache created : {meta.get('created_at', 'unknown')}")
    print(f"   KB rows       : {meta.get('kb_rows', '?')}")
    return bundle


# ─── Main evaluation pipeline ─────────────────────────────────────────────────

def run_eval_pipeline():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Evaluation  (Machine vs Human)")
    print("=" * 62)

    # ── 1. Load KB ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    print(f"      {len(kb_df):,} occupations")

    # ── 2. Load test split ────────────────────────────────────────────────────
    print(f"\n[2/4] Loading test split...")
    if not GT_TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found at:\n  {GT_TEST_SPLIT_PATH}\n"
            "Run finetune_model.py first — it saves this file automatically."
        )
    test_df = pd.read_excel(GT_TEST_SPLIT_PATH, dtype=str).fillna('')
    for col in test_df.columns:
        test_df[col] = test_df[col].str.strip()
    print(f"      {len(test_df):,} test rows")

    # ── 3. Load vector cache ──────────────────────────────────────────────────
    print(f"\n[3/4] Loading KB vector cache...")
    bundle               = load_vector_cache(MODEL_NAME)
    task_embeddings      = bundle['task_embeddings']
    skill_embeddings     = bundle['skill_embeddings']
    tool_embeddings      = bundle['tool_embeddings']
    alt_title_embeddings = bundle['alt_title_embeddings']
    ofc_title_embeddings = bundle['ofc_title_embeddings']
    desc_embeddings      = bundle['desc_embeddings']
    print(f"      All 6 KB vector sets loaded from cache ✓")

    # Encode user queries
    print(f"\n      Encoding {len(test_df):,} test queries...")
    model        = SentenceTransformer(MODEL_NAME, device='cpu')
    user_texts   = (test_df['occupation'] + " " + test_df['Job description']).tolist()
    query_embs   = model.encode(user_texts, convert_to_tensor=True, show_progress_bar=True)

    # ── 4. Score and build output ─────────────────────────────────────────────
    print(f"\n[4/4] Calculating weighted similarity scores...")
    task_scores      = util.cos_sim(query_embs, task_embeddings)
    skill_scores     = util.cos_sim(query_embs, skill_embeddings)
    tool_scores      = util.cos_sim(query_embs, tool_embeddings)
    alt_title_scores = util.cos_sim(query_embs, alt_title_embeddings)
    ofc_title_scores = util.cos_sim(query_embs, ofc_title_embeddings)
    desc_scores      = util.cos_sim(query_embs, desc_embeddings)

    final_scores = (
        (task_scores      * WEIGHT_TASKS)       +
        (desc_scores      * WEIGHT_DESCRIPTION) +
        (skill_scores     * WEIGHT_SKILLS)      +
        (ofc_title_scores * WEIGHT_OFC_TITLE)   +
        (alt_title_scores * WEIGHT_ALT_TITLES)  +
        (tool_scores      * WEIGHT_TOOLS)
    )

    results = []
    for i in range(len(test_df)):
        top1       = torch.argmax(final_scores[i]).item()
        confidence = round(final_scores[i][top1].item() * 100, 2)

        results.append({
            'Respondent_ID'      : test_df.iloc[i].get('ID', ''),
            'User_Occupation'    : test_df.iloc[i]['occupation'],
            'User_Description'   : test_df.iloc[i]['Job description'],
            'Machine_SOC_Code'   : kb_df.iloc[top1]['O*NET-SOC Code'],
            'Machine_Title'      : kb_df.iloc[top1]['Title'],
            'Machine_Confidence' : confidence,
            'Human_SOC_Code'     : test_df.iloc[i]['onet_soc'],
            'Human_Title'        : test_df.iloc[i]['onet_title'],
            'Correct'            : 1 if kb_df.iloc[top1]['O*NET-SOC Code'].strip()
                                       == test_df.iloc[i]['onet_soc'].strip() else 0,
        })

    out_df = pd.DataFrame(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_correct = out_df['Correct'].sum()
    n_total   = len(out_df)
    print(f"\n{'─' * 62}")
    print(f"  EVALUATION RESULTS")
    print(f"{'─' * 62}")
    print(f"  Total test rows          : {n_total}")
    print(f"  Correct (exact SOC)      : {n_correct}  ({n_correct/n_total*100:.1f}%)")
    print(f"  Avg confidence (all)     : {out_df['Machine_Confidence'].mean():.1f}%")
    print(f"  Avg confidence (correct) : {out_df[out_df['Correct']==1]['Machine_Confidence'].mean():.1f}%")
    print(f"  Avg confidence (wrong)   : {out_df[out_df['Correct']==0]['Machine_Confidence'].mean():.1f}%")
    print(f"{'─' * 62}")

    os.makedirs(EVAL_OUTPUT_PATH.parent, exist_ok=True)
    out_df.to_csv(EVAL_OUTPUT_PATH, index=False)
    print(f"\n  Output saved → {EVAL_OUTPUT_PATH}")

    print(f"\n  SAMPLE (first 10 rows):")
    print(out_df[['User_Occupation', 'Machine_Title', 'Machine_Confidence',
                  'Human_Title', 'Correct']].head(10).to_string(index=False))

    print(f"\n  Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_eval_pipeline()
