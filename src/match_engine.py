"""
match_engine.py — MatchMyJob Production Matching Pipeline
==========================================================
Loads pre-computed KB vectors from cache (built by vectorize_kb.py)
instead of re-encoding on every run.

Run order:
    1. master_data_creation.py   → builds onet_knowledge_base.csv
    2. finetune_model.py         → trains the model
    3. vectorize_kb.py           → encodes KB and saves vectors to disk
    4. THIS FILE                 → loads cache, matches user input, saves results
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import textwrap
import time
import os
import json
from pathlib import Path
from config import (
    KB_PATH, MODELS_DIR, FINETUNED_MODEL_PATH,
    USER_INPUT_FILE, MATCH_OUTPUT_PATH, vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ─── MODEL ────────────────────────────────────────────────────────────────────
MODEL_NAME = str(FINETUNED_MODEL_PATH)   # switch to 'all-MiniLM-L6-v2' for baseline

# ─── CATEGORY WEIGHTS (must add up to 1.0) ────────────────────────────────────
# WEIGHT_TASKS       = 0.45   
# WEIGHT_DESCRIPTION = 0.20   # Official O*NET description
# WEIGHT_SKILLS      = 0.15   # Technology Skills
# WEIGHT_OFC_TITLE   = 0.10   # Official O*NET title
# WEIGHT_ALT_TITLES  = 0.05   # Alternate Titles
# WEIGHT_TOOLS       = 0.05   # Tools Used


WEIGHT_TASKS       = 0.2017   # Structured Tasks (importance-weighted)
WEIGHT_DESCRIPTION = 0.1021   # Official O*NET description
WEIGHT_SKILLS      = 0.0063   # Technology Skills
WEIGHT_OFC_TITLE   = 0.3941   # Official O*NET title
WEIGHT_ALT_TITLES  = 0.2898   # Alternate Titles
WEIGHT_TOOLS       = 0.0061   # Tools Used

INPUT_PATH  = USER_INPUT_FILE
OUTPUT_PATH = MATCH_OUTPUT_PATH


# ─── Vector cache loader ──────────────────────────────────────────────────────

def load_vector_cache(model_name: str) -> dict:
    """
    Loads the pre-computed KB vector bundle saved by vectorize_kb.py.
    Raises a clear error if the cache is missing so the user knows to run
    vectorize_kb.py first.
    """
    cache_path = vector_cache_path(model_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Vector cache not found at:\n  {cache_path}\n\n"
            "Run vectorize_kb.py first to build the cache:\n"
            "    python src/vectorize_kb.py\n"
        )
    print(f"   Loading vector cache: {cache_path.name}")
    bundle = torch.load(cache_path, map_location='cpu', weights_only=False)

    # Sanity check
    meta = bundle.get('meta', {})
    print(f"   Cache created : {meta.get('created_at', 'unknown')}")
    print(f"   KB rows       : {meta.get('kb_rows', '?')}")
    print(f"   Model         : {meta.get('model_name', '?')}")
    return bundle


# ─── Query encoder (still needed for user input — no cache here) ──────────────

def encode_long_text(model, text: str, chunk_size: int = 800) -> torch.Tensor:
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(model.get_sentence_embedding_dimension())
    chunks = textwrap.wrap(text, width=chunk_size)
    if not chunks:
        return torch.zeros(model.get_sentence_embedding_dimension())
    return torch.mean(model.encode(chunks, convert_to_tensor=True), dim=0)


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_data():
    print(f"   Loading Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')

    if not INPUT_PATH.exists():
        print("[NOTICE] Creating mock user_survey_input.csv for testing...")
        os.makedirs(INPUT_PATH.parent, exist_ok=True)
        pd.DataFrame({
            'User_ID'             : [1, 2],
            'User_Job_Title'      : ["Budget Director", "Code Writer"],
            'User_Job_Description': [
                "I manage finances, analyze budget reports, and handle investments.",
                "I write python scripts, debug software, and manage databases."
            ]
        }).to_csv(INPUT_PATH, index=False)

    print(f"   Loading User Input...")
    input_df = pd.read_csv(INPUT_PATH).fillna('')
    return kb_df, input_df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_matching_pipeline():
    start = time.time()
    print("=" * 62)
    print("  MatchMyJob — Matching Pipeline")
    print("=" * 62)

    # ── 1. Load KB + user input ───────────────────────────────────────────────
    print(f"\n[1/4] Loading data...")
    kb_df, input_df = load_data()
    print(f"      KB occupations : {len(kb_df):,}")
    print(f"      User rows      : {len(input_df):,}")

    # ── 2. Load vector cache ──────────────────────────────────────────────────
    print(f"\n[2/4] Loading KB vector cache...")
    bundle = load_vector_cache(MODEL_NAME)

    task_embeddings      = bundle['task_embeddings']
    skill_embeddings     = bundle['skill_embeddings']
    tool_embeddings      = bundle['tool_embeddings']
    alt_title_embeddings = bundle['alt_title_embeddings']
    ofc_title_embeddings = bundle['ofc_title_embeddings']
    desc_embeddings      = bundle['desc_embeddings']
    print(f"      All 6 KB vector sets loaded from cache ✓")

    # ── 3. Encode user queries (always fresh — user input changes each run) ───
    print(f"\n[3/4] Encoding user queries...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    user_texts       = (input_df['User_Job_Title'] + " " + input_df['User_Job_Description']).tolist()
    query_embeddings = model.encode(user_texts, convert_to_tensor=True, show_progress_bar=True)

    # ── 4. Score and rank ─────────────────────────────────────────────────────
    print(f"\n[4/4] Calculating weighted similarity scores...")
    task_scores      = util.cos_sim(query_embeddings, task_embeddings)
    skill_scores     = util.cos_sim(query_embeddings, skill_embeddings)
    tool_scores      = util.cos_sim(query_embeddings, tool_embeddings)
    alt_title_scores = util.cos_sim(query_embeddings, alt_title_embeddings)
    ofc_title_scores = util.cos_sim(query_embeddings, ofc_title_embeddings)
    desc_scores      = util.cos_sim(query_embeddings, desc_embeddings)

    final_scores = (
        (task_scores      * WEIGHT_TASKS)       +
        (desc_scores      * WEIGHT_DESCRIPTION) +
        (skill_scores     * WEIGHT_SKILLS)      +
        (ofc_title_scores * WEIGHT_OFC_TITLE)   +
        (alt_title_scores * WEIGHT_ALT_TITLES)  +
        (tool_scores      * WEIGHT_TOOLS)
    )

    # ── Extract top 3 matches ─────────────────────────────────────────────────
    m1_codes, m1_titles, m1_scores = [], [], []
    m2_codes, m2_titles, m2_scores = [], [], []
    m3_codes, m3_titles, m3_scores = [], [], []

    for i in range(len(input_df)):
        top3         = torch.topk(final_scores[i], k=3)
        top3_scores  = top3.values.tolist()
        top3_indices = top3.indices.tolist()

        for rank, (codes, titles, scores_list) in enumerate([
            (m1_codes, m1_titles, m1_scores),
            (m2_codes, m2_titles, m2_scores),
            (m3_codes, m3_titles, m3_scores),
        ]):
            idx = top3_indices[rank]
            codes.append(kb_df.iloc[idx]['O*NET-SOC Code'])
            titles.append(kb_df.iloc[idx]['Title'])
            scores_list.append(round(top3_scores[rank] * 100, 2))

    input_df['Match_1_SOC_Code'] = m1_codes
    input_df['Match_1_Title']    = m1_titles
    input_df['Match_1_Score']    = m1_scores
    input_df['Match_2_SOC_Code'] = m2_codes
    input_df['Match_2_Title']    = m2_titles
    input_df['Match_2_Score']    = m2_scores
    input_df['Match_3_SOC_Code'] = m3_codes
    input_df['Match_3_Title']    = m3_titles
    input_df['Match_3_Score']    = m3_scores

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    input_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[SUCCESS] Finished in {time.time() - start:.1f}s")
    print(f"          Output → {OUTPUT_PATH}")
    print("\n--- SAMPLE PREDICTIONS (Top 3 Matches) ---")
    print(input_df[['User_Job_Title',
                     'Match_1_Title', 'Match_1_Score',
                     'Match_2_Title', 'Match_2_Score',
                     'Match_3_Title', 'Match_3_Score']].head())


if __name__ == "__main__":
    run_matching_pipeline()
