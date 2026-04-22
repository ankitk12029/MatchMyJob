"""
match_engine_eval.py — MatchMyJob Evaluation Pipeline (Scenario 3)
===================================================================
Two improvements added today:
  1. Top-3 accuracy metric  — measures if correct SOC is in top-3 predictions
  2. Cross-encoder reranker — re-scores top-10 candidates by reading survey
                              text + O*NET profile together (no retraining needed)

Expected improvement:
  Top-1 accuracy  : ~40%  (existing)
  Top-3 accuracy  : ~58–65%  (new metric, no model change)
  Top-1 + reranker: ~52–58%  (cross-encoder on top-10)
"""

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

# ─── MODEL ────────────────────────────────────────────────────────────────────
MODEL_NAME          = str(FINETUNED_MODEL_PATH)
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# ─── CATEGORY WEIGHTS ─────────────────────────────────────────────────────────
WEIGHT_TASKS       = 0.25
WEIGHT_DESCRIPTION = 0.20
WEIGHT_SKILLS      = 0.15
WEIGHT_OFC_TITLE   = 0.25
WEIGHT_ALT_TITLES  = 0.10
WEIGHT_TOOLS       = 0.05

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


# ─── Cross-encoder reranker ───────────────────────────────────────────────────

def build_cross_encoder_profile(row: pd.Series) -> str:
    """
    Builds a concise text profile of an O*NET occupation for the cross-encoder.
    Shorter than the full vectorize_kb profile — cross-encoder has a 512 token limit.
    """
    parts = filter(None, [
        str(row.get('Title', '')),
        str(row.get('Description', ''))[:300],
        str(row.get('All_Alt_Titles', ''))[:150],
    ])
    return " ".join(parts).strip()


def rerank_with_cross_encoder(cross_encoder: CrossEncoder,
                               survey_text: str,
                               candidate_indices: list,
                               kb_df: pd.DataFrame) -> int:
    """
    Takes top-N candidate KB indices from cosine similarity,
    re-scores each by reading survey text + O*NET profile together,
    returns the index of the best candidate.

    Cross-encoder sees both texts simultaneously → much more accurate
    than cosine similarity which encodes them independently.
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
      Stage 1 — Weighted cosine similarity → top-10 candidates
      Stage 2 — Cross-encoder reranker     → best of top-10

    Also returns top-3 SOC codes for the top-3 accuracy metric.
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

    print(f"   Running cross-encoder reranker on top-10 candidates per row...")
    for i in range(len(texts)):
        # ── Stage 1: cosine sim top-10 with title boost ────────────────────
        boosted      = apply_title_boost(survey_titles[i], kb_df, final_scores[i])
        top10        = torch.topk(boosted, k=min(10, len(kb_df)))
        top10_idx    = top10.indices.tolist()

        # Top-3 SOC codes for top-3 accuracy metric (from cosine sim stage)
        top3_socs_list.append([kb_df.iloc[idx]['O*NET-SOC Code'] for idx in top10_idx[:3]])

        # ── Stage 2: cross-encoder reranks the top-10 ──────────────────────
        best_idx = rerank_with_cross_encoder(
            cross_encoder, texts[i], top10_idx, kb_df
        )

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


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(out_df: pd.DataFrame):
    print(f"\n{'─' * 62}")
    print(f"  EVALUATION RESULTS — Scenario 3 + Cross-Encoder Reranker")
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
        n      = len(sub)
        c1     = sub['Correct'].sum()
        c3     = sub['Correct_Top3'].sum()
        conf_c = sub[sub['Correct']==1]['Machine_Confidence'].mean() if c1 > 0 else 0
        conf_w = sub[sub['Correct']==0]['Machine_Confidence'].mean() if (n-c1) > 0 else 0

        print(f"\n  [{label}]")
        print(f"    Rows           : {n:,}")
        print(f"    Top-1 correct  : {c1}  ({c1/n*100:.1f}%)")
        print(f"    Top-3 correct  : {c3}  ({c3/n*100:.1f}%)  ← human SOC in top-3 predictions")
        print(f"    Avg confidence : {sub['Machine_Confidence'].mean():.1f}%")
        print(f"    Conf gap (✓/✗) : {abs(conf_c - conf_w):.1f}%")

    no_soc = out_df[out_df['Correct'] == '']
    if len(no_soc) > 0:
        print(f"\n  [No human SOC — {len(no_soc)} rows, machine ran but no comparison possible]")

    print(f"\n{'─' * 62}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval_pipeline():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Evaluation + Cross-Encoder Reranker")
    print("=" * 62)

    # ── 1. Load KB ────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    print(f"      {len(kb_df):,} occupations")

    # ── 2. Load test split ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading test split...")
    test_df = load_test_split()

    # ── 3. Load vector cache ──────────────────────────────────────────────────
    print(f"\n[3/5] Loading vector cache...")
    bundle = load_vector_cache(MODEL_NAME)
    print(f"      All 6 KB vector sets loaded ✓")

    # ── 4. Load models ────────────────────────────────────────────────────────
    print(f"\n[4/5] Loading models...")
    print(f"      Bi-encoder  : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device='cpu')

    print(f"      Cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"      Both models loaded ✓")

    # ── 5. Run matching and save ──────────────────────────────────────────────
    print(f"\n[5/5] Running matching on {len(test_df):,} rows...")
    print(f"      Stage 1: weighted cosine similarity → top-10 candidates")
    print(f"      Stage 2: cross-encoder reranks top-10 per row")

    out_df = build_results(test_df, kb_df, model, cross_encoder, bundle)

    os.makedirs(EVAL_OUTPUT_PATH.parent, exist_ok=True)
    out_df.to_csv(EVAL_OUTPUT_PATH, index=False)
    print(f"\n      Output saved → {EVAL_OUTPUT_PATH}")

    print_summary(out_df)

    # ── Preview ───────────────────────────────────────────────────────────────
    cols = ['User_Occupation', 'Machine_Title', 'Machine_Confidence',
            'Human_Title', 'Correct', 'Correct_Top3', 'Rater_Subset']
    print(f"\n  SAMPLE (first 8 rows):")
    print(out_df[cols].head(8).to_string(index=False))

    print(f"\n  Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_eval_pipeline()

