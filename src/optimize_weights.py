"""
optimize_weights.py — Data-Driven Weight Optimization for MatchMyJob
=====================================================================
Replaces the manually guessed weights with values learned from your
ground truth data using scipy's Nelder-Mead optimizer.

How it works:
  1. Loads the pre-computed KB vector cache (no re-encoding needed)
  2. Encodes the ground truth survey texts once
  3. Computes 6 separate similarity score matrices (one per field)
  4. Hands those matrices to scipy.optimize — it tries thousands of
     weight combinations and picks the one that maximizes accuracy
     on a validation split of your training data
  5. Prints the optimal weights + accuracy improvement
  6. Writes the optimal weights to config so all scripts use them

Critical: uses the TRAIN split only for optimization.
          The TEST split is never touched — stays honest for eval.

Run order:
    1. vectorize_kb.py           (must exist already)
    2. finetune_model.py         (must exist already — provides train split)
    3. THIS SCRIPT               (finds best weights)
    4. match_engine_eval.py      (uses new weights automatically)

Usage:
    python src/optimize_weights.py
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sentence_transformers import SentenceTransformer, util
from config import (
    KB_PATH, FINETUNED_MODEL_PATH,
    GT_TEST_SPLIT_S3_PATH, GT_TEST_SPLIT_PATH,
    vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── WHICH SCENARIO ───────────────────────────────────────────────────────────
# Point to whichever test split you're using — the OPPOSITE of that split
# is the train split used for optimization.
# Scenario 1/2: GT_TEST_SPLIT_PATH  (15% of 1,210 matched)
# Scenario 3  : GT_TEST_SPLIT_S3_PATH (15% of 1,926 combined)
MODEL_NAME        = str(FINETUNED_MODEL_PATH)
TEST_SPLIT_PATH   = GT_TEST_SPLIT_S3_PATH    # ← change to GT_TEST_SPLIT_PATH for S1/S2

# ─── OPTIMIZATION CONFIG ──────────────────────────────────────────────────────
VALIDATION_FRAC   = 0.20     # 20% of train rows used for optimization validation
RANDOM_SEED       = 42
N_FIELDS          = 6        # tasks, desc, skills, ofc_title, alt_titles, tools

# Field names (for readable output)
FIELD_NAMES = ['Tasks', 'Description', 'Skills', 'Ofc_Title', 'Alt_Titles', 'Tools']


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_vector_cache() -> dict:
    cache_path = vector_cache_path(MODEL_NAME)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}\nRun vectorize_kb.py first.")
    print(f"   Cache: {cache_path.name}")
    bundle = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"   KB rows: {bundle['meta'].get('kb_rows', '?')}")
    return bundle


def load_train_rows() -> pd.DataFrame:
    """
    Loads the train rows — everything NOT in the test split.
    Assumes the test split file was saved by finetune_model.py.
    """
    # Load full matched dataset
    from config import GT_PATH, GT_NON_MATCHED_PATH
    matched = pd.read_excel(GT_PATH, dtype=str).fillna('')
    matched['Rater_Subset'] = 'Matched'

    non_matched = pd.DataFrame()
    if GT_NON_MATCHED_PATH.exists():
        non_matched = pd.read_excel(GT_NON_MATCHED_PATH, dtype=str).fillna('')
        non_matched['Rater_Subset'] = 'Non-Matched'

    combined = pd.concat([matched, non_matched], ignore_index=True)
    for col in combined.columns:
        combined[col] = combined[col].astype(str).str.strip()

    # Remove rows that are in the test split
    if TEST_SPLIT_PATH.exists():
        test_df = pd.read_excel(TEST_SPLIT_PATH, dtype=str).fillna('')
        for col in test_df.columns:
            test_df[col] = test_df[col].astype(str).str.strip()
        # Match on occupation + description to identify test rows
        test_keys = set(zip(test_df['occupation'], test_df['Job description']))
        mask = combined.apply(
            lambda r: (r['occupation'], r['Job description']) not in test_keys, axis=1
        )
        combined = combined[mask].copy()
        print(f"   Removed {(~mask).sum()} test rows → {len(combined):,} train rows remaining")

    # Drop rows with no SOC code
    combined = combined[combined['onet_soc'].str.strip() != ''].reset_index(drop=True)
    print(f"   Train rows with valid SOC: {len(combined):,}")
    return combined


def encode_queries(model, texts: list) -> torch.Tensor:
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=True)


def compute_score_matrices(query_embs: torch.Tensor, bundle: dict) -> list:
    """
    Computes 6 raw cosine similarity matrices — one per KB field.
    Shape: (n_queries, n_kb_occupations)
    These are computed ONCE and reused across all optimizer iterations.
    """
    fields = [
        bundle['task_embeddings'],
        bundle['desc_embeddings'],
        bundle['skill_embeddings'],
        bundle['ofc_title_embeddings'],
        bundle['alt_title_embeddings'],
        bundle['tool_embeddings'],
    ]
    print("   Computing 6 similarity matrices (done once)...")
    return [util.cos_sim(query_embs, f).numpy() for f in fields]


def accuracy_from_weights(weights: np.ndarray,
                           score_matrices: list,
                           correct_indices: np.ndarray) -> float:
    """
    Given a weight vector, compute the weighted sum of score matrices
    and return accuracy (fraction of rows where top-1 == correct index).

    correct_indices[i] = the KB row index of the correct SOC for query i.
    Returns negative accuracy because scipy MINIMIZES.
    """
    # Normalize weights to sum to 1
    w = np.abs(weights)          # force non-negative
    w = w / w.sum()              # normalize

    # Weighted sum: shape (n_queries, n_kb)
    combined = sum(w[i] * score_matrices[i] for i in range(N_FIELDS))

    # Top-1 prediction per query
    predicted = np.argmax(combined, axis=1)

    # Accuracy
    acc = np.mean(predicted == correct_indices)
    return -acc   # negative because scipy minimizes


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Weight Optimization")
    print("=" * 62)

    # ── 1. Load KB + cache ────────────────────────────────────────────────────
    print(f"\n[1/5] Loading KB and vector cache...")
    kb_df  = pd.read_csv(KB_PATH).fillna('')
    bundle = load_vector_cache()

    # Build SOC → KB row index map
    soc_to_idx = {str(row['O*NET-SOC Code']).strip(): i
                  for i, row in kb_df.iterrows()}
    print(f"   {len(soc_to_idx):,} SOC codes in KB")

    # ── 2. Load train rows ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading train rows (excluding test split)...")
    train_df = load_train_rows()

    # Take a validation sample for optimization (never test rows)
    val_df = train_df.sample(frac=VALIDATION_FRAC, random_state=RANDOM_SEED)
    print(f"   Validation subset for optimization: {len(val_df):,} rows")

    # Map each row to its correct KB index
    correct_indices = []
    valid_mask = []
    for _, row in val_df.iterrows():
        soc = row['onet_soc'].strip()
        idx = soc_to_idx.get(soc, None)
        valid_mask.append(idx is not None)
        correct_indices.append(idx if idx is not None else 0)

    val_df    = val_df[valid_mask].reset_index(drop=True)
    correct_indices = np.array([i for i, v in zip(correct_indices, valid_mask) if v])
    print(f"   Rows with SOC in KB: {len(val_df):,}")

    # ── 3. Encode queries ─────────────────────────────────────────────────────
    print(f"\n[3/5] Encoding {len(val_df):,} validation queries...")
    model      = SentenceTransformer(MODEL_NAME, device='cpu')
    val_texts  = (val_df['occupation'] + " " + val_df['Job description']).tolist()
    query_embs = encode_queries(model, val_texts)

    # ── 4. Pre-compute score matrices ─────────────────────────────────────────
    print(f"\n[4/5] Pre-computing score matrices...")
    score_matrices = compute_score_matrices(query_embs, bundle)

    # Baseline accuracy with current equal weights
    current_weights = np.array([0.25, 0.20, 0.15, 0.25, 0.10, 0.05])
    baseline_acc = -accuracy_from_weights(current_weights, score_matrices, correct_indices)
    print(f"\n   Baseline accuracy (current weights): {baseline_acc*100:.1f}%")
    print(f"   Current weights: " + "  ".join(
        f"{FIELD_NAMES[i]}={current_weights[i]:.2f}" for i in range(N_FIELDS)
    ))

    # ── 5. Optimize ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Running optimization (this takes 1-3 minutes)...")
    print(f"   Method: Differential Evolution — global search, no gradient needed")

    # Bounds: each weight between 0.0 and 1.0
    bounds = [(0.0, 1.0)] * N_FIELDS

    result = differential_evolution(
        func=accuracy_from_weights,
        bounds=bounds,
        args=(score_matrices, correct_indices),
        seed=RANDOM_SEED,
        maxiter=300,
        popsize=12,
        tol=0.001,
        polish=True,          # Nelder-Mead polish after global search
        disp=True,            # print progress
    )

    # Normalize the winning weights
    best_raw    = np.abs(result.x)
    best_weights = best_raw / best_raw.sum()
    best_acc    = -result.fun

    # ── Print results ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Optimization complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 62}")
    print(f"\n  Baseline accuracy : {baseline_acc*100:.1f}%")
    print(f"  Optimized accuracy: {best_acc*100:.1f}%  (+{(best_acc-baseline_acc)*100:.1f}pp)")
    print(f"\n  Optimal weights (copy these into match_engine_eval.py):")
    print(f"  {'─'*50}")
    for i, name in enumerate(FIELD_NAMES):
        bar = '█' * int(best_weights[i] * 40)
        print(f"  {name:12s} : {best_weights[i]:.4f}  {bar}")
    print(f"  {'─'*50}")
    print(f"  Sum check: {best_weights.sum():.4f}  (should be 1.0)")

    print(f"\n  Paste into match_engine_eval.py and match_engine.py:")
    print(f"  {'─'*50}")
    print(f"  WEIGHT_TASKS       = {best_weights[0]:.4f}")
    print(f"  WEIGHT_DESCRIPTION = {best_weights[1]:.4f}")
    print(f"  WEIGHT_SKILLS      = {best_weights[2]:.4f}")
    print(f"  WEIGHT_OFC_TITLE   = {best_weights[3]:.4f}")
    print(f"  WEIGHT_ALT_TITLES  = {best_weights[4]:.4f}")
    print(f"  WEIGHT_TOOLS       = {best_weights[5]:.4f}")
    print(f"  {'─'*50}")

    # ── Save to a weights file so eval scripts can auto-load ──────────────────
    weights_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'optimal_weights.csv'
    weights_df = pd.DataFrame({
        'field'  : FIELD_NAMES,
        'weight' : best_weights,
    })
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(weights_path, index=False)
    print(f"\n  Weights saved → {weights_path}")
    print(f"  (match_engine_eval.py will auto-load these if the file exists)")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
