"""
finetune_model.py — MatchMyJob Domain Fine-Tuning (Scenario 6)
===============================================================
Datasets:
  Primary  : master_training_data.xlsx → "Combined" sheet (~2,476 clean rows after swap fix)
  Title-only: training_data_without_descrption.xlsx → "Sheet4" (~2,205 rows, 2+ word titles only)
  ──────────────────────────────────────────────────────────────────────
  Total                                             : ~4,681 rows
  Train (80%)                                       : ~3,745 rows
  Test  (20%) — from primary dataset only           :   ~495 rows

Changes from Scenario 5:
  [1] Model upgrade: bge-small-en-v1.5 → bge-base-en-v1.5
      - 3× more parameters (33M → 109M); significantly higher retrieval accuracy
        on MTEB benchmarks while still fitting in Mac unified memory.
      - New vector cache auto-created (kb_vectors_matchmyjob-finetuned.pt) —
        run vectorize_kb.py after training to rebuild.

  [2] BGE query prefix applied consistently in training + inference.
      - Anchor: "Represent this job for retrieval: <title> <desc>"
      - KB profile (positive): unchanged — BGE is asymmetric (prefix on query only).
      - Must keep BGE_QUERY_PREFIX identical in finetune_model.py and match_engine_eval.py.
      - Previously tested at inference-only → -1.1% (wrong). Correct use: train+infer.

  [3] Added title-only dataset (training_data_without_descrption.xlsx, Sheet4)
      - Filtered to 2+ word titles only: removes 847 single-word vague entries
        ("Manager", "sales", "retail") that create contradictory MNR training
        signal when the same word maps to multiple SOC codes in the same batch.
      - Adds 137 new SOC codes not present in primary training data.
      - Adds 277 rows for SOC 15 (Computer/Math) and 378 for SOC 11 (Management)
        — the two weakest major groups at 25% and 35% top-1 accuracy.
      - Title-only rows use empty description: make_survey_text(title, "") = title.
        This is valid — the model learns direct title→SOC mappings, which also
        helps short-description survey responses at inference time.

  [4] Test split sourced from primary dataset only (has descriptions)
      - Title-only rows go entirely into training — no point evaluating on
        description-less rows since real survey data always has descriptions.

  [5] "REMOVE" entries filtered out of Sheet1 (not applicable to Sheet4).
"""

import os
import random
import re
import time
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from config import (
    KB_PATH, GT_MASTER_TRAINING_PATH, GT_TITLE_ONLY_PATH,
    GT_TEST_SPLIT_S3_PATH, FINETUNED_MODEL_PATH, BASE_MODEL_NAME
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────
TEST_SPLIT_RATIO   = 0.20   # 80/20 split
TRAIN_BATCH_SIZE   = 64     # increased from 32: halves steps/epoch + 63 in-batch negatives vs 31
EPOCHS             = 3      # reduced from 5: bge-small peaked at epoch 3 (Spearman 0.7723);
                             # epochs 4-5 showed no gain. save_best_model=True preserves the peak.
WARMUP_STEPS_RATIO = 0.10
RANDOM_SEED        = 42


# ─── 1. LOAD DATASET ──────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    print(f"   File   : {GT_MASTER_TRAINING_PATH.name}  (sheet: Combined)")
    df = pd.read_excel(GT_MASTER_TRAINING_PATH, sheet_name='Combined', dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Rename to internal standard column names
    df = df.rename(columns={
        'Job title'       : 'occupation',
        'Job description' : 'Job description',
        'O*NET_code'      : 'onet_soc',
        'O*NET Job title' : 'onet_title',
    })

    before = len(df)
    df = df[
        (df['occupation'] != '') &
        (df['Job description'] != '') &
        (df['onet_soc'] != '')
    ].copy().reset_index(drop=True)

    if before - len(df):
        print(f"   Dropped {before - len(df)} rows with missing fields.")

    # Fix swapped onet_soc / onet_title columns
    # ~25% of rows have the title text in onet_soc and the code in onet_title
    SOC_PATTERN = re.compile(r'^\d{2}-\d{4}\.\d{2}')
    is_swapped = ~df['onet_soc'].apply(lambda x: bool(SOC_PATTERN.match(str(x).strip())))
    if is_swapped.sum() > 0:
        df.loc[is_swapped, ['onet_soc', 'onet_title']] = (
            df.loc[is_swapped, ['onet_title', 'onet_soc']].values
        )
        print(f"   Fixed  : {is_swapped.sum()} rows with swapped onet_soc/onet_title columns.")

    # Drop any rows still without a valid SOC code after swap fix
    still_invalid = ~df['onet_soc'].apply(lambda x: bool(SOC_PATTERN.match(str(x).strip())))
    if still_invalid.sum():
        df = df[~still_invalid].reset_index(drop=True)
        print(f"   Dropped: {still_invalid.sum()} rows with no valid SOC code.")

    print(f"   Loaded : {len(df):,} rows  |  {df['onet_soc'].nunique()} unique SOC codes")
    return df


# ─── 1b. LOAD TITLE-ONLY DATASET ─────────────────────────────────────────────

def load_title_only_dataset() -> pd.DataFrame:
    """
    Loads training_data_without_descrption.xlsx → Sheet4.

    Change [1]: filtered to 2+ word titles only.
      Single-word titles like "Manager", "sales", "retail" are too vague —
      the same word maps to many different SOC codes across the batch, which
      creates contradictory MNR in-batch negatives and degrades training.
      2+ word titles ("Staff Accountant", "Systems Programmer") are specific
      enough to provide clean anchor→positive signal.

    Change [1]: description set to empty string.
      make_survey_text(title, "") returns just the title, which is valid input.
      This teaches the model direct title→SOC associations that also help
      at inference time when survey descriptions are vague or short.
    """
    print(f"   File   : {GT_TITLE_ONLY_PATH.name}  (sheet: Sheet4)")
    df = pd.read_excel(GT_TITLE_ONLY_PATH, sheet_name='Sheet4', dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    df = df.rename(columns={
        'occupation' : 'occupation',
        'onetJC'     : 'onet_soc',
        'onetJT'     : 'onet_title',
    })
    df['Job description'] = ''   # no description available

    before = len(df)

    # Drop rows missing title or SOC code
    df = df[(df['occupation'] != '') & (df['onet_soc'] != '')].copy()

    # Filter: keep only 2+ word titles — removes 847 ambiguous single-word entries
    df = df[df['occupation'].str.split().str.len() >= 2].copy()

    # Validate SOC code format
    SOC_PATTERN = re.compile(r'^\d{2}-\d{4}\.\d{2}')
    df = df[df['onet_soc'].apply(lambda x: bool(SOC_PATTERN.match(x)))].copy()

    df = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped: {dropped} rows (single-word titles, missing fields, invalid SOC).")

    print(f"   Loaded : {len(df):,} rows  |  {df['onet_soc'].nunique()} unique SOC codes")
    return df[['occupation', 'Job description', 'onet_soc', 'onet_title']]


# ─── 2. O*NET PROFILE MAP ─────────────────────────────────────────────────────

def build_onet_profile_map(kb_df: pd.DataFrame) -> dict:
    profile_map = {}
    for _, row in kb_df.iterrows():
        soc  = str(row['O*NET-SOC Code']).strip()
        text = " ".join(filter(None, [
            str(row.get('Title', '')),
            str(row.get('Description', ''))[:400],
            str(row.get('All_Alt_Titles', ''))[:200],
            str(row.get('All_Tech_Skills', ''))[:300],
        ]))
        profile_map[soc] = text.strip()
    return profile_map


# ─── 3. TRAINING PAIRS (MultipleNegativesRankingLoss format) ─────────────────

# BGE instruction prefix — applied to the query (anchor) side ONLY, not KB profiles.
# Must be used identically at inference time in match_engine_eval.py.
# BGE-base/large models are trained to use this prefix for retrieval tasks.
BGE_QUERY_PREFIX = "Represent this job for retrieval: "


def make_survey_text(title: str, desc: str) -> str:
    return f"{title} {desc}".strip()


def build_training_pairs_mnr(train_df: pd.DataFrame, profile_map: dict) -> list:
    """
    MNR expects only (anchor, positive) pairs — no explicit negatives needed.
    Every other item in the same batch becomes a negative automatically.
    Batch size of 32 gives 31 in-batch negatives per anchor.

    The anchor (survey text) gets the BGE query prefix; the positive (KB profile)
    does not — BGE is asymmetric: query prefix on query side only.
    """
    examples = []
    skipped  = 0

    for _, row in train_df.iterrows():
        soc = row['onet_soc']
        if soc not in profile_map:
            skipped += 1
            continue
        txt = BGE_QUERY_PREFIX + make_survey_text(row['occupation'], row['Job description'])
        examples.append(InputExample(texts=[txt, profile_map[soc]]))

    if skipped:
        print(f"   Warning: {skipped} rows skipped — SOC not in O*NET KB.")

    random.shuffle(examples)
    print(f"   Training pairs : {len(examples):,}")
    print(f"   In-batch negatives per anchor: {TRAIN_BATCH_SIZE - 1}")
    return examples


def build_evaluator(test_df: pd.DataFrame, profile_map: dict):
    """Spearman evaluator — positive pair + same-major-group negative per row.
    Uses BGE query prefix on s1 (queries) to match training conditions.
    """
    s1, s2, scores = [], [], []
    eval_rows = test_df[test_df['onet_soc'] != '']
    for _, row in eval_rows.iterrows():
        soc = row['onet_soc']
        if soc not in profile_map:
            continue
        txt = BGE_QUERY_PREFIX + make_survey_text(row['occupation'], row['Job description'])
        s1.append(txt);  s2.append(profile_map[soc]);  scores.append(1.0)

        # Same-major-group negative for harder eval signal
        pool = [s for s in profile_map if s[:2] == soc[:2] and s != soc] or \
               [s for s in profile_map if s != soc]
        neg  = random.choice(pool)
        s1.append(txt);  s2.append(profile_map[neg]);  scores.append(0.0)

    return evaluation.EmbeddingSimilarityEvaluator(s1, s2, scores, name='matchmyjob-s4-eval')


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Scenario 6 Fine-Tuning")
    print("  Model  : bge-base-en-v1.5  (upgraded from bge-small)")
    print("  Prefix : 'Represent this job for retrieval: ' on anchors")
    print("  Primary (~2,476) + Title-only 2+words (~2,205) = ~4,681 rows")
    print("  Loss   : MultipleNegativesRankingLoss  |  Batch: 64  |  Epochs: 3")
    print("  Split  : 80/20 on primary only; title-only goes fully into train")
    print("=" * 62)

    # ── 1. KB ─────────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading O*NET Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    profile_map = build_onet_profile_map(kb_df)
    print(f"      {len(profile_map):,} occupations loaded.")

    # ── 2. Load primary dataset and split 80/20 ───────────────────────────────
    # Change [2]: test split comes from primary dataset only (has descriptions).
    # Title-only rows are training-only — evaluating on description-less rows
    # would not reflect real inference conditions.
    print(f"\n[2/6] Loading primary dataset and splitting 80/20...")
    primary_df = load_dataset()
    primary_df = primary_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    cut        = int(len(primary_df) * (1 - TEST_SPLIT_RATIO))
    train_df   = primary_df.iloc[:cut].copy()
    test_df    = primary_df.iloc[cut:].copy()

    print(f"   Primary train : {len(train_df):,} rows")
    print(f"   Primary test  : {len(test_df):,} rows  (saved for eval)")

    GT_TEST_SPLIT_S3_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_excel(GT_TEST_SPLIT_S3_PATH, index=False)
    print(f"   Test split saved → {GT_TEST_SPLIT_S3_PATH.name}")

    # ── 3. Load title-only dataset and append to train ────────────────────────
    # Change [1]: title-only rows (2+ word titles) appended to training set only.
    print(f"\n[3/6] Loading title-only dataset...")
    title_df = load_title_only_dataset()
    train_df = pd.concat([train_df, title_df], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"\n   Combined train : {len(train_df):,} rows  |  {train_df['onet_soc'].nunique()} unique SOC codes")
    print(f"   Test           : {len(test_df):,} rows  |  {test_df['onet_soc'].nunique()} unique SOC codes")

    # ── 4. Build training examples ────────────────────────────────────────────
    print(f"\n[4/6] Building training examples...")
    train_examples = build_training_pairs_mnr(train_df, profile_map)
    evaluator      = build_evaluator(test_df, profile_map)

    # ── 5. Fine-tune ──────────────────────────────────────────────────────────
    print(f"\n[5/6] Fine-tuning '{BASE_MODEL_NAME}'...")
    import torch
    device  = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"      Device   : {device}")
    model   = SentenceTransformer(BASE_MODEL_NAME, device=device)
    loader  = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    warmup  = int(len(loader) * EPOCHS * WARMUP_STEPS_RATIO)

    print(f"      Epochs   : {EPOCHS}  |  Batch: {TRAIN_BATCH_SIZE}  |  Warmup: {warmup}")
    print(f"      Spearman score printed after each epoch (higher = better)\n")

    FINETUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup,
        output_path=str(FINETUNED_MODEL_PATH),
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Model saved  → {FINETUNED_MODEL_PATH}")
    print(f"  Test split   → {GT_TEST_SPLIT_S3_PATH.name}")
    print(f"\n  [6/6] NEXT STEPS:")
    print(f"    1. python src/vectorize_kb.py")
    print(f"    2. python src/optimize_weights.py")
    print(f"    3. python src/match_engine_eval.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
