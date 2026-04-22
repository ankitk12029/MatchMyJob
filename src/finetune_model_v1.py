"""
finetune_model.py — MatchMyJob Domain Fine-Tuning (Scenario 3)
===============================================================
Scenario 3: Train on COMBINED dataset (matched + non-matched)

  Matched (filter_raters_matched.xlsx)        : 1,210 rows
  Non-Matched (filtered_raters_non-matched.xlsx):  716 rows  (22 skipped — no SOC)
  ───────────────────────────────────────────────────────────
  Total usable                                : 1,904 rows
  Train (85%)                                 : ~1,618 rows
  Test  (15%)                                 :   ~286 rows  → saved as ground_truth_test_split_s3.xlsx

Test split preserves Rater_Subset column so eval output shows
which rows came from matched vs non-matched data.

Loss     : CosineSimilarityLoss  (TripletLoss collapsed accuracy to 2.2%)
Negatives: 2 same-major + 1 cross-major per positive (label smoothing 0.05)
"""

import os
import random
import time
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from config import (
    KB_PATH, GT_PATH, GT_NON_MATCHED_PATH,
    GT_TEST_SPLIT_S3_PATH, FINETUNED_MODEL_PATH, BASE_MODEL_NAME
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"



# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────
TEST_SPLIT_RATIO          = 0.15
# TRAIN_BATCH_SIZE          = 16
TRAIN_BATCH_SIZE          = 8   # smaller batch size seems to help a lot with convergence on CPU (vs 16 or 32)
EPOCHS                    = 5
WARMUP_STEPS_RATIO        = 0.10
RANDOM_SEED               = 42
POSITIVE_LABEL            = 1.0
NEGATIVE_LABEL            = 0.05   # label smoothing


# ─── 1. LOAD & COMBINE DATASETS ───────────────────────────────────────────────

def load_dataset(path: Path, subset_label: str) -> pd.DataFrame:
    """
    Loads one dataset file, tags rows with Rater_Subset,
    and drops rows that have no SOC code (can't train on those).
    """
    print(f"   Loading {path.name}...")
    df = pd.read_excel(path, dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    required = ['occupation', 'Job description', 'onet_soc', 'onet_title']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df['Rater_Subset'] = subset_label

    before = len(df)
    df = df[df['onet_soc'] != ''].copy().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped {dropped} rows with blank onet_soc.")

    print(f"   {subset_label}: {len(df):,} rows  |  {df['onet_soc'].nunique()} unique SOC codes")
    return df


def load_combined() -> pd.DataFrame:
    """
    Loads and concatenates both datasets.
    Matched rows come first, then non-matched.
    """
    matched     = load_dataset(GT_PATH,              'Matched')
    non_matched = load_dataset(GT_NON_MATCHED_PATH,  'Non-Matched')
    combined    = pd.concat([matched, non_matched], ignore_index=True)
    print(f"   Combined total: {len(combined):,} rows")
    return combined


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


# ─── 3. TRAINING PAIRS ────────────────────────────────────────────────────────

def make_survey_text(title: str, desc: str) -> str:
    return f"{title} {desc}".strip()


def build_training_pairs(train_df: pd.DataFrame, profile_map: dict) -> list:
    """
    For each row:
      1 positive  (label=1.0)  — survey text ↔ correct O*NET profile
      2 negatives (label=0.05) — same major group  (fine-grained signal)
      1 negative  (label=0.05) — different major group  (broad anchoring)
    """
    examples  = []
    all_socs  = list(profile_map.keys())
    major_idx: dict[str, list] = {}
    for soc in all_socs:
        major_idx.setdefault(soc[:2], []).append(soc)

    skipped = 0
    for _, row in train_df.iterrows():
        txt = make_survey_text(row['occupation'], row['Job description'])
        soc = row['onet_soc']
        if soc not in profile_map:
            skipped += 1
            continue

        # Positive
        examples.append(InputExample(texts=[txt, profile_map[soc]], label=POSITIVE_LABEL))

        # 2 same-major negatives
        same_major = [s for s in major_idx.get(soc[:2], []) if s != soc]
        random.shuffle(same_major)
        for neg in same_major[:2]:
            examples.append(InputExample(texts=[txt, profile_map[neg]], label=NEGATIVE_LABEL))

        # 1 cross-major negative
        cross = [s for s in all_socs if s[:2] != soc[:2]]
        if cross:
            examples.append(InputExample(
                texts=[txt, profile_map[random.choice(cross)]], label=NEGATIVE_LABEL
            ))

    if skipped:
        print(f"   Warning: {skipped} rows skipped — SOC not in O*NET KB.")
    random.shuffle(examples)
    return examples


def build_evaluator(test_df: pd.DataFrame, profile_map: dict):
    """
    Evaluator uses only rows that have a valid SOC code.
    Uses same-major negatives for a harder evaluation signal.
    """
    s1, s2, scores = [], [], []
    eval_rows = test_df[test_df['onet_soc'] != '']
    for _, row in eval_rows.iterrows():
        soc = row['onet_soc']
        if soc not in profile_map:
            continue
        txt = make_survey_text(row['occupation'], row['Job description'])
        s1.append(txt);  s2.append(profile_map[soc]);  scores.append(1.0)

        pool = [s for s in profile_map if s[:2] == soc[:2] and s != soc] or \
               [s for s in profile_map if s != soc]
        neg  = random.choice(pool)
        s1.append(txt);  s2.append(profile_map[neg]);  scores.append(0.0)

    return evaluation.EmbeddingSimilarityEvaluator(s1, s2, scores, name='matchmyjob-s3-eval')


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — Scenario 3 Fine-Tuning")
    print("  Training on Matched + Non-Matched (combined)")
    print("=" * 62)

    # ── 1. KB ─────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading O*NET Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    profile_map = build_onet_profile_map(kb_df)
    print(f"      {len(profile_map):,} occupations loaded.")

    # ── 2. Load combined dataset ───────────────────────────────────────────────
    print(f"\n[2/5] Loading combined dataset...")
    combined_df = load_combined()

    # ── 3. Stratified 85/15 split — preserve Rater_Subset balance ─────────────
    print(f"\n[3/5] Splitting → 85% train  |  15% test  (stratified by Rater_Subset)...")
    train_parts, test_parts = [], []

    for subset in ['Matched', 'Non-Matched']:
        sub = combined_df[combined_df['Rater_Subset'] == subset].sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        cut = int(len(sub) * (1 - TEST_SPLIT_RATIO))
        train_parts.append(sub.iloc[:cut])
        test_parts.append(sub.iloc[cut:])
        print(f"   {subset:12s} → train: {cut:,}  |  test: {len(sub)-cut:,}")

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    test_df  = pd.concat(test_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"\n   Total train : {len(train_df):,} rows")
    print(f"   Total test  : {len(test_df):,} rows")

    # Save test split with Rater_Subset column for eval script
    GT_TEST_SPLIT_S3_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_excel(GT_TEST_SPLIT_S3_PATH, index=False)
    print(f"   Test split saved → {GT_TEST_SPLIT_S3_PATH.name}")

    # ── 4. Build training examples ────────────────────────────────────────────
    print(f"\n[4/5] Building training examples...")
    train_examples = build_training_pairs(train_df, profile_map)
    n_pos = sum(1 for e in train_examples if e.label == POSITIVE_LABEL)
    n_neg = sum(1 for e in train_examples if e.label == NEGATIVE_LABEL)
    print(f"      Positives : {n_pos:,}")
    print(f"      Negatives : {n_neg:,}  (2 same-major + 1 cross-major each)")
    print(f"      Total     : {len(train_examples):,}")

    evaluator = build_evaluator(test_df, profile_map)

    # ── 5. Fine-tune ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning '{BASE_MODEL_NAME}'...")
    model   = SentenceTransformer(BASE_MODEL_NAME)
    # AND force the model to CPU
    # model = SentenceTransformer(BASE_MODEL_NAME, device='cpu')
    loader  = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    loss_fn = losses.CosineSimilarityLoss(model)
    warmup  = int(len(loader) * EPOCHS * WARMUP_STEPS_RATIO)

    print(f"      Epochs  : {EPOCHS}  |  Batch: {TRAIN_BATCH_SIZE}  |  Warmup: {warmup}")
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
    print(f"\n  NEXT STEPS:")
    print(f"    1. python src/vectorize_kb.py")
    print(f"    2. python src/match_engine_eval.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
