"""
finetune_model.py — MatchMyJob Domain Fine-Tuning
===================================================
Dataset : filter_raters_matched.xlsx  (1,210 rows, pre-filtered)
Split   : 85% train / 15% test (test split saved for match_engine_eval.py)

After training:
    1. Run vectorize_kb.py to rebuild the vector cache with the new model.
    2. match_engine.py and match_engine_eval.py will automatically use it.
"""

import os
import random
import time
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from config import (
    KB_PATH, GT_PATH, GT_TEST_SPLIT_PATH,
    FINETUNED_MODEL_PATH, BASE_MODEL_NAME
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────
TEST_SPLIT_RATIO          = 0.15
TRAIN_BATCH_SIZE          = 16
EPOCHS                    = 4
WARMUP_STEPS_RATIO        = 0.10
HARD_NEGATIVES_PER_SAMPLE = 2
RANDOM_SEED               = 42


# ─── 1. LOAD GROUND TRUTH ─────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> pd.DataFrame:
    print(f"   File: {path.name}")
    df = pd.read_excel(path, dtype=str).fillna('')
    for col in df.columns:
        df[col] = df[col].str.strip()

    required = ['occupation', 'Job description', 'onet_soc', 'onet_title']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    before = len(df)
    df = df[df['onet_soc'] != ''].copy().reset_index(drop=True)
    if len(df) < before:
        print(f"   Dropped {before - len(df)} rows with blank onet_soc.")

    print(f"   Loaded {len(df):,} rows  |  {df['onet_soc'].nunique()} unique SOC codes")
    return df


# ─── 2. O*NET PROFILE MAP ─────────────────────────────────────────────────────

def build_onet_profile_map(kb_df: pd.DataFrame) -> dict:
    """
    Builds profile text that matches exactly what vectorize_kb.py encodes
    so fine-tuning and inference see the same text.
    Includes all 6 fields used in the weighted scoring scheme.
    """
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
    examples = []
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

        examples.append(InputExample(texts=[txt, profile_map[soc]], label=1.0))

        pool = [s for s in major_idx.get(soc[:2], []) if s != soc]
        if len(pool) < HARD_NEGATIVES_PER_SAMPLE:
            extras = [s for s in all_socs if s != soc and s not in pool]
            pool  += random.sample(extras, min(HARD_NEGATIVES_PER_SAMPLE, len(extras)))

        for neg in random.sample(pool, min(HARD_NEGATIVES_PER_SAMPLE, len(pool))):
            examples.append(InputExample(texts=[txt, profile_map[neg]], label=0.0))

    if skipped:
        print(f"   Warning: {skipped} rows skipped — SOC not in O*NET KB.")
    return examples


def build_evaluator(test_df: pd.DataFrame, profile_map: dict):
    s1, s2, scores = [], [], []
    for _, row in test_df.iterrows():
        soc = row['onet_soc']
        if soc not in profile_map:
            continue
        txt = make_survey_text(row['occupation'], row['Job description'])
        s1.append(txt);  s2.append(profile_map[soc]);  scores.append(1.0)
        neg = random.choice([s for s in profile_map if s != soc])
        s1.append(txt);  s2.append(profile_map[neg]);  scores.append(0.0)
    return evaluation.EmbeddingSimilarityEvaluator(s1, s2, scores, name='matchmyjob-eval')


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — S-BERT Domain Fine-Tuning")
    print("=" * 62)

    print(f"\n[1/5] Loading O*NET Knowledge Base...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    profile_map = build_onet_profile_map(kb_df)
    print(f"      {len(profile_map):,} occupations loaded.")

    print(f"\n[2/5] Loading ground truth...")
    gt_df = load_ground_truth(GT_PATH)

    print(f"\n[3/5] Splitting  →  85% train  |  15% test ...")
    gt_df    = gt_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    cut      = int(len(gt_df) * (1 - TEST_SPLIT_RATIO))
    train_df = gt_df.iloc[:cut].copy()
    test_df  = gt_df.iloc[cut:].copy()
    print(f"      Train : {len(train_df):,}  |  Test : {len(test_df):,}")

    # Save test split for match_engine_eval.py
    GT_TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_excel(GT_TEST_SPLIT_PATH, index=False)
    print(f"      Test split saved → {GT_TEST_SPLIT_PATH.name}")

    print(f"\n[4/5] Building training examples...")
    train_examples = build_training_pairs(train_df, profile_map)
    n_pos = sum(1 for e in train_examples if e.label == 1.0)
    n_neg = sum(1 for e in train_examples if e.label == 0.0)
    print(f"      Positives: {n_pos:,}  |  Negatives: {n_neg:,}  |  Total: {len(train_examples):,}")
    evaluator = build_evaluator(test_df, profile_map)

    print(f"\n[5/5] Fine-tuning '{BASE_MODEL_NAME}'...")
    model   = SentenceTransformer(BASE_MODEL_NAME)
    loader  = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    loss_fn = losses.CosineSimilarityLoss(model)
    warmup  = int(len(loader) * EPOCHS * WARMUP_STEPS_RATIO)

    print(f"      Epochs: {EPOCHS}  |  Batch: {TRAIN_BATCH_SIZE}  |  Warmup: {warmup}")
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
    print(f"  Model saved → {FINETUNED_MODEL_PATH}")
    print(f"\n  NEXT STEP — rebuild the vector cache:")
    print(f"    python src/vectorize_kb.py")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
