"""
vectorize_kb.py — One-Time KB Vectorisation
=============================================
Run this ONCE after:
  1. master_data_creation.py  (builds onet_knowledge_base.csv)
  2. finetune_model.py        (trains the fine-tuned model)

Encodes all 6 KB fields and saves them as a single .pt bundle in vectors/.
match_engine.py and match_engine_eval.py load from this cache instead of
re-encoding every run — saves ~3-5 minutes per run.

Re-run this script whenever you:
  • Rebuild the knowledge base (master_data_creation.py)
  • Switch MODEL_NAME to a different model

Usage:
    python vectorize_kb.py
"""

import os
import json
import time
import textwrap
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import (
    KB_PATH, VECTORS_DIR, FINETUNED_MODEL_PATH,
    BASE_MODEL_NAME, vector_cache_path
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── Which model to vectorise with ────────────────────────────────────────────
# Switch to BASE_MODEL_NAME before fine-tuning to create a baseline cache.
# Switch to FINETUNED_MODEL_PATH after fine-tuning to create the production cache.
MODEL_NAME = str(FINETUNED_MODEL_PATH)   # ← change to BASE_MODEL_NAME for baseline


# ─── Encoding helpers (identical to match_engine.py) ──────────────────────────

def encode_weighted_tasks(model, df: pd.DataFrame) -> torch.Tensor:
    """Weighted task vector using O*NET importance ratings."""
    print("   Encoding Tasks (importance-weighted)...")
    all_vectors = []
    for _, row in df.iterrows():
        json_str = row['Structured_Tasks']
        if not isinstance(json_str, str) or not json_str.strip():
            all_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue
        try:
            tasks = json.loads(json_str)
        except Exception:
            all_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue

        texts   = [t.get('task', '')    for t in tasks if t.get('task', '')]
        weights = [t.get('weight', 0.5) for t in tasks if t.get('task', '')]

        if not texts:
            all_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue

        task_vecs     = model.encode(texts, convert_to_tensor=True)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        weighted      = task_vecs * weight_tensor
        avg           = torch.sum(weighted, dim=0) / torch.sum(weight_tensor)
        all_vectors.append(avg)

    return torch.stack(all_vectors)


def encode_long_text(model, text: str, chunk_size: int = 800) -> torch.Tensor:
    """Chunking encoder for long text fields."""
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(model.get_sentence_embedding_dimension())
    chunks = textwrap.wrap(text, width=chunk_size)
    if not chunks:
        return torch.zeros(model.get_sentence_embedding_dimension())
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return torch.mean(embeddings, dim=0)


def encode_column(model, series: pd.Series, label: str) -> torch.Tensor:
    """Encodes a whole KB column and shows progress."""
    print(f"   Encoding {label}...")
    return torch.stack([encode_long_text(model, t) for t in series])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print("  MatchMyJob — KB Vectorisation (one-time)")
    print("=" * 62)

    # ── Check output path ─────────────────────────────────────────────────────
    cache_path = vector_cache_path(MODEL_NAME)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model      : {MODEL_NAME}")
    print(f"  Cache path : {cache_path}")

    if cache_path.exists():
        print(f"\n  [INFO] Cache already exists at:\n    {cache_path}")
        answer = input("  Re-vectorise and overwrite? [y/N]: ").strip().lower()
        if answer != 'y':
            print("  Aborted — existing cache kept.")
            return

    # ── Load KB ───────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading Knowledge Base from {KB_PATH.name}...")
    kb_df = pd.read_csv(KB_PATH).fillna('')
    print(f"      {len(kb_df):,} occupations loaded.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading model...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print(f"      Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ── Encode all 6 fields ───────────────────────────────────────────────────
    print(f"\n[3/3] Encoding 6 KB fields...")
    vectors = {
        'task_embeddings'      : encode_weighted_tasks(model, kb_df),
        'skill_embeddings'     : encode_column(model, kb_df['All_Tech_Skills'],  'All_Tech_Skills'),
        'tool_embeddings'      : encode_column(model, kb_df['All_Tools'],        'All_Tools'),
        'alt_title_embeddings' : encode_column(model, kb_df['All_Alt_Titles'],   'All_Alt_Titles'),
        'ofc_title_embeddings' : encode_column(model, kb_df['Title'],            'Title (official)'),
        'desc_embeddings'      : encode_column(model, kb_df['Description'],      'Description'),
        # Metadata stored alongside so loaders can sanity-check
        'meta': {
            'model_name' : MODEL_NAME,
            'kb_rows'    : len(kb_df),
            'kb_path'    : str(KB_PATH),
            'created_at' : time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    torch.save(vectors, cache_path)

    elapsed = time.time() - t0
    size_mb = cache_path.stat().st_size / 1_000_000
    print(f"\n{'=' * 62}")
    print(f"  Done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Cache saved → {cache_path}  ({size_mb:.1f} MB)")
    print(f"\n  Next runs of match_engine.py and match_engine_eval.py")
    print(f"  will load this cache instantly instead of re-encoding.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
