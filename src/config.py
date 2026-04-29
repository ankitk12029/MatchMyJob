from pathlib import Path

# Root = MatchMyJob/  (this file lives in MatchMyJob/src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Directories ──────────────────────────────────────────────────────────────
DATA_DIR           = ROOT_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR         = DATA_DIR / "output"
MODELS_DIR         = ROOT_DIR / "models"
VECTORS_DIR        = ROOT_DIR / "vectors"          # ← cached KB embeddings live here

# ─── Input files ──────────────────────────────────────────────────────────────
USER_INPUT_FILE      = RAW_DATA_DIR / "user_survey_input.csv"
GT_PATH              = RAW_DATA_DIR / "filtered_raters_matched.xlsx"
GT_NON_MATCHED_PATH      = RAW_DATA_DIR / "filtered_raters_non-matched.xlsx"  # 100% non-matched
GT_MASTER_TRAINING_PATH  = RAW_DATA_DIR / "master_training_data.xlsx"          # primary dataset (Combined sheet, 3,305 rows)
GT_TITLE_ONLY_PATH       = RAW_DATA_DIR / "training_data_without_descrption.xlsx"  # title-only dataset (Sheet4, ~3,052 rows)

# ─── Test split paths (one per scenario) ──────────────────────────────────────
GT_TEST_SPLIT_PATH       = RAW_DATA_DIR / "ground_truth_test_split.xlsx"         # Scenario 1 & 2 — 15% of matched only
GT_TEST_SPLIT_S3_PATH    = RAW_DATA_DIR / "ground_truth_test_split_s3.xlsx"      # Scenario 3 — 15% of combined (1,926 rows)

# ─── Processed / model files ──────────────────────────────────────────────────
KB_PATH              = PROCESSED_DATA_DIR / "onet_knowledge_base.csv"
# BASE_MODEL_NAME      = "BAAI/bge-base-en-v1.5"   # upgraded from bge-small (33M→109M params)
BASE_MODEL_NAME      = "BAAI/bge-small-en-v1.5"  # previous — 43.3% top-1
# BASE_MODEL_NAME      = "all-MiniLM-L6-v2"         # previous — 43% top-1
# BASE_MODEL_NAME      = "all-mpnet-base-v2"         # OOM on Mac 20GB MPS
FINETUNED_MODEL_PATH = MODELS_DIR / "matchmyjob-finetuned"

# ─── Output files ─────────────────────────────────────────────────────────────
EVAL_OUTPUT_PATH     = OUTPUT_DIR / "eval_machine_vs_human.csv"
MATCH_OUTPUT_PATH    = OUTPUT_DIR / "matched_jobs_results.csv"

# ─── Vector cache file names (saved inside VECTORS_DIR) ───────────────────────
# One cache file per model so switching models auto-invalidates the old cache.
# vectorize_kb.py writes these; match_engine.py and match_engine_eval.py read them.
def vector_cache_path(model_name: str) -> Path:
    """Returns path to the .pt cache bundle for a given model."""
    safe_name = str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    # Use only the last folder name for local paths so the filename stays short
    safe_name = Path(safe_name).name
    return VECTORS_DIR / f"kb_vectors_{safe_name}.pt"

