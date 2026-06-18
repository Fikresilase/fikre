"""Phase 0 — central config: paths, seeds, model routing, retrieval/gen constants."""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
BASE = Path(os.environ.get("FIKRE_BASE", "/workspace"))
DATA_DIR = BASE / "data"
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / os.environ.get("TRAIN_FILE", "Train.csv")
VAL_CSV = DATA_DIR / os.environ.get("VAL_FILE", "Val.csv")
TEST_CSV = DATA_DIR / os.environ.get("TEST_FILE", "Test.csv")

# cached artifacts (each phase writes one; next phase reads it -> crash-resumable)
SAMPLE_CSV = ARTIFACTS / "sample.csv"
TRAIN_EMB_NPY = ARTIFACTS / "train_input_emb.npy"
TRAIN_IDS_NPY = ARTIFACTS / "train_ids.npy"
CANDIDATES_JSON = ARTIFACTS / "candidates.json"
EXEMPLARS_JSON = ARTIFACTS / "exemplars.json"
ANSWERS_CSV = ARTIFACTS / "answers.csv"
SCORES_TABLE_CSV = ARTIFACTS / "scores_table.csv"

# data columns
COL_ID, COL_IN, COL_OUT, COL_SUBSET = "ID", "input", "output", "subset"

# ----------------------------------------------------------------------------- repro
SEED = 42

# ----------------------------------------------------------------------------- models
# Confirmed repo IDs (verified on HuggingFace). Override via env without editing code.
MODEL_AFRIQWEN = os.environ.get("MODEL_AFRIQWEN", "McGill-NLP/AfriqueQwen-14B")  # English + Aka/Swa/Amh
MODEL_SUNFLOWER = os.environ.get("MODEL_SUNFLOWER", "Sunbird/Sunflower-14B")     # Luganda
# Optional English specialist — only wire back in if Val shows English ROUGE lagging.
MODEL_QWEN = os.environ.get("MODEL_QWEN", "Qwen/Qwen3.5-35B-A3B")

EMBEDDER = os.environ.get("EMBEDDER", "BAAI/bge-m3")
RERANKER = os.environ.get("RERANKER", "BAAI/bge-reranker-v2-m3")

# subset -> human language name for the prompt
SUBSET_LANG = {
    "Eng_Uga": "English", "Eng_Gha": "English", "Eng_Ken": "English", "Eng_Eth": "English",
    "Aka_Gha": "Akan (Twi)", "Swa_Ken": "Swahili", "Amh_Eth": "Amharic", "Lug_Uga": "Luganda",
}
# AfriqueQwen handles English + Akan/Swahili/Amharic; Sunflower handles Luganda.
SUBSET_MODEL = {
    "Eng_Uga": MODEL_AFRIQWEN, "Eng_Gha": MODEL_AFRIQWEN, "Eng_Ken": MODEL_AFRIQWEN,
    "Eng_Eth": MODEL_AFRIQWEN, "Aka_Gha": MODEL_AFRIQWEN, "Swa_Ken": MODEL_AFRIQWEN,
    "Amh_Eth": MODEL_AFRIQWEN, "Lug_Uga": MODEL_SUNFLOWER,
}
# generation order = grouped by model, so each model loads once then is torn down.
# 2 passes now (no 70GB Qwen): AfriqueQwen for 7 subsets, Sunflower for Luganda.
MODEL_PASSES = [
    (MODEL_AFRIQWEN, ["Eng_Uga", "Eng_Gha", "Eng_Ken", "Eng_Eth", "Aka_Gha", "Swa_Ken", "Amh_Eth"]),
    (MODEL_SUNFLOWER, ["Lug_Uga"]),
]

# ----------------------------------------------------------------------------- retrieval
N_CANDIDATES = 20      # dense top-k before rerank
N_EXEMPLARS = 8        # after rerank, into the prompt

# ----------------------------------------------------------------------------- generation
GEN_BACKEND = os.environ.get("GEN_BACKEND", "vllm").lower()   # vllm | hf
DTYPE = "bfloat16"
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
# per-subset answer length cap (African langs tokenize longer -> a bit more headroom)
MAX_NEW_TOKENS = {
    "Eng_Uga": 256, "Eng_Gha": 256, "Eng_Ken": 256, "Eng_Eth": 256,
    "Aka_Gha": 384, "Swa_Ken": 384, "Amh_Eth": 512, "Lug_Uga": 384,
}
CHECKPOINT_EVERY = 50   # rows, during generation
