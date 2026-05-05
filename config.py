"""
config.py — Central configuration for the Cisco RAG pipeline.

Edit the paths below before running any script.
"""

import os

# ── Data paths ─────────────────────────────────────────────────────────────────

# Directory containing the raw JSON knowledge-base files (one JSON per PDF)
DATA_JSON_DIR = "data_json/"
DATA_PDF_DIR = "data_pdf/"

# Output of chunking.py — consumed by build_vectorstore.py
CHUNKS_JSON = "chunks.json"

# ── Model paths ────────────────────────────────────────────────────────────────

# Directory that contains the fine-tuned embedding checkpoints
# (sub-folders with metadata.json + model weights)
CHECKPOINT_DIR = "./e5-finetuned/"


# HuggingFace model ID for the LLM
LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# ── HuggingFace auth ───────────────────────────────────────────────────────────
#HF_TOKEN = "PUT YOUR HF TOKEN HERE" 
#less risky to export the HF_TOKEN in command line
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# ── Chunking hyperparameters ───────────────────────────────────────────────────

CHUNK_MAX_TOKENS     = 512
CHUNK_OVERLAP_TOKENS = 50

# ── RAG hyperparameters ────────────────────────────────────────────────────────

RAG_K = 4   # number of chunks passed to the LLM (dense-only pipelines)

# ── Batch evaluation ───────────────────────────────────────────────────────────

INPUT_CSV  = "datasets/NetConfRAG_17.csv"    # must contain a 'question' column
OUTPUT_CSV = "results/results_17.csv"

SAVE_RAW             = False   # also save the raw (pre-sanitize) LLM output
SLEEP_BETWEEN_CALLS  = 0.0    # seconds between rows (helps on memory-constrained GPUs)

# ── Reranker (baseline 5 only) ─────────────────────────────────────────────────

RERANKER_MODEL   = "BAAI/bge-reranker-base"
RERANKER_K_DENSE = 30   # FAISS candidates before reranking
RERANKER_K_BM25  = 50   # BM25 candidates before reranking
RERANKER_K_FUSED = 20   # candidates after RRF fusion
RERANKER_K_FINAL = 4    # docs passed to LLM after reranking
