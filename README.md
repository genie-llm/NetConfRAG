# NetConfRAG

> **RAG-based pipeline for autonomous Cisco IOS configuration generation using a lightweight 3B-parameter LLM.**

This repository accompanies the paper below and please cite it if you use this code. 
It investigates how Retrieval-Augmented Generation (RAG) combined with hierarchical chunking, fine-tuned embeddings, and domain-adapted representations can bridge the gap between human intent and low-level Cisco IOS device configuration — while running entirely on a resource-constrained, 3B-parameter LLM.


Yasmine Ouni, Kamal Singh, Antoine Gourru, Farouk Mhamdi, "WIP: Large Language Models for Network Automation: A Hierarchical Retrieval-Augmented Approach", 27th IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM), Bologna, Italy, June 16-19, 2026

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Baselines](#baselines)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Fine-tuning](#fine-tuning)
- [Configuration Reference](#configuration-reference)

---

## Overview

Traditional network management is increasingly bottlenecked by the manual effort required to translate high-level intent into low-level device configuration. This project explores whether a lightweight LLM — conditioned at inference time on external Cisco documentation via RAG — can generate technically viable IOS configurations from natural-language questions.

Key findings from the paper:
- **Hierarchy-based RAG with fine-tuned embeddings (HRAG-FT)** outperforms both the LLM-only baseline and standard flat RAG.
- The 3B-parameter Llama 3.2 Instruct model remains viable in resource-constrained settings when combined with RAG.
- Hallucination and completeness remain the primary challenges toward fully autonomous configuration.

---

## Architecture

The pipeline is composed of four sequential stages:

```
PDF documentation
      │
      ▼
 0. PDF → JSON  (data_pdf/pdf_to_json.py)           ← optional, if starting from PDFs
    Font-driven extraction, 5 chunking strategies,
    flat + hierarchical JSON output
      │
      ▼
 1. Chunking (chunking.py)
    Hierarchical chunking with semantic headers,
    CLI-block-aware splitting, tiktoken budgeting
      │
      ▼
 2. Vector Store (build_vectorstore.py)
    Fine-tuned intfloat/e5-small-v2 embeddings
    + FAISS index
      │
      ▼
 3. RAG Inference (rag_inference.py / hierarchical_rag.py)
    Retrieval → context assembly → LLaMA 3.2-3B-Instruct (4-bit)
      │
      ▼
 4. Evaluation (evaluate_ollama.py / evaluate_gemini.py)
    LLM-as-a-judge scoring on 5 metrics
```

**Embedding note:** The pipeline uses `intfloat/e5-small-v2` with mandatory `"query: "` / `"passage: "` prefixes. A custom `E5PrefixEmbeddings` wrapper handles these automatically and includes a double-prefix guard for LangChain's internal FAISS call pattern.

---

## Project Structure

```
NetConfRAG/
├── config.py                        # Central configuration (paths, models, hyperparameters)
├── chunking.py                      # Stage 1 — hierarchical chunking of JSON docs
├── build_vectorstore.py             # Stage 2 — fine-tuned embeddings + FAISS index build
├── rag_inference.py                 # Stage 3 — RAG query + LLM inference
├── batch_evaluate.py                # Stage 3b — batch CSV inference runner
├── hierarchical_rag.py              # Hierarchical RAG with vanilla e5-small-v2 embeddings
├── hierarchical_rag_ft.py           # Hierarchical RAG with fine-tuned embeddings (local checkpoint)
├── utils.py                         # Shared utilities (sanitiser, chunking helpers, prompts)
├── evaluate_ollama.py               # LLM-as-a-judge via local Ollama models
├── evaluate_gemini.py               # LLM-as-a-judge via Google Gemini API
│
├── baselines/
│   ├── llm_only.py                  # Baseline 1 — LLM only, no retrieval
│   ├── basic_rag.py                 # Baseline 2 — flat RAG from raw PDFs
│   └── hierarchical_rag_finetuned_reranker.py  # Baseline 5 — HRAG + FT + BM25 + reranking
│
├── data_pdf/
│   ├── pdf_to_json.py               # PDF → JSON corpus builder (font-driven extraction)
│   └── README.md                    # Instructions for placing source PDFs
│
├── data_json/                       # JSON knowledge base (one file per PDF)
│   └── README.md
│
├── finetuning/
│   └── finetune_e5.py               # Fine-tuning script for intfloat/e5-small-v2
│
├── datasets/
│   ├── NetConfRAG_17.csv            # Small evaluation set (17 questions)
│   ├── NetConfRAG_144.csv           # Full evaluation set (144 questions)
│   └── NetConfRAG_144_ground_truth.csv
├── results/
│   └── results_17.csv
└── graphs/
    ├── graph.py                     # Matplotlib bar chart generation
    ├── comparison_bar_chart.pdf
    └── comparison_bar_chart.png
```

---

## Baselines

| ID | Script | Method |
|----|--------|--------|
| B1 | `baselines/llm_only.py` | LLM only — no retrieval |
| B2 | `baselines/basic_rag.py` | Flat RAG from raw PDFs, vanilla e5-base-v2 |
| B3 | `hierarchical_rag.py` | Hierarchical RAG, vanilla e5-small-v2 |
| B4 | `hierarchical_rag_ft.py` / `batch_evaluate.py` | Hierarchical RAG + fine-tuned e5 embeddings |
| B5 | `baselines/hierarchical_rag_finetuned_reranker.py` | B4 + BM25 + RRF fusion + BGE cross-encoder reranking |

---

## Setup

### Requirements

```bash
pip install torch transformers langchain langchain-community faiss-cpu \
            sentence-transformers huggingface_hub tiktoken pandas tqdm \
            rank_bm25 pymupdf
```

For Gemini evaluation:
```bash
pip install google-generativeai
```

For PDF extraction (`data_pdf/pdf_to_json.py`):
```bash
pip install pdfplumber langchain-text-splitters tqdm
```

For Ollama evaluation, install [Ollama](https://ollama.com) and pull at least one judge model:
```bash
ollama pull llama3.3:70b
```

### HuggingFace Access

The LLM (`meta-llama/Llama-3.2-3B-Instruct`) is a gated model. You need a HuggingFace account with access granted, then set your token as an environment variable:

```bash
export HF_TOKEN="your_HuggingFace_token_here"
```

For the Gemini evaluator:
```bash
export GEMINI_API_KEY="your_key_here"
```

> **Security note:** Never commit API keys or tokens to version control. Both secrets are read via `os.environ.get(...)` in `config.py` — no hardcoded values.

---

## Usage

### 0. Build the JSON corpus from PDFs (optional)

If you are starting from raw PDF documentation rather than pre-built JSON files, use the font-driven extractor:

```bash
# Recommended: font-driven, block-preserving, both output formats
python data_pdf/pdf_to_json.py --input data_pdf/ --strategy block_preserve --deduplicate --verbose

# Ablation sweep over chunk sizes
for size in 64 128 256 512; do
  python data_pdf/pdf_to_json.py --input data_pdf/ --strategy block_preserve \
      --chunk-size $size --hier-out data_json/hier_${size}.json --mode hierarchical
done
```

Available strategies: `paragraph`, `fixed`, `semantic`, `section`, `block_preserve` (default, recommended — never splits mid-command or mid-step table).

Place the resulting JSON files in `data_json/` before running stage 1.

### 1. Chunk the knowledge base

Place your JSON files (one per PDF, structured with `chapters` → `sections`) in `data_json/`, then:

```bash
python chunking.py
```

Outputs `chunks.json`.

### 2. Build the FAISS vector store

```bash
python build_vectorstore.py
```

Outputs `faiss_index/`. Requires a fine-tuned checkpoint in `e5-finetuned/` (see [Fine-tuning](#fine-tuning) or download below).

### 3. Run RAG inference

Single question (interactive):
```bash
python rag_inference.py --question "How to configure an IPv6 ACL on an interface?"
```

Batch over a CSV:
```bash
python batch_evaluate.py
```

Debug mode (verbose prefix / FAISS / LLM logging):
```bash
DEBUG_RAG=1 python rag_inference.py --question "..."
```

### 4. Run a baseline

```bash
python baselines/llm_only.py
python baselines/basic_rag.py
python hierarchical_rag.py
python hierarchical_rag_ft.py
python baselines/hierarchical_rag_finetuned_reranker.py
```

---

## Evaluation

Evaluation uses an **LLM-as-a-judge** approach scoring each generated configuration on five weighted metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Accuracy | 20% | Syntactic correctness of CLI commands |
| Relevance | 20% | Adherence to the specific task asked |
| Correctness | 30% | Functional / logical correctness of networking logic |
| Hallucination | 20% | Absence of invented or non-existent commands |
| Completeness | 10% | Readiness for real deployment |

### Evaluate with local Ollama models

```bash
# Single pair
python evaluate_ollama.py \
    --question "How to configure OSPF area 0?" \
    --answer "enable\nconfigure terminal\n..."

# Batch CSV
python evaluate_ollama.py \
    --csv results/results_17.csv \
    --answer-col "rag2+finetuning" \
    --question-col "question"
```

### Evaluate with Gemini

```bash
export GEMINI_API_KEY="your_key_here"

python evaluate_gemini.py \
    --csv results/results_17.csv \
    --answer-col "rag2+finetuning"
```

Output is saved to `<input>_evaluated.csv` / `<input>_evaluated_gemini.csv` with per-metric score columns appended.

---

## Datasets

| File | Questions | Description |
|------|-----------|-------------|
| `NetConfRAG_17.csv` | 17 | Small dev/test set used during development |
| `NetConfRAG_144.csv` | 144 | Full benchmark set |
| `NetConfRAG_144_ground_truth.csv` | 144 | With reference configurations |

Each CSV must contain at minimum a `question` column. Inference scripts append an answer column and are fully resumable (already-filled rows are skipped on re-run).

---

## Fine-tuning

The `finetuning/` directory contains the training script for domain-adapting `intfloat/e5-small-v2` on Cisco RAG triplets.

### Script

`finetuning/finetune_e5.py` — fine-tunes e5-small-v2 using in-batch negatives + margin loss on `(query, positive_passage, negative_passage)` triplets. Checkpoints are saved in SentenceTransformer/HuggingFace format and are directly consumable by `build_vectorstore.py`.

> **Note:** The triplet generation script is not yet included in this repository and will be added in a future release.

### Usage

```bash
# Full training run (auto-split into train/val by topic)
python finetuning/finetune_e5.py --triplets triplets.jsonl --output-dir ./e5-finetuned

# Pre-split files
python finetuning/finetune_e5.py \
    --train triplets_train.jsonl \
    --val   triplets_val.jsonl \
    --output-dir ./e5-finetuned

# Smoke test (fast sanity check, 1 epoch, truncated data)
python finetuning/finetune_e5.py --triplets triplets.jsonl --smoke-test

# Eval-only mode (score an existing checkpoint)
python finetuning/finetune_e5.py --triplets triplets.jsonl --eval-only \
    --output-dir ./e5-finetuned
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 32 | Per-device batch size |
| `--lr` | 2e-5 | Peak learning rate |
| `--temperature` | 0.02 | InfoNCE softmax temperature |
| `--bf16` / `--fp16` | off | Mixed-precision training |
| `--score-threshold` | 2 | Minimum triplet quality score to keep |
| `--eval-every` | 200 | Evaluate on validation set every N steps |
| `--save-every` | 400 | Save `checkpoint_latest` every N steps |

The script automatically resumes from `checkpoint_latest` if it exists in the output directory.

### Pre-trained checkpoint

The fine-tuned checkpoint used in the paper is available for download:

**[Download `e5-finetuned/` from Google Drive](https://drive.google.com/your-link-here)**

Place the contents in `./e5-finetuned/` (i.e., `config.py`'s `CHECKPOINT_DIR`). The directory must contain `checkpoint_best/` with model weights and `sentence_bert_config.json`.

### Additional requirements for fine-tuning

```bash
pip install torch transformers
# Optional: 8-bit Adam for reduced GPU memory usage
pip install bitsandbytes
```

---

## Configuration Reference

All settings are in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_JSON_DIR` | `data_json/` | JSON knowledge-base directory |
| `DATA_PDF_DIR` | `data_pdf/` | Source PDF directory (Baseline 2 / pdf_to_json only) |
| `CHUNKS_JSON` | `chunks.json` | Output of chunking stage |
| `CHECKPOINT_DIR` | `./e5-finetuned/` | Fine-tuned e5 embedding checkpoints |
| `LLM_MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID |
| `HF_TOKEN` | *(env var)* | HuggingFace auth token — set via `export HF_TOKEN=...` |
| `CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between adjacent chunks |
| `RAG_K` | `4` | Number of retrieved chunks passed to LLM |
| `INPUT_CSV` | `datasets/NetConfRAG_17.csv` | Evaluation input |
| `OUTPUT_CSV` | `results/results_17.csv` | Evaluation output |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder for Baseline 5 |
| `RERANKER_K_DENSE` | `30` | Dense FAISS candidates before reranking |
| `RERANKER_K_BM25` | `50` | BM25 candidates before reranking |
| `RERANKER_K_FINAL` | `4` | Documents passed to LLM after reranking |
