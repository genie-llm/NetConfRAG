# NetConfRAG

NetConfRAG is a research codebase for generating Cisco IOS configurations from natural-language questions using Retrieval-Augmented Generation (RAG).

The project focuses on a small local model setup (Llama 3.2 3B) and compares multiple baselines, including hierarchical retrieval and fine-tuned E5 embeddings.

If this repository helps your work, please cite:

> Yasmine Ouni, Kamal Singh, Antoine Gourru, Farouk Mhamdi, "WIP: Large Language Models for Network Automation: A Hierarchical Retrieval-Augmented Approach", 27th IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM), Bologna, Italy, June 16-19, 2026. [HAL link](https://hal.science/hal-05613494v1/file/Towards_Autonomous_Network_Configuration_with_LLMs-15.pdf)

## What is in this repo

- Data preparation from PDF to JSON (optional)
- Hierarchical chunking for network docs
- FAISS vector store building
- RAG inference scripts (single query and batch)
- Baselines (LLM-only, flat RAG, hierarchical variants)
- LLM-based evaluation (Ollama or Gemini)
- Fine-tuning script for E5 embeddings

## Quick start

Install the main dependencies:

```bash
pip install torch transformers langchain langchain-community faiss-cpu \
            sentence-transformers huggingface_hub tiktoken pandas tqdm \
            rank_bm25 pymupdf
```

Optional extras:

```bash
# Gemini evaluator
pip install google-generativeai

# PDF to JSON extraction
pip install pdfplumber langchain-text-splitters
```

Set environment variables:

```bash
export HF_TOKEN="your_huggingface_token"
# only if you use Gemini evaluation
export GEMINI_API_KEY="your_gemini_key"
```

## Typical workflow

1. (Optional) Convert PDFs to JSON:

```bash
python data_pdf/pdf_to_json.py --input data_pdf/ --strategy block_preserve --deduplicate --verbose
```

2. Build chunks from JSON docs:

```bash
python chunking.py
```

3. Build vector store:

```bash
python build_vectorstore.py
```

4. Run inference:

```bash
python rag_inference.py --question "How to configure an IPv6 ACL on an interface?"
```

5. Run batch inference (CSV):

```bash
python batch_evaluate.py
```

## Baselines

- LLM only: baselines/llm_only.py
- Flat RAG: baselines/basic_rag.py
- Hierarchical RAG: hierarchical_rag.py
- Hierarchical RAG + fine-tuned embeddings: hierarchical_rag_ft.py
- HRAG + FT + reranking: baselines/hierarchical_rag_finetuned_reranker.py

## Evaluation

Run local evaluation with Ollama:

```bash
python evaluate_ollama.py --csv results/results_17.csv --answer-col "rag2+finetuning" --question-col "question"
```

Run evaluation with Gemini:

```bash
python evaluate_gemini.py --csv results/results_17.csv --answer-col "rag2+finetuning"
```

Both evaluators score generated configs on accuracy, relevance, correctness, hallucination, and completeness.

## Fine-tuning embeddings

Fine-tuning script:

```bash
python finetuning/finetune_e5.py --triplets triplets.jsonl --output-dir ./e5-finetuned
```

If you already have train/val files:

```bash
python finetuning/finetune_e5.py --train triplets_train.jsonl --val triplets_val.jsonl --output-dir ./e5-finetuned
```

Note: triplet generation is not included yet.

## Datasets

- datasets/NetConfRAG_17.csv
- datasets/NetConfRAG_144.csv
- datasets/NetConfRAG_144_ground_truth.csv

Each dataset needs at least a question column.

## Configuration

Main settings live in config.py (paths, models, chunking params, retrieval params).

## Notes

- Keep secrets in environment variables. Do not hardcode keys in source files.
- The HF model meta-llama/Llama-3.2-3B-Instruct is gated and requires access approval.
