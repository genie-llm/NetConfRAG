"""
baselines/hierarchical_rag_finetuned_reranker.py
    — Baseline 5: LLM + hierarchical RAG + fine-tuned embeddings + reranking.

Hybrid retrieval pipeline:
  1. Dense (FAISS, fine-tuned embeddings) — top k_dense
  2. Lexical (BM25)                       — top k_bm25
  3. Reciprocal Rank Fusion (RRF)         — top k_fused
  4. Cross-encoder reranking (BGE)        — top k_final
  5. LLM generation

Usage:
    python baselines/hierarchical_rag_finetuned_reranker.py
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Any

import torch
import pandas as pd
from huggingface_hub import login
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from build_vectorstore import load_embeddings, load_chunks_as_documents, FAISS_INDEX_DIR
from rag_inference import load_llm, build_langchain_llm, PROMPT
from utils import (
    extract_cisco_config,
    sanitize_ios_only,
    retrieve_with_scores,
    build_context_and_provenance_full,
)


# ── BM25 retrieval ─────────────────────────────────────────────────────────────

def retrieve_bm25_with_scores(
    retriever: BM25Retriever, query: str, k: int = 50
) -> List[Tuple[Document, float]]:
    """Return [(Document, pseudo_score)] from BM25 (score = 1/rank)."""
    docs = retriever.get_relevant_documents(query)[:k]
    return [(doc, 1.0 / (1 + i)) for i, doc in enumerate(docs)]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def rrf_fusion(
    dense_list: List[Tuple[Document, float]],
    bm25_list:  List[Tuple[Document, float]],
    k_rrf: int = 60,
    top_k: int = 20,
) -> List[Tuple[Document, float]]:
    """Merge dense + BM25 rankings using Reciprocal Rank Fusion."""

    def doc_key(doc: Document):
        return (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content[:64],
        )

    def to_rank_map(pairs):
        return {doc_key(doc): (idx + 1) for idx, (doc, _) in enumerate(pairs)}

    dense_ranks = to_rank_map(dense_list)
    bm25_ranks  = to_rank_map(bm25_list)

    all_docs: Dict[tuple, Document] = {}
    for doc, _ in dense_list + bm25_list:
        all_docs[doc_key(doc)] = doc

    fused = []
    for key, doc in all_docs.items():
        score = 0.0
        if key in dense_ranks:
            score += 1.0 / (k_rrf + dense_ranks[key])
        if key in bm25_ranks:
            score += 1.0 / (k_rrf + bm25_ranks[key])
        fused.append((doc, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


# ── Cross-encoder reranking ────────────────────────────────────────────────────

def rerank_documents(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    reranker: CrossEncoder,
    top_k: int = 4,
) -> List[Tuple[Document, float]]:
    """Re-score candidate documents with a cross-encoder and keep top_k."""
    if not docs_with_scores:
        return []

    texts = []
    for doc, _ in docs_with_scores:
        md   = doc.metadata or {}
        text = md.get("display_text") or doc.page_content or ""
        texts.append(text)

    pairs         = [(query, t) for t in texts]
    rerank_scores = reranker.predict(pairs)
    reranked      = [
        (doc, float(sc)) for (doc, _), sc in zip(docs_with_scores, rerank_scores)
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# ── Full hybrid pipeline ───────────────────────────────────────────────────────

def ask_rag_reranked(
    question: str,
    db,
    bm25_retriever: BM25Retriever,
    reranker: CrossEncoder,
    llm_chain: LLMChain,
    k_dense: int  = None,
    k_bm25: int   = None,
    k_fused: int  = None,
    k_final: int  = None,
) -> Dict[str, Any]:
    """
    Full hybrid pipeline:
      dense (FAISS) → BM25 → RRF fusion → cross-encoder reranking → LLM.
    """
    k_dense = k_dense or config.RERANKER_K_DENSE
    k_bm25  = k_bm25  or config.RERANKER_K_BM25
    k_fused = k_fused or config.RERANKER_K_FUSED
    k_final = k_final or config.RERANKER_K_FINAL

    print(f"🔍 FAISS (dense) top {k_dense}…")
    dense_results = retrieve_with_scores(db, question, k=k_dense)

    print(f"🔎 BM25 (lexical) top {k_bm25}…")
    bm25_results = retrieve_bm25_with_scores(bm25_retriever, question, k=k_bm25)

    print(f"🧪 RRF fusion → {k_fused} candidates…")
    fused = rrf_fusion(dense_results, bm25_results, k_rrf=60, top_k=k_fused)

    print(f"🎯 Reranking → top {k_final}…")
    reranked = rerank_documents(question, fused, reranker, top_k=k_final)

    context, used_chunks = build_context_and_provenance_full(reranked)
    for i, chunk_info in enumerate(used_chunks):
        if i < len(reranked):
            chunk_info["rerank_score"]      = reranked[i][1]
            chunk_info["retrieval_method"]  = "hybrid_rrf_reranked"

    print("🤖 Generating answer…")
    raw  = llm_chain.invoke({"context": context, "question": question})
    text = raw["text"] if isinstance(raw, dict) and "text" in raw else str(raw)
    try:
        answer = sanitize_ios_only(text)
    except Exception:
        answer = text

    return {
        "answer":       answer,
        "context_used": context,
        "used_chunks":  used_chunks,
        "count":        len(used_chunks),
        "retrieval_config": {
            "mode":           "hybrid_rrf_reranked",
            "k_dense":        k_dense,
            "k_bm25":         k_bm25,
            "k_fused":        k_fused,
            "k_final":        k_final,
            "reranker_model": config.RERANKER_MODEL,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    login(config.HF_TOKEN)

    # ── Embeddings ──
    embeddings = load_embeddings(config.CHECKPOINT_DIR)
    if embeddings is None:
        raise RuntimeError(
            "Could not load fine-tuned embeddings.\n"
            "Check CHECKPOINT_DIR in config.py."
        )

    # ── FAISS ──
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"Loading FAISS index from {FAISS_INDEX_DIR}/…")
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"FAISS index not found. Building from {config.CHUNKS_JSON}…")
        docs = load_chunks_as_documents(config.CHUNKS_JSON)
        db   = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_INDEX_DIR)
    print("✅ FAISS index ready.\n")

    # ── BM25 ──
    print("Building BM25 index…")
    docs_for_bm25  = load_chunks_as_documents(config.CHUNKS_JSON)
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = config.RERANKER_K_BM25
    print("✅ BM25 index ready.\n")

    # ── Reranker ──
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(config.RERANKER_MODEL, device=device, max_length=512)
    print(f"✅ Reranker ({config.RERANKER_MODEL}) loaded.\n")

    # ── LLM ──
    model, tokenizer = load_llm()
    llm              = build_langchain_llm(model, tokenizer)
    llm_chain        = LLMChain(llm=llm, prompt=PROMPT)

    # ── CSV ──
    if not os.path.exists(config.INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV)
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    TARGET_COL = "hrag+finetuning+reranking"
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""
    rows_to_process = df.index[df[TARGET_COL].astype(str).str.strip() == ""].tolist()
    print(f"Total rows: {len(df)} | To generate: {len(rows_to_process)}\n")

    output_csv = config.OUTPUT_CSV.replace(
        ".csv", "_hierarchical_rag_finetuned_reranker.csv"
    )

    for i in rows_to_process:
        q = str(df.at[i, "question"]).strip()
        if not q:
            df.at[i, TARGET_COL] = ""
            continue
        try:
            res     = ask_rag_reranked(q, db, bm25_retriever, reranker, llm_chain)
            cleaned = extract_cisco_config(res["answer"])
            df.at[i, TARGET_COL] = cleaned
            print(f"[{i+1}/{len(df)}] ✅ {q[:60]}")
        except KeyboardInterrupt:
            print("\nInterrupted. Saving progress…")
            break
        except Exception as e:
            print(f"[{i+1}/{len(df)}] ⚠️  Failed (row {i}): {e}")

        df.to_csv(output_csv, index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if config.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS)

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Done → {output_csv}")


if __name__ == "__main__":
    main()
