"""
baselines/basic_rag.py — Baseline 2: LLM + basic RAG.

Uses flat paragraph-level chunking (from raw PDFs via PyMuPDF), a vanilla
sentence-transformer embedding model (intfloat/e5-base-v2), and FAISS similarity
search — no hierarchical structure, no fine-tuned embeddings.

Usage:
    python baselines/basic_rag.py
"""

import os
import sys
import json
import time

import fitz  # PyMuPDF
import torch
import pandas as pd
from huggingface_hub import login
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils import extract_cisco_config, sanitize_ios_only, PROMPT_TEMPLATE_STR

# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str, header_ratio=0.1, footer_ratio=0.1) -> str:
    """Extract clean text from a PDF, skipping headers and footers."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        blocks     = page.get_text("blocks")
        page_h     = page.rect.height
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if y0 > page_h * header_ratio and y1 < page_h * (1 - footer_ratio):
                full_text += text + "\n"
    return full_text.strip()


def build_documents_from_pdf_dir(pdf_dir: str) -> list:
    """Convert all PDFs in a directory to LangChain Documents (flat chunking)."""
    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    lines_per_chunk = 20
    max_chars       = 1000
    overlap_chars   = 100

    for pdf_file in pdf_files:
        raw_text   = extract_text_from_pdf(os.path.join(pdf_dir, pdf_file))
        paragraphs = [
            " ".join(p.strip().splitlines())
            for p in raw_text.split("\n\n") if p.strip()
        ]

        i        = 0
        chunk_id = 0
        while i < len(paragraphs):
            current = paragraphs[i : i + lines_per_chunk]
            chunk   = "\n".join(current).strip()

            if len(chunk) > max_chars:
                for j in range(0, len(chunk), max_chars):
                    documents.append(Document(
                        page_content=chunk[j : j + max_chars],
                        metadata={"source": pdf_file, "chunk_id": f"{chunk_id}.{j//max_chars}"},
                    ))
            else:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": pdf_file, "chunk_id": chunk_id},
                ))

            if len(chunk) > overlap_chars:
                overlap_count = 0
                char_total    = 0
                for k in range(1, len(current) + 1):
                    char_total += len(current[-k]) + 1
                    if char_total >= overlap_chars:
                        overlap_count = k
                        break
                i += max(1, lines_per_chunk - overlap_count)
            else:
                i += lines_per_chunk
            chunk_id += 1

    print(f"✅ {len(documents)} documents created from {len(pdf_files)} PDFs.")
    return documents


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm():
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading LLM: {config.LLM_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ LLM loaded.")
    return model, tokenizer


def build_langchain_llm(model, tokenizer) -> HuggingFacePipeline:
    gen_pipe = hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


# ── RAG function ───────────────────────────────────────────────────────────────

def ask_basic_rag(question: str, db, llm_chain: LLMChain, k: int = 4) -> str:
    """Simple RAG: FAISS similarity search → LLM."""
    results  = db.similarity_search_with_score(question, k=k)
    seen     = set()
    parts    = []
    for doc, _ in results:
        txt = doc.page_content.strip()
        key = txt[:200]
        if key not in seen:
            seen.add(key)
            parts.append(txt)
    context = "\n\n---\n\n".join(parts)[:4000]

    raw  = llm_chain.invoke({"context": context, "question": question})
    text = raw["text"] if isinstance(raw, dict) and "text" in raw else str(raw)
    try:
        return sanitize_ios_only(text)
    except Exception:
        return text


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    login(config.HF_TOKEN)

    # NOTE: this baseline uses PDFs directly.
    # Set DATA_PDF_DIR in config.py if you want to use this baseline with PDFs.
    pdf_dir = getattr(config, "DATA_PDF_DIR", config.DATA_JSON_DIR)
    if not os.path.isdir(pdf_dir):
        raise RuntimeError(
            f"PDF directory not found: {pdf_dir}\n"
            "Set DATA_PDF_DIR in config.py to your PDF folder."
        )

    print("Building flat chunk index from PDFs…")
    documents  = build_documents_from_pdf_dir(pdf_dir)
    emb_model  = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    db         = FAISS.from_documents(documents, emb_model)
    print("✅ FAISS index built.\n")

    model, tokenizer = load_llm()
    llm       = build_langchain_llm(model, tokenizer)
    prompt    = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE_STR)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    if not os.path.exists(config.INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV)
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    TARGET_COL = "basic_rag"
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""
    rows_to_process = df.index[df[TARGET_COL].astype(str).str.strip() == ""].tolist()
    print(f"Total rows: {len(df)} | To generate: {len(rows_to_process)}\n")

    output_csv = config.OUTPUT_CSV.replace(".csv", "_basic_rag.csv")

    for i in rows_to_process:
        q = str(df.at[i, "question"]).strip()
        if not q:
            df.at[i, TARGET_COL] = ""
            continue
        try:
            raw     = ask_basic_rag(q, db, llm_chain, k=config.RAG_K)
            cleaned = extract_cisco_config(raw)
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
