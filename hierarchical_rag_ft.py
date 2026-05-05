"""
hierarchical_rag_ft.py

Drop-in replacement for hierarchical_rag.py that loads a **locally
fine-tuned** e5-small-v2 checkpoint instead of the vanilla HuggingFace
Hub model.

CHECKPOINT SELECTION
────────────────────
Set the env var FT_CHECKPOINT_DIR to override the default:

    FT_CHECKPOINT_DIR=tmp/checkpoint_final python hierarchical_rag_ft.py

Defaults to  tmp/checkpoint_best  (best validation loss during training).

Available checkpoints produced by the fine-tuning run:
    tmp/checkpoint_best    ← default (lowest validation loss)
    tmp/checkpoint_final   ← weights at the last training step
    tmp/checkpoint_latest  ← most-recently-saved checkpoint (same as final
                             unless training was interrupted)

HOW TO ENABLE DEBUG
───────────────────
Option A — module-level flag (flip before running):
    DEBUG = True

Option B — environment variable:
    DEBUG_RAG=1 python hierarchical_rag_ft.py

Option C — runtime toggle:
    import hierarchical_rag_ft as rag_ft
    rag_ft.DEBUG = True

WHAT CHANGED vs hierarchical_rag.py
─────────────────────────────────────
1.  FT_CHECKPOINT_DIR constant / env-var added.
2.  FTE5PrefixEmbeddings replaces E5PrefixEmbeddings.
    It loads the SentenceTransformer directly from the local checkpoint
    directory instead of downloading from HuggingFace Hub.
3.  The FAISS index is built with the fine-tuned encoder.
4.  Debug blocks updated to show the checkpoint path being used.
5.  TARGET_COL in main() changed to "hierarchical_rag_ft" so outputs
    land in a separate column and do not overwrite the vanilla results.
6.  Output CSV gets the suffix _hierarchical_rag_ft.csv.

Everything else — prefix logic, LLM loading, context assembly, sanitize,
CSV loop — is identical to hierarchical_rag.py.

PREFIX SAFETY NOTE (unchanged)
──────────────────────────────
FTe5PrefixEmbeddings.embed_query()     adds "query: "   once.
FTe5PrefixEmbeddings.embed_documents() adds "passage: " once (with
  double-prefix guard for LangChain's internal call pattern).
ask_hierarchical_rag() passes the raw question to FAISS — no manual
  prefix added there.
"""

import os
import sys
import textwrap
import time
from typing import List

import torch
import pandas as pd
from huggingface_hub import login
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils import (
    extract_cisco_config,
    sanitize_ios_only,
    PROMPT_TEMPLATE_STR,
    load_chunks_as_documents,
    build_context_and_provenance_full,
)

# ── Fine-tuned checkpoint ──────────────────────────────────────────────────────
# Override with env var FT_CHECKPOINT_DIR if desired.
FT_CHECKPOINT_DIR: str = os.environ.get(
    "FT_CHECKPOINT_DIR",
    os.path.join("tmp", "checkpoint_best"),
)

# ── Debug flag ─────────────────────────────────────────────────────────────────
DEBUG: bool = os.environ.get("DEBUG_RAG", "0").strip() == "1"

_SEP_THICK = "═" * 72
_SEP_THIN  = "─" * 72
_ANSI = {
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
}


def _dbg(tag: str, label: str, body: str = "", *, width: int = 100) -> None:
    if not DEBUG:
        return
    colour = _ANSI["cyan"] if tag not in ("ERROR", "WARN") else _ANSI["red"]
    reset  = _ANSI["reset"]
    bold   = _ANSI["bold"]
    print(f"\n{colour}{bold}[DEBUG:{tag}]{reset} {label}")
    print(_SEP_THIN)
    if body:
        for line in body.splitlines():
            if len(line) > width:
                print(textwrap.fill(line, width=width, subsequent_indent="  "))
            else:
                print(line)
    print(_SEP_THIN)


def _dbg_section(title: str) -> None:
    if not DEBUG:
        return
    bold  = _ANSI["bold"]
    reset = _ANSI["reset"]
    print(f"\n{_ANSI['yellow']}{_SEP_THICK}")
    print(f"  {bold}{title}{reset}{_ANSI['yellow']}")
    print(f"{_SEP_THICK}{reset}")


# ── Fine-tuned e5 prefix wrapper ───────────────────────────────────────────────

class FTE5PrefixEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper that loads a fine-tuned
    SentenceTransformer checkpoint from a **local directory** and applies
    the intfloat/e5 prefix convention:

        "query: <text>"   for queries  (embed_query)
        "passage: <text>" for documents (embed_documents)

    Double-prefix guard
    ───────────────────
    LangChain's FAISS internals sometimes call embed_documents() with
    the already-prefixed query string (the one returned by embed_query).
    The guard detects any text that already starts with "query: " or
    "passage: " and passes it through unchanged, preventing double-prefixing.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the local SentenceTransformer checkpoint directory.
        Must contain config.json, model.safetensors (or pytorch_model.bin),
        tokenizer files, modules.json, and the 1_Pooling sub-directory.
    device : str
        "cuda" or "cpu".
    normalize_embeddings : bool
        Whether to L2-normalise output vectors (recommended for e5 models).
    batch_size : int
        Encoding batch size.
    """

    _E5_PREFIXES = ("query: ", "passage: ")

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        self.checkpoint_dir       = checkpoint_dir
        self.device               = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size           = batch_size

        _dbg_section("FTE5PrefixEmbeddings — loading fine-tuned checkpoint")
        _dbg(
            "FT_EMB",
            "Loading SentenceTransformer from local checkpoint",
            f"  checkpoint_dir       : {os.path.abspath(checkpoint_dir)}\n"
            f"  device               : {device}\n"
            f"  normalize_embeddings : {normalize_embeddings}\n"
            f"  batch_size           : {batch_size}",
        )

        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"Fine-tuned checkpoint directory not found: {checkpoint_dir!r}\n"
                "Set FT_CHECKPOINT_DIR env var or update FT_CHECKPOINT_DIR constant."
            )

        self._model = SentenceTransformer(checkpoint_dir, device=device)
        _dbg(
            "FT_EMB",
            "SentenceTransformer loaded successfully",
            f"  max_seq_length : {self._model.max_seq_length}\n"
            f"  embedding dim  : {self._model.get_sentence_embedding_dimension()}",
        )

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return vecs.tolist()

    # ------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = []
        actions  = []
        for t in texts:
            if t.startswith(self._E5_PREFIXES):
                prefixed.append(t)
                actions.append("PASS-THROUGH (already prefixed)")
            else:
                prefixed.append(f"passage: {t}")
                actions.append("ADD 'passage: '")

        _dbg_section("FTE5PrefixEmbeddings.embed_documents() called")
        _dbg(
            "FT_EMB",
            f"embed_documents — {len(texts)} text(s)",
            "\n".join(
                f"  [{i}] action   : {actions[i]}\n"
                f"       original : {repr(t[:110])}\n"
                f"       final    : {repr(p[:118])}"
                for i, (t, p) in enumerate(zip(texts[:5], prefixed[:5]))
            ) + ("\n  … (showing first 5 only)" if len(texts) > 5 else "")
            + "\n\n  ✅ Double-prefix guard active — texts already prefixed pass through.",
        )

        result = self._encode(prefixed)

        _dbg(
            "FT_EMB",
            f"embed_documents — done. {len(result)} vectors "
            f"(dim={len(result[0]) if result else 'N/A'})",
        )
        return result

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"

        _dbg_section("FTE5PrefixEmbeddings.embed_query() called")
        _dbg(
            "FT_EMB",
            "embed_query — adding 'query: ' prefix",
            f"  original question : {repr(text)}\n"
            f"  prefixed string   : {repr(prefixed)}\n"
            f"\n"
            f"  NOTE: FAISS may call embed_documents([prefixed_query]) internally.\n"
            f"  Double-prefix guard detects 'query: ' and passes through unchanged.",
        )

        result = self._encode([prefixed])[0]

        _dbg(
            "FT_EMB",
            f"embed_query — done. Returning vector (dim={len(result)})",
        )
        return result


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm():
    _dbg_section("Loading LLM")
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
    _dbg(
        "LLM",
        "BitsAndBytes config",
        f"  model_id      : {config.LLM_MODEL_ID}\n"
        f"  compute_dtype : {compute_dtype}\n"
        f"  quant_type    : nf4\n"
        f"  double_quant  : True\n"
        f"  device_map    : auto",
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
    _dbg(
        "LLM",
        "Tokenizer details",
        f"  pad_token    : {repr(tokenizer.pad_token)}\n"
        f"  eos_token    : {repr(tokenizer.eos_token)}\n"
        f"  padding_side : {tokenizer.padding_side}\n"
        f"  vocab_size   : {tokenizer.vocab_size}",
    )
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
    _dbg(
        "LLM",
        "HuggingFace pipeline built",
        "  task             : text-generation\n"
        "  max_new_tokens   : 400\n"
        "  do_sample        : False\n"
        "  temperature      : 0.1\n"
        "  repetition_pen.  : 1.1\n"
        "  return_full_text : False  ← model output = NEW tokens only",
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


# ── RAG function ───────────────────────────────────────────────────────────────

def ask_hierarchical_rag(question: str, db, llm_chain: LLMChain, k: int = 4) -> dict:
    """
    Core RAG query function — identical logic to hierarchical_rag.py,
    but the FAISS index was built with the fine-tuned encoder.

    PREFIX FLOW
    ───────────
    1. `question` arrives here as a plain string — NO prefix yet.
    2. db.similarity_search_with_score(question, k=k) calls
       FTE5PrefixEmbeddings.embed_query(question).
    3. embed_query() prepends "query: " → encoder sees "query: <question>".
    4. Chunks in the FAISS index were encoded with "passage: " prefix.
    → Exactly one prefix on each side, consistent with e5 training convention.
    """
    _dbg_section(f"ask_hierarchical_rag() — question: {question[:80]!r}")

    _dbg(
        "FAISS",
        "Calling db.similarity_search_with_score()",
        f"  raw question : {repr(question)}\n"
        f"  k            : {k}\n"
        f"\n"
        f"  FTE5PrefixEmbeddings.embed_query() will prepend 'query: ' automatically.",
    )

    results = db.similarity_search_with_score(question, k=k)

    _dbg("RETRIEVE", f"FAISS returned {len(results)} chunk(s)")
    for rank, (doc, score) in enumerate(results, start=1):
        md = doc.metadata or {}
        chunk_text = (
            md.get("display_text")
            or getattr(doc, "display_text", None)
            or doc.page_content
            or ""
        )
        _dbg(
            "RETRIEVE",
            f"Chunk #{rank}  score={score:.6f}  (L2 distance, lower = more similar)",
            f"  chunk_id    : {md.get('chunk_id')}\n"
            f"  source_file : {md.get('source_file')}\n"
            f"  pdf_name    : {md.get('pdf_name')}\n"
            f"  chapter     : {md.get('chapter')}\n"
            f"  section     : {md.get('section')}\n"
            f"  part        : {md.get('part')} / {md.get('parts_total')}\n"
            f"  word_count  : {md.get('word_count')}\n"
            f"  token_count : {md.get('token_count')}\n"
            f"  has_sem_hdr : {md.get('has_semantic_header')}\n"
            f"  header_style: {md.get('header_style')}\n"
            f"  sha1        : {md.get('sha1')}\n"
            f"  text[:300]  :\n"
            + textwrap.indent(chunk_text[:300], "    "),
        )

    _dbg_section("Building context from retrieved chunks")
    context, used_chunks = build_context_and_provenance_full(results)

    _dbg(
        "CONTEXT",
        f"Context assembled — {len(used_chunks)} chunk(s) used, {len(context)} chars total",
        f"  chunks_requested : {k}\n"
        f"  chunks_used      : {len(used_chunks)}\n"
        f"  context_len      : {len(context)} chars\n"
        f"\n--- FULL CONTEXT ---\n"
        + context
        + "\n--- END CONTEXT ---",
    )

    _dbg_section("LLM inference")
    full_prompt_preview = PROMPT_TEMPLATE_STR.replace("{context}", context).replace(
        "{question}", question
    )
    _dbg(
        "LLM",
        "Full prompt sent to model (reconstructed for debug)",
        full_prompt_preview,
    )

    raw  = llm_chain.invoke({"context": context, "question": question})
    text = raw["text"] if isinstance(raw, dict) and "text" in raw else str(raw)

    _dbg(
        "LLM",
        "Raw model output (return_full_text=False → new tokens only)",
        f"  type : {type(raw).__name__}\n"
        f"  key  : {'text' if isinstance(raw, dict) and 'text' in raw else 'str(raw)'}\n"
        f"\n--- RAW OUTPUT ---\n"
        + text
        + "\n--- END RAW OUTPUT ---",
    )

    _dbg_section("Sanitizing LLM output")
    try:
        answer = sanitize_ios_only(text)
        _dbg(
            "SANITIZE",
            "sanitize_ios_only() succeeded",
            f"--- SANITIZED OUTPUT ---\n{answer}\n--- END SANITIZED OUTPUT ---",
        )
    except Exception as exc:
        _dbg(
            "SANITIZE",
            f"sanitize_ios_only() raised {type(exc).__name__}: {exc} — using raw text",
        )
        answer = text

    return {"answer": answer, "context_used": context, "used_chunks": used_chunks}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if DEBUG:
        _dbg_section("DEBUG MODE ENABLED — hierarchical_rag_ft.py")
        _dbg(
            "CONFIG",
            "Runtime configuration",
            f"  DEBUG              : {DEBUG}\n"
            f"  FT_CHECKPOINT_DIR  : {os.path.abspath(FT_CHECKPOINT_DIR)}\n"
            f"  LLM_MODEL_ID       : {config.LLM_MODEL_ID}\n"
            f"  CHUNKS_JSON        : {config.CHUNKS_JSON}\n"
            f"  INPUT_CSV          : {config.INPUT_CSV}\n"
            f"  RAG_K              : {config.RAG_K}\n"
            f"  SLEEP_BTW_CALLS    : {config.SLEEP_BETWEEN_CALLS}\n"
            f"\n"
            f"  PREFIX STRATEGY:\n"
            f"  • embed_documents() → 'passage: ' + chunk text  (index-build time)\n"
            f"  • embed_query()     → 'query: '   + question    (retrieval time)\n"
            f"  • Double-prefix guard active in embed_documents().",
        )

    login(config.HF_TOKEN)

    if not os.path.exists(config.CHUNKS_JSON):
        raise FileNotFoundError(
            f"Chunks file not found: {config.CHUNKS_JSON}\n"
            "Run chunking.py first."
        )

    print(f"Loading chunks from {config.CHUNKS_JSON}…")
    docs = load_chunks_as_documents(config.CHUNKS_JSON)

    _dbg(
        "INDEX",
        f"Loaded {len(docs)} documents for FAISS index",
        f"  Sample metadata from doc[0]: {docs[0].metadata if docs else 'N/A'}\n"
        f"  Sample text from doc[0][:200]: {docs[0].page_content[:200] if docs else 'N/A'}",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Building FAISS index with fine-tuned e5 embeddings\n"
        f"  checkpoint : {os.path.abspath(FT_CHECKPOINT_DIR)}\n"
        f"  device     : {device}"
    )

    _dbg(
        "INDEX",
        "Creating FTE5PrefixEmbeddings",
        f"  checkpoint_dir       : {os.path.abspath(FT_CHECKPOINT_DIR)}\n"
        f"  device               : {device}\n"
        f"  normalize_embeddings : True\n"
        f"  batch_size           : 32\n"
        f"\n"
        f"  When FAISS.from_documents() fires, embed_documents() will encode\n"
        f"  all {len(docs)} chunks with 'passage: ' prefix using the FT model.",
    )

    embeddings = FTE5PrefixEmbeddings(
        checkpoint_dir=FT_CHECKPOINT_DIR,
        device=device,
        normalize_embeddings=True,
        batch_size=32,
    )
    db = FAISS.from_documents(docs, embeddings)
    print("✅ FAISS index built.\n")

    _dbg(
        "INDEX",
        "FAISS index built successfully",
        f"  Total vectors indexed : {db.index.ntotal if hasattr(db, 'index') else 'N/A'}\n"
        f"  Embedding dimension   : {db.index.d if hasattr(db, 'index') else 'N/A'}",
    )

    model, tokenizer = load_llm()
    llm       = build_langchain_llm(model, tokenizer)
    prompt    = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE_STR)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    if not os.path.exists(config.INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV)
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    # Separate output column so vanilla and FT results can be compared side-by-side.
    TARGET_COL = "hierarchical_rag_ft"
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""
    rows_to_process = df.index[df[TARGET_COL].astype(str).str.strip() == ""].tolist()
    print(f"Total rows: {len(df)} | To generate: {len(rows_to_process)}\n")

    _dbg(
        "LOOP",
        "CSV loop starting",
        f"  total rows      : {len(df)}\n"
        f"  rows to process : {len(rows_to_process)}\n"
        f"  target column   : {TARGET_COL}\n"
        f"  RAG k           : {config.RAG_K}",
    )

    output_csv = config.OUTPUT_CSV.replace(".csv", "_hierarchical_rag_ft.csv")

    for loop_idx, i in enumerate(rows_to_process):
        q = str(df.at[i, "question"]).strip()
        if not q:
            _dbg("LOOP", f"Row {i} — empty question, skipping")
            df.at[i, TARGET_COL] = ""
            continue

        _dbg_section(
            f"ROW {loop_idx + 1}/{len(rows_to_process)}  (df row {i})  "
            f"question={q[:70]!r}"
        )

        try:
            res = ask_hierarchical_rag(q, db, llm_chain, k=config.RAG_K)

            _dbg(
                "EXTRACT",
                "Calling extract_cisco_config() on sanitized answer",
                f"  input:\n{textwrap.indent(res['answer'], '    ')}",
            )
            cleaned = extract_cisco_config(res["answer"])
            _dbg(
                "EXTRACT",
                "extract_cisco_config() output",
                f"  output:\n{textwrap.indent(cleaned, '    ')}",
            )

            df.at[i, TARGET_COL] = cleaned
            print(f"[{loop_idx + 1}/{len(rows_to_process)}] ✅ {q[:60]}")

            _dbg(
                "LOOP",
                f"Row {i} complete",
                f"  question : {q}\n"
                f"  answer   :\n{textwrap.indent(cleaned, '    ')}",
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Saving progress…")
            break
        except Exception as e:
            _dbg("ERROR", f"Row {i} failed with {type(e).__name__}: {e}")
            print(f"[{loop_idx + 1}/{len(rows_to_process)}] ⚠️  Failed (row {i}): {e}")

        df.to_csv(output_csv, index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if config.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS)

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Done → {output_csv}")
    _dbg_section("Pipeline complete")


if __name__ == "__main__":
    main()
