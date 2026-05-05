"""
hierarchical_rag.py 

HOW TO ENABLE DEBUG
───────────────────
Option A — module-level flag (flip before running):
    DEBUG = True   ← change the constant at the top of this file

Option B — environment variable (no code change needed):
    DEBUG_RAG=1 python hierarchical_rag.py

Option C — runtime toggle from another script or notebook:
    import hierarchical_rag as b3
    b3.DEBUG = True

WHAT IS LOGGED WHEN DEBUG=True
───────────────────────────────
[E5Prefix]      Every call to embed_query / embed_documents showing the
                EXACT string (with or without prefix) that hits the model.
                This is the definitive answer to "is the prefix added once,
                twice, or not at all?"

[FAISS]         The raw similarity_search_with_score call: question string
                passed in (no manual prefix added here — E5PrefixEmbeddings
                handles it transparently), and k value used.

[RETRIEVE]      Each retrieved chunk: rank, L2 distance score, chunk_id,
                source_file, chapter/section, word_count, and first 300 chars
                of text.

[CONTEXT]       Deduplication decisions, per-chunk budget tracking, final
                context string length and chunk count.

[LLM]           The full prompt sent to the model and the raw output before
                sanitization.

[SANITIZE]      Lines accepted / rejected / deduplicated by sanitize_ios_only.

[EXTRACT]       Lines accepted / skipped by extract_cisco_config.

[LOOP]          Per-row question/answer summary in the CSV loop.

PREFIX SAFETY NOTE
──────────────────
E5PrefixEmbeddings.embed_query()  adds "query: "   once (here).
E5PrefixEmbeddings.embed_documents() adds "passage: " once (here, at index time).
ask_hierarchical_rag() calls db.similarity_search_with_score(question, k=k)
  → question has NO manual prefix here; E5PrefixEmbeddings.embed_query() adds it.
utils.retrieve_with_scores() is NOT used in this baseline, so its query_prefix
  parameter is irrelevant and there is NO double-prefixing.
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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils import (
    extract_cisco_config,
    sanitize_ios_only,
    PROMPT_TEMPLATE_STR,
    load_chunks_as_documents,
    build_context_and_provenance_full,
)

VANILLA_EMB_MODEL = "intfloat/e5-small-v2"

# ── Debug flag ─────────────────────────────────────────────────────────────────
# Flip this to True, or set env var DEBUG_RAG=1 before running.
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
    """
    Emit a coloured, tagged debug block.

    tag   — short ALL-CAPS category, e.g. "E5PREFIX", "RETRIEVE", "LLM"
    label — one-line description of what this block shows
    body  — multi-line detail (omit for header-only blocks)
    """
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
    """Print a thick separator with a title — used at major pipeline stages."""
    if not DEBUG:
        return
    bold  = _ANSI["bold"]
    reset = _ANSI["reset"]
    print(f"\n{_ANSI['yellow']}{_SEP_THICK}")
    print(f"  {bold}{title}{reset}{_ANSI['yellow']}")
    print(f"{_SEP_THICK}{reset}")


# ── e5 prefix wrapper ─────────────────────────────────────────────────────────

class E5PrefixEmbeddings(HuggingFaceEmbeddings):
    """
    Thin subclass of HuggingFaceEmbeddings that prepends the required e5 prefixes.

    intfloat/e5-* models must receive:
        "query: <text>"   for queries / questions
        "passage: <text>" for documents / chunks

    *** BUG THAT WAS CAUGHT AND FIXED ***
    LangChain's FAISS.similarity_search_with_score() internally calls BOTH
    embed_query() AND embed_documents() on the query string (the latter for
    its internal score-computation path). If embed_documents() blindly prepends
    "passage: ", the query ends up as "passage: query: <text>" — double-prefixed.

    FIX: embed_documents() checks whether each text already starts with a known
    e5 prefix ("query: " or "passage: "). If it does, the text is passed through
    unchanged (it came from embed_query() internally). If it does not, "passage: "
    is prepended (normal document indexing path).

    This makes both call sites safe:
      • Index-build (FAISS.from_documents)     → "passage: <chunk>"      ✅
      • Retrieval (similarity_search_with_score internal embed_documents call)
                                               → "query: <question>"     ✅  (unchanged)
    """

    _E5_PREFIXES = ("query: ", "passage: ")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = []
        actions  = []   # for debug logging only
        for t in texts:
            if t.startswith(self._E5_PREFIXES):
                # Already prefixed — came from embed_query() internal FAISS call.
                # Do NOT add another prefix.
                prefixed.append(t)
                actions.append("PASS-THROUGH (already prefixed)")
            else:
                prefixed.append(f"passage: {t}")
                actions.append("ADD 'passage: '")

        _dbg_section("E5PrefixEmbeddings.embed_documents() called")
        _dbg(
            "E5PREFIX",
            f"embed_documents — {len(texts)} text(s)",
            "\n".join(
                f"  [{i}] action   : {actions[i]}\n"
                f"       original : {repr(t[:110])}\n"
                f"       final    : {repr(p[:118])}"
                for i, (t, p) in enumerate(zip(texts[:5], prefixed[:5]))
            ) + ("\n  … (showing first 5 only)" if len(texts) > 5 else "")
            + "\n\n  ✅ Double-prefix guard active: texts already starting with "
              "'query: ' or 'passage: ' are passed through unchanged.",
        )

        result = super().embed_documents(prefixed)

        _dbg(
            "E5PREFIX",
            f"embed_documents — done. Returning {len(result)} embedding vectors "
            f"(dim={len(result[0]) if result else 'N/A'})",
        )
        return result

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}"

        _dbg_section("E5PrefixEmbeddings.embed_query() called")
        _dbg(
            "E5PREFIX",
            "embed_query — adding 'query: ' prefix",
            f"  original question : {repr(text)}\n"
            f"  prefixed string   : {repr(prefixed)}\n"
            f"\n"
            f"  NOTE: FAISS will also call embed_documents([prefixed_query]) internally.\n"
            f"  The double-prefix guard in embed_documents() detects the 'query: ' prefix\n"
            f"  and passes it through unchanged → final string stays 'query: <question>'.",
        )

        result = super().embed_query(prefixed)

        _dbg(
            "E5PREFIX",
            f"embed_query — done. Returning embedding vector (dim={len(result)})",
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
        f"  pad_token     : {repr(tokenizer.pad_token)}\n"
        f"  eos_token     : {repr(tokenizer.eos_token)}\n"
        f"  padding_side  : {tokenizer.padding_side}\n"
        f"  vocab_size    : {tokenizer.vocab_size}",
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
    Core RAG query function.

    PREFIX FLOW (important for e5-small-v2 correctness):
    ─────────────────────────────────────────────────────
    1. `question` arrives here as a plain string — NO prefix yet.
    2. db.similarity_search_with_score(question, k=k) internally calls
       E5PrefixEmbeddings.embed_query(question).
    3. embed_query() prepends "query: " → encoder sees "query: <question>".
    4. Document chunks were indexed with "passage: " prefix (embed_documents).
    → Result: exactly ONE prefix on each side, correct for intfloat/e5-small-v2.
    """
    _dbg_section(f"ask_hierarchical_rag() — question: {question[:80]!r}")

    # ── Step 1: sanity-check the raw question ────────────────────────────────
    _dbg(
        "FAISS",
        "Calling db.similarity_search_with_score()",
        f"  raw question passed in : {repr(question)}\n"
        f"  k                      : {k}\n"
        f"\n"
        f"  NOTE: No manual prefix is added here.\n"
        f"  E5PrefixEmbeddings.embed_query() will prepend 'query: ' automatically.\n"
        f"  utils.retrieve_with_scores() is NOT called — no risk of double-prefixing.",
    )

    # ── Step 2: FAISS retrieval ───────────────────────────────────────────────
    # embed_query() inside E5PrefixEmbeddings fires here (logged there).
    results = db.similarity_search_with_score(question, k=k)

    # ── Step 3: log each retrieved chunk ─────────────────────────────────────
    _dbg(
        "RETRIEVE",
        f"FAISS returned {len(results)} chunk(s)",
    )
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

    # ── Step 4: build context ─────────────────────────────────────────────────
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

    # ── Step 5: build prompt and call LLM ─────────────────────────────────────
    _dbg_section("LLM inference")

    # Reconstruct what the prompt looks like (for debug only)
    full_prompt_preview = PROMPT_TEMPLATE_STR.replace("{context}", context).replace(
        "{question}", question
    )
    _dbg(
        "LLM",
        "Full prompt sent to model (reconstructed for debug)",
        full_prompt_preview,
    )

    raw = llm_chain.invoke({"context": context, "question": question})
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

    # ── Step 6: sanitize ──────────────────────────────────────────────────────
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
        _dbg_section("DEBUG MODE ENABLED — baseline3_hierarchical_rag_v2_debug.py")
        _dbg(
            "CONFIG",
            "Runtime configuration",
            f"  DEBUG              : {DEBUG}\n"
            f"  VANILLA_EMB_MODEL  : {VANILLA_EMB_MODEL}\n"
            f"  LLM_MODEL_ID       : {config.LLM_MODEL_ID}\n"
            f"  CHUNKS_JSON        : {config.CHUNKS_JSON}\n"
            f"  INPUT_CSV          : {config.INPUT_CSV}\n"
            f"  RAG_K              : {config.RAG_K}\n"
            f"  SLEEP_BTW_CALLS    : {config.SLEEP_BETWEEN_CALLS}\n"
            f"\n"
            f"  PREFIX STRATEGY:\n"
            f"  • embed_documents() → 'passage: ' + chunk text  (at index-build time)\n"
            f"  • embed_query()     → 'query: '   + question     (at retrieval time)\n"
            f"  • ask_hierarchical_rag() passes raw question to FAISS;\n"
            f"    E5PrefixEmbeddings adds the prefix transparently.\n"
            f"  • utils.retrieve_with_scores() is NOT used here → no double-prefix.",
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

    print(f"Building FAISS index with e5 prefix embeddings ({VANILLA_EMB_MODEL})…")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _dbg(
        "INDEX",
        "Creating E5PrefixEmbeddings",
        f"  model_name           : {VANILLA_EMB_MODEL}\n"
        f"  device               : {device}\n"
        f"  normalize_embeddings : True\n"
        f"  batch_size           : 32\n"
        f"\n"
        f"  When FAISS.from_documents() is called next, embed_documents() will\n"
        f"  be invoked on all {len(docs)} chunks — each prefixed with 'passage: '.\n"
        f"  Watch for [DEBUG:E5PREFIX] embed_documents blocks above.",
    )

    # E5PrefixEmbeddings.embed_documents() fires inside from_documents() below.
    # DEBUG blocks from that call will appear in the output.
    embeddings = E5PrefixEmbeddings(
        model_name=VANILLA_EMB_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
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

    TARGET_COL = "hierarchical_rag"
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

    output_csv = config.OUTPUT_CSV.replace(".csv", "_hierarchical_rag.csv")

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
            _dbg(
                "ERROR",
                f"Row {i} failed with {type(e).__name__}: {e}",
            )
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
