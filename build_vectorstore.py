"""
02 build_vectorstore.py — Load fine-tuned embeddings and build a FAISS vectorstore.

Reads chunks from config.CHUNKS_JSON, builds a FAISS index using the fine-tuned
embedding model found under config.CHECKPOINT_DIR, and saves the index to disk
as "faiss_index/".

Usage:
    python build_vectorstore.py

DEBUG MODE
──────────
Option A — env var (no code change):
    DEBUG_RAG=1 python build_vectorstore.py

Option B — flip the constant in this file:
    DEBUG = True

Option C — runtime toggle:
    import importlib, build_vectorstore as bv; bv.DEBUG = True

BUG FIXED IN THIS VERSION
──────────────────────────
The original E5PrefixEmbeddings.embed_documents() blindly prepended "passage: " to
every text it received. LangChain's FAISS.similarity_search_with_score() internally
calls embed_documents() on the query string (not just embed_query()), so the query
was being encoded as "passage: query: <text>" — a double-prefix that silently
degraded retrieval quality.

Fix: embed_documents() now checks whether each text already starts with a known
e5 prefix ("query: " or "passage: "). If it does, the text is passed through
unchanged. If it does not, "passage: " is prepended as normal.

Correct behaviour:
  Index-build time (FAISS.from_documents):
      embed_documents(["chunk text", …])
      → ["passage: chunk text", …]           ✅  one prefix

  Retrieval time (FAISS.similarity_search_with_score internal call):
      embed_query("user question")
      → super().embed_query("query: user question")   ← called first, produces vector
      embed_documents(["query: user question"])        ← FAISS internal call
      → guard fires, text passed through unchanged     ✅  no double-prefix
"""

import os
import json
import textwrap
from pathlib import Path
from typing import List

# Re-export so baseline scripts can do: from build_vectorstore import load_chunks_as_documents
from utils import load_chunks_as_documents  # noqa: F401

import torch
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import login

import config

# ── Debug flag ─────────────────────────────────────────────────────────────────
DEBUG: bool = os.environ.get("DEBUG_RAG", "0").strip() == "1"

_SEP_THICK = "═" * 72
_SEP_THIN  = "─" * 72
_C = {
    "cyan":   "\033[96m",
    "yellow": "\033[93m",
    "green":  "\033[92m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _dbg(tag: str, label: str, body: str = "", *, width: int = 100) -> None:
    """Emit a tagged, coloured debug block. No-op when DEBUG is False."""
    if not DEBUG:
        return
    colour = _C["red"] if tag in ("ERROR", "WARN") else _C["cyan"]
    print(f"\n{colour}{_C['bold']}[DEBUG:{tag}]{_C['reset']} {label}")
    print(_SEP_THIN)
    if body:
        for line in body.splitlines():
            print(line if len(line) <= width else textwrap.fill(line, width, subsequent_indent="  "))
    print(_SEP_THIN)


def _dbg_section(title: str) -> None:
    """Print a thick separator — used at major pipeline stages."""
    if not DEBUG:
        return
    print(f"\n{_C['yellow']}{_SEP_THICK}")
    print(f"  {_C['bold']}{title}{_C['reset']}{_C['yellow']}")
    print(f"{_SEP_THICK}{_C['reset']}")


# ── E5 prefix wrapper ──────────────────────────────────────────────────────────

class E5PrefixEmbeddings(HuggingFaceEmbeddings):
    """
    Subclass of HuggingFaceEmbeddings that prepends mandatory e5 prefixes:
      - "passage: " when embedding documents (index build time)
      - "query: "   when embedding a search query (retrieval time)

    intfloat/e5-* models were trained with these prefixes and produce
    significantly worse embeddings without them.

    *** DOUBLE-PREFIX BUG — FIXED HERE ***
    LangChain's FAISS.similarity_search_with_score() internally calls BOTH
    embed_query() AND embed_documents() on the query string. If embed_documents()
    blindly prepends "passage: ", the query becomes "passage: query: <text>".

    Fix: embed_documents() inspects each text before prefixing.
      • Already starts with "query: " or "passage: "?  → pass through unchanged.
      • Otherwise?                                      → prepend "passage: ".

    This makes both call sites safe:
      Index build (FAISS.from_documents)
          chunks arrive without any prefix → "passage: " is prepended ✅
      Retrieval (FAISS internal embed_documents call on the query string)
          query already has "query: " → guard fires, no second prefix ✅

    query_prefix / passage_prefix are Pydantic fields (HuggingFaceEmbeddings
    is a Pydantic BaseModel) so they are readable from outside the class and
    by the debug helpers in 03_rag_inference.py.
    """

    query_prefix:   str = "query: "
    passage_prefix: str = "passage: "

    # Tuple used by the guard — add new prefixes here if the model ever changes.
    _E5_PREFIXES: tuple = ("query: ", "passage: ")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = []
        actions  = []   # populated only when DEBUG is on

        for t in texts:
            if t.startswith(self._E5_PREFIXES):
                # Already has a valid e5 prefix — this is the FAISS-internal
                # call that passes back the output of embed_query().
                # Do NOT add another prefix.
                prefixed.append(t)
                actions.append("PASS-THROUGH (already prefixed — double-prefix guard)")
            else:
                prefixed.append(f"{self.passage_prefix}{t}")
                actions.append(f"ADD '{self.passage_prefix}'")

        if DEBUG:
            _dbg_section("E5PrefixEmbeddings.embed_documents() called")
            sample_lines = []
            for i, (t, p, act) in enumerate(zip(texts[:5], prefixed[:5], actions[:5])):
                sample_lines.append(
                    f"  [{i}] action   : {act}\n"
                    f"       original : {repr(t[:110])}\n"
                    f"       final    : {repr(p[:118])}"
                )
            if len(texts) > 5:
                sample_lines.append(f"  … ({len(texts) - 5} more, showing first 5 only)")

            guard_hits = sum(1 for a in actions if "PASS-THROUGH" in a)
            add_hits   = len(actions) - guard_hits

            _dbg(
                "E5PREFIX",
                f"embed_documents — {len(texts)} text(s)  "
                f"[added prefix: {add_hits}, passed through: {guard_hits}]",
                "\n".join(sample_lines)
                + "\n\n  Double-prefix guard: texts already starting with "
                  "'query: ' or 'passage: ' are passed through unchanged.",
            )

        result = super().embed_documents(prefixed)

        _dbg(
            "E5PREFIX",
            f"embed_documents — done. {len(result)} vectors (dim={len(result[0]) if result else 'N/A'})",
        )
        return result

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"{self.query_prefix}{text}"

        if DEBUG:
            _dbg_section("E5PrefixEmbeddings.embed_query() called")
            _dbg(
                "E5PREFIX",
                "embed_query — adding query prefix",
                f"  original question : {repr(text)}\n"
                f"  prefixed string   : {repr(prefixed)}\n"
                f"\n"
                f"  FLOW: FAISS will also call embed_documents([prefixed_query]) internally.\n"
                f"  The double-prefix guard in embed_documents() detects '{self.query_prefix}'\n"
                f"  and passes the string through unchanged → encoder sees exactly:\n"
                f"    '{self.query_prefix}<question>'    (one prefix, never two)",
            )

        result = super().embed_query(prefixed)

        _dbg(
            "E5PREFIX",
            f"embed_query — done. Embedding vector dim={len(result)}",
        )
        return result


FAISS_INDEX_DIR = "faiss_index"


# ── Checkpoint discovery ───────────────────────────────────────────────────────

def find_best_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Scan checkpoint_dir for the best model checkpoint.

    Priority:
      1. Sub-directories with metadata.json + checkpoint_type == 'best'
         (highest triplet_accuracy wins).
      2. Sub-directories with model files but no metadata (first found).
    Returns the path to the chosen checkpoint, or None if nothing is found.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        _dbg("WARN", f"Checkpoint directory missing: {checkpoint_dir}")
        return None

    print(f"Scanning checkpoint directory: {checkpoint_dir}")
    _dbg_section(f"find_best_checkpoint — scanning: {checkpoint_dir}")

    MODEL_FILES = {"pytorch_model.bin", "model.safetensors"}
    best_checkpoints = []
    fallback          = None
    fallback_score    = -1.0
    scanned           = []

    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if not os.path.isdir(item_path):
            continue

        transformer_subdir = os.path.join(item_path, "0_Transformer")
        has_model = (
            os.path.exists(os.path.join(item_path, "sentence_bert_config.json"))
            or any(os.path.exists(os.path.join(transformer_subdir, f)) for f in MODEL_FILES)
            or any(os.path.exists(os.path.join(item_path, f)) for f in MODEL_FILES)
        )
        if not has_model:
            _dbg("CKPT", f"Skipping '{item}' — no model files found", "")
            continue

        metadata_path = os.path.join(item_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as mf:
                    meta = json.load(mf)
                ckpt_type = meta.get("checkpoint_type", "unknown")
                score     = meta.get("metrics", {}).get("triplet_accuracy", 0.0)
                print(f"  📁 {item}  type={ckpt_type}  score={score:.4f}")
                scanned.append((item, ckpt_type, score))
                if ckpt_type == "best":
                    best_checkpoints.append((item_path, score))
                elif score > fallback_score:
                    fallback_score = score
                    fallback       = item_path
            except Exception as e:
                print(f"  ⚠️  Could not read metadata for {item}: {e}")
                _dbg("WARN", f"metadata.json parse error for '{item}'", str(e))
                if fallback is None:
                    fallback = item_path
        else:
            print(f"  📁 {item}  (no metadata.json)")
            scanned.append((item, "no-metadata", None))
            if fallback is None:
                fallback = item_path

    _dbg(
        "CKPT",
        f"Checkpoint scan complete — {len(scanned)} candidate(s)",
        "\n".join(
            f"  {name:40s}  type={t:12s}  score={s:.4f}" if s is not None
            else f"  {name:40s}  type={t:12s}  score=N/A"
            for name, t, s in scanned
        ) or "  (none found)",
    )

    if best_checkpoints:
        best_checkpoints.sort(key=lambda x: x[1], reverse=True)
        chosen, score = best_checkpoints[0]
        print(f"\n✅ Best checkpoint (type='best'): {chosen}  score={score:.4f}")
        _dbg("CKPT", f"Selected checkpoint: {chosen}  score={score:.4f}")
        return chosen

    if fallback:
        print(f"\n✅ Fallback checkpoint: {fallback}")
        _dbg("CKPT", f"Using fallback checkpoint: {fallback}")
        return fallback

    print("❌ No valid checkpoint found.")
    return None


# ── Embedding loader ───────────────────────────────────────────────────────────

def load_embeddings(checkpoint_dir: str) -> "E5PrefixEmbeddings | None":
    """Load E5PrefixEmbeddings from the best checkpoint in checkpoint_dir."""
    _dbg_section(f"load_embeddings — checkpoint_dir: {checkpoint_dir}")

    best_path = find_best_checkpoint(checkpoint_dir)
    if not best_path:
        return None

    model_files = ["pytorch_model.bin", "model.safetensors"]
    transformer_sub = os.path.join(best_path, "0_Transformer")
    has_sentinel = os.path.exists(os.path.join(best_path, "sentence_bert_config.json"))
    has_weights  = (
        any(os.path.exists(os.path.join(best_path, f)) for f in model_files)
        or any(os.path.exists(os.path.join(transformer_sub, f)) for f in model_files)
    )

    _dbg(
        "CKPT",
        f"Checkpoint validation: {best_path}",
        f"  has sentence_bert_config.json : {has_sentinel}\n"
        f"  has weights (direct or in 0_Transformer/) : {has_weights}",
    )

    if not has_sentinel and not has_weights:
        print(f"❌ No valid SentenceTransformer checkpoint found in {best_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading embeddings from: {best_path}  (device={device})")

    _dbg(
        "EMBED",
        "Creating E5PrefixEmbeddings",
        f"  model_name           : {best_path}\n"
        f"  device               : {device}\n"
        f"  normalize_embeddings : True\n"
        f"  batch_size           : 32\n"
        f"  trust_remote_code    : True\n"
        f"\n"
        f"  This class adds 'passage: ' at index-build time and 'query: ' at\n"
        f"  retrieval time. The double-prefix guard prevents FAISS's internal\n"
        f"  embed_documents() call from re-prefixing the query vector.",
    )

    try:
        embeddings = E5PrefixEmbeddings(
            model_name=best_path,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        _dbg(
            "EMBED",
            "E5PrefixEmbeddings loaded successfully",
            f"  query_prefix   : {repr(embeddings.query_prefix)}\n"
            f"  passage_prefix : {repr(embeddings.passage_prefix)}\n"
            f"  model_name     : {embeddings.model_name}",
        )
        print("✅ E5PrefixEmbeddings loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        _dbg("ERROR", f"load_embeddings failed: {type(e).__name__}: {e}")
        return None


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if DEBUG:
        _dbg_section("DEBUG MODE ENABLED — 02_build_vectorstore.py")
        _dbg(
            "CONFIG",
            "Runtime configuration",
            f"  CHECKPOINT_DIR : {config.CHECKPOINT_DIR}\n"
            f"  CHUNKS_JSON    : {config.CHUNKS_JSON}\n"
            f"  FAISS_INDEX_DIR: {FAISS_INDEX_DIR}\n"
            f"\n"
            f"  PREFIX STRATEGY (E5PrefixEmbeddings):\n"
            f"  • embed_documents() → 'passage: ' + chunk  (index build)\n"
            f"  • embed_query()     → 'query: '   + text   (retrieval)\n"
            f"  • Double-prefix guard: texts already prefixed are passed through unchanged.",
        )

    login(config.HF_TOKEN)

    # 1. Load embeddings
    embeddings = load_embeddings(config.CHECKPOINT_DIR)
    if embeddings is None:
        raise RuntimeError("Could not load embedding model. Check CHECKPOINT_DIR in config.py.")

    # 2. Load chunks
    docs = load_chunks_as_documents(config.CHUNKS_JSON)
    _dbg(
        "INDEX",
        f"Loaded {len(docs)} documents for FAISS indexing",
        f"  Sample metadata doc[0]: {docs[0].metadata if docs else 'N/A'}\n"
        f"  Sample text    doc[0][:200]: {docs[0].page_content[:200] if docs else 'N/A'}",
    )

    # 3. Build FAISS index
    # embed_documents() is called here on all chunks.
    # DEBUG will show [DEBUG:E5PREFIX] blocks for the first 5 chunks.
    print(f"\nBuilding FAISS index over {len(docs)} documents…")
    _dbg(
        "INDEX",
        "Calling FAISS.from_documents() — embed_documents() fires now",
        f"  {len(docs)} chunks will be prefixed with 'passage: ' and encoded.\n"
        f"  Watch for [DEBUG:E5PREFIX] embed_documents blocks below.",
    )
    db = FAISS.from_documents(docs, embeddings)

    _dbg(
        "INDEX",
        "FAISS index built successfully",
        f"  Total vectors indexed : {db.index.ntotal if hasattr(db, 'index') else 'N/A'}\n"
        f"  Embedding dimension   : {db.index.d if hasattr(db, 'index') else 'N/A'}",
    )

    # 4. Save index
    db.save_local(FAISS_INDEX_DIR)
    print(f"✅ FAISS index saved → {FAISS_INDEX_DIR}/")
    _dbg("INDEX", f"FAISS index saved to disk: {FAISS_INDEX_DIR}/")

    return db, embeddings


if __name__ == "__main__":
    main()
