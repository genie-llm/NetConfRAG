"""
04 batch_evaluate.py — Batch RAG evaluation over a validation CSV.

Reads questions from config.INPUT_CSV, runs the full RAG pipeline for each,
writes cleaned Cisco IOS answers to config.OUTPUT_CSV column "rag2+finetuning".

Supports incremental resume: rows that already have a non-empty answer are skipped.

Usage:
    python batch_evaluate.py

DEBUG MODE
──────────
Option A — env var (no code change):
    DEBUG_RAG=1 python batch_evaluate.py

Option B — flip constant in this file:
    DEBUG = True

When DEBUG=True, all [DEBUG:*] output from 03_rag_inference.py and
02_build_vectorstore.py is also active (the DEBUG flag is propagated to both
imported modules at startup).

WHAT IS LOGGED WHEN DEBUG=True
───────────────────────────────
[DEBUG:CONFIG]    Runtime configuration (CSV paths, model IDs, k, etc.)
[DEBUG:LOOP]      Per-row summary: row index, question, answer, timing.
[DEBUG:EXTRACT]   Lines accepted / skipped by extract_cisco_config().
[DEBUG:ERROR]     Exception details for failed rows.

All [DEBUG:E5PREFIX], [DEBUG:FAISS], [DEBUG:RETRIEVE], [DEBUG:CONTEXT],
[DEBUG:LLM], [DEBUG:SANITIZE] blocks from the underlying modules fire
automatically because DEBUG is propagated.
"""

import os
import time
import importlib
import textwrap

import torch
import pandas as pd
from huggingface_hub import login
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS

import config

# ── Debug flag ─────────────────────────────────────────────────────────────────
DEBUG: bool = os.environ.get("DEBUG_RAG", "0").strip() == "1"

_SEP_THICK = "═" * 72
_SEP_THIN  = "─" * 72
_C = {
    "cyan":   "\033[96m",
    "yellow": "\033[93m",
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


# Both upstream modules start with a digit and cannot be imported normally.
_bv = importlib.import_module("build_vectorstore")
load_embeddings = _bv.load_embeddings

_ri = importlib.import_module("rag_inference")
load_llm                     = _ri.load_llm
build_langchain_llm          = _ri.build_langchain_llm
PROMPT                       = _ri.PROMPT
FAISS_INDEX_DIR              = _ri.FAISS_INDEX_DIR
ask_rag_with_provenance_full = _ri.ask_rag_with_provenance_full

# Propagate DEBUG flag to upstream modules so their debug blocks fire too.
if DEBUG:
    _bv.DEBUG = True
    _ri.DEBUG = True


# ── Cisco config extractor ─────────────────────────────────────────────────────

def extract_cisco_config(model_response: str) -> str:
    """
    Strip code-fence markers and return only valid-looking Cisco IOS lines.

    NOTE: utils.extract_cisco_config() does the same thing — this local copy
    is kept for backward compatibility. If DEBUG is on it logs every decision.
    """
    if not model_response or not model_response.strip():
        _dbg("EXTRACT", "Input is empty — returning ''")
        return ""

    text  = model_response.replace("```", "").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    IGNORE = {"bash", "shell", "python", "javascript", "html", "css"}
    START  = {"enable", "configure terminal", "config t", "conf t"}
    CISCO_CMDS = (
        "interface", "ip", "ipv6", "router", "access-list",
        "line", "hostname", "username", "service", "no ",
    )

    config_lines   = []
    config_started = False
    skipped_lines  = []

    for line in lines:
        ll = line.lower()
        if any(ign in ll for ign in IGNORE):
            skipped_lines.append(f"  [ignored lang keyword] {line}")
            continue
        if any(s in ll for s in START):
            config_started = True
        if not config_started:
            if any(cmd in ll for cmd in CISCO_CMDS):
                config_started = True
        if config_started:
            config_lines.append(line)
            if ll in ("end", "exit"):
                break
        else:
            skipped_lines.append(f"  [pre-config, skipped] {line}")

    _dbg(
        "EXTRACT",
        f"extract_cisco_config — {len(config_lines)} lines accepted, "
        f"{len(skipped_lines)} skipped",
        "  Accepted lines:\n" + "\n".join(f"    {l}" for l in config_lines) if config_lines
        else "  (no lines accepted)",
    )
    if skipped_lines:
        _dbg("EXTRACT", "Skipped lines", "\n".join(skipped_lines))

    return "\n".join(config_lines)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if DEBUG:
        _dbg_section("DEBUG MODE ENABLED — batch_evaluate.py")
        _dbg(
            "CONFIG",
            "Runtime configuration",
            f"  DEBUG              : {DEBUG}\n"
            f"  LLM_MODEL_ID       : {config.LLM_MODEL_ID}\n"
            f"  CHECKPOINT_DIR     : {config.CHECKPOINT_DIR}\n"
            f"  INPUT_CSV          : {config.INPUT_CSV}\n"
            f"  OUTPUT_CSV         : {config.OUTPUT_CSV}\n"
            f"  RAG_K              : {config.RAG_K}\n"
            f"  SLEEP_BTW_CALLS    : {config.SLEEP_BETWEEN_CALLS}\n"
            f"  FAISS_INDEX_DIR    : {FAISS_INDEX_DIR}\n"
            f"\n"
            f"  DEBUG propagated to: build_vectorstore, rag_inference\n"
            f"  → All [DEBUG:E5PREFIX], [DEBUG:RETRIEVE], [DEBUG:LLM] etc. are active.",
        )

    login(config.HF_TOKEN)

    # ── Load components ────────────────────────────────────────────────────────
    _dbg_section("Loading embeddings")
    embeddings = load_embeddings(config.CHECKPOINT_DIR)
    if embeddings is None:
        raise RuntimeError("Could not load embeddings. Check CHECKPOINT_DIR in config.py.")

    _dbg(
        "CONFIG",
        "E5PrefixEmbeddings loaded",
        f"  type           : {type(embeddings).__name__}\n"
        f"  query_prefix   : {repr(getattr(embeddings, 'query_prefix', 'N/A'))}\n"
        f"  passage_prefix : {repr(getattr(embeddings, 'passage_prefix', 'N/A'))}\n"
        f"  _E5_PREFIXES   : {getattr(embeddings, '_E5_PREFIXES', 'N/A')}  ← double-prefix guard",
    )

    _dbg_section(f"Loading FAISS index from {FAISS_INDEX_DIR}/")
    print(f"Loading FAISS index from {FAISS_INDEX_DIR}/…")
    db = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    _dbg(
        "CONFIG",
        "FAISS index loaded",
        f"  Total vectors : {db.index.ntotal if hasattr(db, 'index') else 'N/A'}\n"
        f"  Dimension     : {db.index.d if hasattr(db, 'index') else 'N/A'}",
    )
    print("✅ FAISS index loaded.\n")

    _dbg_section("Loading LLM")
    model, tokenizer = load_llm()
    llm       = build_langchain_llm(model, tokenizer)
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    if not os.path.exists(config.INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {config.INPUT_CSV}")

    df = pd.read_csv(config.INPUT_CSV)
    required = {"question"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {config.INPUT_CSV}: {missing}")

    TARGET_COL = "rag2+finetuning"
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""

    rows_to_process = df.index[
        df[TARGET_COL].astype(str).str.strip() == ""
    ].tolist()

    print(f"Total rows: {len(df)} | To generate: {len(rows_to_process)}\n")

    _dbg(
        "LOOP",
        "CSV loop starting",
        f"  total rows      : {len(df)}\n"
        f"  rows to process : {len(rows_to_process)}\n"
        f"  target column   : {TARGET_COL}\n"
        f"  RAG k           : {config.RAG_K}\n"
        f"  output CSV      : {config.OUTPUT_CSV}",
    )

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

        t_start = time.time()
        try:
            # Pass embeddings_obj so the E5 prefix debug fires inside retrieve_with_scores.
            res = ask_rag_with_provenance_full(
                q, db, llm_chain,
                k=config.RAG_K,
                embeddings_obj=embeddings,
            )
            raw     = res["answer"]
            cleaned = extract_cisco_config(raw)

            df.at[i, TARGET_COL] = cleaned
            if config.SAVE_RAW:
                if "rag2+finetuning-raw" not in df.columns:
                    df["rag2+finetuning-raw"] = ""
                df.at[i, "rag2+finetuning-raw"] = raw

            elapsed = time.time() - t_start
            print(f"[{loop_idx + 1}/{len(rows_to_process)}] ✅ Done — {q[:60]}")

            _dbg(
                "LOOP",
                f"Row {i} complete  ({elapsed:.1f}s)",
                f"  question : {q}\n"
                f"  answer   :\n{textwrap.indent(cleaned, '    ')}",
            )

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress…")
            break
        except Exception as e:
            elapsed = time.time() - t_start
            _dbg(
                "ERROR",
                f"Row {i} failed after {elapsed:.1f}s — {type(e).__name__}: {e}",
            )
            print(f"[{loop_idx + 1}/{len(rows_to_process)}] ⚠️  Failed (row {i}): {e}")

        # Incremental save after every row
        df.to_csv(config.OUTPUT_CSV, index=False)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if config.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS)

    # Final save
    df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"\n✅ Done. Output saved → {config.OUTPUT_CSV}")
    _dbg_section("04_batch_evaluate.py complete")


if __name__ == "__main__":
    main()
