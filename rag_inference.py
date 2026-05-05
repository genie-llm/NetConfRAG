"""
03 rag_inference.py — RAG + LLM inference for Cisco IOS configuration.

Loads the FAISS vectorstore and LLM, then exposes:
  - ask_rag_with_provenance_full(question, k) → dict
  - ask_model(question)                       → str  (no RAG, baseline)

Can also be run directly for interactive single-question testing:
    python rag_inference.py --question "How to configure IPv6 ACL on an interface?"

Requires build_vectorstore.py to have been run (FAISS index on disk).

DEBUG MODE
──────────
Option A — env var (no code change):
    DEBUG_RAG=1 python rag_inference.py [--question "..."]

Option B — CLI flag:
    python rag_inference.py --debug --question "..."

Option C — flip constant in this file:
    DEBUG = True

Option D — runtime toggle:
    import importlib; ri = importlib.import_module("rag_inference"); ri.DEBUG = True

WHAT IS LOGGED WHEN DEBUG=True
───────────────────────────────
[DEBUG:E5PREFIX]   Prefix config, string construction, byte-level hex proof,
                   embedding vector statistics (norm + cosine similarity between
                   prefixed and un-prefixed vectors so you can confirm the prefix
                   is actually changing the embedding).

[DEBUG:FAISS]      The exact raw query string passed to FAISS — confirms no manual
                   prefix is added at this level (E5PrefixEmbeddings handles it).

[DEBUG:RETRIEVE]   Every retrieved chunk: rank, L2 score, chunk_id, source_file,
                   chapter/section, part, word/token counts, sha1, text[:300].

[DEBUG:CONTEXT]    Deduplication decisions, per-chunk character budget, final
                   context string (full text).

[DEBUG:LLM]        Full rendered prompt + raw model output before sanitisation.

[DEBUG:SANITIZE]   Lines accepted / rejected / deduplicated by IOS_LINE regex.

[DEBUG:NORAG]      Prompt and outputs for the no-RAG ask_model() baseline.

PREFIX SAFETY NOTE (important for intfloat/e5-* models)
────────────────────────────────────────────────────────
E5PrefixEmbeddings (defined in 02_build_vectorstore.py) adds prefixes:
  • embed_query("question")  → encoder sees "query: question"
  • embed_documents(["chunk"]) → encoder sees "passage: chunk"

FAISS.similarity_search_with_score() calls BOTH embed_query() AND
embed_documents() on the query internally. Without the double-prefix guard
in embed_documents(), the query would become "passage: query: question".
The guard in 02_build_vectorstore.py detects texts that already start with
a known e5 prefix and passes them through unchanged.

retrieve_with_scores() in this file passes the RAW question to FAISS with
no manual prefix — E5PrefixEmbeddings handles everything transparently.
"""

import re
import os
import json
import argparse
import importlib
import textwrap
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from huggingface_hub import login

import config

_bv = importlib.import_module("build_vectorstore")
load_embeddings    = _bv.load_embeddings
E5PrefixEmbeddings = _bv.E5PrefixEmbeddings

FAISS_INDEX_DIR = "faiss_index"

# ── Debug flag ─────────────────────────────────────────────────────────────────
# Set to True here, flip at runtime, or pass --debug on the CLI.
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


# ── LLM setup ─────────────────────────────────────────────────────────────────

def load_llm():
    """Load the quantised LLaMA model and tokeniser."""
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
        "BitsAndBytes quantisation config",
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
    """Wrap the HF model in a LangChain-compatible pipeline."""
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


# ── Prompt template ────────────────────────────────────────────────────────────

PROMPT_TEMPLATE_STR = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Cisco network engineer. Reply with ONLY a single fenced code block labeled ios.
NO explanations, NO notes, NO comments, NO extra text.

Rules:
- Start with: enable, configure terminal
- End with: end
- Use only valid Cisco IOS configuration commands (one per line).
- If IPv6 ACL is asked, use ipv6 syntax and apply it to an interface.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Context (from docs):
{context}

Question:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE_STR,
)


# ── IOS-only sanitiser ─────────────────────────────────────────────────────────

IOS_LINE = re.compile(
    r'^\s*(?:'
    r'(?:no|default)\s+)?'
    r'(?:'
    r'(?:interface|router|line|ip|ipv6|vlan|spanning-tree|hostname|username|aaa|snmp|logging|clock|ntp|crypto|banner|mls|lldp|cdp|switchport|control-plane)\b'
    r'|'
    r'(?:access-list|ip\s+access-list|ipv6\s+access-list|class-map|policy-map|service-policy|route-map|object-group|track|ip\s+sla)\b'
    r'|'
    r'(?:permit|deny|remark|match|set|neighbor|network|area|redistribute|address-family|passive-interface|shutdown|no\s+shutdown)\b'
    r'|'
    r'(?:[a-z][\w-]*)\s+(?:[\w./:-]*\d[\w./:-]*|[\w./:-]*:[\w./:-]+|[\w./:-]+/\d+\S*)'
    r')'
    r'(?:\s+\S+)*\s*$',
    re.IGNORECASE,
)


def sanitize_ios_only(generated: str) -> str:
    """Extract the first ``` block and keep only IOS-plausible lines."""
    _dbg_section("sanitize_ios_only()")
    m   = re.search(r'```(?:[a-zA-Z]+)?\n(.*?)```', generated, flags=re.S)
    raw = m.group(1) if m else generated

    _dbg(
        "SANITIZE",
        "Raw block extracted from model output",
        (raw if raw.strip() else "(empty — no fenced block found in generated text)")
        + f"\n\n  fence match : {'found ```...``` block' if m else 'no fence found — using raw output'}",
    )

    kept     = []
    rejected = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln_clean = re.split(r'\s[!#].*$', ln)[0].strip()
        if IOS_LINE.match(ln_clean):
            kept.append(ln_clean)
        else:
            rejected.append(ln_clean)

    _dbg(
        "SANITIZE",
        f"IOS_LINE filter: {len(kept)} kept, {len(rejected)} rejected",
        ("  Rejected lines:\n" + "\n".join(f"    {r}" for r in rejected))
        if rejected else "  No lines rejected.",
    )

    if not kept or kept[0].lower() != "enable":
        kept.insert(0, "enable")
    if len(kept) < 2 or kept[1].lower() != "configure terminal":
        kept.insert(1, "configure terminal")
    if kept[-1].lower() != "end":
        kept.append("end")

    result = "```ios\n" + "\n".join(kept) + "\n```"
    _dbg("SANITIZE", "Final sanitised output", result)
    return result


# ── No-RAG baseline ────────────────────────────────────────────────────────────

def ask_model(question: str, model, tokenizer) -> str:
    """Direct LLM answer without retrieval (no-RAG baseline)."""
    _dbg_section(f"ask_model() — no-RAG baseline  question={question[:70]!r}")

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a Cisco network engineer. Generate ONLY valid Cisco IOS commands "
        "in a single code block.\n\n"
        "Rules:\n"
        "- Start with: enable, configure terminal\n"
        "- End with: end\n"
        "- Use real Cisco commands only\n"
        "- No explanations or comments\n"
        "- No multiple choice answers\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```"
    )

    _dbg("NORAG", "Full prompt (no-RAG)", prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    _dbg("NORAG", f"Prompt token count: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    _dbg("NORAG", f"Generated token count: {len(generated_ids)}")

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    _dbg("NORAG", "Raw decoded output", answer)

    if answer.startswith("```"):
        answer = answer[3:]
    if answer.endswith("```"):
        answer = answer[:-3]

    final = f"```\n{answer.strip()}\n```"
    _dbg("NORAG", "Final output", final)
    return final


# ── E5 embedding debug helpers ─────────────────────────────────────────────────

def _debug_embedding_prefix(embeddings_obj, raw_query: str) -> None:
    """
    Verify E5 prefix injection end-to-end. Called from retrieve_with_scores()
    when DEBUG=True and embeddings_obj is provided.

    Checks three things and prints a verdict for each:

    1. PREFIX CONFIGURATION
       Reads query_prefix / passage_prefix from the E5PrefixEmbeddings wrapper.

    2. PREFIXED STRING CONSTRUCTION
       Shows raw vs prefixed query and a byte-level hex dump of the first 32
       bytes of each — zero ambiguity about whether the prefix is present.

    3. DOUBLE-PREFIX GUARD VERIFICATION
       Confirms that embed_documents() will NOT add a second "passage: " prefix
       when it receives a query string that already starts with "query: ".

    4. EMBEDDING VECTOR STATISTICS
       Embeds the raw query and the prefixed query separately, then prints:
         - L2 norm of each vector
         - Cosine similarity between them
         - First 8 dimensions side by side
       If cosine similarity ≈ 1.0, the prefix has no effect (wrong — investigate).
       If it differs, the prefix IS changing the embedding (correct).
    """
    if not DEBUG:
        return

    _dbg_section("_debug_embedding_prefix() — E5 prefix end-to-end verification")

    # ── 1. Prefix configuration ──────────────────────────────────────────────
    query_prefix   = getattr(embeddings_obj, "query_prefix",   None)
    passage_prefix = getattr(embeddings_obj, "passage_prefix", None)
    e5_prefixes    = getattr(embeddings_obj, "_E5_PREFIXES",   None)

    cfg_lines = [
        f"  embeddings type   : {type(embeddings_obj).__name__}",
        f"  query_prefix      : {query_prefix!r}",
        f"  passage_prefix    : {passage_prefix!r}",
        f"  _E5_PREFIXES      : {e5_prefixes!r}  ← used by double-prefix guard",
    ]
    if query_prefix is None:
        cfg_lines.append("  ⚠️  WARNING: query_prefix is None — prefix may not be applied!")
    elif query_prefix == "":
        cfg_lines.append("  ⚠️  WARNING: query_prefix is empty string — no prefix will be added.")
    else:
        cfg_lines.append(f"  ✅ query_prefix is set: {query_prefix!r}")
    _dbg("E5PREFIX", "1 — Prefix configuration", "\n".join(cfg_lines))

    # ── 2. String construction ────────────────────────────────────────────────
    prefixed_query = f"{query_prefix}{raw_query}" if query_prefix else raw_query
    raw_bytes      = raw_query.encode("utf-8")[:32]
    pref_bytes     = prefixed_query.encode("utf-8")[:32]

    str_lines = [
        f"  raw_query         : {raw_query!r}",
        f"  prefixed_query    : {prefixed_query!r}",
        f"  raw   hex[:32]    : {raw_bytes.hex(' ')}",
        f"  pref  hex[:32]    : {pref_bytes.hex(' ')}",
        f"  strings differ    : {raw_query != prefixed_query}",
    ]
    if raw_query == prefixed_query:
        str_lines.append("  ⚠️  raw == prefixed — prefix was NOT applied!")
    else:
        str_lines.append("  ✅ prefix IS present in the string sent to the encoder.")
    _dbg("E5PREFIX", "2 — String construction", "\n".join(str_lines))

    # ── 3. Double-prefix guard verification ──────────────────────────────────
    already_prefixed = prefixed_query.startswith(tuple(e5_prefixes)) if e5_prefixes else False
    guard_lines = [
        f"  embed_documents() receives : {repr(prefixed_query[:80])}",
        f"  starts with known e5 prefix: {already_prefixed}",
    ]
    if already_prefixed:
        guard_lines.append(
            "  ✅ Guard FIRES — embed_documents() will pass this through unchanged.\n"
            "     Final string reaching encoder: same as above (no second prefix)."
        )
    else:
        guard_lines.append(
            "  ⚠️  Guard does NOT fire — 'passage: ' would be prepended.\n"
            "     This path should only happen for actual document chunks, not queries."
        )
    _dbg("E5PREFIX", "3 — Double-prefix guard verification", "\n".join(guard_lines))

    # ── 4. Embedding vector statistics ───────────────────────────────────────
    try:
        # embed_query() applies the prefix internally; this gives us the
        # production vector that FAISS will compare against document vectors.
        vec_prefixed = np.array(embeddings_obj.embed_query(raw_query), dtype=np.float32)

        # Embed WITHOUT prefix by reaching the underlying SentenceTransformer.
        underlying = (
            getattr(embeddings_obj, "client", None)
            or getattr(embeddings_obj, "_model", None)
            or getattr(embeddings_obj, "model", None)
        )
        if underlying is not None and hasattr(underlying, "encode"):
            vec_raw = np.array(
                underlying.encode(raw_query, normalize_embeddings=True),
                dtype=np.float32,
            )
        else:
            vec_raw = vec_prefixed
            _dbg(
                "E5PREFIX",
                "4 — Vector stats (fallback)",
                "  ⚠️  Could not reach underlying encoder — showing prefixed vec only.",
            )
            return

        norm_pref = float(np.linalg.norm(vec_prefixed))
        norm_raw  = float(np.linalg.norm(vec_raw))
        cos_sim   = float(
            np.dot(vec_prefixed, vec_raw) / (norm_pref * norm_raw + 1e-9)
        )

        vec_lines = [
            f"  dim                 : {vec_prefixed.shape[0]}",
            f"  norm (prefixed)     : {norm_pref:.6f}",
            f"  norm (raw/no-pfx)   : {norm_raw:.6f}",
            f"  cosine similarity   : {cos_sim:.6f}  (1.0 = identical, <1 = prefix changed vec)",
            f"",
            f"  first 8 dims (prefixed) : {vec_prefixed[:8].tolist()}",
            f"  first 8 dims (raw)      : {vec_raw[:8].tolist()}",
            f"",
        ]
        if cos_sim > 0.9999:
            vec_lines.append(
                "  ⚠️  Vectors are near-identical — prefix has NO effect on the embedding!\n"
                "     Check E5PrefixEmbeddings.embed_query in 02_build_vectorstore.py."
            )
        else:
            vec_lines.append(
                f"  ✅ Vectors differ (cosine={cos_sim:.4f}) — prefix IS changing the embedding."
            )
        _dbg("E5PREFIX", "4 — Embedding vector statistics", "\n".join(vec_lines))

    except Exception as exc:
        _dbg(
            "ERROR",
            f"Vector stats failed: {type(exc).__name__}: {exc}",
            "  Skipping vector comparison — check embed_query / underlying encoder.",
        )


# ── RAG retrieval helpers ──────────────────────────────────────────────────────

def retrieve_with_scores(
    db,
    query: str,
    k: int = 4,
    embeddings_obj=None,
) -> List[Tuple[Document, float]]:
    """
    Return [(Document, distance_score)] from FAISS (lower = closer).

    query        — raw question string; NO manual prefix added here.
    embeddings_obj — pass the E5PrefixEmbeddings instance to enable deep
                   prefix verification in DEBUG mode.

    PREFIX FLOW:
    ────────────
    FAISS.similarity_search_with_score(query) internally calls
    embeddings_obj.embed_query(query), which prepends "query: " once.
    The double-prefix guard in embed_documents() prevents a second
    "passage: " being added if FAISS calls embed_documents on the result.
    """
    _dbg_section(f"retrieve_with_scores() — query={query[:70]!r}")

    if DEBUG and embeddings_obj is not None:
        _debug_embedding_prefix(embeddings_obj, query)

    _dbg(
        "FAISS",
        "Calling db.similarity_search_with_score()",
        f"  raw query (no manual prefix added here) : {repr(query)}\n"
        f"  k                                       : {k}\n"
        f"\n"
        f"  E5PrefixEmbeddings.embed_query() will prepend 'query: ' internally.\n"
        f"  The double-prefix guard in embed_documents() prevents re-prefixing.\n"
        f"  No call to utils.retrieve_with_scores() here — no risk of overlap.",
    )

    results = db.similarity_search_with_score(query, k=k)

    if DEBUG:
        lines = [
            f"  query (raw) : {query!r}",
            f"  k           : {k}",
            f"  returned    : {len(results)} chunk(s)",
            "",
        ]
        for i, (doc, score) in enumerate(results, start=1):
            md = doc.metadata or {}
            chunk_text = (
                md.get("display_text")
                or getattr(doc, "display_text", None)
                or doc.page_content
                or ""
            )
            lines.append(f"  ── Chunk #{i} {'─' * 50}")
            lines.append(f"    FAISS L2 score : {score:.6f}  (lower = more similar)")
            lines.append(f"    chunk_id       : {md.get('chunk_id')}")
            lines.append(f"    source_file    : {md.get('source_file')}")
            lines.append(f"    pdf_name       : {md.get('pdf_name')}")
            lines.append(f"    chapter        : {md.get('chapter')}")
            lines.append(f"    section        : {md.get('section')}")
            lines.append(f"    part           : {md.get('part')} / {md.get('parts_total')}")
            lines.append(f"    word_count     : {md.get('word_count')}")
            lines.append(f"    token_count    : {md.get('token_count')}")
            lines.append(f"    has_sem_hdr    : {md.get('has_semantic_header')}")
            lines.append(f"    header_style   : {md.get('header_style')}")
            lines.append(f"    sha1           : {md.get('sha1')}")
            lines.append(f"    text[:300]     :")
            lines.append(textwrap.indent(chunk_text[:300], "      "))
            lines.append("")
        _dbg("RETRIEVE", f"FAISS results — {len(results)} chunk(s)", "\n".join(lines))

    return results


def build_context_and_provenance_full(
    results: List[Tuple[Document, float]],
    max_chars: int = 8000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build the LLM context string and a full provenance list."""
    _dbg_section("build_context_and_provenance_full()")
    parts  = []
    used   = []
    seen   = set()

    for rank, (doc, score) in enumerate(results, start=1):
        md = doc.metadata or {}
        chunk_text = (
            md.get("display_text")
            or getattr(doc, "display_text", None)
            or doc.page_content
            or ""
        )
        key = chunk_text[:200]

        if key in seen:
            _dbg(
                "CONTEXT",
                f"Chunk #{rank} SKIPPED — duplicate",
                f"  score={score:.6f}  key[:80]={key[:80]!r}",
            )
            continue
        seen.add(key)

        projected = sum(len(p) for p in parts) + len(chunk_text)
        if parts and projected > max_chars:
            _dbg(
                "CONTEXT",
                f"Chunk #{rank} SKIPPED — max_chars budget reached",
                f"  projected={projected}  limit={max_chars}\n"
                f"  Remaining chunks will NOT be included.",
            )
            break

        parts.append(chunk_text.strip())
        used.append({
            "rank":               rank,
            "score":              float(score),
            "chunk_id":           md.get("chunk_id"),
            "source_file":        md.get("source_file"),
            "pdf_name":           md.get("pdf_name"),
            "chapter":            md.get("chapter"),
            "section":            md.get("section"),
            "part":               md.get("part"),
            "parts_total":        md.get("parts_total"),
            "word_count":         md.get("word_count"),
            "token_count":        md.get("token_count"),
            "has_semantic_header": md.get("has_semantic_header"),
            "header_style":       md.get("header_style"),
            "sha1":               md.get("sha1"),
            "chunk_text":         chunk_text,
        })

        cumulative = sum(len(p) for p in parts)
        _dbg(
            "CONTEXT",
            f"Chunk #{rank} ACCEPTED",
            f"  score={score:.6f}  chunk_chars={len(chunk_text)}  "
            f"cumulative={cumulative}/{max_chars}\n"
            f"  chunk_id={md.get('chunk_id')}  source={md.get('source_file')}",
        )

    context = "\n\n---\n\n".join(parts)
    _dbg(
        "CONTEXT",
        f"Final context — {len(used)} chunk(s), {len(context)} chars",
        f"  chunks_requested : {len(results)}\n"
        f"  chunks_used      : {len(used)}\n"
        f"  total_chars      : {len(context)}\n"
        f"\n--- FULL CONTEXT ---\n{context}\n--- END CONTEXT ---",
    )
    return context, used


# ── Main RAG function ──────────────────────────────────────────────────────────

def ask_rag_with_provenance_full(
    question: str,
    db,
    llm_chain: LLMChain,
    k: int = 4,
    save_path: str | None = None,
    embeddings_obj=None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
      1. Retrieve k chunks  (E5 prefix debug if embeddings_obj provided)
      2. Build context + provenance
      3. Call LLM
      4. Post-process to IOS-only output

    Returns {"answer", "context_used", "used_chunks", "count"}.
    Optionally saves the full payload to save_path as JSON.

    Pass embeddings_obj (the E5PrefixEmbeddings instance) to enable deep
    prefix verification in DEBUG mode.
    """
    _dbg_section(f"ask_rag_with_provenance_full() — question={question[:70]!r}")
    _dbg(
        "LLM",
        "Pipeline call parameters",
        f"  question       : {question!r}\n"
        f"  k              : {k}\n"
        f"  save_path      : {save_path!r}\n"
        f"  embeddings_obj : {type(embeddings_obj).__name__ if embeddings_obj else 'None (prefix debug disabled)'}",
    )

    # ── Step 1: retrieval ─────────────────────────────────────────────────────
    results = retrieve_with_scores(db, question, k=k, embeddings_obj=embeddings_obj)

    # ── Step 2: context ───────────────────────────────────────────────────────
    context, used_chunks = build_context_and_provenance_full(results)

    # ── Step 3: LLM call ──────────────────────────────────────────────────────
    rendered_prompt = PROMPT_TEMPLATE_STR.format(context=context, question=question)
    _dbg(
        "LLM",
        "Full prompt sent to LLM (reconstructed for debug)",
        f"  prompt_length_chars : {len(rendered_prompt)}\n"
        f"\n--- FULL PROMPT ---\n{rendered_prompt}\n--- END PROMPT ---",
    )

    raw  = llm_chain.invoke({"context": context, "question": question})
    text = raw["text"] if isinstance(raw, dict) and "text" in raw else str(raw)

    _dbg(
        "LLM",
        "Raw LLM output (return_full_text=False → new tokens only)",
        f"  type : {type(raw).__name__}\n"
        f"  key  : {'text' if isinstance(raw, dict) and 'text' in raw else 'str(raw)'}\n"
        f"\n--- RAW OUTPUT ---\n{text}\n--- END RAW OUTPUT ---",
    )

    # ── Step 4: sanitise ──────────────────────────────────────────────────────
    try:
        answer = sanitize_ios_only(text)
    except Exception as exc:
        _dbg(
            "ERROR",
            f"sanitize_ios_only raised {type(exc).__name__}: {exc} — using raw text",
        )
        answer = text

    _dbg("LLM", "Final answer after sanitisation", answer)

    payload = {
        "answer":       answer,
        "context_used": context,
        "used_chunks":  used_chunks,
        "count":        len(used_chunks),
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _dbg("LLM", f"Payload saved to: {save_path}")

    return payload


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Cisco RAG inference")
    parser.add_argument(
        "--question", "-q",
        default="How to Configure the IPv6 ACL to an Interface?",
        help="Question to ask the RAG pipeline",
    )
    parser.add_argument("--k", type=int, default=config.RAG_K,
                        help="Number of chunks to retrieve")
    parser.add_argument("--save", default=None,
                        help="Optional JSON path to save full RAG result")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print retrieved chunks and scores")
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Enable verbose debug output: E5 prefix verification, "
            "raw prompt, raw LLM output, chunk details, context, etc."
        ),
    )
    args = parser.parse_args()

    if args.debug:
        DEBUG = True
        # Propagate to 02_build_vectorstore so its E5PrefixEmbeddings debug blocks fire too.
        _bv.DEBUG = True
        print(f"{_C['yellow']}[DEBUG] Debug mode enabled (rag_inference + build_vectorstore).{_C['reset']}")

    if DEBUG:
        _dbg_section("DEBUG MODE ENABLED — rag_inference.py")
        _dbg(
            "CONFIG",
            "Runtime configuration",
            f"  LLM_MODEL_ID    : {config.LLM_MODEL_ID}\n"
            f"  CHECKPOINT_DIR  : {config.CHECKPOINT_DIR}\n"
            f"  FAISS_INDEX_DIR : {FAISS_INDEX_DIR}\n"
            f"  RAG_K           : {config.RAG_K}\n"
            f"\n"
            f"  PREFIX FLOW (intfloat/e5-*):\n"
            f"  • embed_query(q)     → 'query: q'     (in E5PrefixEmbeddings)\n"
            f"  • embed_documents(d) → 'passage: d'   (in E5PrefixEmbeddings)\n"
            f"  • Double-prefix guard: texts already prefixed are passed through.\n"
            f"  • retrieve_with_scores() passes RAW query to FAISS — no manual prefix.",
        )

    login(config.HF_TOKEN)

    embeddings = load_embeddings(config.CHECKPOINT_DIR)
    if embeddings is None:
        raise RuntimeError("Could not load embedding model. Run build_vectorstore.py first.")

    print(f"\nLoading FAISS index from {FAISS_INDEX_DIR}/…")
    _dbg(
        "INDEX",
        f"Loading FAISS index from disk: {FAISS_INDEX_DIR}/",
        "  allow_dangerous_deserialization=True (required for FAISS .pkl files)",
    )
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    _dbg(
        "INDEX",
        "FAISS index loaded",
        f"  Total vectors : {db.index.ntotal if hasattr(db, 'index') else 'N/A'}\n"
        f"  Dimension     : {db.index.d if hasattr(db, 'index') else 'N/A'}",
    )
    print("✅ FAISS index loaded.\n")

    model, tokenizer = load_llm()
    llm       = build_langchain_llm(model, tokenizer)
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")

    result = ask_rag_with_provenance_full(
        args.question,
        db,
        llm_chain,
        k=args.k,
        save_path=args.save,
        embeddings_obj=embeddings,   # enables E5 prefix debug
    )

    print("── Answer ──────────────────────────────────────────────────")
    print(result["answer"])

    if args.show_chunks:
        print("\n── Retrieved Chunks ────────────────────────────────────────")
        for u in result["used_chunks"]:
            print(f"\n[#{u['rank']}] score={u['score']:.4f} | "
                  f"source={u.get('source_file')} | "
                  f"chapter={u.get('chapter')} | section={u.get('section')}")
            print("--- Chunk text ---")
            print(u["chunk_text"][:2000])

    if args.save:
        print(f"\n✅ Full result saved → {args.save}")


if __name__ == "__main__":
    main()
