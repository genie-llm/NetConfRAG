"""
utils.py — Shared utilities for the Cisco RAG pipeline.

Imported by the baseline scripts so that all shared logic lives in one place.

NOTE: The main pipeline script (03_rag_inference.py) is kept self-contained
(no imports from utils) so that it can be run as a standalone script without
any circular dependency. The implementations here are identical to those in
03_rag_inference.py — if you modify one, update the other.

DEBUG MODE
──────────
Set DEBUG = True below, or toggle it at runtime from any importer:
    import utils; utils.DEBUG = True

Or via env var:
    DEBUG_RAG=1 python <your_script>.py

Debug output includes:
  [DEBUG:SANITIZE]  Per-line accept/reject/duplicate decisions in sanitize_ios_only().
  [DEBUG:EXTRACT]   Per-line accept/skip decisions in extract_cisco_config().
  [DEBUG:RETRIEVE]  FAISS results: scores, chunk metadata, text previews.
  [DEBUG:CONTEXT]   Deduplication, budget tracking, final assembled context.
"""

import json
import os
import re
import textwrap
from typing import List, Dict, Any, Tuple

from langchain.docstore.document import Document

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


# ── Prompt template (shared across all RAG baselines) ─────────────────────────

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


# ── IOS sanitiser ──────────────────────────────────────────────────────────────

IOS_LINE = re.compile(
    r'^\s*(?:'
    r'(?:enable|end|exit|do)\s*$'
    r'|'
    r'(?:configure\s+terminal|config\s+t|conf\s+t)\s*$'
    r'|'
    r'(?:(?:no|default)\s+)?'
    r'(?:'
    r'(?:interface|router|line|ip|ipv6|vlan|spanning-tree|hostname|username|aaa|snmp|logging|clock|ntp|crypto|banner|mls|lldp|cdp|switchport|control-plane)\b'
    r'|'
    r'(?:access-list|ip\s+access-list|ipv6\s+access-list|class-map|policy-map|service-policy|route-map|object-group|track|ip\s+sla)\b'
    r'|'
    r'(?:permit|deny|remark|match|set|neighbor|network|area|redistribute|address-family|passive-interface|shutdown|no\s+shutdown)\b'
    r'|'
    r'(?:[a-z][\w-]*)\s+(?:[\w./:-]*\d[\w./:-]*|[\w./:-]*:[\w./:-]+|[\w./:-]+/\d+\S*)'
    r')'
    r'(?:\s+\S+)*'
    r')'
    r'\s*$',
    re.IGNORECASE,
)


def sanitize_ios_only(generated: str) -> str:
    """Extract IOS commands from the model's continuation output.

    The prompt template closes with an opening ``` fence so the model continues
    *inside* that fence. LangChain's return_full_text=False means `generated`
    contains only the NEW tokens — everything after the opening ```.

    Steps:
      1. Strip the leading language tag line ("ios", "IOS XE", etc.) if present.
      2. Strip the trailing closing ``` and anything after it.
      3. Filter each remaining line through IOS_LINE.
      4. Deduplicate on exact full line — preserves repeated commands with
         different args.
    """
    _dbg_section("sanitize_ios_only()")

    lines = generated.splitlines()
    if lines and re.fullmatch(r'[ \t]*ios[\w ]*[ \t]*', lines[0], flags=re.IGNORECASE):
        _dbg("SANITIZE", "Stripped leading language tag", f"  removed: {repr(lines[0])}")
        lines = lines[1:]
    raw = "\n".join(lines)
    raw = re.sub(r'```.*$', '', raw, flags=re.S).strip()

    _dbg(
        "SANITIZE",
        "Raw block after stripping lang-tag + closing fence",
        raw if raw.strip() else "(empty — nothing usable found)",
    )

    kept      = []
    rejected  = []
    duplicate = []
    seen_lines: set = set()

    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln_clean = re.split(r'\s[!#].*$', ln)[0].strip()
        if not IOS_LINE.match(ln_clean):
            rejected.append(ln_clean)
            continue
        dedup_key = ln_clean.lower()
        if dedup_key in seen_lines:
            duplicate.append(ln_clean)
            continue
        seen_lines.add(dedup_key)
        kept.append(ln_clean)

    _dbg(
        "SANITIZE",
        f"IOS_LINE filter: {len(kept)} kept, {len(rejected)} rejected, {len(duplicate)} duplicates",
        ("  REJECTED:\n" + "\n".join(f"    {r}" for r in rejected) if rejected else "  No rejections.")
        + ("\n  DUPLICATES (dropped):\n" + "\n".join(f"    {d}" for d in duplicate) if duplicate else ""),
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


def extract_cisco_config(model_response: str) -> str:
    """
    Strip code-fence markers and return only valid-looking Cisco IOS lines.
    Used to clean the LLM output before writing to the CSV.
    """
    _dbg_section("extract_cisco_config()")

    if not model_response or not model_response.strip():
        _dbg("EXTRACT", "Input is empty — returning ''")
        return ""

    text  = model_response.replace("```", "").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    IGNORE     = {"bash", "shell", "python", "javascript", "html", "css"}
    START      = {"enable", "configure terminal", "config t", "conf t"}
    CISCO_CMDS = ("interface", "ip", "ipv6", "router", "access-list",
                  "line", "hostname", "username", "service", "no ")

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
        "  Accepted:\n" + "\n".join(f"    {l}" for l in config_lines)
        if config_lines else "  (no lines accepted)",
    )
    if skipped_lines:
        _dbg("EXTRACT", "Skipped lines", "\n".join(skipped_lines))

    return "\n".join(config_lines)


# ── Chunk / document helpers ───────────────────────────────────────────────────

def load_chunks_as_documents(json_path: str) -> List[Document]:
    """Load a chunks JSON file and return LangChain Document objects.

    The chunk schema stores the unique id as a top-level key ("id") rather
    than inside "metadata".  FAISS only preserves the metadata dict, so we
    copy the top-level id into metadata["chunk_id"] here so every retrieved
    Document carries it without any special-casing downstream.

    Extra top-level scalar fields (token_count, sha1, has_semantic_header,
    header_style, header_tokens) are also merged in so provenance logs and
    debug output can surface them.
    """
    EXTRA_TOP_LEVEL_FIELDS = {
        "token_count", "sha1", "has_semantic_header",
        "header_style", "header_tokens",
    }

    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    docs = []
    for c in chunks_data:
        meta = dict(c.get("metadata") or {})

        if "chunk_id" not in meta:
            meta["chunk_id"] = c.get("id")

        for field in EXTRA_TOP_LEVEL_FIELDS:
            if field not in meta and field in c:
                meta[field] = c[field]

        docs.append(Document(page_content=c["text"], metadata=meta))

    print(f"Loaded {len(docs)} documents from {json_path}")
    _dbg(
        "CONTEXT",
        f"load_chunks_as_documents — {len(docs)} docs from {json_path}",
        f"  Sample metadata doc[0]: {docs[0].metadata if docs else 'N/A'}\n"
        f"  Sample text    doc[0][:200]: {docs[0].page_content[:200] if docs else 'N/A'}",
    )
    return docs


# ── Retrieval helpers (shared by baselines 3–5) ────────────────────────────────

def retrieve_with_scores(
    db, query: str, k: int = 4, query_prefix: str = ""
) -> List[Tuple[Document, float]]:
    """FAISS: return [(Document, distance_score)] — lower score = closer.

    query_prefix — if non-empty, prepended to query before the FAISS call.
    NOTE: When used with E5PrefixEmbeddings (02_build_vectorstore.py), leave
    query_prefix='' — the embedding class handles the 'query: ' prefix itself.
    Passing a prefix here AND using E5PrefixEmbeddings would cause double-prefixing.
    """
    _dbg_section(f"retrieve_with_scores() — query={query[:70]!r}")

    effective_query = f"{query_prefix}{query}" if query_prefix else query

    _dbg(
        "RETRIEVE",
        "Query construction",
        f"  raw query       : {repr(query)}\n"
        f"  query_prefix    : {repr(query_prefix)}\n"
        f"  effective_query : {repr(effective_query)}\n"
        f"  k               : {k}\n"
        f"\n"
        f"  ⚠️  PREFIX NOTE: if db uses E5PrefixEmbeddings, leave query_prefix='' here.\n"
        f"     The embedding class adds 'query: ' automatically inside embed_query().\n"
        f"     Passing a prefix here AND using E5PrefixEmbeddings = double-prefix.",
    )

    results = db.similarity_search_with_score(effective_query, k=k)

    if DEBUG:
        lines = [
            f"  query (raw)     : {query!r}",
            f"  effective_query : {effective_query!r}",
            f"  k               : {k}",
            f"  returned        : {len(results)} chunk(s)",
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
            lines.append(f"    score       : {score:.6f}  (lower = more similar)")
            lines.append(f"    chunk_id    : {md.get('chunk_id')}")
            lines.append(f"    source_file : {md.get('source_file')}")
            lines.append(f"    pdf_name    : {md.get('pdf_name')}")
            lines.append(f"    chapter     : {md.get('chapter')}")
            lines.append(f"    section     : {md.get('section')}")
            lines.append(f"    part        : {md.get('part')} / {md.get('parts_total')}")
            lines.append(f"    word_count  : {md.get('word_count')}")
            lines.append(f"    token_count : {md.get('token_count')}")
            lines.append(f"    text[:300]  :")
            lines.append(textwrap.indent(chunk_text[:300], "      "))
            lines.append("")
        _dbg("RETRIEVE", f"FAISS results — {len(results)} chunk(s)", "\n".join(lines))

    return results


def build_context_and_provenance_full(
    results: List[Tuple[Document, float]],
    max_chars: int = 4000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build the LLM context string and a full provenance list from FAISS results.
    Deduplicates on the first 200 chars of each chunk.
    """
    _dbg_section("build_context_and_provenance_full()")

    parts: List[str]            = []
    used:  List[Dict[str, Any]] = []
    seen  = set()

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
