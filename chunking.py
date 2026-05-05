"""
01 chunking.py — Hierarchical chunking of JSON knowledge-base files.

Reads JSON docs from config.DATA_JSON_DIR, applies hierarchical chunking with
semantic headers, and saves the result to config.CHUNKS_JSON.

Usage:
    python chunking.py
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Any

import config
import tiktoken

# ── Header helpers ─────────────────────────────────────────────────────────────

def create_semantic_header(
    pdf_name: str,
    chapter_title: str,
    section_title: str,
    part_info: str = "",
    style: str = "ignored",
    enable: bool = True,
    sep: str = " > ",
    n_words_chapter: int = 5,
) -> str:
    """
    Uniform header for ALL chunks:
        <pdf_name> > <section_title> >
    Returns an empty string when enable=False.
    """
    if not enable:
        return ""
    pdf = (pdf_name or "").strip()
    sec = (section_title or "").strip()
    parts = [p for p in (pdf, sec) if p]
    return sep.join(parts) + " >"


def reserve_body_budget(
    max_tokens: int,
    tokens_len,
    pdf_name: str,
    chapter_title: str,
    section_title: str,
    header_style: str = "ignored",
    include_header: bool = True,
    min_body_floor: int = 200,
    sep: str = " > ",
    n_words_chapter: int = 5,
):
    """
    Calculates the token budget for the body text, accounting for the header.
    Always returns the same header text to guarantee uniform headers.
    """
    if not include_header:
        return max_tokens, "", 0

    hdr = create_semantic_header(pdf_name, chapter_title, section_title,
                                 enable=True, sep=sep, n_words_chapter=n_words_chapter)
    cost   = tokens_len(hdr)
    budget = max(1, max(min_body_floor, max_tokens - cost))
    return budget, hdr, cost


# ── CLI block preservation ─────────────────────────────────────────────────────

CLI_BLOCK_START = re.compile(
    r'^(?:interface\s+\S+|router\s+\S+|ip\s+access-list(?:\s+\S+)?|line\s+\S+)\b',
    re.I,
)


def split_preserving_cli_blocks(text: str) -> list:
    lines = text.splitlines()
    blocks, buf = [], []
    for ln in lines:
        if CLI_BLOCK_START.match(ln.strip()) and buf:
            blocks.append("\n".join(buf))
            buf = []
        buf.append(ln)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


# ── Boilerplate filter ─────────────────────────────────────────────────────────

EXCLUDE_PHRASES = [
    "Finding Feature Information", "Additional References", "Related Documentation",
    "Document Conventions", "Technical Assistance", "Support Information",
    "Documentation Updates", "Obtaining Documentation", "Bug Search Tool",
]


def is_boilerplate(s: str) -> bool:
    low = s.lower()
    return any(p.lower() in low for p in EXCLUDE_PHRASES)


def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()


# ── Main chunking function ─────────────────────────────────────────────────────

def chunking_hierarchical(
    data_paths: List[str],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
    minimize_context: bool = True,
    adaptive_overlap: bool = True,
    max_context_tokens: int = 300,
    include_semantic_header: bool = True,
    header_style: str = "cisco",
    min_words_per_chunk: int = 15,
) -> List[Dict[str, Any]]:
    """
    Hierarchical chunking with:
    - Semantic headers embedded in the indexed text
    - CLI-block-aware splitting
    - Boilerplate filtering and deduplication
    - Enriched metadata
    """
    # Tokeniser (tiktoken preferred, character-based fallback)
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        def to_tokens(s):   return enc.encode(s)
        def tokens_len(s):  return len(enc.encode(s))
        def from_tokens(t): return enc.decode(t)
    except Exception:
        def to_tokens(s):   return [s[i:i+4] for i in range(0, len(s), 4)]
        def tokens_len(s):  return max(1, len(s) // 4)
        def from_tokens(t): return "".join(t)

    assert max_tokens > 0
    assert 0 <= overlap_tokens < max_tokens
    assert max_context_tokens > 0

    def create_optimized_context(pdf_name, chapter_title, chapter_intro=""):
        context = f"PDF: {pdf_name}"
        if chapter_title and chapter_title.strip():
            title = chapter_title.strip()
            if title.upper().startswith(("CHAPTER", "CHAPITRE")):
                parts = title.split(" ", 3)
                num   = " ".join(parts[:2])
                main  = parts[2] if len(parts) > 2 else ""
                context += f"\n{num}: {main}"
            else:
                context += f"\nChapter: {title[:100]}{'...' if len(title)>100 else ''}"
        if chapter_intro and minimize_context:
            intro = chapter_intro.strip()
            if len(intro) <= 150:
                context += f"\n{intro}"
        return context

    def should_apply_overlap(content_size, budget_size):
        return not adaptive_overlap or (content_size / budget_size) > 1.3

    def calculate_adaptive_overlap(content_size, budget_size, base_overlap):
        if not adaptive_overlap:
            return base_overlap
        ratio = content_size / budget_size
        if ratio <= 1.3:   return min(50,  base_overlap // 4)
        elif ratio <= 2.0: return min(100, base_overlap // 2)
        return base_overlap

    def sliding_windows_tokens(body_tokens, max_len, overlap):
        if not body_tokens:
            return []
        stride  = max(1, max_len - overlap)
        windows = []
        start   = 0
        n       = len(body_tokens)
        while start < n:
            end = min(n, start + max_len)
            windows.append(body_tokens[start:end])
            if end == n:
                break
            start += stride
        return windows

    chunks     = []
    chunk_id   = 0
    seen_hashes = set()
    stats = dict(contexts_optimized=0, overlaps_reduced=0, tokens_saved=0,
                 headers_added=0, header_tokens=0)

    print(f"📁 {len(data_paths)} JSON files to process:")
    for p in data_paths:
        print(f"  - {os.path.basename(p)}")
    print()

    for file_index, file_path in enumerate(data_paths, 1):
        print(f"📂 File {file_index}/{len(data_paths)}: {os.path.basename(file_path)}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_chunk_count = 0
            pdf_name = ""

            if isinstance(data, dict):
                pdf_name = (data.get("pdf_name") or data.get("document_name")
                            or data.get("source") or
                            os.path.splitext(os.path.basename(file_path))[0])

            for main_key, chapters in data.items():
                if main_key in ("pdf_name", "document_name", "source") or not isinstance(chapters, list):
                    continue

                print(f"  📖 Section: {main_key} ({len(chapters)} chapters)")

                for chapter in chapters:
                    chapter_title   = chapter.get("chapter") or chapter.get("title") or ""
                    chapter_intro   = chapter.get("intro") or chapter.get("introduction") or ""
                    chapter_context = create_optimized_context(pdf_name, chapter_title, chapter_intro)

                    if minimize_context:
                        orig = f"Document: {main_key}\n{chapter_title}".strip()
                        if chapter_intro:
                            orig += f"\n{chapter_intro}"
                        saved = tokens_len(orig) - tokens_len(chapter_context)
                        if saved > 0:
                            stats["tokens_saved"]       += saved
                            stats["contexts_optimized"] += 1

                    ctx_toks = to_tokens(chapter_context + "\n\n")
                    if len(ctx_toks) > max_context_tokens:
                        ctx_toks        = ctx_toks[:max_context_tokens]
                        chapter_context = from_tokens(ctx_toks).rstrip() + "..."

                    for section in chapter.get("sections", []):
                        section_title   = section.get("title") or section.get("section") or ""
                        section_content = (section.get("content") or section.get("text")
                                           or section.get("body") or "")

                        if not section_content or is_boilerplate(section_content):
                            continue

                        section_content = re.sub(r'[ \t]+', ' ', section_content)
                        section_content = re.sub(r'\n{3,}', '\n\n', section_content).strip()
                        cli_blocks      = split_preserving_cli_blocks(section_content)
                        section_content = "\n".join(cli_blocks) if cli_blocks else section_content

                        budget_for_body, header_text, header_cost = reserve_body_budget(
                            max_tokens=max_tokens,
                            tokens_len=tokens_len,
                            pdf_name=pdf_name,
                            chapter_title=chapter_title,
                            section_title=section_title,
                            header_style=header_style,
                            include_header=include_semantic_header,
                        )

                        body_tokens = to_tokens(section_content)

                        def make_chunk(text_body, part_idx=None, parts_total=None, p_header=None):
                            nonlocal chunk_id
                            used_hdr = p_header if p_header is not None else header_text
                            final    = (used_hdr + text_body) if include_semantic_header else text_body
                            if len(final.split()) < min_words_per_chunk:
                                return
                            h = sha1(final)
                            if h in seen_hashes:
                                return
                            seen_hashes.add(h)

                            meta = dict(
                                source_file=os.path.basename(file_path),
                                pdf_name=pdf_name,
                                chapter=chapter_title,
                                section=section_title,
                                token_count=tokens_len(final),
                                word_count=len(final.split()),
                                has_semantic_header=include_semantic_header,
                                header_style=header_style if include_semantic_header else None,
                                header_tokens=header_cost if include_semantic_header else 0,
                                sha1=h,
                            )
                            if part_idx is not None:
                                meta.update(part=part_idx, parts_total=parts_total,
                                            context_optimized=minimize_context,
                                            has_overlap=True,
                                            overlap_tokens=0,
                                            header_tokens=tokens_len(used_hdr) if include_semantic_header else 0)

                            chunks.append({"id": f"chunk{chunk_id:06d}",
                                           "text": final,
                                           "metadata": meta})
                            chunk_id += 1
                            nonlocal file_chunk_count
                            file_chunk_count += 1
                            if include_semantic_header:
                                stats["headers_added"] += 1
                                stats["header_tokens"] += meta["header_tokens"]

                        if len(body_tokens) <= budget_for_body:
                            make_chunk(section_content)
                            print(f"  ✅ Chunk: {section_title[:50]} (1/1)")
                        else:
                            use_overlap       = should_apply_overlap(len(body_tokens), budget_for_body)
                            eff_overlap       = calculate_adaptive_overlap(
                                len(body_tokens), budget_for_body, overlap_tokens) if use_overlap else 0
                            if eff_overlap < overlap_tokens:
                                stats["overlaps_reduced"] += 1

                            windows     = sliding_windows_tokens(body_tokens, budget_for_body, eff_overlap)
                            parts_total = len(windows)

                            for part_idx, win in enumerate(windows, 1):
                                body_part   = from_tokens(win)
                                part_info   = f"Part {part_idx}/{parts_total}" if parts_total > 1 else ""
                                part_header = create_semantic_header(
                                    pdf_name, chapter_title, section_title, part_info,
                                    header_style, enable=include_semantic_header,
                                )
                                make_chunk(body_part, part_idx, parts_total, part_header)

                            print(f"  ✅ {parts_total} chunks: {section_title[:50]}"
                                  f" (overlap={eff_overlap})"
                                  f"{' +headers' if include_semantic_header else ''}")

            print(f"  📊 {file_chunk_count} chunks from this file\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")

    # Summary
    total_tokens = sum(c["metadata"]["token_count"] for c in chunks) or 1
    print("\n📊 Optimisation stats:")
    if stats["headers_added"]:
        avg = stats["header_tokens"] / stats["headers_added"]
        pct = stats["header_tokens"] / total_tokens * 100
        print(f"  - Semantic headers: {stats['headers_added']} (avg {avg:.1f} tok, {pct:.1f}% of total)")
    if stats["tokens_saved"]:
        print(f"  - Tokens saved via context minimisation: {stats['tokens_saved']}")
    print(f"\n📈 Final: {len(chunks)} chunks, {total_tokens} tokens total")
    return chunks


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    data_paths = [
        os.path.join(config.DATA_JSON_DIR, f)
        for f in os.listdir(config.DATA_JSON_DIR)
        if f.endswith(".json")
    ]
    print(f"Found {len(data_paths)} JSON files in {config.DATA_JSON_DIR}\n")

    chunks = chunking_hierarchical(
        data_paths,
        max_tokens=config.CHUNK_MAX_TOKENS,
        overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
    )

    with open(config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(chunks)} chunks saved → {config.CHUNKS_JSON}")


if __name__ == "__main__":
    main()
