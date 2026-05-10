"""
pdf_to_json.py – PDF → JSON corpus builder for RAG and HRAG ingestion
======================================================================

Converts one or more Cisco IOS configuration guide PDFs into JSON corpora
ready to be consumed by hrag 

Design principle
----------------
All structural decisions are driven by FONT  Metadata

    Font                        Size(s)   Role
    ─────────────────────────── ────────  ──────────────────────────────────────
    CourierNew / CourierNewBold  8 pt     CLI code (commands + show output)
    Univers-CondensedBold        ≥17 pt   Chapter / section heading
    Univers-CondensedBold       10-14 pt  Sub-heading, table label (DETAILED STEPS)
    Times-Bold                  10 pt     Step label + command syntax in step rows
    TimesNewRoman               10 pt     Prose body, Purpose column text
    Univers-CondensedBold        8 pt     Running header / footer → SKIP
    Univers-Bold / Univers       any      Variant body / step numbering

This font-driven approach replaces fragile keyword and indent-based heuristics.
It generalises to any Cisco IOS XE configuration guide produced by DITA Open
Toolkit / XEP (the same toolchain used across the full Cisco documentation set).

Two output modes
----------------
  --mode flat          → data.json         (dict[pdf_name → list[str]])
  --mode hierarchical  → hierarchical_data.json (list of chunk dicts)
  --mode both          → both files (default)

Chunking strategies (--strategy)
---------------------------------
  paragraph      Paragraph-boundary splits (intern baseline, no font metadata)
  fixed          Fixed token-window with overlap (classic RAG baseline)
  semantic       Sentence-aware recursive splits targeting a token budget
  section        Font-driven chapter/section detection + prose sub-chunking
  block_preserve Font-driven section detection + atomic CLI/table protection
                 ← recommended: never splits mid-command or mid-step table

Research levers (all exposed as CLI flags for ablation studies)
---------------------------------------------------------------
  --chunk-size          Target prose chunk size in tokens     [default: 256]
  --chunk-overlap       Overlap between prose chunks          [default: 32]
  --min-chunk-tokens    Drop chunks shorter than this         [default: 20]
  --max-chunk-tokens    Hard ceiling; oversized CLI kept whole[default: 512]
  --header-ratio        Page-height fraction to skip at top   [default: 0.08]
  --footer-ratio        Page-height fraction to skip at bottom[default: 0.08]
  --deduplicate         Remove exact-match duplicate chunks
  --add-semantic-header Prepend "chapter > section" breadcrumb
  --oversized-action    keep | warn | drop  (block_preserve only)
  --strategy            paragraph|fixed|semantic|section|block_preserve
  --mode                flat|hierarchical|both

Usage examples
--------------
  # Recommended: font-driven, block-preserving, both output formats
  python pdf_to_json.py --input ./pdf/ --strategy block_preserve --deduplicate --verbose

  # Ablation sweep over chunk sizes
  for size in 64 128 256 512; do
    python pdf_to_json.py --input ./pdf/ --strategy block_preserve \\
        --chunk-size $size --hier-out hier_${size}.json --mode hierarchical
  done

  # Baseline
  python pdf_to_json.py --input ./pdf/ --strategy paragraph --mode flat

Dependencies
------------
  pip install pdfplumber langchain-text-splitters tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# pdfplumber – primary extraction engine (font metadata per word)
# ---------------------------------------------------------------------------
try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    _PDFPLUMBER_OK = False

# PyMuPDF – fallback for strategies that don't need font metadata
try:
    import fitz
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

# langchain splitters – needed for fixed / semantic / section / block_preserve
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_OK = True
except ImportError:
    _LANGCHAIN_OK = False


def _require(lib: str, install: str) -> None:
    sys.exit(f"{lib} is required: pip install {install}")

def _need_pdfplumber() -> None:
    if not _PDFPLUMBER_OK: _require("pdfplumber", "pdfplumber")

def _need_fitz() -> None:
    if not _FITZ_OK: _require("PyMuPDF", "pymupdf")

def _need_langchain() -> None:
    if not _LANGCHAIN_OK: _require("langchain-text-splitters", "langchain-text-splitters")


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class RawPage:
    page_num: int   # 1-based
    text: str
    pdf_name: str


@dataclass
class FontLine:
    """One logical line with its dominant font role, extracted via pdfplumber."""
    text: str
    role: str       # _FontRole constant
    page_num: int   # 1-based
    x0: float       # left edge of first word (indentation signal)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Font-role taxonomy  (grounded in real PDF inspection)
# ===========================================================================

class _FontRole:
    """
    Font roles inferred from the Cisco IOS XE documentation font taxonomy.

    Determination logic per logical line (words at the same y-position):

    1. ALL words in Courier* font                  → CLI
       (monospace = verbatim code, whether regular or bold)

    2. Dominant font is Univers-CondensedBold:
         size ≤ 9 pt                               → SKIP (running header/footer)
         size ≥ 17 pt                              → HEADING (chapter/section title)
         10 ≤ size ≤ 14 pt                         → LABEL (DETAILED STEPS, table headers)

    3. Dominant font is Times-Bold at 10 pt        → STEP
       (step-row command syntax and "Step N" labels in DETAILED STEPS tables)

    4. Everything else (TimesNewRoman, mixed)       → PROSE
    """
    CLI     = "cli"      # Courier → verbatim command / show output
    HEADING = "heading"  # Univers-CondensedBold ≥17 pt → chapter / section title
    LABEL   = "label"    # Univers-CondensedBold 10–14 pt → DETAILED STEPS, table label
    STEP    = "step"     # Times-Bold 10 pt → step syntax / "Step N"
    PROSE   = "prose"    # TimesNewRoman 10 pt → body text / Purpose column
    SKIP    = "skip"     # Univers-CondensedBold 8 pt → running header / footer


def _classify_font_line(words: list[dict]) -> str:
    """Return a _FontRole for a list of pdfplumber word dicts (fontname, size)."""
    if not words:
        return _FontRole.SKIP

    # 1. Courier check: all words in a monospace font → CLI
    courier_n = sum(1 for w in words if "courier" in w["fontname"].lower())
    if courier_n == len(words):
        return _FontRole.CLI

    # 2. Dominant font by word count
    dominant_font = Counter(w["fontname"] for w in words).most_common(1)[0][0]
    fn = dominant_font.lower()

    # Median size across the line
    sizes = sorted(w["size"] for w in words)
    size  = sizes[len(sizes) // 2]

    # Univers-CondensedBold (the structural font in Cisco DITA guides)
    is_ucb = (
        ("univers" in fn and "condensed" in fn and "bold" in fn)
        or "univers-condensedbold" in fn
    )
    if is_ucb:
        if size <= 9:   return _FontRole.SKIP
        if size >= 17:  return _FontRole.HEADING
        return _FontRole.LABEL   # 10–14 pt

    # Times-Bold (step-row label and command syntax)
    is_tb = "times-bold" in fn or "timesnewromanps-boldmt" in fn or fn == "times-bold"
    if is_tb:
        return _FontRole.STEP

    # Mixed line with some Courier words → treat as PROSE (inline code in prose)
    return _FontRole.PROSE


# ===========================================================================
# Font-aware page extraction (pdfplumber)
# ===========================================================================

def extract_font_lines(
    pdf_path: str,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
) -> list[FontLine]:
    """
    Extract all body lines as FontLine objects using pdfplumber.

    Words are grouped into logical lines by y-position (2 pt tolerance) and
    sorted left-to-right.  The SKIP font role provides a second deduplication
    pass on top of the geometric header/footer bands.
    """
    _need_pdfplumber()
    pdf_name = Path(pdf_path).stem
    all_lines: list[FontLine] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            h_band = page.height * header_ratio
            f_band = page.height * (1 - footer_ratio)

            words = page.extract_words(
                extra_attrs=["fontname", "size"],
                keep_blank_chars=False,
                x_tolerance=3,
                y_tolerance=3,
            )

            # Bucket words by y-position into logical lines (2 pt buckets)
            buckets: dict[int, list[dict]] = {}
            for w in words:
                if w["top"] < h_band or w["bottom"] > f_band:
                    continue
                bucket = round(w["top"] / 2) * 2
                buckets.setdefault(bucket, []).append(w)

            for bucket_y in sorted(buckets):
                line_words = sorted(buckets[bucket_y], key=lambda w: w["x0"])
                role = _classify_font_line(line_words)
                if role == _FontRole.SKIP:
                    continue
                text = " ".join(w["text"] for w in line_words).strip()
                if not text:
                    continue
                all_lines.append(FontLine(
                    text=text,
                    role=role,
                    page_num=page.page_number,
                    x0=line_words[0]["x0"],
                ))

    return all_lines


# ===========================================================================
# Plain text extraction (for paragraph / fixed / semantic strategies)
# ===========================================================================

def extract_pages_plain(
    pdf_path: str,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
) -> list[RawPage]:
    """Extract plain text per page using pdfplumber (or fitz as fallback)."""
    pdf_name = Path(pdf_path).stem

    if _PDFPLUMBER_OK:
        pages: list[RawPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                h_cut = page.height * header_ratio
                f_cut = page.height * (1 - footer_ratio)
                cropped = page.within_bbox((0, h_cut, page.width, f_cut))
                text = cropped.extract_text() or ""
                if text.strip():
                    pages.append(RawPage(
                        page_num=page.page_number, text=text, pdf_name=pdf_name
                    ))
        return pages

    _need_fitz()
    pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        ph = page.rect.height
        blocks = page.get_text("blocks", sort=True)
        body = "\n".join(
            b[4].strip() for b in blocks
            if b[1] >= ph * header_ratio and b[3] <= ph * (1 - footer_ratio) and b[4].strip()
        )
        if body:
            pages.append(RawPage(page_num=page.number + 1, text=body, pdf_name=pdf_name))
    doc.close()
    return pages


# ===========================================================================
# Utilities
# ===========================================================================

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[\.\-\s]{5,}$", "", text, flags=re.MULTILINE)  # TOC leaders
    return text.strip()


def is_noise(text: str) -> bool:
    """True for TOC entries, boilerplate, or single-word fragments."""
    s = text.strip()
    if not s or len(s.split()) < 3:
        return True
    if re.search(r"[\s\.]{3,}\d+\s*$", s):   # "Some Title . . . 12"
        return True
    if re.fullmatch(r"\d+", s):
        return True
    boilerplate = (
        "Americas Headquarters", "Cisco Systems, Inc.",
        "THE SPECIFICATIONS", "THE SOFTWARE LICENSE",
        "NOTWITHSTANDING ANY OTHER", "IN NO EVENT SHALL CISCO",
        "© 20", "All rights reserved",
    )
    return any(s.startswith(b) for b in boilerplate)


def approx_tokens(text: str) -> int:
    """BPE approximation: words × 4/3."""
    return len(text.split()) * 4 // 3


def _fingerprint(text: str) -> str:
    n = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(n.encode()).hexdigest()


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out:  list[Chunk] = []
    for ch in chunks:
        fp = _fingerprint(ch.text)
        if fp not in seen:
            seen.add(fp)
            out.append(ch)
    return out


class _ChunkCounter:
    _n = 0

    @classmethod
    def next(cls, pdf_name: str) -> str:
        cls._n += 1
        safe = re.sub(r"[^a-z0-9]", "_", pdf_name.lower())
        return f"{safe}_chunk{cls._n:06d}"


def _make_chunk(
    text: str,
    pdf_name: str,
    chapter: str = "",
    section: str = "",
    page_num: int = 0,
    add_semantic_header: bool = False,
    strategy: str = "",
    chunk_index: int = 0,
    **extra_meta,
) -> Chunk:
    if add_semantic_header and (chapter or section):
        breadcrumb = " > ".join(p for p in [chapter, section] if p)
        text = breadcrumb + "\n" + text
    return Chunk(
        chunk_id=_ChunkCounter.next(pdf_name),
        text=text.strip(),
        metadata={
            "pdf_name":    pdf_name,
            "source_file": f"{pdf_name}.json",
            "chapter":     chapter,
            "section":     section,
            "page_num":    page_num,
            "token_count": approx_tokens(text),
            "strategy":    strategy,
            "chunk_index": chunk_index,
            **extra_meta,
        },
    )


# ===========================================================================
# Strategy 1 – paragraph  (baseline)
# ===========================================================================

def chunk_paragraph(
    pages: list[RawPage],
    min_tokens: int = 20,
    max_tokens: int = 512,
    add_semantic_header: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for idx, para in enumerate(page.text.split("\n\n")):
            para = clean_text(para)
            if is_noise(para) or approx_tokens(para) < min_tokens:
                continue
            if approx_tokens(para) > max_tokens:
                para = " ".join(para.split()[:max_tokens * 3 // 4])
            chunks.append(_make_chunk(para, page.pdf_name,
                                      page_num=page.page_num,
                                      add_semantic_header=add_semantic_header,
                                      strategy="paragraph", chunk_index=idx))
    return chunks


# ===========================================================================
# Strategy 2 – fixed (token-window with overlap)
# ===========================================================================

def chunk_fixed(
    pages: list[RawPage],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    min_tokens: int = 20,
    add_semantic_header: bool = False,
) -> list[Chunk]:
    _need_langchain()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        body = clean_text(page.text)
        if not body:
            continue
        for idx, split in enumerate(splitter.split_text(body)):
            if is_noise(split) or approx_tokens(split) < min_tokens:
                continue
            chunks.append(_make_chunk(split, page.pdf_name,
                                      page_num=page.page_num,
                                      add_semantic_header=add_semantic_header,
                                      strategy="fixed", chunk_index=idx))
    return chunks


# ===========================================================================
# Strategy 3 – semantic  (sentence-aware recursive split)
# ===========================================================================

def chunk_semantic(
    pages: list[RawPage],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    min_tokens: int = 20,
    add_semantic_header: bool = False,
) -> list[Chunk]:
    _need_langchain()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    for page in pages:
        body = clean_text(page.text)
        if not body:
            continue
        for idx, split in enumerate(splitter.split_text(body)):
            if is_noise(split) or approx_tokens(split) < min_tokens:
                continue
            chunks.append(_make_chunk(split, page.pdf_name,
                                      page_num=page.page_num,
                                      add_semantic_header=add_semantic_header,
                                      strategy="semantic", chunk_index=idx))
    return chunks


# ===========================================================================
# Shared: font-driven structure parser  (used by section + block_preserve)
# ===========================================================================

@dataclass
class _Section:
    chapter:    str
    section:    str
    page_num:   int
    font_lines: list[FontLine] = field(default_factory=list)


def _parse_font_structure(font_lines: list[FontLine]) -> list[_Section]:
    """
    Walk FontLine objects and emit _Section objects at each HEADING boundary.

    Chapter vs section distinction:
      • "CHAPTER" in the heading text → new chapter (resets section to "Overview")
      • Any other HEADING            → new section within current chapter
    """
    sections: list[_Section] = []
    current_chapter = "Preamble"
    current_section = "Overview"
    current_page    = 1
    buffer: list[FontLine] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            sections.append(_Section(
                chapter=current_chapter,
                section=current_section,
                page_num=current_page,
                font_lines=list(buffer),
            ))
            buffer = []

    for fl in font_lines:
        if fl.role == _FontRole.HEADING:
            flush()
            # Detect chapter-level headings by the "CHAPTER" keyword
            if re.search(r"\bchapter\b", fl.text, re.IGNORECASE):
                current_chapter = fl.text
                current_section = "Overview"
            else:
                current_section = fl.text
            current_page = fl.page_num
        else:
            buffer.append(fl)

    flush()
    return sections


# ===========================================================================
# Strategy 4 – section  (font-driven heading detection + prose sub-chunking)
# ===========================================================================

def chunk_section(
    pdf_path: str,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    min_tokens: int = 20,
    max_tokens: int = 512,
    add_semantic_header: bool = True,
) -> list[Chunk]:
    """
    Font-driven section chunking without atomic-block protection.
    Detects chapter/section boundaries accurately but may split CLI blocks.
    Use 'block_preserve' for full integrity.
    """
    _need_langchain()
    pdf_name   = Path(pdf_path).stem
    font_lines = extract_font_lines(pdf_path, header_ratio, footer_ratio)
    sections   = _parse_font_structure(font_lines)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for sec in sections:
        body = clean_text("\n".join(fl.text for fl in sec.font_lines))
        if not body or is_noise(body):
            continue
        splits = [body] if approx_tokens(body) <= chunk_size else splitter.split_text(body)
        for idx, split in enumerate(splits):
            if is_noise(split) or approx_tokens(split) < min_tokens:
                continue
            if approx_tokens(split) > max_tokens:
                split = " ".join(split.split()[:max_tokens * 3 // 4])
            chunks.append(_make_chunk(split, pdf_name,
                                      chapter=sec.chapter, section=sec.section,
                                      page_num=sec.page_num,
                                      add_semantic_header=add_semantic_header,
                                      strategy="section", chunk_index=idx))
    return chunks


# ===========================================================================
# Strategy 5 – block_preserve  (font-driven + atomic CLI/table protection)
# ===========================================================================
#
# After _parse_font_structure() partitions lines into sections, each section's
# FontLines are segmented into typed *spans* using font role:
#
#   SPAN KIND    SOURCE ROLES          NEVER SPLIT?   WHAT IT CAPTURES
#   ──────────   ─────────────────     ────────────   ──────────────────────────────
#   cli_code     CLI                   YES            Config examples, show output
#   step_table   STEP, LABEL, CLI*     YES            DETAILED/SUMMARY STEPS tables
#                (*CLI within a step   block is       (command syntax, Example:,
#                absorbed into it)     kept whole)    Device# cmd — all one unit)
#   prose        PROSE                 NO             Body text, split by token budget
#
# The CLI-into-step absorption rule is critical: in DETAILED STEPS tables the
# "Example:\n  Device# cmd" lines carry the LABEL and CLI roles respectively,
# but they belong to the same step row and must not be separated.
# ===========================================================================

class _SpanKind:
    CLI_CODE   = "cli_code"    # standalone config example / show output
    STEP_TABLE = "step_table"  # DETAILED/SUMMARY STEPS table row (+ its Example CLI)
    PROSE      = "prose"

    ATOMIC: frozenset = frozenset()


_SpanKind.ATOMIC = frozenset({_SpanKind.CLI_CODE, _SpanKind.STEP_TABLE})


@dataclass
class _Span:
    kind:  str
    lines: list[FontLine] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(fl.text for fl in self.lines)


def _role_to_span_kind(role: str, prev_kind: str) -> str:
    """
    Map a FontLine role to a _SpanKind with step-continuation logic.

    A CLI line that immediately follows a STEP_TABLE span is absorbed into
    it (covers the "Example: / Device# cmd" sub-pattern in DETAILED STEPS).
    A standalone CLI line (outside a step block) becomes CLI_CODE.
    """
    if role == _FontRole.CLI:
        return _SpanKind.STEP_TABLE if prev_kind == _SpanKind.STEP_TABLE else _SpanKind.CLI_CODE
    if role in (_FontRole.STEP, _FontRole.LABEL):
        return _SpanKind.STEP_TABLE
    return _SpanKind.PROSE


def _segment_section(font_lines: list[FontLine]) -> list[_Span]:
    """
    Segment a section's FontLines into typed _Span objects.
    Consecutive lines of the same kind are merged into one span.
    """
    spans: list[_Span] = []
    prev_kind = _SpanKind.PROSE

    for fl in font_lines:
        kind = _role_to_span_kind(fl.role, prev_kind)
        if spans and spans[-1].kind == kind:
            spans[-1].lines.append(fl)
        else:
            spans.append(_Span(kind=kind, lines=[fl]))
        prev_kind = kind

    # Strip leading/trailing blank lines within each span
    for span in spans:
        while span.lines and not span.lines[0].text.strip():
            span.lines.pop(0)
        while span.lines and not span.lines[-1].text.strip():
            span.lines.pop()

    return [s for s in spans if s.lines]


def chunk_block_preserve(
    pdf_path: str,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    min_tokens: int = 20,
    max_tokens: int = 512,
    add_semantic_header: bool = True,
    oversized_action: str = "keep",
) -> list[Chunk]:
    """
    Font-driven, block-preserving chunking.  Recommended for Cisco IOS docs.

    CLI code blocks and DETAILED/SUMMARY STEPS tables are emitted as
    indivisible atomic chunks — they are never split mid-command or mid-row.
    Only prose paragraphs are split by the token budget splitter.

    Extra metadata per chunk:
        block_type : "cli_code" | "step_table" | "prose"
        oversized  : True if atomic block exceeds max_tokens (kept whole anyway)

    The block_type field enables targeted retrieval in hrag.py:
    e.g. filtering to block_type="step_table" gives only structured procedure
    content, useful as an ablation axis in retrieval experiments.
    """
    _need_langchain()
    pdf_name   = Path(pdf_path).stem
    font_lines = extract_font_lines(pdf_path, header_ratio, footer_ratio)
    sections   = _parse_font_structure(font_lines)

    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []

    for sec in sections:
        spans = _segment_section(sec.font_lines)
        idx   = 0

        for span in spans:
            raw = clean_text(span.text())
            if not raw or is_noise(raw):
                continue
            tok = approx_tokens(raw)

            if span.kind in _SpanKind.ATOMIC:
                # ── Atomic: emit whole, never split ──────────────────────
                if tok < min_tokens:
                    continue
                oversized = tok > max_tokens
                if oversized:
                    if oversized_action == "drop":
                        continue
                    if oversized_action == "warn":
                        print(
                            f"  [block_preserve] WARN oversized {span.kind} "
                            f"({tok} tok > {max_tokens}) in section "
                            f"'{sec.section[:60]}' — kept whole.",
                            file=sys.stderr,
                        )
                chunks.append(_make_chunk(
                    raw, pdf_name,
                    chapter=sec.chapter, section=sec.section,
                    page_num=sec.page_num,
                    add_semantic_header=add_semantic_header,
                    strategy="block_preserve", chunk_index=idx,
                    block_type=span.kind, oversized=oversized,
                ))
                idx += 1

            else:
                # ── Prose: split by token budget ──────────────────────────
                splits = [raw] if tok <= chunk_size else prose_splitter.split_text(raw)
                for split in splits:
                    if is_noise(split) or approx_tokens(split) < min_tokens:
                        continue
                    if approx_tokens(split) > max_tokens:
                        split = " ".join(split.split()[:max_tokens * 3 // 4])
                    chunks.append(_make_chunk(
                        split, pdf_name,
                        chapter=sec.chapter, section=sec.section,
                        page_num=sec.page_num,
                        add_semantic_header=add_semantic_header,
                        strategy="block_preserve", chunk_index=idx,
                        block_type=_SpanKind.PROSE, oversized=False,
                    ))
                    idx += 1

    return chunks


# ===========================================================================
# Per-PDF dispatch
# ===========================================================================

def process_pdf(
    pdf_path: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    min_tokens: int,
    max_tokens: int,
    header_ratio: float,
    footer_ratio: float,
    add_semantic_header: bool,
    deduplicate: bool,
    oversized_action: str = "keep",
) -> list[Chunk]:
    pdf_name = Path(pdf_path).stem

    font_kw = dict(
        pdf_path=pdf_path,
        header_ratio=header_ratio,
        footer_ratio=footer_ratio,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        add_semantic_header=add_semantic_header,
    )

    if strategy == "section":
        chunks = chunk_section(**font_kw)
    elif strategy == "block_preserve":
        chunks = chunk_block_preserve(**font_kw, oversized_action=oversized_action)
    else:
        pages = extract_pages_plain(pdf_path, header_ratio=header_ratio, footer_ratio=footer_ratio)
        if strategy == "paragraph":
            chunks = chunk_paragraph(pages, min_tokens, max_tokens, add_semantic_header)
        elif strategy == "fixed":
            chunks = chunk_fixed(pages, chunk_size, chunk_overlap, min_tokens, add_semantic_header)
        elif strategy == "semantic":
            chunks = chunk_semantic(pages, chunk_size, chunk_overlap, min_tokens, add_semantic_header)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    if deduplicate:
        before = len(chunks)
        chunks = deduplicate_chunks(chunks)
        removed = before - len(chunks)
        if removed:
            print(f"    [{pdf_name}] dedup: {before} → {len(chunks)} (-{removed})")

    return chunks


# ===========================================================================
# Output builders
# ===========================================================================

def build_flat_json(chunks: list[Chunk]) -> dict:
    """Format for rag.py: {pdf_name.json: [text, ...]}"""
    flat: dict = {}
    for ch in chunks:
        key = ch.metadata.get("source_file", "unknown.json")
        flat.setdefault(key, []).append(ch.text)
    return flat


def build_hierarchical_json(chunks: list[Chunk]) -> list[dict]:
    """Format for hrag.py: [{id, text, metadata}, ...]"""
    return [ch.to_dict() for ch in chunks]


# ===========================================================================
# CLI
# ===========================================================================

def collect_pdfs(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*.pdf"))
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [str(p)]
    sys.exit(f"[ERROR] --input must be a PDF file or directory: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cisco PDF → JSON for RAG / HRAG  (font-driven extraction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────
    parser.add_argument("--input", "-i", required=True,
                        help="PDF file or directory of PDF files")
    parser.add_argument("--flat-out", default="data.json",
                        help="Output path for flat JSON (rag.py)")
    parser.add_argument("--hier-out", default="hierarchical_data.json",
                        help="Output path for hierarchical JSON")
    parser.add_argument("--mode", choices=["flat", "hierarchical", "both"], default="both")

    # ── Chunking strategy ─────────────────────────────────────────────────
    parser.add_argument(
        "--strategy",
        choices=["paragraph", "fixed", "semantic", "section", "block_preserve"],
        default="block_preserve",
        help=(
            "paragraph: blank-line splits (intern baseline, no font metadata); "
            "fixed: token-window + overlap (classic RAG baseline); "
            "semantic: sentence-aware recursive split; "
            "section: font-driven heading detection + prose sub-chunking; "
            "block_preserve: font-driven + atomic CLI/step-table protection [default]"
        ),
    )

    # ── Chunk sizing (key ablation parameters) ────────────────────────────
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help=(
            "Target prose chunk size in tokens. "
            "Ablation range: 64, 128, 256, 512. "
            "CLI/step blocks are kept whole regardless of this value."
        ),
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=32,
        help="Token overlap between consecutive prose chunks. Ablation: 0, 16, 32, 64.",
    )
    parser.add_argument("--min-chunk-tokens", type=int, default=20,
                        help="Discard chunks shorter than this (noise filter).")
    parser.add_argument(
        "--max-chunk-tokens", type=int, default=512,
        help=(
            "Hard ceiling for prose chunks (truncated). "
            "CLI/step blocks that exceed this are flagged 'oversized' but kept whole "
            "unless --oversized-action drop is set."
        ),
    )

    # ── Extraction quality ────────────────────────────────────────────────
    parser.add_argument(
        "--header-ratio", type=float, default=0.08,
        help=(
            "Page-height fraction treated as header zone and skipped. "
            "The SKIP font role provides a second filter. Ablation: 0.05–0.15."
        ),
    )
    parser.add_argument(
        "--footer-ratio", type=float, default=0.08,
        help="Page-height fraction treated as footer zone and skipped. Ablation: 0.05–0.12.",
    )

    # ── Quality flags ─────────────────────────────────────────────────────
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="Remove exact-match duplicate chunks (recommended for multi-PDF corpora).",
    )
    parser.add_argument(
        "--add-semantic-header", dest="add_semantic_header",
        action="store_true", default=True,
        help=(
            "Prepend 'chapter > section' breadcrumb to each chunk text. "
            "Improves embedding quality for hierarchical retrieval. "
            "Disable with --no-add-semantic-header for ablation."
        ),
    )
    parser.add_argument("--no-add-semantic-header",
                        dest="add_semantic_header", action="store_false")

    # ── block_preserve ────────────────────────────────────────────────────
    parser.add_argument(
        "--oversized-action", choices=["keep", "warn", "drop"], default="keep",
        help=(
            "Action for CLI/step blocks exceeding --max-chunk-tokens. "
            "keep: emit whole + flag metadata['oversized']=True (default, safest); "
            "warn: same + print to stderr; "
            "drop: discard (may lose long config examples — ablation use only)."
        ),
    )

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-file statistics.")

    args = parser.parse_args()

    pdf_files = collect_pdfs(args.input)
    if not pdf_files:
        sys.exit("[ERROR] No PDF files found.")

    print(f"[pdf_to_json] PDFs found       : {len(pdf_files)}")
    print(f"[pdf_to_json] Strategy         : {args.strategy}")
    print(f"[pdf_to_json] Chunk size        : {args.chunk_size} tokens  overlap={args.chunk_overlap}")
    print(f"[pdf_to_json] Header/footer     : {args.header_ratio:.0%} / {args.footer_ratio:.0%}")
    print(f"[pdf_to_json] Semantic header   : {args.add_semantic_header}")
    print(f"[pdf_to_json] Deduplicate       : {args.deduplicate}")
    if args.strategy == "block_preserve":
        print(f"[pdf_to_json] Oversized action  : {args.oversized_action}")
    print(f"[pdf_to_json] Output mode       : {args.mode}")
    print()

    all_chunks: list[Chunk] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks = process_pdf(
                pdf_path=pdf_path,
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_tokens=args.min_chunk_tokens,
                max_tokens=args.max_chunk_tokens,
                header_ratio=args.header_ratio,
                footer_ratio=args.footer_ratio,
                add_semantic_header=args.add_semantic_header,
                deduplicate=args.deduplicate,
                oversized_action=args.oversized_action,
            )
            all_chunks.extend(chunks)

            if args.verbose:
                chapters = {ch.metadata.get("chapter") for ch in chunks}
                sections = {ch.metadata.get("section") for ch in chunks}
                avg_tok  = (
                    sum(ch.metadata["token_count"] for ch in chunks) / len(chunks)
                    if chunks else 0
                )
                extra = ""
                if args.strategy == "block_preserve":
                    by_type: dict[str, int] = {}
                    for ch in chunks:
                        bt = ch.metadata.get("block_type", "prose")
                        by_type[bt] = by_type.get(bt, 0) + 1
                    over_n = sum(1 for ch in chunks if ch.metadata.get("oversized"))
                    parts  = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
                    extra  = f" | {parts}, oversized={over_n}"
                print(
                    f"  {Path(pdf_path).name}: "
                    f"{len(chunks)} chunks | {len(chapters)} chap | "
                    f"{len(sections)} sec | avg {avg_tok:.0f} tok{extra}"
                )
        except Exception as exc:
            print(f"  [WARNING] {Path(pdf_path).name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    print(f"\n[pdf_to_json] Total chunks      : {len(all_chunks)}")

    if args.deduplicate:
        before = len(all_chunks)
        all_chunks = deduplicate_chunks(all_chunks)
        print(f"[pdf_to_json] After global dedup : {len(all_chunks)} (-{before - len(all_chunks)})")

    # ── Write outputs ─────────────────────────────────────────────────────
    if args.mode in ("flat", "both"):
        flat = build_flat_json(all_chunks)
        with open(args.flat_out, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)
        total = sum(len(v) for v in flat.values())
        print(f"[pdf_to_json] Flat JSON         → {args.flat_out}  ({total} entries)")

    if args.mode in ("hierarchical", "both"):
        hier = build_hierarchical_json(all_chunks)
        with open(args.hier_out, "w", encoding="utf-8") as f:
            json.dump(hier, f, ensure_ascii=False, indent=2)
        print(f"[pdf_to_json] HRAG JSON         → {args.hier_out}  ({len(hier)} chunks)")

    # ── Summary statistics (for dataset table) ───────────────────
    if all_chunks:
        toks = sorted(ch.metadata["token_count"] for ch in all_chunks)
        n    = len(toks)
        print(
            f"\n[pdf_to_json] Token stats:"
            f"\n  min={toks[0]}  p25={toks[n//4]}  median={toks[n//2]}"
            f"  p75={toks[3*n//4]}  max={toks[-1]}  mean={sum(toks)/n:.1f}"
        )
        if args.strategy in ("section", "block_preserve"):
            chaps = {ch.metadata.get("chapter") for ch in all_chunks}
            sects = {ch.metadata.get("section") for ch in all_chunks}
            print(f"  Unique chapters : {len(chaps)}")
            print(f"  Unique sections : {len(sects)}")
        if args.strategy == "block_preserve":
            by_type: dict[str, int] = {}
            for ch in all_chunks:
                bt = ch.metadata.get("block_type", "prose")
                by_type[bt] = by_type.get(bt, 0) + 1
            over_n = sum(1 for ch in all_chunks if ch.metadata.get("oversized"))
            print("  Block-type breakdown:")
            for bt, cnt in sorted(by_type.items()):
                print(f"    {bt:<16} : {cnt}")
            print(f"    {'oversized (kept)':<16} : {over_n}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
