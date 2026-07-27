"""Production-grade Markdown chunker for optimization knowledge base.

Strategy:
  1. MarkdownHeaderTextSplitter — split on ## / ### headers, keep header path
  2. RecursiveCharacterTextSplitter — fallback for sections exceeding chunk_size
  3. Code-block preservation — ``` fences kept intact during splitting
  4. Metadata inheritance — each chunk carries parent document tags/type/name

Parameters:
  chunk_size: 512 tokens   (text-embedding-3-small optimal range)
  chunk_overlap: 64 tokens (cross-chunk context continuity)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)")
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
TABLE_ROW_RE = re.compile(r"^\|.*\|$")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int
    parent_header: str        # "## Parameters > ### Crossover"
    header_path: List[str]    # ["Parameters", "Crossover"]
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ── Header-level splitting ──────────────────────────────────────────────────

def _parse_markdown_sections(markdown_text: str) -> List[dict]:
    """Split Markdown by ## / ### headers into sections with accumulated header paths.

    A preamble before any header is collected as a section with an empty path.
    """
    lines = markdown_text.split("\n")
    sections: List[dict] = []
    current_lines: List[str] = []
    current_path: List[str] = []

    def _emit():
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({"header_path": list(current_path), "content": text})

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            _emit()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Drill up to the new level then replace at that depth
            current_path = current_path[: level - 1] + [title]
            current_lines = [line]
        else:
            current_lines.append(line)

    _emit()
    return sections


# ── Recursive splitting ─────────────────────────────────────────────────────

def _recursive_split_text(text: str, headers: List[str]) -> List[str]:
    """Split a single section that exceeds chunk_size into sub-chunks.

    Tries separators in order: double-newline → single-newline → sentence-end.
    Code fence blocks are protected before splitting and restored after.
    """
    tokens = count_tokens(text)
    if tokens <= CHUNK_SIZE:
        return [text]

    # Protect code blocks from being torn apart
    code_blocks: List[str] = []

    def _protect(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"\n__CB{len(code_blocks) - 1}__\n"

    protected = CODE_FENCE_RE.sub(_protect, text)

    separators = ["\n\n", "\n", ". ", "。 ", " "]
    for sep in separators:
        if sep in protected:
            pieces = _split_with_separator(protected, sep)
            # Restore code blocks in each piece
            pieces = [_restore_code(p, code_blocks) for p in pieces]
            return _merge_short_pieces(pieces)

    # Last resort — character split with overlap
    return _character_split(text)


def _split_with_separator(text: str, sep: str) -> List[str]:
    """Split text by separator, merging adjacent pieces that are under chunk_size."""
    raw = text.split(sep)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for piece in raw:
        piece_tokens = count_tokens(piece + sep)
        if current_tokens + piece_tokens > CHUNK_SIZE and current:
            chunks.append(sep.join(current))
            # Overlap: keep last ~CHUNK_OVERLAP tokens worth of pieces
            overlap_tokens = 0
            overlap_start = len(current)
            for i in range(len(current) - 1, -1, -1):
                overlap_tokens += count_tokens(current[i] + sep)
                if overlap_tokens >= CHUNK_OVERLAP:
                    overlap_start = i
                    break
            current = current[overlap_start:]
            current_tokens = sum(count_tokens(p + sep) for p in current)

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append(sep.join(current))

    return chunks


def _restore_code(text: str, blocks: List[str]) -> str:
    for i, b in enumerate(blocks):
        text = text.replace(f"__CB{i}__", b)
    return text


def _merge_short_pieces(pieces: List[str]) -> List[str]:
    """Merge undersized tail pieces into the preceding chunk when possible."""
    if len(pieces) <= 1:
        return pieces

    merged = []
    buf = ""
    buf_tokens = 0

    for piece in pieces:
        pt = count_tokens(piece)
        if buf_tokens + pt <= CHUNK_SIZE:
            buf = buf + "\n\n" + piece if buf else piece
            buf_tokens = count_tokens(buf)
        else:
            if buf:
                merged.append(buf)
            buf = piece
            buf_tokens = pt

    if buf:
        if merged and count_tokens(merged[-1]) + buf_tokens <= CHUNK_SIZE:
            merged[-1] = merged[-1] + "\n\n" + buf
        else:
            merged.append(buf)

    return merged


def _character_split(text: str) -> List[str]:
    """Last-resort: split around CHUNK_SIZE characters with overlap."""
    chars_per_chunk = CHUNK_SIZE * 2  # rough char estimate
    overlap_chars = CHUNK_OVERLAP * 2
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        chunks.append(text[start:end])
        start = end - overlap_chars
        if start >= len(text):
            break
    return chunks


# ── Public API ──────────────────────────────────────────────────────────────

def chunk_markdown(
    content: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    doc_metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Main entry: chunk a Markdown README into embedding-ready pieces.

    Returns a flat list of Chunk objects ordered by their position in the document.
    """
    global CHUNK_SIZE, CHUNK_OVERLAP
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    sections = _parse_markdown_sections(content)
    chunks: List[Chunk] = []
    chunk_idx = 0

    for section in sections:
        header_path = section["header_path"]
        header_str = " > ".join(header_path) if header_path else "(preamble)"

        tokens = count_tokens(section["content"])
        if tokens <= chunk_size:
            chunks.append(Chunk(
                content=section["content"],
                chunk_index=chunk_idx,
                token_count=tokens,
                parent_header=header_str,
                header_path=header_path,
                metadata=dict(doc_metadata or {}),
            ))
            chunk_idx += 1
        else:
            sub_texts = _recursive_split_text(section["content"], header_path)
            for sub in sub_texts:
                chunks.append(Chunk(
                    content=sub,
                    chunk_index=chunk_idx,
                    token_count=count_tokens(sub),
                    parent_header=header_str,
                    header_path=header_path,
                    metadata=dict(doc_metadata or {}),
                ))
                chunk_idx += 1

    return chunks


def chunk_markdown_file(
    file_path: str,
    doc_metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Convenience: read a Markdown file and chunk it."""
    content = open(file_path, encoding="utf-8", errors="ignore").read()
    return chunk_markdown(content, doc_metadata=doc_metadata)
