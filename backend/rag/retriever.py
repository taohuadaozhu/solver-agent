"""Chunk-level retrieval with context assembly.

All search functions query rag_chunks (not rag_documents).  Each result
includes document-level metadata (name, type, tags, problem_types) so
downstream callers work without change.

Context assembly (small-to-big): after finding matching chunks, optionally
fetch adjacent sibling chunks from the same document and merge them into
a richer context window for the LLM.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional

from backend.llm.openai_client import embeddings, chat_completion
from backend.database.postgres import fetch_all, fetch_one, execute

logger = logging.getLogger(__name__)

RERANK_PROMPT = (
    "You are a search relevance evaluator. Given a user query and a list of documents, "
    "rank each document by its relevance to the query on a scale of 0-10, where 0 means "
    "completely irrelevant and 10 means perfectly relevant.\n\n"
    "Output a JSON array of objects with fields: index (0-based position in input list), "
    "score (0-10), reason (one short sentence in the query's language).\n"
    "Only include documents that are actually relevant (score >= 3). "
    "Sort by score descending.\n\n"
    "User query: {query}\n\nDocuments:\n"
)

# Columns returned from chunk queries — includes document-level fields via JOIN
_CHUNK_SELECT = """
SELECT
    c.id, c.document_id, c.chunk_index, c.content AS chunk_content,
    c.token_count, c.parent_header, c.header_path, c.metadata AS chunk_metadata,
    d.source_type AS type, d.name, d.path, d.summary, d.tags, d.problem_types
FROM rag_chunks c
JOIN rag_documents d ON d.id = c.document_id
"""


def _row_to_result(row: dict) -> dict:
    """Normalize a chunk row into a result dict (backward-compatible shape)."""
    return {
        "name": row.get("name", ""),
        "type": row.get("type", ""),
        "path": row.get("path", ""),
        "summary": row.get("summary", ""),
        "tags": row.get("tags") or [],
        "problem_types": row.get("problem_types") or [],
        # Chunk-level
        "chunk_id": row.get("id"),
        "chunk_index": row.get("chunk_index"),
        "chunk_content": row.get("chunk_content", ""),
        "token_count": row.get("token_count"),
        "parent_header": row.get("parent_header", ""),
        "header_path": row.get("header_path") or [],
        "similarity": 0.0,
        "score": 0.0,
    }


# ── Single-chunk retrieval ──────────────────────────────────────────────────

async def semantic_search(
    query: str, top_k: int = 5, doc_type: Optional[str] = None
) -> List[dict]:
    """Embed query → cosine similarity on rag_chunks.embedding via IVFFlat."""
    logger.info("Semantic search (chunks): query=%.80s top_k=%d type=%s",
                query, top_k, doc_type or "all")

    execute("SET ivfflat.probes = 5;")
    q_emb = (await embeddings([query]))[0]
    vec_str = "[" + ",".join(str(v) for v in q_emb) + "]"

    if doc_type:
        rows = fetch_all(
            f"""{_CHUNK_SELECT}
               WHERE d.source_type = %s
               ORDER BY c.embedding <=> %s::vector
               LIMIT %s""",
            (doc_type, vec_str, top_k),
        )
    else:
        rows = fetch_all(
            f"""{_CHUNK_SELECT}
               ORDER BY c.embedding <=> %s::vector
               LIMIT %s""",
            (vec_str, top_k),
        )

    results = [_row_to_result(dict(r)) for r in rows]
    for r in results:
        r["similarity"] = round(1.0 - float(r.get("_distance", 0)), 4)  # approximate
    logger.info("Semantic search returned %d chunks", len(results))
    return results


async def keyword_search(
    query: str, top_k: int = 10, doc_type: Optional[str] = None
) -> List[dict]:
    """PostgreSQL full-text search on rag_chunks.search_vector via ts_rank."""
    logger.info("Keyword search (chunks): query=%.80s top_k=%d type=%s",
                query, top_k, doc_type or "all")

    if doc_type:
        rows = fetch_all(
            f"""{_CHUNK_SELECT},
                   ts_rank(c.search_vector, plainto_tsquery('english', %s)) AS score
               WHERE c.search_vector @@ plainto_tsquery('english', %s)
                 AND d.source_type = %s
               ORDER BY score DESC
               LIMIT %s""",
            (query, query, doc_type, top_k),
        )
    else:
        rows = fetch_all(
            f"""{_CHUNK_SELECT},
                   ts_rank(c.search_vector, plainto_tsquery('english', %s)) AS score
               WHERE c.search_vector @@ plainto_tsquery('english', %s)
               ORDER BY score DESC
               LIMIT %s""",
            (query, query, top_k),
        )

    results = [_row_to_result(dict(r)) for r in rows]
    for r, row in zip(results, rows):
        r["score"] = round(float(dict(row).get("score", 0)), 4)
    logger.info("Keyword search returned %d chunks", len(results))
    return results


# ── Hybrid + Rerank ─────────────────────────────────────────────────────────

async def hybrid_search(
    query: str, top_k: int = 5, doc_type: Optional[str] = None, rrf_k: float = 60.0
) -> List[dict]:
    """Dense + sparse chunk retrieval with Reciprocal Rank Fusion."""
    candidate_count = max(top_k * 2, 10)

    dense, sparse = await asyncio.gather(
        semantic_search(query, candidate_count, doc_type),
        keyword_search(query, candidate_count, doc_type),
    )

    rrf_scores: dict = {}
    all_chunks: dict = {}

    for rank, doc in enumerate(dense):
        key = (doc["path"], doc["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        all_chunks[key] = doc

    for rank, doc in enumerate(sparse):
        key = (doc["path"], doc["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        all_chunks[key] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = [all_chunks[key] for key, _ in merged[:top_k]]
    for r, (_, score) in zip(results, merged[:top_k]):
        r["rrf_score"] = round(score, 6)

    logger.info("Hybrid: %d dense + %d sparse → %d chunks", len(dense), len(sparse), len(results))
    return results


def _parse_rerank_json(raw: str) -> List[dict]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


async def rerank_candidates(
    query: str, candidates: List[dict], top_k: int = 5
) -> List[dict]:
    """LLM listwise rerank using chunk_content + parent_header."""
    if len(candidates) <= top_k:
        return candidates

    docs_text = []
    for i, doc in enumerate(candidates):
        header = doc.get("parent_header", "")
        content = (doc.get("chunk_content") or "")[:400]
        docs_text.append(f"[{i}] {doc['name']} | {header}\n    {content}")

    prompt = RERANK_PROMPT.format(query=query) + "\n".join(docs_text)

    try:
        response = await chat_completion(
            prompt=prompt,
            system="You are a precise search relevance evaluator. Be strict and objective.",
            temperature=0.1,
            max_tokens=500,
        )
    except Exception:
        logger.exception("Rerank LLM call failed")
        return candidates[:top_k]

    ranked = _parse_rerank_json(response)
    if not ranked:
        return candidates[:top_k]

    reordered = []
    for item in ranked:
        idx = item.get("index", -1)
        if 0 <= idx < len(candidates):
            doc = candidates[idx].copy()
            doc["rerank_score"] = item.get("score")
            doc["rerank_reason"] = item.get("reason", "")
            reordered.append(doc)

    logger.info("Rerank: %d candidates → %d ranked → top %d",
                len(candidates), len(reordered), min(top_k, len(reordered)))
    return reordered[:top_k]


async def hybrid_search_with_rerank(
    query: str, top_k: int = 5, doc_type: Optional[str] = None
) -> List[dict]:
    """Full pipeline: hybrid RRF → chunk candidates → LLM rerank → top_k."""
    candidates = await hybrid_search(query, top_k=top_k * 3, doc_type=doc_type)
    if len(candidates) <= top_k:
        return candidates
    return await rerank_candidates(query, candidates, top_k)


# ── Context assembly (small-to-big) ─────────────────────────────────────────

async def assemble_context(
    results: List[dict],
    sibling_window: int = 1,
    deduplicate_by: str = "document_id",
) -> List[dict]:
    """Enrich matched chunks with adjacent siblings from the same document.

    For each result, fetch sibling_window chunks before and after.
    Merge contiguous siblings into a single context block per document.
    """
    if not results:
        return results

    # Build (doc_id, set of chunk_indices to fetch)
    doc_chunks: dict = {}
    for r in results:
        doc_id = r.get("document_id")
        if not doc_id:
            continue
        ci = r.get("chunk_index", 0)
        indices = set()
        for offset in range(-sibling_window, sibling_window + 1):
            indices.add(ci + offset)
        if doc_id in doc_chunks:
            doc_chunks[doc_id] |= indices
        else:
            doc_chunks[doc_id] = indices

    # Fetch sibling chunks
    siblings: dict = {}  # (doc_id, chunk_index) → row
    for doc_id, indices in doc_chunks.items():
        if not indices:
            continue
        idx_list = sorted(indices)
        rows = fetch_all(
            """SELECT document_id, chunk_index, content, parent_header, header_path
               FROM rag_chunks
               WHERE document_id = %s AND chunk_index = ANY(%s)
               ORDER BY chunk_index""",
            (doc_id, idx_list),
        )
        for row in rows:
            r = dict(row)
            siblings[(doc_id, r["chunk_index"])] = r

    # Assemble: for each result, merge siblings into a context string
    enriched = []
    for r in results:
        doc_id = r.get("document_id")
        ci = r.get("chunk_index", 0)
        doc = dict(r)
        parts = []
        for offset in range(-sibling_window, sibling_window + 1):
            sib = siblings.get((doc_id, ci + offset))
            if sib:
                header = sib.get("parent_header", "")
                parts.append(f"[{header}]\n{sib['content']}")
        if parts:
            doc["context"] = "\n\n".join(parts)
        enriched.append(doc)

    return enriched
