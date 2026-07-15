import logging
from typing import List, Optional

from backend.llm.openai_client import embeddings
from backend.database.postgres import fetch_all, execute

logger = logging.getLogger(__name__)


async def semantic_search(
    query: str, top_k: int = 5, doc_type: Optional[str] = None
) -> List[dict]:
    """Embed query and retrieve top-k documents by cosine similarity via pgvector."""
    logger.info("Semantic search: query=%.80s... top_k=%d type=%s", query, top_k, doc_type or "all")

    execute('SET ivfflat.probes = 5;')
    query_embedding = (await embeddings([query]))[0]
    vector_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if doc_type:
        rows = fetch_all(
            """
            SELECT type, name, path, summary, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_documents
            WHERE type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_str, doc_type, vector_str, top_k),
        )
    else:
        rows = fetch_all(
            """
            SELECT type, name, path, summary, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_str, vector_str, top_k),
        )

    results = [dict(row) for row in rows]
    logger.info("Semantic search returned %d results", len(results))
    return results
