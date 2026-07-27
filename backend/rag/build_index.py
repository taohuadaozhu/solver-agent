"""Production-grade knowledge base indexing with Markdown chunking.

Two-level schema:
  rag_documents    — document metadata (one row per README, no embedding)
  rag_chunks       — retrieval units (one row per chunk, with embedding + tsvector)

Supports incremental updates via SHA256 document hashes.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.llm.openai_client import embeddings
from backend.database.postgres import execute, fetch_all, fetch_one
from backend.rag.chunker import chunk_markdown, count_tokens

DEFAULT_TYPES = ["algorithms", "projects", "datasets"]


def migrate_schema():
    """One-time migration: drop old single-table schema, create chunked schema.

    The old rag_documents had embedding + search_vector columns inline.
    The new schema splits into rag_documents (metadata) + rag_chunks (embeddings).
    """
    execute("DROP TABLE IF EXISTS rag_chunks CASCADE;")
    execute("DROP TABLE IF EXISTS rag_documents CASCADE;")
    create_tables()
    print("Schema migrated: rag_documents + rag_chunks created.")


# ── Table creation ───────────────────────────────────────────────────────────

def create_tables():
    """Create rag_documents + rag_chunks with indexes (idempotent)."""
    execute("CREATE EXTENSION IF NOT EXISTS vector;")
    execute("""
CREATE TABLE IF NOT EXISTS rag_documents (
    id            SERIAL PRIMARY KEY,
    source_type   VARCHAR(20) NOT NULL,
    name          TEXT NOT NULL,
    path          TEXT NOT NULL UNIQUE,
    content       TEXT NOT NULL,
    summary       TEXT,
    metadata      JSONB DEFAULT '{}',
    tags          TEXT[],
    problem_types TEXT[],
    chunk_count   INTEGER DEFAULT 0,
    document_hash TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);
""")
    execute("""
CREATE TABLE IF NOT EXISTS rag_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    embedding       VECTOR(1536),
    search_vector   TSVECTOR,
    parent_header   TEXT,
    header_path     TEXT[],
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);
""")
    # FTS trigger on chunks (auto-populate search_vector from content)
    execute("""
CREATE OR REPLACE FUNCTION update_chunk_search_vector()
RETURNS TRIGGER AS $$
BEGIN
  NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""")
    execute("""
DROP TRIGGER IF EXISTS trg_chunk_search_vector ON rag_chunks;
CREATE TRIGGER trg_chunk_search_vector
  BEFORE INSERT OR UPDATE OF content ON rag_chunks
  FOR EACH ROW EXECUTE FUNCTION update_chunk_search_vector();
""")


def ensure_indexes():
    """Create vector + FTS indexes on rag_chunks (call AFTER data is loaded)."""
    row_count = fetch_one("SELECT COUNT(*) AS cnt FROM rag_chunks")
    if not row_count or row_count["cnt"] == 0:
        return

    # GIN index for keyword search
    execute("CREATE INDEX IF NOT EXISTS idx_chunks_fts ON rag_chunks USING GIN (search_vector);")
    execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON rag_chunks(document_id);")

    # IVFFlat for vector search (skip if exists)
    existing = fetch_one("SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding'")
    if not existing:
        lists = max(100, int(row_count["cnt"] ** 0.5))
        execute(f"""
CREATE INDEX idx_chunks_embedding
    ON rag_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {lists});
""")
    execute("SET ivfflat.probes = 5;")

    # Backfill existing null search_vectors (if any)
    execute("""
UPDATE rag_chunks SET search_vector = to_tsvector('english', COALESCE(content, ''))
 WHERE search_vector IS NULL;
""")


# ── Document collection ─────────────────────────────────────────────────────

def collect_readmes(source_dir: Path, doc_types: List[str]) -> Iterator[Path]:
    for doc_type in doc_types:
        type_root = source_dir / doc_type
        if not type_root.is_dir():
            continue
        for readme_path in sorted(type_root.rglob("README.md")):
            yield readme_path


def read_metadata(readme_path: Path) -> Dict:
    metadata_path = readme_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            pass
    return {}


def document_name(readme_path: Path, metadata: Dict) -> str:
    if isinstance(metadata.get("name"), str) and metadata["name"].strip():
        return metadata["name"].strip()
    return readme_path.parent.name


def document_summary(metadata: Dict, content: str, max_length: int = 280) -> str:
    for key in ("description", "overview", "summary"):
        v = metadata.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return content.strip().replace("\n", " ")[:max_length]


def document_hash(content: str, metadata: Dict) -> str:
    payload = content + json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── CRUD operations ─────────────────────────────────────────────────────────

def get_existing(path: str) -> Optional[Dict]:
    return fetch_one("SELECT id, document_hash FROM rag_documents WHERE path = %s", (path,))


def upsert_document(
    doc_type: str, name: str, path: str, summary: str,
    content: str, metadata: Dict, doc_hash: str, tags: List[str],
    problem_types: List[str], chunk_count: int,
) -> int:
    """Insert or update a document row, return its id."""
    row = fetch_one("SELECT id FROM rag_documents WHERE path = %s", (path,))
    if row:
        doc_id = row["id"]
        execute("""
UPDATE rag_documents SET source_type=%s, name=%s, content=%s, summary=%s,
  metadata=%s, tags=%s, problem_types=%s, chunk_count=%s,
  document_hash=%s, updated_at=NOW()
WHERE id=%s
""", (doc_type, name, content, summary, json.dumps(metadata, ensure_ascii=False),
      tags, problem_types, chunk_count, doc_hash, doc_id))
    else:
        row = fetch_one("""
INSERT INTO rag_documents (source_type, name, path, content, summary, metadata,
  tags, problem_types, chunk_count, document_hash)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
""", (doc_type, name, path, content, summary, json.dumps(metadata, ensure_ascii=False),
      tags, problem_types, chunk_count, doc_hash))
        doc_id = row["id"]
    return doc_id


def delete_chunks(doc_id: int):
    execute("DELETE FROM rag_chunks WHERE document_id = %s", (doc_id,))


def insert_chunk(doc_id: int, chunk, embedding: List[float]):
    vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
    execute("""
INSERT INTO rag_chunks (document_id, chunk_index, content, token_count,
  embedding, parent_header, header_path, metadata)
VALUES (%s,%s,%s,%s,%s::vector,%s,%s,%s)
ON CONFLICT (document_id, chunk_index) DO UPDATE SET
  content = EXCLUDED.content,
  token_count = EXCLUDED.token_count,
  embedding = EXCLUDED.embedding,
  parent_header = EXCLUDED.parent_header,
  header_path = EXCLUDED.header_path,
  metadata = EXCLUDED.metadata
""", (
    doc_id, chunk.chunk_index, chunk.content, chunk.token_count,
    vector_str, chunk.parent_header, chunk.header_path,
    json.dumps(chunk.metadata, ensure_ascii=False),
))


# ── Main indexing loop ──────────────────────────────────────────────────────

_BATCH_SIZE = 20  # embed up to 20 chunks per API call


async def index_knowledge_base(source_dir: str, doc_types: List[str]):
    source_root = Path(source_dir)
    readme_paths = list(collect_readmes(source_root, doc_types))
    if not readme_paths:
        raise FileNotFoundError(f"No README.md files found in {source_dir} for types: {doc_types}")

    create_tables()
    total_chunks = 0

    for readme_path in readme_paths:
        content = readme_path.read_text(encoding="utf-8", errors="ignore")
        metadata = read_metadata(readme_path)
        doc_type = readme_path.relative_to(source_root).parts[0]
        name = document_name(readme_path, metadata)
        rel_path = str(readme_path.relative_to(source_root))
        summary = document_summary(metadata, content)
        doc_hash = document_hash(content, metadata)

        existing = get_existing(rel_path)
        if existing and existing.get("document_hash") == doc_hash:
            print(f"Skipped (unchanged): {rel_path}")
            continue

        # Extract structured tags from metadata
        tags = metadata.get("tags") or metadata.get("keywords") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        problem_types = metadata.get("problemType") or []

        # Chunk the document
        doc_meta = {
            "source_type": doc_type,
            "source_name": name,
            "source_path": rel_path,
            "tags": tags,
            "problem_types": problem_types,
        }
        chunks = chunk_markdown(content, doc_metadata=doc_meta)

        # Upsert document record
        doc_id = upsert_document(
            doc_type, name, rel_path, summary, content,
            metadata, doc_hash, tags, problem_types, len(chunks),
        )

        # Delete old chunks, insert new ones with embeddings
        delete_chunks(doc_id)

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]
            texts = [c.content for c in batch]
            batch_embeddings = await embeddings(texts)
            for chunk, emb in zip(batch, batch_embeddings):
                insert_chunk(doc_id, chunk, emb)

        total_chunks += len(chunks)
        print(f"Indexed {rel_path}: {len(chunks)} chunks [{name}]")

    ensure_indexes()
    print(f"\nDone. {len(readme_paths)} documents → {total_chunks} chunks")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan knowledge-base, chunk README files, compute embeddings, store in PostgreSQL."
    )
    parser.add_argument("--source", default="knowledge-base", help="Source directory to scan.")
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES),
                        help="Comma-separated subdirectories, e.g. algorithms,datasets.")
    parser.add_argument("--migrate", action="store_true",
                        help="Drop old schema and create new chunked schema (destroys existing data).")
    args = parser.parse_args()
    selected_types = [t.strip() for t in args.types.split(",") if t.strip()]
    if args.migrate:
        migrate_schema()
    asyncio.run(index_knowledge_base(args.source, selected_types))
