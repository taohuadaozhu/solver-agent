import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.llm.openai_client import embeddings
from backend.database.postgres import execute, fetch_one


DEFAULT_TYPES = ['algorithms', 'projects', 'datasets']


def collect_readmes(source_dir: Path, doc_types: List[str]) -> Iterator[Path]:
    for doc_type in doc_types:
        type_root = source_dir / doc_type
        if not type_root.exists():
            continue
        for readme_path in sorted(type_root.rglob('README.md')):
            yield readme_path


def read_content(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def read_metadata(readme_path: Path) -> Dict:
    metadata_path = readme_path.parent / 'metadata.json'
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding='utf-8', errors='ignore'))
        except json.JSONDecodeError:
            pass
    return {}


def document_type_for_path(readme_path: Path, source_root: Path) -> str:
    relative = readme_path.relative_to(source_root)
    return relative.parts[0] if len(relative.parts) > 1 else 'unknown'


def document_name(readme_path: Path, metadata: Dict) -> str:
    if isinstance(metadata.get('name'), str) and metadata['name'].strip():
        return metadata['name'].strip()
    return readme_path.parent.name


def document_summary(metadata: Dict, content: str, max_length: int = 280) -> str:
    summary = metadata.get('description') or metadata.get('overview') or metadata.get('summary')
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return content.strip().replace('\n', ' ')[:max_length]


def compute_document_hash(content: str, metadata: Dict) -> str:
    payload = content + json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def create_pgvector_table():
    execute('CREATE EXTENSION IF NOT EXISTS vector;')
    execute('''
CREATE TABLE IF NOT EXISTS rag_documents (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20),
    name TEXT,
    path TEXT NOT NULL UNIQUE,
    summary TEXT,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536),
    document_hash TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
''')


def ensure_vector_index(lists: int = 100):
    """Create IVFFlat index on the embedding column for fast approximate nearest neighbor search.

    IVFFlat partitions vectors into ``lists`` clusters via k-means. At query time only the
    closest ``probes`` clusters are scanned, trading a tiny amount of recall for orders-of-magnitude
    speed improvement over exact search.

    Must be called AFTER the table has data, or the centroids will be empty.
    """
    row_count = fetch_one('SELECT COUNT(*) AS cnt FROM rag_documents')
    if not row_count or row_count['cnt'] == 0:
        return

    # Check if index already exists
    existing = fetch_one(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_rag_embedding_ivfflat'"
    )
    if existing:
        return

    # lists ≈ sqrt(n) is a common starting point; cap at a reasonable floor
    auto_lists = max(lists, int((row_count['cnt'] ** 0.5)))
    execute(f'''
CREATE INDEX idx_rag_embedding_ivfflat
    ON rag_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = {auto_lists});
''')
    # Tell pgvector how many clusters to probe at query time. Default is 1 (too low).
    execute('SET ivfflat.probes = 5;')


def get_existing_document(path: str) -> Optional[Dict]:
    return fetch_one('SELECT document_hash FROM rag_documents WHERE path = %s', (path,))


def upsert_document(
    doc_type: str,
    name: str,
    path: str,
    summary: str,
    content: str,
    metadata: Dict,
    embedding: List[float],
    document_hash: str,
):
    vector_value = '[' + ','.join(str(v) for v in embedding) + ']'
    execute('''
INSERT INTO rag_documents (type, name, path, summary, content, metadata, embedding, document_hash, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, NOW())
ON CONFLICT (path) DO UPDATE SET
    type = EXCLUDED.type,
    name = EXCLUDED.name,
    summary = EXCLUDED.summary,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    document_hash = EXCLUDED.document_hash,
    updated_at = NOW();
''', (doc_type, name, path, summary, content, json.dumps(metadata, ensure_ascii=False), vector_value, document_hash))


async def index_knowledge_base(source_dir: str, doc_types: List[str]):
    source_root = Path(source_dir)
    readme_paths = list(collect_readmes(source_root, doc_types))
    if not readme_paths:
        raise FileNotFoundError(f'No README.md files found in {source_dir} for types: {doc_types}')

    create_pgvector_table()

    for readme_path in readme_paths:
        content = read_content(readme_path)
        metadata = read_metadata(readme_path)
        doc_type = document_type_for_path(readme_path, source_root)
        name = document_name(readme_path, metadata)
        summary = document_summary(metadata, content)
        path = str(readme_path.relative_to(source_root))
        document_hash = compute_document_hash(content, metadata)

        existing = get_existing_document(path)
        if existing and existing.get('document_hash') == document_hash:
            print(f'Skipped unchanged document: {path}')
            continue

        embedding_text = f"""# {name}\n{summary}\n{content}"""
        embedding_vector = (await embeddings([embedding_text]))[0]
        upsert_document(
            doc_type=doc_type,
            name=name,
            path=path,
            summary=summary,
            content=content,
            metadata=metadata,
            embedding=embedding_vector,
            document_hash=document_hash,
        )
        print(f'Indexed {path} ({doc_type})')

    ensure_vector_index()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scan knowledge-base, read README files, compute embeddings, and store documents in PostgreSQL with pgvector.'
    )
    parser.add_argument('--source', default='knowledge-base', help='Source directory to scan for README files.')
    parser.add_argument(
        '--types', default=','.join(DEFAULT_TYPES),
        help='Comma-separated list of knowledge-base subdirectories to index, e.g. algorithms,projects.',
    )
    args = parser.parse_args()

    selected_types = [item.strip() for item in args.types.split(',') if item.strip()]
    asyncio.run(index_knowledge_base(args.source, selected_types))
