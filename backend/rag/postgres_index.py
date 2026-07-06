import json
from pathlib import Path
from typing import List

from backend.llm.openai_client import embeddings
from backend.database.postgres import execute


CREATE_VECTOR_EXTENSION_SQL = 'CREATE EXTENSION IF NOT EXISTS vector;'
CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS rag_documents (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    path TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
'''


def create_table():
    execute(CREATE_VECTOR_EXTENSION_SQL)
    execute(CREATE_TABLE_SQL)


def insert_document(source: str, path: str, content: str, embedding: List[float]):
    insert_sql = '''
INSERT INTO rag_documents (source, path, content, embedding)
VALUES (%s, %s, %s, %s)
'''
    execute(insert_sql, (source, path, content, embedding))


def index_knowledge_base(source_dir: str):
    source_root = Path(source_dir)
    readme_paths = sorted(source_root.rglob('README.md'))
    if not readme_paths:
        raise FileNotFoundError(f'No README.md files found in {source_dir}')

    create_table()

    for readme_path in readme_paths:
        content = readme_path.read_text(encoding='utf-8', errors='ignore')
        embeddings_list = embeddings([content])
        insert_document(
            source=str(readme_path.parent.relative_to(source_root)),
            path=str(readme_path.relative_to(source_root)),
            content=content,
            embedding=embeddings_list[0],
        )
        print(f'Indexed {readme_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build pgvector index from knowledge-base README files.')
    parser.add_argument('--source', default='knowledge-base', help='Source directory to scan for README files.')
    args = parser.parse_args()

    index_knowledge_base(args.source)
