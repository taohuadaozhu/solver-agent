"""PostgreSQL-based memory store — uses the shared psycopg2 pool from backend.database."""

import logging

from backend.database.postgres import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)


def init_db():
    """Create memory tables in PostgreSQL (idempotent)."""
    execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS workflow_sessions (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
            problem TEXT NOT NULL,
            current_step TEXT DEFAULT 'STEP_1_PROBLEM',
            state_json JSONB DEFAULT '{}',
            step_outputs JSONB DEFAULT '{}',
            dataset_name TEXT DEFAULT '',
            algorithm TEXT DEFAULT '',
            execution_result JSONB DEFAULT '{}',
            status TEXT DEFAULT 'in_progress' CHECK(status IN ('in_progress', 'completed', 'archived')),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conv
        ON conversation_messages(conversation_id, created_at);
    """)
    execute("""
        CREATE INDEX IF NOT EXISTS idx_workflow_conv
        ON workflow_sessions(conversation_id);
    """)
    logger.info("Memory tables initialized in PostgreSQL")


try:
    init_db()
except Exception:
    logger.exception("Failed to init memory tables")
