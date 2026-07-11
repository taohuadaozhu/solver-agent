"""SQLite-based memory store for conversations and workflow sessions."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DB_DIR / "memory.db"


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS workflow_sessions (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
            problem TEXT NOT NULL,
            current_step TEXT DEFAULT 'STEP_1_PROBLEM',
            state_json TEXT DEFAULT '{}',
            step_outputs TEXT DEFAULT '{}',
            dataset_name TEXT DEFAULT '',
            algorithm TEXT DEFAULT '',
            execution_result TEXT DEFAULT '',
            status TEXT DEFAULT 'in_progress' CHECK(status IN ('in_progress', 'completed', 'archived')),
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv ON conversation_messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_workflow_conv ON workflow_sessions(conversation_id);
    """)
    conn.commit()
    conn.close()
    logger.info("Memory DB initialized at %s", DB_PATH)


# Ensure DB is initialized on import
try:
    init_db()
except Exception:
    logger.exception("Failed to init memory DB")
