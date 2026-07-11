"""CRUD operations for conversation and workflow memory."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.memory.db import _connect

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20


# ── Conversation ────────────────────────────────────────────────────────────

def create_conversation(title: str = "") -> str:
    conv_id = uuid.uuid4().hex[:12]
    conn = _connect()
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (conv_id, title or f"Chat {conv_id[:6]}"))
    conn.commit()
    conn.close()
    return conv_id


def list_conversations(limit: int = 20) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_conversation(conv_id: str):
    conn = _connect()
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def add_message(conv_id: str, role: str, content: str, metadata: Optional[Dict] = None):
    conn = _connect()
    conn.execute(
        "INSERT INTO conversation_messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
    )
    conn.execute("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?", (conv_id,))

    # Enforce max messages — delete oldest beyond limit
    count = conn.execute(
        "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = ?", (conv_id,)
    ).fetchone()[0]
    if count > MAX_MESSAGES:
        to_delete = count - MAX_MESSAGES
        conn.execute(
            "DELETE FROM conversation_messages WHERE id IN ("
            "  SELECT id FROM conversation_messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?"
            ")",
            (conv_id, to_delete),
        )

    conn.commit()
    conn.close()


def get_messages(conv_id: str, limit: int = MAX_MESSAGES) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, metadata, created_at FROM conversation_messages "
        "WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
        (conv_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Workflow ────────────────────────────────────────────────────────────────

def save_workflow(
    conv_id: str,
    problem: str,
    state: Dict,
    step_outputs: Dict[str, str],
    status: str = "in_progress",
) -> str:
    conn = _connect()
    wf_id = uuid.uuid4().hex[:12]

    current_step = "STEP_1_PROBLEM"
    for step in ["STEP_7_ALGO", "STEP_6_CLASSIFY", "STEP_5_CONSTRAINT", "STEP_4_OBJECTIVE",
                  "STEP_3_VARIABLE", "STEP_2_DATASET", "STEP_1_PROBLEM"]:
        if step in step_outputs:
            current_step = step
            break

    conn.execute(
        """INSERT OR REPLACE INTO workflow_sessions
           (id, conversation_id, problem, current_step, state_json, step_outputs,
            dataset_name, algorithm, execution_result, status, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            wf_id, conv_id, problem, current_step,
            json.dumps(state, ensure_ascii=False),
            json.dumps(step_outputs, ensure_ascii=False),
            state.get("dataset_name", ""),
            state.get("recommended_algorithm", ""),
            json.dumps(state.get("execution_result") or {}, ensure_ascii=False),
            status,
        ),
    )
    conn.commit()
    conn.close()
    return wf_id


def get_workflow(wf_id: str) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM workflow_sessions WHERE id = ?", (wf_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result["state_json"] = json.loads(result["state_json"] or "{}")
    result["step_outputs"] = json.loads(result["step_outputs"] or "{}")
    result["execution_result"] = json.loads(result["execution_result"] or "{}")
    return result


def list_workflows(conv_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    conn = _connect()
    if conv_id:
        rows = conn.execute(
            "SELECT id, problem, current_step, dataset_name, algorithm, status, created_at "
            "FROM workflow_sessions WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT ?",
            (conv_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, problem, current_step, dataset_name, algorithm, status, created_at "
            "FROM workflow_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_workflow_status(wf_id: str, status: str):
    conn = _connect()
    conn.execute(
        "UPDATE workflow_sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
        (status, wf_id),
    )
    conn.commit()
    conn.close()
