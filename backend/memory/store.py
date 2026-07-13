"""CRUD operations for conversation and workflow memory — PostgreSQL backend."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.database.postgres import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20


# ── Conversation ────────────────────────────────────────────────────────────

def create_conversation(title: str = "") -> str:
    conv_id = uuid.uuid4().hex[:12]
    execute(
        "INSERT INTO conversations (id, title) VALUES (%s, %s)",
        (conv_id, title or f"Chat {conv_id[:6]}"),
    )
    return conv_id


def list_conversations(limit: int = 20) -> List[Dict]:
    rows = fetch_all(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT %s",
        (limit,),
    )
    return [dict(r) for r in rows]


def delete_conversation(conv_id: str):
    execute("DELETE FROM conversations WHERE id = %s", (conv_id,))


def add_message(conv_id: str, role: str, content: str, metadata: Optional[Dict] = None):
    execute(
        "INSERT INTO conversation_messages (conversation_id, role, content, metadata) "
        "VALUES (%s, %s, %s, %s)",
        (conv_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
    )
    execute("UPDATE conversations SET updated_at = NOW() WHERE id = %s", (conv_id,))

    # Enforce max messages — delete oldest beyond limit
    existing = fetch_one(
        "SELECT COUNT(*) AS cnt FROM conversation_messages WHERE conversation_id = %s",
        (conv_id,),
    )
    count = existing["cnt"] if existing else 0
    if count > MAX_MESSAGES:
        execute(
            "DELETE FROM conversation_messages WHERE id IN ("
            "  SELECT id FROM conversation_messages WHERE conversation_id = %s "
            "  ORDER BY created_at ASC LIMIT %s"
            ")",
            (conv_id, count - MAX_MESSAGES),
        )


def get_messages(conv_id: str, limit: int = MAX_MESSAGES) -> List[Dict]:
    rows = fetch_all(
        "SELECT role, content, metadata, created_at FROM conversation_messages "
        "WHERE conversation_id = %s ORDER BY created_at ASC LIMIT %s",
        (conv_id, limit),
    )
    return [dict(r) for r in rows]


# ── Workflow ────────────────────────────────────────────────────────────────

def save_workflow(
    conv_id: str,
    problem: str,
    state: Dict,
    step_outputs: Dict[str, str],
    status: str = "in_progress",
) -> str:
    wf_id = uuid.uuid4().hex[:12]

    current_step = "STEP_1_PROBLEM"
    for step in [
        "STEP_7_ALGO", "STEP_6_CLASSIFY", "STEP_5_CONSTRAINT",
        "STEP_4_OBJECTIVE", "STEP_3_VARIABLE", "STEP_2_DATASET", "STEP_1_PROBLEM",
    ]:
        if step in step_outputs:
            current_step = step
            break

    execute(
        """INSERT INTO workflow_sessions
           (id, conversation_id, problem, current_step, state_json, step_outputs,
            dataset_name, algorithm, execution_result, status, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
           ON CONFLICT (id) DO UPDATE SET
            problem = EXCLUDED.problem,
            current_step = EXCLUDED.current_step,
            state_json = EXCLUDED.state_json,
            step_outputs = EXCLUDED.step_outputs,
            dataset_name = EXCLUDED.dataset_name,
            algorithm = EXCLUDED.algorithm,
            execution_result = EXCLUDED.execution_result,
            status = EXCLUDED.status,
            updated_at = NOW()""",
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
    return wf_id


def get_workflow(wf_id: str) -> Optional[Dict]:
    row = fetch_one("SELECT * FROM workflow_sessions WHERE id = %s", (wf_id,))
    if not row:
        return None
    result = dict(row)
    result["state_json"] = result.get("state_json") or {}
    result["step_outputs"] = result.get("step_outputs") or {}
    result["execution_result"] = result.get("execution_result") or {}
    return result


def list_workflows(conv_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    if conv_id:
        rows = fetch_all(
            "SELECT id, problem, current_step, dataset_name, algorithm, status, created_at "
            "FROM workflow_sessions WHERE conversation_id = %s ORDER BY updated_at DESC LIMIT %s",
            (conv_id, limit),
        )
    else:
        rows = fetch_all(
            "SELECT id, problem, current_step, dataset_name, algorithm, status, created_at "
            "FROM workflow_sessions ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        )
    return [dict(r) for r in rows]


def update_workflow_status(wf_id: str, status: str):
    execute(
        "UPDATE workflow_sessions SET status = %s, updated_at = NOW() WHERE id = %s",
        (status, wf_id),
    )
