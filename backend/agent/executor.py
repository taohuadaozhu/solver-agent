"""Algorithm executor — thin wrapper forwarding to the Tool Dispatcher."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from backend.agent.dispatcher import dispatch

logger = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parents[2] / "knowledge-base"


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Find and load a dataset from the knowledge base."""
    for data_dir in KB_ROOT.rglob("data"):
        for data_file in data_dir.iterdir():
            if data_file.is_file():
                if dataset_name.lower().replace(" ", "") in str(data_file).lower().replace(" ", ""):
                    return {
                        "name": dataset_name,
                        "file": str(data_file.relative_to(KB_ROOT)),
                        "size": data_file.stat().st_size,
                        "preview": data_file.read_text(errors="ignore")[:500],
                    }

    logger.warning("Dataset not found: %s", dataset_name)
    return {"name": dataset_name, "file": "", "size": 0, "preview": ""}


def execute_solver(
    algorithm: str,
    dataset_name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an algorithm on a dataset via the Tool Dispatcher."""
    dataset = load_dataset(dataset_name)
    return dispatch("execute_solver", algorithm, dataset, parameters)
