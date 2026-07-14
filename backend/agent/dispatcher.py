"""Layer 2 — Tool Dispatcher. Routes function calling results to Solver classes."""

import logging
from typing import Any, Dict, Optional

from backend.agent.registry import TOOLS, resolve
from backend.agent.solver import Solver
from backend.agent.validator import validate_result

logger = logging.getLogger(__name__)


def dispatch(tool_name: str, algorithm: str, dataset: Dict[str, Any],
             parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Dispatch a function-calling request to the appropriate solver.

    Args:
        tool_name: "execute_solver" (from function calling schema)
        algorithm: algorithm name, e.g. "NSGA-II", "GA"
        dataset: dataset dict from load_dataset()
        parameters: solver hyperparameters
    """
    if tool_name != "execute_solver":
        raise ValueError(f"Unknown tool: {tool_name}")

    logger.info("Dispatch: tool=%s algorithm=%s dataset=%s", tool_name, algorithm, dataset.get("name"))

    # Resolve solver from registry
    solver: Solver = resolve(algorithm)

    # Inject parameters if provided
    if parameters:
        solver.parameters.update(parameters)

    # Execute
    result = solver.solve(dataset)

    # Validate
    validation = validate_result(result)

    return {
        "execution": result,
        "validation": validation,
    }
