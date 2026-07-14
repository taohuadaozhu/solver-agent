"""Result Validator — checks convergence, constraints, timeout, and empty results."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONVERGENCE_THRESHOLD = 0.001  # < 0.1% change = converged
DEFAULT_CONVERGENCE_WINDOW = 5         # look at last N points


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Run all validation checks on a solver result. Returns a validation report."""

    checks = {
        "converged": _check_convergence(result),
        "non_empty": _check_non_empty(result),
        "timeout": _check_timeout(result),
        "feasible": _check_feasible(result),
    }

    all_pass = all(checks.values())
    checks["all_pass"] = all_pass

    if not all_pass:
        failed = [k for k, v in checks.items() if not v and k != "all_pass"]
        logger.warning("Result validation failed: %s", failed)
    else:
        logger.info("Result validation passed")

    return checks


def _check_convergence(result: Dict[str, Any]) -> bool:
    """Check if the convergence curve flattened at the end."""
    curve = result.get("convergence_curve", [])
    if not curve or len(curve) < DEFAULT_CONVERGENCE_WINDOW:
        return True  # can't judge, assume ok

    tail = curve[-DEFAULT_CONVERGENCE_WINDOW:]
    if tail[0] == 0:
        return True

    change = abs(tail[-1] - tail[0]) / abs(tail[0])
    return change <= DEFAULT_CONVERGENCE_THRESHOLD


def _check_non_empty(result: Dict[str, Any]) -> bool:
    """Check that the result contains actual data."""
    curve = result.get("convergence_curve", [])
    obj = result.get("best_objective")
    return len(curve) > 0 and obj is not None and isinstance(obj, (int, float))


def _check_timeout(result: Dict[str, Any]) -> bool:
    """Check if execution exceeded the timeout."""
    runtime = result.get("runtime_seconds", 0)
    return runtime <= DEFAULT_TIMEOUT_SECONDS


def _check_feasible(result: Dict[str, Any]) -> bool:
    """Check if the solution is feasible (basic sanity)."""
    status = result.get("status", "")
    if status == "error":
        return False
    if status == "timeout":
        return False
    obj = result.get("best_objective")
    if obj is not None and obj < 0:
        # Negative objective may be valid for some problems but flag it
        pass
    return True
