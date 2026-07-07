"""Algorithm executor — simulates running optimization solvers."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parents[2] / "knowledge-base"


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Find and load a dataset from the knowledge base."""
    for data_dir in KB_ROOT.rglob("data"):
        for data_file in data_dir.iterdir():
            if data_file.is_file():
                content = data_file.read_text(errors="ignore")
                if dataset_name.lower().replace(" ", "") in str(data_file).lower().replace(" ", ""):
                    return {
                        "name": dataset_name,
                        "file": str(data_file.relative_to(KB_ROOT)),
                        "size": len(content),
                        "preview": content[:500],
                    }

    logger.warning("Dataset not found: %s", dataset_name)
    return {"name": dataset_name, "file": "unknown", "size": 0, "preview": ""}


def execute_solver(
    algorithm: str,
    dataset_name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simulate running an optimization algorithm on a dataset."""
    params = parameters or {}
    logger.info("Executing %s on %s with params=%s", algorithm, dataset_name, params)

    # Simulate computation time
    time.sleep(0.5)

    # Generate simulated results based on algorithm and dataset
    import random
    rng = random.Random(hash(algorithm + dataset_name) % (2**31))

    iterations = params.get("max_iterations", 100)
    population_size = params.get("population_size", 50)

    # Simulated convergence curve
    best_values = []
    current = rng.uniform(800, 2000)
    for i in range(min(iterations, 50)):
        current *= (0.92 + rng.uniform(0, 0.08))
        best_values.append(round(current, 2))

    final_value = best_values[-1]

    return {
        "algorithm": algorithm,
        "dataset": dataset_name,
        "parameters": params,
        "status": "completed",
        "best_objective": final_value,
        "iterations": iterations,
        "population_size": population_size,
        "convergence_curve": best_values,
        "runtime_seconds": round(rng.uniform(0.3, 3.0), 2),
        "solution_quality": "excellent" if final_value < 800 else "good" if final_value < 1200 else "acceptable",
        "raw_output": (
            f"=== {algorithm} on {dataset_name} ===\n"
            f"Best Objective: {final_value}\n"
            f"Iterations: {iterations}\n"
            f"Population: {population_size}\n"
            f"Runtime: {rng.uniform(0.3, 3.0):.2f}s\n"
            f"Status: Converged\n"
            f"Solution: feasible"
        ),
    }
