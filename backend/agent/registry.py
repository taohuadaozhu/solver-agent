"""Layer 1 — Tool Registry. Maps algorithm names to Solver classes."""

import logging
from typing import Dict, Optional, Type

from backend.agent.solver import (
    Solver,
    GASolver,
    NSGA2Solver,
    PSOSolver,
    ALNSSolver,
    ACOPermutationSolver,
    SAPermutationSolver,
    DummySolver,
)

logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────────
# Maps algorithm names (as they appear in RAG metadata.name) to Solver classes.
# Add new solvers here.

TOOLS: Dict[str, Type[Solver]] = {
    "GA": GASolver,
    "GA-Permutation": GASolver,
    "NSGA-II": NSGA2Solver,
    "NSGA2": NSGA2Solver,
    "PSO": PSOSolver,
    "PSO-Permutation": PSOSolver,
    "ALNS": ALNSSolver,
    "ACO": ACOPermutationSolver,
    "ACO-Permutation": ACOPermutationSolver,
    "SA": SAPermutationSolver,
    "SA-Permutation": SAPermutationSolver,
    "Simulated Annealing": SAPermutationSolver,
    "Particle Swarm Optimization": PSOSolver,
    "Genetic Algorithm": GASolver,
    "Ant Colony Optimization": ACOPermutationSolver,
}


def register(algorithm_name: str, solver_cls: Type[Solver]):
    """Dynamically register a new solver."""
    TOOLS[algorithm_name] = solver_cls
    logger.info("Registered solver: %s -> %s", algorithm_name, solver_cls.__name__)


def resolve(algorithm_name: str) -> Solver:
    """Look up the best matching solver class for an algorithm name."""
    # Exact match
    if algorithm_name in TOOLS:
        return TOOLS[algorithm_name]()

    # Fuzzy match: check if any registered key is a substring
    name_lower = algorithm_name.lower()
    for key, cls in TOOLS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            logger.info("Fuzzy match: '%s' -> '%s'", algorithm_name, key)
            return cls()

    # Fallback to DummySolver
    logger.warning("No solver registered for '%s', using DummySolver", algorithm_name)
    return DummySolver(name=algorithm_name)
