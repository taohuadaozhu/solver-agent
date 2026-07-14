"""Layer 3 — Unified Solver Interface. All algorithms subclass Solver."""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parents[2] / "knowledge-base"


class Solver(ABC):
    """Abstract base class for all optimization solvers."""

    name: str = ""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    @abstractmethod
    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the solver on a dataset. Returns a standard result dict."""
        ...

    def _load_dataset_file(self, dataset: Dict[str, Any]) -> str:
        """Load raw dataset content from the knowledge base."""
        file_path = dataset.get("file", "")
        full_path = KB_ROOT / file_path
        if full_path.exists():
            return full_path.read_text(errors="ignore")
        # Fallback: search by name
        for data_file in KB_ROOT.rglob("data/*"):
            if data_file.is_file():
                if dataset.get("name", "").lower().replace(" ", "") in str(data_file).lower().replace(" ", ""):
                    return data_file.read_text(errors="ignore")
        return ""


# ── Built-in solvers ────────────────────────────────────────────────────────
# Each corresponds to an algorithm in knowledge-base/algorithms/


class GASolver(Solver):
    name = "GA"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        pop = self.parameters.get("population_size", 100)
        gen = self.parameters.get("num_generations", 200)
        logger.info("[GA] Solving %s pop=%d gen=%d", dataset.get("name"), pop, gen)
        time.sleep(0.3)
        return _sim_convergence(self.name, dataset.get("name", ""), pop, gen)


class NSGA2Solver(Solver):
    name = "NSGA-II"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        pop = self.parameters.get("population_size", 100)
        gen = self.parameters.get("num_generations", 250)
        logger.info("[NSGA-II] Solving %s pop=%d gen=%d", dataset.get("name"), pop, gen)
        time.sleep(0.4)
        return _sim_convergence(self.name, dataset.get("name", ""), pop, gen)


class PSOSolver(Solver):
    name = "PSO"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        swarm = self.parameters.get("swarm_size", 50)
        gen = self.parameters.get("num_iterations", 200)
        logger.info("[PSO] Solving %s swarm=%d iter=%d", dataset.get("name"), swarm, gen)
        time.sleep(0.25)
        return _sim_convergence(self.name, dataset.get("name", ""), swarm, gen)


class ALNSSolver(Solver):
    name = "ALNS"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        iters = self.parameters.get("max_iterations", 500)
        logger.info("[ALNS] Solving %s iterations=%d", dataset.get("name"), iters)
        time.sleep(0.35)
        return _sim_convergence(self.name, dataset.get("name", ""), 1, iters)


class ACOPermutationSolver(Solver):
    name = "ACO-Permutation"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        ants = self.parameters.get("num_ants", 30)
        gen = self.parameters.get("num_iterations", 200)
        logger.info("[ACO-Permutation] Solving %s ants=%d", dataset.get("name"), ants)
        time.sleep(0.3)
        return _sim_convergence(self.name, dataset.get("name", ""), ants, gen)


class SAPermutationSolver(Solver):
    name = "SA-Permutation"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        temp = self.parameters.get("initial_temperature", 1000)
        iters = self.parameters.get("max_iterations", 500)
        logger.info("[SA] Solving %s temp=%d iter=%d", dataset.get("name"), temp, iters)
        time.sleep(0.2)
        return _sim_convergence(self.name, dataset.get("name", ""), 1, iters)


class DummySolver(Solver):
    """Fallback solver for algorithms without a dedicated implementation."""

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.name = name

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Solving %s (dummy)", self.name, dataset.get("name"))
        time.sleep(0.5)
        return _sim_convergence(self.name, dataset.get("name", ""), 50, 100)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sim_convergence(
    name: str, dataset_name: str, pop_size: int, iterations: int
) -> Dict[str, Any]:
    """Generate simulated convergence results (placeholder until real solvers exist)."""
    import random
    rng = random.Random(hash(f"{name}{dataset_name}") % (2**31))

    n_pts = min(iterations, 50)
    curve = []
    val = rng.uniform(600, 2000)
    for _ in range(n_pts):
        val *= 0.90 + rng.uniform(0, 0.10)
        curve.append(round(val, 2))

    runtime = round(rng.uniform(0.2, 3.5), 2)

    return {
        "algorithm": name,
        "dataset": dataset_name,
        "status": "completed",
        "best_objective": curve[-1],
        "iterations": iterations,
        "population_size": pop_size,
        "convergence_curve": curve,
        "runtime_seconds": runtime,
    }
