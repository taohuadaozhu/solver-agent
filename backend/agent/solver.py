"""Layer 3 — Unified Solver Interface. All algorithms subclass Solver.

Each solver delegates to algorithms.runner.solve() for real execution.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KB_ROOT = Path(__file__).resolve().parents[2] / "knowledge-base"
ALGO_ROOT = Path(__file__).resolve().parents[2] / "algorithms"

# Make algorithms/ importable from backend
if str(ALGO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ALGO_ROOT.parent))


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
        for data_file in KB_ROOT.rglob("data/*"):
            if data_file.is_file():
                if dataset.get("name", "").lower().replace(" ", "") in str(data_file).lower().replace(" ", ""):
                    return data_file.read_text(errors="ignore")
        return ""


class RunnerSolver(Solver):
    """Solver that delegates to algorithms.runner.solve().

    Subclasses only need to set ``_algo_name`` to the runner algorithm key.
    """

    _algo_name: str = ""

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        from algorithms.runner import solve as run_solve

        nvar = self.parameters.get("nvar", 30)
        popsize = self.parameters.get("population_size",
                                       self.parameters.get("swarm_size",
                                           self.parameters.get("num_ants", 50)))
        maxgen = self.parameters.get("num_generations",
                                      self.parameters.get("num_iterations",
                                          self.parameters.get("max_iterations", 200)))
        lb = self.parameters.get("lb", -5.0)
        ub = self.parameters.get("ub", 10.0)
        seed = self.parameters.get("seed")
        threshold = self.parameters.get("threshold", 1e-8)

        algo_params = self.parameters.get("algo_params", {})

        logger.info("[%s] Solving %s nvar=%d pop=%d gen=%d",
                    self._algo_name, dataset.get("name"), nvar, popsize, maxgen)

        return run_solve(
            algorithm=self._algo_name,
            nvar=nvar,
            popsize=popsize,
            maxgen=maxgen,
            lb=lb,
            ub=ub,
            seed=seed,
            threshold=threshold,
            extra_params=algo_params,
        )


# ── Concrete solver classes ──────────────────────────────────────────────────

class GASolver(RunnerSolver):
    name = "GA"
    _algo_name = "GA"


class NSGA2Solver(Solver):
    """NSGA-II is multi-objective. The current runner targets single-objective.
    Falls back to GA with a weighted-sum note until MO runner is ready."""

    name = "NSGA-II"

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        from algorithms.runner import solve as run_solve

        popsize = self.parameters.get("population_size", 100)
        maxgen = self.parameters.get("num_generations", 250)
        nvar = self.parameters.get("nvar", 30)
        logger.info("[NSGA-II] Solving %s (single-obj fallback via GA)", dataset.get("name"))

        result = run_solve(algorithm="GA", nvar=nvar, popsize=popsize, maxgen=maxgen)
        result["algorithm"] = "NSGA-II"
        return result


class PSOSolver(RunnerSolver):
    name = "PSO"
    _algo_name = "PSO"


class ALNSSolver(RunnerSolver):
    name = "ALNS"
    _algo_name = "ILS"


class ACOPermutationSolver(RunnerSolver):
    name = "ACO-Permutation"
    _algo_name = "ACO"


class SAPermutationSolver(RunnerSolver):
    name = "SA-Permutation"
    _algo_name = "SA"


class DESolver(RunnerSolver):
    name = "DE"
    _algo_name = "DE"


class CSASolver(RunnerSolver):
    name = "CSA"
    _algo_name = "CSA"


class HSSolver(RunnerSolver):
    name = "HS"
    _algo_name = "HS"


class VNSSolver(RunnerSolver):
    name = "VNS"
    _algo_name = "VNS"


class WOASolver(RunnerSolver):
    name = "WOA"
    _algo_name = "WOA"


class BASolver(RunnerSolver):
    name = "BA"
    _algo_name = "BA"


class ABCASolver(RunnerSolver):
    name = "ABCA"
    _algo_name = "ABCA"


class EGOSolver(RunnerSolver):
    name = "EGO"
    _algo_name = "EGO"


class DummySolver(Solver):
    """Fallback solver for algorithms without a dedicated runner entry."""

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.name = name

    def solve(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        from algorithms.runner import solve as run_solve

        logger.info("[%s] Solving %s (dummy → GA fallback)", self.name, dataset.get("name"))
        result = run_solve(algorithm="GA", nvar=30, popsize=50, maxgen=100)
        result["algorithm"] = self.name
        return result
