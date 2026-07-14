"""Unified execution interface for all optimization algorithms.

Provides a single solve() entry point that takes an algorithm name plus
problem configuration and returns a standardized result dict.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from .multimethod import MultiMet
from .problems import Sphere

logger = logging.getLogger(__name__)


def solve(
    algorithm: str,
    nvar: int = 30,
    popsize: int = 50,
    maxgen: int = 200,
    lb: float = -5.0,
    ub: float = 10.0,
    evaluate: Callable | None = None,
    seed: int | None = None,
    threshold: float = 1e-8,
    verbose: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an optimization algorithm and return a standardized result dict.

    Args:
        algorithm: Algorithm name, e.g. "GA", "PSO", "DE", "SA", "CSA".
        nvar: Number of decision variables (problem dimension).
        popsize: Population / swarm size.
        maxgen: Maximum generations / iterations.
        lb: Lower bound for decision variables.
        ub: Upper bound for decision variables.
        evaluate: Fitness function f(x, nvar) -> float.  Defaults to Sphere.
        seed: Random seed for reproducibility.
        threshold: Convergence threshold on objective value.
        verbose: Print per-generation objective if True.
        extra_params: Override default algorithm hyperparameters.

    Returns:
        Dict with keys: algorithm, status, best_objective, generations,
        runtime_seconds, convergence_curve, population_size, nvar.
    """
    if seed is not None:
        random.seed(seed)

    fitness_func = evaluate or Sphere

    solver = MultiMet(popsize, nvar, lb, ub, evaluate=fitness_func)
    solver.Initial()

    generation = 0
    stagnant = 0
    best = solver.gbest_fit
    curve: list[float] = [best]
    curve_interval = max(1, maxgen // 50)

    started = time.perf_counter()

    while generation < maxgen and solver.gbest_fit > threshold and stagnant < maxgen / 10:
        _dispatch(solver, algorithm, extra_params or {}, generation, maxgen, popsize)
        solver.Evaluation(True, 0, popsize)
        solver.pop_update(0, popsize)
        solver.worst_and_best()
        solver.Elist()

        if solver.gbest_fit < best - 1e-15:
            stagnant = 0
        else:
            stagnant += 1
        generation += 1
        best = solver.gbest_fit

        if generation % curve_interval == 0 or generation == 1:
            curve.append(best)

        if verbose and generation % 10 == 0:
            logger.info("[%s] gen=%d best=%.6g", algorithm, generation, best)

    curve.append(solver.gbest_fit)
    elapsed = time.perf_counter() - started

    return {
        "algorithm": algorithm,
        "status": "completed" if solver.gbest_fit <= threshold else "max_generations",
        "best_objective": solver.gbest_fit,
        "generations": generation,
        "runtime_seconds": round(elapsed, 4),
        "convergence_curve": curve,
        "population_size": popsize,
        "nvar": nvar,
    }


# ── Dispatch table ──────────────────────────────────────────────────────────
# Each entry: (method_name, is_gen_based, arg_builder)
# arg_builder(params, gen, maxgen, popsize) -> (*args, **kwargs)

def _dispatch(
    solver: MultiMet,
    algorithm: str,
    params: dict[str, Any],
    gen: int,
    maxgen: int,
    popsize: int,
) -> None:
    """Call the right algorithm method on the solver instance."""
    method_name = _ALGO_TABLE.get(algorithm)
    if method_name is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    method = getattr(solver, method_name)
    args, kwargs = _build_args(algorithm, method_name, params, gen, maxgen, popsize)
    method(*args, **kwargs)


def _build_args(
    algo: str,
    method_name: str,
    params: dict[str, Any],
    gen: int,
    maxgen: int,
    popsize: int,
) -> tuple[tuple, dict]:
    """Build positional and keyword arguments for an algorithm method."""
    p = params  # shorthand

    # ── Generation-based algorithms (Gen, MaxGen, p_start, p_end) ──────
    if algo in _GEN_BASED:
        return (gen, maxgen, 0, popsize), {}

    # ── Per-algorithm special cases ────────────────────────────────────
    if algo == "GA":
        return (p.get("pc", 0.9), p.get("pm", 0.1), 0, popsize), {}
    if algo == "PSO":
        return (p.get("w", 0.7), p.get("c1", 2.0), p.get("c2", 2.0),
                p.get("max_ve", 2.0), 0, popsize), {}
    if algo == "DE":
        return (p.get("F", 0.5), p.get("S", 1), p.get("cr", 0.9), 0, popsize), {}
    if algo == "CSA":
        return (p.get("pa", 0.25), 0, popsize), {}
    if algo == "ACO":
        return (p.get("epsl", 0.1), 0, popsize), {}
    if algo == "HS":
        return (p.get("srate", 0.9), p.get("trate", 0.3),
                p.get("bw", 0.01), 0, popsize), {}
    if algo == "ILS":
        return (p.get("nst", 3), 0, popsize), {}
    if algo == "VNS":
        return (p.get("nst", 3), 0, popsize), {}
    if algo == "BA":
        return (0, popsize), {}
    if algo == "ABCA":
        return (p.get("limit", 30), 0, popsize), {}
    if algo == "POA":
        return (0, popsize), {}
    if algo == "BATA":
        return (gen, 0, popsize), {}
    if algo == "CMAES":
        return (0, 0, popsize), {}
    if algo == "COA":
        return (p.get("chaos_n", 5), 0, popsize), {}
    if algo == "CSO":
        return (p.get("phi", 0.1), 0, popsize), {}
    if algo == "FA":
        return (p.get("gama", 1.0), p.get("alpha0", 0.5),
                p.get("betamin", 0.2), maxgen, 0, popsize), {}
    if algo == "PBILC":
        return (0, popsize, p.get("learn_rate", 0.05)), {}

    # ── Algorithms with default-only kwargs ────────────────────────────
    kw = {}
    if algo == "EGO":
        kw["infill"] = p.get("infill", 3)
    elif algo == "FEP":
        pass
    elif algo == "FRCG":
        kw["grad_dims"] = p.get("grad_dims")
    elif algo == "FROFI":
        pass
    elif algo == "GPSO":
        return (p.get("w", 0.729), p.get("c1", 1.49445), p.get("c2", 1.49445),
                p.get("max_ve", 0.1), 0, popsize), {}
    elif algo == "GRASP":
        return (p.get("nst", 3), 0, popsize), {}
    elif algo == "KMA":
        kw["clusters"] = p.get("clusters", 3)
    elif algo == "L2SMEA":
        kw["group_size"] = p.get("group_size")
    elif algo == "MFEA":
        kw["tasks"] = p.get("tasks", 2)
    elif algo == "MFEA_II":
        kw["tasks"] = p.get("tasks", 2)
    elif algo == "MVPA":
        pass
    elif algo == "MiSACO":
        pass
    elif algo == "Nelder_Mead":
        kw["stepn"] = p.get("stepn", 4)
    elif algo == "SACC_EAM_II":
        kw["group_size"] = p.get("group_size")
    elif algo == "SACOSO":
        pass
    elif algo == "SD":
        kw["grad_dims"] = p.get("grad_dims")
    elif algo == "SHADE":
        pass
    elif algo == "SQP":
        kw["grad_dims"] = p.get("grad_dims")
    elif algo == "SSIO_RL":
        pass
    return (0, popsize), kw


# Map algorithm name → method name on MultiMet
_ALGO_TABLE: dict[str, str] = {
    "GA": "GA", "PSO": "PSO", "DE": "DE", "CSA": "CSA",
    "SA": "SA", "ACO": "ACO", "HS": "HS", "ILS": "ILS",
    "VNS": "VNS", "WOA": "WOA", "BA": "BA", "ABCA": "ABCA",
    "EGO": "EGO", "BATA": "BATA", "CMAES": "CMAES", "COA": "COA",
    "CSO": "CSO", "DOA": "DOA", "FA": "FA", "FEP": "FEP",
    "FRCG": "FRCG", "FROFI": "FROFI", "GPSO": "GPSO", "GRASP": "GRASP",
    "GWO": "GWO", "IMODE": "IMODE", "KMA": "KMA", "L2SMEA": "L2SMEA",
    "MFEA": "MFEA", "MFEA_II": "MFEA_II", "MGO": "MGO",
    "MVPA": "MVPA", "MiSACO": "MiSACO", "OFA": "OFA",
    "PBILC": "PBILC", "POA": "POA", "SACC_EAM_II": "SACC_EAM_II",
    "SACOSO": "SACOSO", "SADE_ATDSC": "SADE_ATDSC", "SAMSO": "SAMSO",
    "SAPO": "SAPO", "SD": "SD", "SHADE": "SHADE", "SQP": "SQP",
    "SSIO_RL": "SSIO_RL", "Nelder_Mead": "Nelder_Mead",
}

# Algorithms whose method signature is (Gen, MaxGen, p_start, p_end)
_GEN_BASED: set[str] = {
    "SA", "WOA", "DOA", "GWO", "IMODE", "MGO", "OFA",
    "SADE_ATDSC", "SAMSO", "SAPO",
}
