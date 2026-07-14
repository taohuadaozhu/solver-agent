from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections.abc import Callable, Iterable
from typing import TextIO

from .multimethod import MultiMet
from .problems import Ackley, Griewank, Rosenbrock

ProblemFunc = Callable[[list[float], int], float]
ResultRow = tuple[str, str, str, int, float, float, float]

PROBLEMS: dict[str, ProblemFunc] = {
    "rosenbrock": Rosenbrock,
    "griewank": Griewank,
    "ackley": Ackley,
}

LOCAL_METHODS = [
    "elite_contraction",
    "coordinate_descent",
    "mcts_local_search",
    "elite_line_search",
    "multi_scale_gaussian",
    "random_subspace_pattern",
    "newton_subspace",
    "gradient_backtracking",
    "cauchy_basin_hop",
    "opposition_elite_blend",
]

ALL_METHODS = ["ADE", *LOCAL_METHODS]


def _refresh_best(solver: MultiMet) -> None:
    solver.worst_and_best()
    solver.Elist()


def _call_local(
    solver: MultiMet,
    method: str,
    index: int,
    iterations: int,
    scale: float,
) -> None:
    if method not in LOCAL_METHODS:
        raise ValueError(f"unknown local-search method: {method}")
    iter_count = iterations * 2 if method in {
        "coordinate_descent",
        "mcts_local_search",
        "multi_scale_gaussian",
    } else iterations
    getattr(solver, f"newpop_{method}")(index, iter_count, scale)


def _make_solver(problem: str, nvar: int, popsize: int) -> MultiMet:
    try:
        evaluator = PROBLEMS[problem]
    except KeyError as exc:
        raise ValueError(f"unknown problem: {problem}") from exc
    return MultiMet(popsize, nvar, -5.0, 10.0, evaluator)


def run_local_method(
    problem: str,
    method: str,
    nvar: int = 1000,
    popsize: int = 20,
    generations: int = 5,
    seed: int = 20260623,
    repeat: int = 0,
) -> ResultRow:
    random.seed(seed)
    solver = _make_solver(problem, nvar, popsize)
    start = time.perf_counter()
    solver.Initial()
    initial = solver.gbest_fit
    for gen in range(generations):
        progress = (gen + 1) / max(1, generations)
        scale = max(1e-6, 0.06 * (1.0 - progress) * (1.0 - progress) + 0.001)
        for index in range(popsize):
            _call_local(solver, method, index, 3, scale)
        solver.pop_better_update(0, popsize)
        _refresh_best(solver)
    seconds = time.perf_counter() - start
    return ("python_pure", problem, method, repeat, initial, solver.gbest_fit, seconds)


def run_ade_method(
    problem: str,
    nvar: int = 1000,
    popsize: int = 20,
    generations: int = 5,
    seed: int = 20260623,
    repeat: int = 0,
) -> ResultRow:
    random.seed(seed)
    solver = _make_solver(problem, nvar, popsize)
    start = time.perf_counter()
    solver.Initial()
    initial = solver.gbest_fit
    seri = [[0.0 for _ in range(solver.ade_seri_size)] for _ in range(popsize)]
    success_flags = [True for _ in range(popsize)]
    mf1 = [0.35 for _ in range(solver.ade_memory_size)]
    mf2 = [0.20 for _ in range(solver.ade_memory_size)]
    mcr = [0.85 for _ in range(solver.ade_memory_size)]
    memory_pos = 0
    stagnant = 0
    last_best = solver.gbest_fit
    solver.reset_seri_pool(seri, mf1, mf2, mcr, 0.0, 0)
    for gen in range(generations):
        active = solver.ade_active_population_size()
        solver.ADE(seri, 0, active)
        solver.Evaluation(True, 0, active)
        progress = (gen + 1) / max(1, generations)
        state = solver.search_state(progress, stagnant)
        stats = solver.collect_ade_trial_stats(seri, success_flags, None, 0, active)
        solver.update_ade_memory(mf1, mf2, mcr, memory_pos, stats)
        solver.update_seri_policy(state, stats)
        solver.share_seri_policy(state, 0.011 if stats.get("success_count", 0) > 0 else 0.014)
        for index in range(active):
            if not success_flags[index]:
                solver.adapt_successful_seri(seri[index], mf1, mf2, mcr, progress)
            else:
                solver.sample_seri(seri[index], mf1, mf2, mcr, progress, state)
        solver.pop_better_update(0, active)
        _refresh_best(solver)
        solver.update_shade_population_size(progress, stagnant, stats)
        if last_best - solver.gbest_fit > max(1e-12, abs(last_best) * 1e-10):
            last_best = solver.gbest_fit
            stagnant = 0
        else:
            stagnant += 1
    seconds = time.perf_counter() - start
    return ("python_pure", problem, "ADE", repeat, initial, solver.gbest_fit, seconds)


def run_method(
    problem: str,
    method: str,
    nvar: int = 1000,
    popsize: int = 20,
    generations: int = 5,
    seed: int = 20260623,
) -> ResultRow:
    if method == "ADE":
        return run_ade_method(problem, nvar, popsize, generations, seed)
    return run_local_method(problem, method, nvar, popsize, generations, seed)


def run_benchmark(
    nvar: int = 1000,
    popsize: int = 20,
    generations: int = 5,
    repeats: int = 1,
    problems: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    problem_names = list(problems or PROBLEMS)
    method_names = list(methods or ALL_METHODS)
    for problem in problem_names:
        for repeat in range(repeats):
            seed = 20260623 + repeat * 17
            for method in method_names:
                rows.append(run_method(problem, method, nvar, popsize, generations, seed))
    return rows


def write_csv(rows: Iterable[ResultRow], stream: TextIO = sys.stdout) -> None:
    writer = csv.writer(stream)
    writer.writerow(["runtime", "problem", "method", "repeat", "initial", "best", "seconds"])
    for row in rows:
        writer.writerow(row)


def rows_to_csv(rows: Iterable[ResultRow]) -> str:
    from io import StringIO

    stream = StringIO()
    write_csv(rows, stream)
    return stream.getvalue().rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure-Python ADE/local-search benchmark.")
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--problems", nargs="*", choices=sorted(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--methods", nargs="*", choices=ALL_METHODS, default=ALL_METHODS)
    args = parser.parse_args()
    write_csv(run_benchmark(args.nvar, args.popsize, args.generations, args.repeats, args.problems, args.methods))


if __name__ == "__main__":
    main()
