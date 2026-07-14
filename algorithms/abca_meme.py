from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .multimethod import MultiMet
from .problems import (
    Ackley,
    Dixon_Price,
    Griewank,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Sum_Squares,
    Trid,
    Zakharov,
)


ProblemFunc = Callable[[list[float], int], float]
MemeFunc = Callable[[MultiMet, int, int, float], None]


PROBLEMS: dict[str, ProblemFunc] = {
    "sphere": Sphere,
    "rosenbrock": Rosenbrock,
    "dixon_price": Dixon_Price,
    "griewank": Griewank,
    "ackley": Ackley,
    "michalewicz": Michalewicz,
    "rastrigin": Rastrigin,
    "schwefel": Schwefel,
    "sum_squares": Sum_Squares,
    "trid": Trid,
    "zakharov": Zakharov,
}


def _meme_none(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    return None


def _meme_random_walk(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_random_walk(scale)


def _meme_simple_random(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_simple_random(scale)


def _meme_randperm(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_randperm(scale)


def _meme_inheritance(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_inheritance(scale)


def _meme_subprob_decomposition(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_subprob_decomposition(gen, max(1, maxgen // 2), max(1, solver.Popsize // 4), scale)


def _meme_biasd_roulette(solver: MultiMet, gen: int, maxgen: int, scale: float) -> None:
    solver.meme_biasd_roulette(gen, max(1, maxgen // 2), scale)


MEMES: dict[str, MemeFunc] = {
    "none": _meme_none,
    "random_walk": _meme_random_walk,
    "simple_random": _meme_simple_random,
    "randperm": _meme_randperm,
    "inheritance": _meme_inheritance,
    "subprob_decomposition": _meme_subprob_decomposition,
    "biasd_roulette": _meme_biasd_roulette,
}


@dataclass
class SolveResult:
    problem: str
    meme: str
    seed: int
    initial: float
    best: float
    generations: int
    seconds: float

    @property
    def improvement(self) -> float:
        if self.best <= 0:
            return float("inf")
        return self.initial / self.best


Task = tuple[int, int, int, str, str, int, int, int, int, float, float, int, float, int]


def run_abca_meme(
    problem_name: str,
    meme_name: str,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    lb: float,
    ub: float,
    limit: int,
    scale: float,
    meme_interval: int = 1,
) -> SolveResult:
    random.seed(seed)
    evaluate = PROBLEMS[problem_name]
    meme = MEMES[meme_name]
    solver = MultiMet(popsize, nvar, lb, ub, evaluate)
    solver.Initial()
    initial = solver.gbest_fit

    started = time.perf_counter()
    for gen in range(generations):
        solver.ABCA(limit, 0, popsize)
        if meme_interval > 0 and gen % meme_interval == 0:
            meme(solver, gen, generations, scale)
        solver.Evaluation(True, 0, popsize)
        solver.pop_update(0, popsize)
        solver.worst_and_best()
        solver.Elist()
    return SolveResult(
        problem=problem_name,
        meme=meme_name,
        seed=seed,
        initial=initial,
        best=solver.gbest_fit,
        generations=generations,
        seconds=time.perf_counter() - started,
    )


def _run_task(task: Task) -> tuple[int, int, int, SolveResult]:
    problem_index, meme_index, repeat, problem, meme, seed, nvar, popsize, generations, lb, ub, limit, scale, meme_interval = task
    result = run_abca_meme(problem, meme, seed, nvar, popsize, generations, lb, ub, limit, scale, meme_interval)
    return problem_index, meme_index, repeat, result


def _run_tasks(tasks: list[Task], workers: int) -> tuple[str, list[tuple[int, int, int, SolveResult]]]:
    if workers <= 1 or len(tasks) <= 1:
        return "serial", [_run_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return "process", list(executor.map(_run_task, tasks))
    except (OSError, PermissionError):
        return "serial_fallback", [_run_task(task) for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve benchmark functions with ABCA plus meme local search selection.")
    parser.add_argument("--problems", nargs="+", default=["sphere", "rosenbrock", "griewank", "ackley"], choices=sorted(PROBLEMS))
    parser.add_argument("--memes", nargs="+", default=["none", "random_walk", "simple_random", "randperm", "inheritance", "subprob_decomposition", "biasd_roulette"], choices=sorted(MEMES))
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--meme-interval", type=int, default=1, help="Apply meme local search every N generations; 0 disables meme local search.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--lb", type=float, default=-5.0)
    parser.add_argument("--ub", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker count; 0 uses available CPU cores.")
    args = parser.parse_args()

    tasks: list[Task] = []
    for problem_index, problem in enumerate(args.problems):
        for meme_index, meme in enumerate(args.memes):
            for repeat in range(args.repeats):
                tasks.append(
                    (
                        problem_index,
                        meme_index,
                        repeat,
                        problem,
                        meme,
                        args.seed + problem_index * 100000 + meme_index * 1000 + repeat * 37,
                        args.nvar,
                        args.popsize,
                        args.generations,
                        args.lb,
                        args.ub,
                        args.limit,
                        args.scale,
                        args.meme_interval,
                    )
                )

    cpu_count = os.cpu_count() or 1
    workers = min(len(tasks), cpu_count) if args.workers == 0 else max(1, args.workers)
    backend, results = _run_tasks(tasks, workers)
    grouped: dict[tuple[int, int], list[SolveResult]] = {}
    for problem_index, meme_index, _repeat, result in results:
        grouped.setdefault((problem_index, meme_index), []).append(result)

    print(f"parallel_backend={backend} parallel_workers={workers}")
    for problem_index, problem in enumerate(args.problems):
        print(
            f"[{problem}] ABCA + meme nvar={args.nvar} popsize={args.popsize} "
            f"generations={args.generations} repeats={args.repeats} meme_interval={args.meme_interval}"
        )
        for meme_index, meme in enumerate(args.memes):
            runs = grouped[(problem_index, meme_index)]
            bests = [run.best for run in runs]
            times = [run.seconds for run in runs]
            initials = [run.initial for run in runs]
            median_best = statistics.median(bests)
            median_time = statistics.median(times)
            median_initial = statistics.median(initials)
            improvement = float("inf") if median_best <= 0 else median_initial / median_best
            print(
                f"{meme:22s} initial={median_initial:.6g} best={median_best:.6g} "
                f"improve={improvement:.3g} time={median_time:.4f}s"
            )
        print()


if __name__ == "__main__":
    main()
