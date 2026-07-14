from __future__ import annotations

import argparse
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

from .multimethod import RunResult, run_csa


RunTask = tuple[int, int, int, int, int, float]


def _run_task(task: RunTask) -> tuple[int, int, RunResult]:
    offset, seed, nvar, popsize, maxgen, threshold = task
    result = run_csa(
        nvar=nvar,
        popsize=popsize,
        maxgen=maxgen,
        threshold=threshold,
        seed=seed,
    )
    return offset, seed, result


def _run_tasks(tasks: list[RunTask], workers: int) -> tuple[str, list[tuple[int, int, RunResult]]]:
    if workers <= 1 or len(tasks) <= 1:
        return "serial", [_run_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return "process", list(executor.map(_run_task, tasks))
    except (OSError, PermissionError):
        return "serial_fallback", [_run_task(task) for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated Python CSA quality/timing checks.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--maxgen", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker count; 0 uses available CPU cores.")
    args = parser.parse_args()

    tasks = [
        (offset, args.seed + offset, args.nvar, args.popsize, args.maxgen, args.threshold)
        for offset in range(args.runs)
    ]
    cpu_count = os.cpu_count() or 1
    workers = min(len(tasks), cpu_count) if args.workers == 0 else max(1, args.workers)
    backend, outputs = _run_tasks(tasks, workers)
    outputs.sort(key=lambda item: item[0])

    print(f"parallel_backend={backend} parallel_workers={workers}")
    results = []
    for offset, seed, result in outputs:
        results.append(result)
        print(
            f"run={offset + 1} seed={seed} "
            f"best={result.best:.12g} generations={result.generations} time={result.seconds:.6f}s",
            flush=True,
        )

    best_values = [result.best for result in results]
    times = [result.seconds for result in results]
    generations = [result.generations for result in results]
    print("summary")
    print(f"best_min={min(best_values):.12g}")
    print(f"best_median={statistics.median(best_values):.12g}")
    print(f"best_max={max(best_values):.12g}")
    print(f"time_median={statistics.median(times):.6f}s")
    print(f"generations_median={statistics.median(generations):.1f}")


if __name__ == "__main__":
    main()
