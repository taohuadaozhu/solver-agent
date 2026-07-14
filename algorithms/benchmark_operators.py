from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import queue
import time
from pathlib import Path

from .evaluate_methods import (
    EvalResult,
    PROBLEM_BOUNDS,
    PROBLEM_MAP,
    _run_operator,
    algorithm_operators,
    local_operators,
)


def _select_operator(group: str, name: str):
    operators = algorithm_operators() if group == "algorithms" else local_operators()
    for operator_name, operator in operators:
        if operator_name == name:
            return operator
    raise KeyError(f"unknown operator: {group}:{name}")


def _run_one(
    result_queue: mp.Queue,
    problem_name: str,
    group: str,
    operator_name: str,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    lb: float,
    ub: float,
) -> None:
    evaluate = PROBLEM_MAP[problem_name]
    operator = _select_operator(group, operator_name)
    result_queue.put(_run_operator(problem_name, evaluate, operator_name, operator, seed, nvar, popsize, generations, lb, ub))


def run_with_timeout(
    problem_name: str,
    group: str,
    operator_name: str,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    timeout: float,
) -> EvalResult:
    lb, ub = PROBLEM_BOUNDS[problem_name]
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_one,
        args=(result_queue, problem_name, group, operator_name, seed, nvar, popsize, generations, lb, ub),
    )
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    seconds = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        return EvalResult(problem_name, operator_name, float("nan"), float("nan"), generations, seconds, error="timeout")
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return EvalResult(problem_name, operator_name, float("nan"), float("nan"), generations, seconds, error="no result")


def write_csv(path: Path, rows: list[EvalResult], groups: list[str]) -> None:
    group_by_name = {}
    for group in groups:
        operators = algorithm_operators() if group == "algorithms" else local_operators()
        group_by_name.update({name: group for name, _ in operators})
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["problem", "group", "operator", "initial", "best", "improvement", "seconds", "status"])
        for row in rows:
            status = "OK" if not row.error and row.finite and row.in_bounds else row.error or "WARN"
            writer.writerow(
                [
                    row.problem,
                    group_by_name.get(row.name, ""),
                    row.name,
                    f"{row.initial:.12g}",
                    f"{row.best:.12g}",
                    f"{row.improvement:.12g}",
                    f"{row.seconds:.6f}",
                    status,
                ]
            )


def print_summary(rows: list[EvalResult], top: int) -> None:
    for problem in sorted({row.problem for row in rows}):
        print(f"\n[{problem}]")
        for group in ["algorithms", "local_search"]:
            group_rows = [
                row for row in rows
                if row.problem == problem and not row.error and row.finite and row.in_bounds
            ]
            names = {name for name, _ in (algorithm_operators() if group == "algorithms" else local_operators())}
            group_rows = [row for row in group_rows if row.name in names]
            group_rows.sort(key=lambda row: row.best)
            print(f"  {group} top {top}:")
            for row in group_rows[:top]:
                print(f"    {row.name:26s} best={row.best:.6g} improve={row.improvement:.3g} time={row.seconds:.4f}s")
        timeouts = [row.name for row in rows if row.problem == problem and row.error == "timeout"]
        if timeouts:
            print(f"  timeout: {', '.join(timeouts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeout-protected benchmark for evolutionary and local operators.")
    parser.add_argument("--problems", nargs="+", default=["rosenbrock", "griewank", "ackley", "rastrigin", "schwefel"], choices=sorted(PROBLEM_MAP))
    parser.add_argument("--groups", nargs="+", default=["algorithms", "local_search"], choices=["algorithms", "local_search"])
    parser.add_argument("--nvar", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--local-generations", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--output", type=Path, default=Path("/tmp/multialg_operator_benchmark.csv"))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    groups = args.groups
    rows: list[EvalResult] = []
    total = sum(len(algorithm_operators() if group == "algorithms" else local_operators()) for group in groups) * len(args.problems)
    done = 0
    for problem_index, problem_name in enumerate(args.problems):
        for group in groups:
            operators = algorithm_operators() if group == "algorithms" else local_operators()
            generations = args.generations if group == "algorithms" else args.local_generations
            for operator_index, (operator_name, _) in enumerate(operators):
                seed = args.seed + problem_index * 100000 + operator_index * 997 + (30000 if group == "local_search" else 0)
                result = run_with_timeout(problem_name, group, operator_name, seed, args.nvar, args.popsize, generations, args.timeout)
                rows.append(result)
                done += 1
                status = "OK" if not result.error and result.finite and result.in_bounds else result.error or "WARN"
                print(
                    f"{done:4d}/{total} {problem_name:13s} {group:12s} {operator_name:26s} "
                    f"{status:8s} best={result.best:.6g} time={result.seconds:.4f}s",
                    flush=True,
                )
    write_csv(args.output, rows, groups)
    print_summary(rows, args.top)
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
