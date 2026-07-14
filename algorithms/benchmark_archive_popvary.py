from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from .evaluate_methods import PROBLEM_BOUNDS, PROBLEM_MAP, _finish_generation
from .multimethod import MultiMet


ARCHIVE_STRATEGIES = ["best", "mixed", "diverse", "random", "worst"]
POP_STRATEGIES = ["best", "rank_diverse", "tournament", "roulette"]


def run_ecpo_variant(
    problem_name: str,
    archive_strategy: str,
    pop_strategy: str,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
) -> dict[str, str | float]:
    random.seed(seed)
    lb, ub = PROBLEM_BOUNDS[problem_name]
    solver = MultiMet(popsize, nvar, lb, ub, PROBLEM_MAP[problem_name])
    solver.Initial()
    initial = solver.gbest_fit
    started = time.perf_counter()
    error = ""
    try:
        for _ in range(generations):
            solver.ECPO(
                2,
                3,
                max(1, round(popsize / 3)),
                0,
                solver.Popsize,
                archive_strategy=archive_strategy,
                pop_strategy=pop_strategy,
            )
            _finish_generation(solver, 0, solver.Popsize)
    except Exception as exc:
        error = repr(exc)
    seconds = time.perf_counter() - started
    return {
        "problem": problem_name,
        "archive": archive_strategy,
        "popvary": pop_strategy,
        "initial": initial,
        "best": solver.gbest_fit,
        "improvement": float("inf") if solver.gbest_fit <= 0 else initial / solver.gbest_fit,
        "seconds": seconds,
        "status": "OK" if not error else error,
    }


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["problem", "archive", "popvary", "initial", "best", "improvement", "seconds", "status"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | float]], top: int) -> None:
    for problem in sorted({str(row["problem"]) for row in rows}):
        ok_rows = [row for row in rows if row["problem"] == problem and row["status"] == "OK"]
        ok_rows.sort(key=lambda row: float(row["best"]))
        print(f"\n[{problem}] top {top}")
        for row in ok_rows[:top]:
            print(
                f"  {row['archive']}+{row['popvary']:12s} "
                f"best={float(row['best']):.6g} improve={float(row['improvement']):.3g} "
                f"time={float(row['seconds']):.4f}s"
            )
        base = next((row for row in ok_rows if row["archive"] == "best" and row["popvary"] == "best"), None)
        if base is not None:
            print(f"  baseline best+best: best={float(base['best']):.6g} time={float(base['seconds']):.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ECPO archive and variable-population mechanisms.")
    parser.add_argument("--problems", nargs="+", default=["rosenbrock", "griewank", "ackley", "rastrigin", "schwefel"], choices=sorted(PROBLEM_MAP))
    parser.add_argument("--archive-strategies", nargs="+", default=ARCHIVE_STRATEGIES, choices=ARCHIVE_STRATEGIES)
    parser.add_argument("--pop-strategies", nargs="+", default=POP_STRATEGIES, choices=POP_STRATEGIES)
    parser.add_argument("--nvar", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--output", type=Path, default=Path("/tmp/multialg_archive_popvary.csv"))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    total = len(args.problems) * len(args.archive_strategies) * len(args.pop_strategies)
    done = 0
    for problem_index, problem_name in enumerate(args.problems):
        for archive_index, archive_strategy in enumerate(args.archive_strategies):
            for pop_index, pop_strategy in enumerate(args.pop_strategies):
                seed = args.seed + problem_index * 100000 + archive_index * 1009 + pop_index * 97
                row = run_ecpo_variant(problem_name, archive_strategy, pop_strategy, seed, args.nvar, args.popsize, args.generations)
                rows.append(row)
                done += 1
                print(
                    f"{done:4d}/{total} {problem_name:13s} {archive_strategy:7s} {pop_strategy:12s} "
                    f"{row['status']:8s} best={float(row['best']):.6g} time={float(row['seconds']):.4f}s",
                    flush=True,
                )
    write_csv(args.output, rows)
    print_summary(rows, args.top)
    print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
