from __future__ import annotations

import argparse

from .abca_meme import MEMES, PROBLEMS, run_abca_meme
from .island_meme import run_island_abca_meme


DEFAULT_PROBLEMS = [
    "rosenbrock",
    "dixon_price",
    "griewank",
    "ackley",
    "michalewicz",
    "rastrigin",
    "schwefel",
    "sum_squares",
    "trid",
    "zakharov",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare serial ABCA+meme and parallel island-model ABCA+meme.")
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS, choices=sorted(PROBLEMS))
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20, help="Backward-compatible alias for serial popsize.")
    parser.add_argument("--serial-popsize", type=int, default=None)
    parser.add_argument("--island-popsize", type=int, default=None)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--serial-meme", default="randperm", choices=sorted(MEMES))
    parser.add_argument("--meme-interval", type=int, default=20)
    parser.add_argument("--island-memes", nargs="+", default=["randperm", "inheritance", "simple_random", "biasd_roulette"], choices=sorted(MEMES))
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--migration-interval", type=int, default=20)
    parser.add_argument("--meme-elites", type=int, default=8)
    parser.add_argument("--migrants", type=int, default=1)
    parser.add_argument("--topology", choices=["ring", "global", "pool"], default="ring")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lb", type=float, default=-5.0)
    parser.add_argument("--ub", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()
    serial_popsize = args.serial_popsize if args.serial_popsize is not None else args.popsize
    island_popsize = args.island_popsize if args.island_popsize is not None else max(1, serial_popsize // args.islands)

    print(
        "problem,serial_best,parallel_best,delta_serial_minus_parallel,"
        "serial_time,parallel_time,speedup_serial_over_parallel,"
        "serial_total_pop,parallel_total_pop"
    )
    for offset, problem in enumerate(args.problems):
        seed = args.seed + offset * 100000
        serial = run_abca_meme(
            problem,
            args.serial_meme,
            seed,
            args.nvar,
            serial_popsize,
            args.generations,
            args.lb,
            args.ub,
            args.limit,
            args.scale,
            args.meme_interval,
        )
        parallel = run_island_abca_meme(
            problem_name=problem,
            meme_name=args.serial_meme,
            seed=seed,
            nvar=args.nvar,
            popsize=island_popsize,
            generations=args.generations,
            islands=args.islands,
            migration_interval=args.migration_interval,
            migrants=args.migrants,
            topology=args.topology,
            workers=args.workers,
            lb=args.lb,
            ub=args.ub,
            limit=args.limit,
            scale=args.scale,
            meme_names=args.island_memes,
            persistent=True,
            meme_interval=args.meme_interval,
            meme_elites=args.meme_elites,
        )
        speedup = serial.seconds / parallel.seconds if parallel.seconds > 0 else float("inf")
        delta = serial.best - parallel.best
        print(
            f"{problem},{serial.best:.12g},{parallel.best:.12g},{delta:.12g},"
            f"{serial.seconds:.6f},{parallel.seconds:.6f},{speedup:.6f},"
            f"{serial_popsize},{island_popsize * args.islands}",
            flush=True,
        )


if __name__ == "__main__":
    main()
