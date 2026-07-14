from __future__ import annotations

import argparse
import math
import random
import time
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


Operator = Callable[[MultiMet, int, int], None]

PROBLEM_MAP = {
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

PROBLEM_BOUNDS = {
    "sphere": (-100.0, 100.0),
    "rosenbrock": (-5.0, 10.0),
    "dixon_price": (-10.0, 10.0),
    "griewank": (-600.0, 600.0),
    "ackley": (-32.768, 32.768),
    "michalewicz": (0.0, math.pi),
    "rastrigin": (-5.12, 5.12),
    "schwefel": (-500.0, 500.0),
    "sum_squares": (-10.0, 10.0),
    "trid": (-10000.0, 10000.0),
    "zakharov": (-5.0, 10.0),
}


@dataclass
class EvalResult:
    problem: str
    name: str
    initial: float
    best: float
    generations: int
    seconds: float
    finite: bool = True
    in_bounds: bool = True
    error: str = ""

    @property
    def improvement(self) -> float:
        if self.best <= 0:
            return float("inf")
        return self.initial / self.best


def _finish_generation(solver: MultiMet, p_start: int, p_end: int) -> None:
    solver.Evaluation(True, p_start, p_end)
    solver.pop_update(p_start, p_end)
    solver.worst_and_best()
    solver.Elist()


def _run_operator(
    problem_name: str,
    evaluate: Callable[[list[float], int], float],
    name: str,
    operator: Operator,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    lb: float,
    ub: float,
) -> EvalResult:
    random.seed(seed)
    solver = MultiMet(popsize, nvar, lb, ub, evaluate)
    solver.Initial()
    initial = solver.gbest_fit
    started = time.perf_counter()
    try:
        for gen in range(generations):
            operator(solver, gen, generations)
            _finish_generation(solver, 0, popsize)
        finite = all(math.isfinite(value) for value in [*solver.pop_fit, *solver.newpop_fit, solver.gbest_fit])
        in_bounds = all(lb <= value <= ub for row in [*solver.pop, *solver.newpop] for value in row[:nvar])
        return EvalResult(problem_name, name, initial, solver.gbest_fit, generations, time.perf_counter() - started, finite, in_bounds)
    except Exception as exc:
        return EvalResult(problem_name, name, initial, solver.gbest_fit, generations, time.perf_counter() - started, True, True, repr(exc))


def algorithm_operators() -> list[tuple[str, Operator]]:
    return [
        ("GA", lambda s, g, m: s.GA(0.8, 0.15, 0, s.Popsize)),
        ("NGA", lambda s, g, m: s.NGA(0.8, 0.15, 1.0, 0, s.Popsize)),
        ("LGA", lambda s, g, m: s.LGA(0.8, 0.15, 0.1, 5, 0, s.Popsize)),
        ("VAGA", lambda s, g, m: s.VAGA(0, s.Popsize)),
        ("EAGA", lambda s, g, m: s.EAGA(0, s.Popsize)),
        ("PSO", lambda s, g, m: s.PSO(0.8, 2.0, 2.0, 0.1, 0, s.Popsize)),
        ("CPSO", lambda s, g, m: s.CPSO(0.8, 2.0, 2.0, 0.1, 0, s.Popsize)),
        ("SPSO", lambda s, g, m: s.SPSO(0.8, 2.0, 2.0, 0.1, g + 1, m, 0, s.Popsize)),
        ("CMPSO", lambda s, g, m: s.CMPSO(0.8, 2.0, 2.0, 0.1, g + 1, m, 0, s.Popsize)),
        ("APSO_1", lambda s, g, m: s.APSO_1(2.0, 2.0, 0.1, g + 1, m, 0, s.Popsize)),
        ("APSO_2", lambda s, g, m: s.APSO_2(2.0, 2.0, 0.1, 0, s.Popsize)),
        ("APSO_3", lambda s, g, m: s.APSO_3(0.1, 0, s.Popsize)),
        ("APSO_4", lambda s, g, m: s.APSO_4(0.1, g + 1, m, 0, s.Popsize)),
        ("APSO_5", lambda s, g, m: s.APSO_5(0.1, g + 1, m, 0, s.Popsize)),
        ("OLPSO", lambda s, g, m: s.OLPSO(0.8, 2.0, 0.1, 0, s.Popsize)),
        ("HS", lambda s, g, m: s.HS(0.9, 0.3, 0.1, 0, s.Popsize)),
        ("ACO", lambda s, g, m: s.ACO(0.85, 0, s.Popsize)),
        ("COA", lambda s, g, m: s.COA(10, 0, s.Popsize)),
        ("DE", lambda s, g, m: s.DE(0.8, 1, 0.5, 0, s.Popsize)),
        ("DE-Archive-Surrogate", lambda s, g, m: s.DE_ARCHIVE_SURROGATE(0.6, 1, 0.8, 0, s.Popsize, "radius_density", "quad")),
        ("ADE", lambda s, g, m: s.ADE(None, 0, s.Popsize)),
        ("AESSPSO", lambda s, g, m: s.AESSPSO(2.05, 2.05, 0, s.Popsize)),
        ("Adam", lambda s, g, m: s.Adam(0.1, 0.9, 0.999, 0, s.Popsize, min(4, s.Nvar))),
        ("AutoV", lambda s, g, m: s.AutoV(None, 0, s.Popsize)),
        ("BFGS", lambda s, g, m: s.BFGS(0.6, 0.4, 0, s.Popsize, min(4, s.Nvar))),
        ("CSO", lambda s, g, m: s.CSO(0.1, 0, s.Popsize)),
        ("DOA", lambda s, g, m: s.DOA(g + 1, m, 0, s.Popsize)),
        ("ECPO", lambda s, g, m: s.ECPO(2, 3, max(1, round(s.Popsize / 3)), 0, s.Popsize)),
        ("CSA", lambda s, g, m: s.CSA(0.3, 0, s.Popsize)),
        ("ABCA", lambda s, g, m: s.ABCA(100, 0, s.Popsize)),
        ("POA", lambda s, g, m: s.POA(0, s.Popsize)),
        ("ILS", lambda s, g, m: s.ILS(5, 0, s.Popsize)),
        ("VNS", lambda s, g, m: s.VNS(5, 0, s.Popsize)),
        ("GRASP", lambda s, g, m: s.GRASP(0.5, 5, 0, s.Popsize)),
        ("PBILC", lambda s, g, m: s.PBILC(0, s.Popsize, 0.015)),
        ("BATA", lambda s, g, m: s.BATA(g + 1, 0, s.Popsize)),
        ("FA", lambda s, g, m: s.FA(1.0 / (s.Nvar**0.5), 0.5, 0.5, max(1, m), 0, s.Popsize)),
        ("CMAES", lambda s, g, m: s.CMAES(g, 0, s.Popsize)),
        ("BA", lambda s, g, m: s.BA(0, s.Popsize)),
        ("EGO", lambda s, g, m: s.EGO(0, s.Popsize, 1)),
        ("FEP", lambda s, g, m: s.FEP(0, s.Popsize)),
        ("FRCG", lambda s, g, m: s.FRCG(0, s.Popsize, min(4, s.Nvar))),
        ("FROFI", lambda s, g, m: s.FROFI(0, s.Popsize)),
        ("GPSO", lambda s, g, m: s.GPSO(0.729, 1.49445, 1.49445, 0.1, 0, s.Popsize)),
        ("GWO", lambda s, g, m: s.GWO(g + 1, m, 0, s.Popsize)),
        ("IMODE", lambda s, g, m: s.IMODE(g + 1, m, 0, s.Popsize)),
        ("KMA", lambda s, g, m: s.KMA(0, s.Popsize)),
        ("L2SMEA", lambda s, g, m: s.L2SMEA(0, s.Popsize, max(1, min(32, int(s.Nvar**0.5) + 1)))),
        ("MFEA-II", lambda s, g, m: s.MFEA_II(0, s.Popsize, 2)),
        ("MFEA", lambda s, g, m: s.MFEA(0, s.Popsize, 2)),
        ("MGO", lambda s, g, m: s.MGO(g + 1, m, 0, s.Popsize)),
        ("MVPA", lambda s, g, m: s.MVPA(0, s.Popsize)),
        ("MiSACO", lambda s, g, m: s.MiSACO(0, s.Popsize)),
        ("Nelder-Mead", lambda s, g, m: s.Nelder_Mead(0, s.Popsize, 4)),
        ("OFA", lambda s, g, m: s.OFA(g + 1, m, 0, s.Popsize)),
        ("RMSProp", lambda s, g, m: s.RMSProp(0.05, 0.9, 0, s.Popsize, min(4, s.Nvar))),
        ("SACC-EAM-II", lambda s, g, m: s.SACC_EAM_II(0, s.Popsize, max(1, min(32, int(s.Nvar**0.5) + 1)))),
        ("SACOSO", lambda s, g, m: s.SACOSO(0, s.Popsize)),
        ("SA", lambda s, g, m: s.SA(g + 1, m, 0, s.Popsize)),
        ("SADE-AMSS", lambda s, g, m: s.SADE_AMSS(g + 1, m, 0, s.Popsize, min(8, s.Popsize), min(8, s.Nvar), 1)),
        ("SADE-AMSS-Orig", lambda s, g, m: s.SADE_AMSS_ORIG(g + 1, m, 0, s.Popsize, min(8, s.Popsize), min(8, s.Nvar), 1, max(s.Popsize, min(80, 3 * s.Popsize)))),
        ("SADE-ATDSC", lambda s, g, m: s.SADE_ATDSC(g + 1, m, 0, s.Popsize)),
        ("SADE-Sammon", lambda s, g, m: s.SADE_Sammon(0, s.Popsize, min(32, s.Nvar))),
        ("SAMSO", lambda s, g, m: s.SAMSO(g + 1, m, 0, s.Popsize)),
        ("SAPO", lambda s, g, m: s.SAPO(g + 1, m, 0, s.Popsize)),
        ("SD", lambda s, g, m: s.SD(0, s.Popsize, min(4, s.Nvar))),
        ("SHADE", lambda s, g, m: s.SHADE(0, s.Popsize)),
        ("SQP", lambda s, g, m: s.SQP(0, s.Popsize, min(4, s.Nvar))),
        ("SSIO-RL", lambda s, g, m: s.SSIO_RL(0, s.Popsize)),
        ("WOA", lambda s, g, m: s.WOA(g + 1, m, 0, s.Popsize)),
    ]


def local_operators() -> list[tuple[str, Operator]]:
    return [
        ("bit_climbing", lambda s, g, m: [s.newpop_bit_climbing(i, 5, 0.1) for i in range(s.Popsize)]),
        ("simplex", lambda s, g, m: [s.newpop_simplex(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("box_complex", lambda s, g, m: [s.newpop_box_complex(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("powell", lambda s, g, m: [s.newpop_powell(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("newton", lambda s, g, m: [s.newpop_newton(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("mcts", lambda s, g, m: [s.newpop_mcts(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("simulated_annealing", lambda s, g, m: [s.newpop_simulated_annealing(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("tabu_search", lambda s, g, m: [s.newpop_tabu_search(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("pattern_search", lambda s, g, m: [s.newpop_pattern_search(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("threshold_accepting", lambda s, g, m: [s.newpop_threshold_accepting(i, 5, 5, 0.1) for i in range(s.Popsize)]),
        ("elite_contraction", lambda s, g, m: [s.newpop_elite_contraction(i, 5, 0.1) for i in range(s.Popsize)]),
        ("coordinate_descent", lambda s, g, m: [s.newpop_coordinate_descent(i, 5, 0.1) for i in range(s.Popsize)]),
        ("mcts_local_search", lambda s, g, m: [s.newpop_mcts_local_search(i, 5, 0.1) for i in range(s.Popsize)]),
        ("elite_line_search", lambda s, g, m: [s.newpop_elite_line_search(i, 5, 0.1) for i in range(s.Popsize)]),
        ("multi_scale_gaussian", lambda s, g, m: [s.newpop_multi_scale_gaussian(i, 5, 0.1) for i in range(s.Popsize)]),
        ("random_subspace_pattern", lambda s, g, m: [s.newpop_random_subspace_pattern(i, 5, 0.1) for i in range(s.Popsize)]),
        ("newton_subspace", lambda s, g, m: [s.newpop_newton_subspace(i, 5, 0.1) for i in range(s.Popsize)]),
        ("gradient_backtracking", lambda s, g, m: [s.newpop_gradient_backtracking(i, 5, 0.1) for i in range(s.Popsize)]),
        ("cauchy_basin_hop", lambda s, g, m: [s.newpop_cauchy_basin_hop(i, 5, 0.1) for i in range(s.Popsize)]),
        ("opposition_elite_blend", lambda s, g, m: [s.newpop_opposition_elite_blend(i, 5, 0.1) for i in range(s.Popsize)]),
    ]


def meme_hyper_operators() -> list[tuple[str, Operator]]:
    return [
        ("meme_random_walk", lambda s, g, m: s.meme_random_walk(0.1)),
        ("meme_simple_random", lambda s, g, m: s.meme_simple_random(0.1)),
        ("meme_randperm", lambda s, g, m: s.meme_randperm(0.1)),
        ("meme_inheritance", lambda s, g, m: s.meme_inheritance(0.1)),
        ("meme_subprob_decomposition", lambda s, g, m: s.meme_subprob_decomposition(g, max(1, m // 2), max(1, s.Popsize // 4), 0.1)),
        ("meme_biasd_roulette", lambda s, g, m: s.meme_biasd_roulette(g, max(1, m // 2), 0.1)),
        ("meme_q_learning", lambda s, g, m: s.meme_q_learning(g, max(1, m), 0.1)),
        ("meme_biasd_roulette_generated_eo", lambda s, g, m: s.meme_biasd_roulette_generated_eo(g, max(1, m), 0.1)),
        ("meme_q_learning_generated_eo", lambda s, g, m: s.meme_q_learning_generated_eo(g, max(1, m), 0.1)),
    ]


def meme_operators() -> list[tuple[str, Operator]]:
    return meme_hyper_operators()


def run_group(
    problem_name: str,
    evaluate: Callable[[list[float], int], float],
    group: str,
    operators: list[tuple[str, Operator]],
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    repeats: int,
    lb: float,
    ub: float,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    print(f"[{problem_name}:{group}] nvar={nvar} popsize={popsize} generations={generations} repeats={repeats}")
    for name, operator in operators:
        runs = [
            _run_operator(problem_name, evaluate, name, operator, seed + repeat * 1009, nvar, popsize, generations, lb, ub)
            for repeat in range(repeats)
        ]
        errored = next((run for run in runs if run.error or not run.finite or not run.in_bounds), None)
        if errored is not None:
            result = errored
        else:
            runs.sort(key=lambda item: item.best)
            result = runs[len(runs) // 2]
            result.name = name
        results.append(result)
        status = "ERROR" if result.error else ("WARN" if not result.finite or not result.in_bounds else "OK")
        warn = ""
        if not result.finite:
            warn += " finite=false"
        if not result.in_bounds:
            warn += " in_bounds=false"
        print(
            f"{status:5s} {name:26s} initial={result.initial:.6g} "
            f"best={result.best:.6g} improve={result.improvement:.3g} "
            f"time={result.seconds:.4f}s"
            + warn
            + (f" error={result.error}" if result.error else ""),
            flush=True,
        )
    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate migrated MultiMet Python operators.")
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--local-generations", type=int, default=20)
    parser.add_argument("--meme-generations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--lb", type=float)
    parser.add_argument("--ub", type=float)
    parser.add_argument("--groups", nargs="+", default=["algorithms", "local_search", "meme"], choices=["algorithms", "local_search", "meme"])
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["rosenbrock", "griewank", "ackley"],
        choices=sorted(PROBLEM_MAP),
    )
    args = parser.parse_args()

    for offset, problem_name in enumerate(args.problems):
        evaluate = PROBLEM_MAP[problem_name]
        default_lb, default_ub = PROBLEM_BOUNDS[problem_name]
        lb = default_lb if args.lb is None else args.lb
        ub = default_ub if args.ub is None else args.ub
        if "algorithms" in args.groups:
            run_group(
                problem_name,
                evaluate,
                "algorithms",
                algorithm_operators(),
                args.seed + offset * 100000,
                args.nvar,
                args.popsize,
                args.generations,
                args.repeats,
                lb,
                ub,
            )
        if "local_search" in args.groups:
            run_group(
                problem_name,
                evaluate,
                "local_search",
                local_operators(),
                args.seed + 30000 + offset * 100000,
                args.nvar,
                args.popsize,
                args.local_generations,
                args.repeats,
                lb,
                ub,
            )
        if "meme" in args.groups:
            run_group(
                problem_name,
                evaluate,
                "meme",
                meme_operators(),
                args.seed + 60000 + offset * 100000,
                args.nvar,
                args.popsize,
                args.meme_generations,
                args.repeats,
                lb,
                ub,
            )


if __name__ == "__main__":
    main()
