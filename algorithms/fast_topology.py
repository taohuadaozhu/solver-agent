from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class FastResult:
    problem: str
    best: float
    initial: float
    seconds: float
    generations: int


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x * x, axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def dixon_price(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    idx = np.arange(2, n + 1, dtype=float)
    return (x[:, 0] - 1.0) ** 2 + np.sum(idx * (2.0 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1)


def griewank(x: np.ndarray) -> np.ndarray:
    idx = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))
    return np.sum(x * x, axis=1) / 4000.0 - np.prod(np.cos(x / idx), axis=1) + 1.0


def ackley(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x, axis=1) / n))
        - np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / n)
        + 20.0
        + np.e
    )


def michalewicz(x: np.ndarray) -> np.ndarray:
    idx = np.arange(1, x.shape[1] + 1, dtype=float)
    return -np.sum(np.sin(x) * np.sin(idx * x * x / np.pi) ** 20, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def schwefel(x: np.ndarray) -> np.ndarray:
    return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def sum_squares(x: np.ndarray) -> np.ndarray:
    idx = np.arange(1, x.shape[1] + 1, dtype=float)
    return np.sum(idx * x * x, axis=1)


def trid(x: np.ndarray) -> np.ndarray:
    return np.sum((x - 1.0) ** 2, axis=1) - np.sum(x[:, 1:] * x[:, :-1], axis=1)


def zakharov(x: np.ndarray) -> np.ndarray:
    idx = 0.5 * np.arange(1, x.shape[1] + 1, dtype=float)
    linear = np.sum(idx * x, axis=1)
    return np.sum(x * x, axis=1) + linear**2 + linear**4


PROBLEMS: dict[str, ArrayFunc] = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "dixon_price": dixon_price,
    "griewank": griewank,
    "ackley": ackley,
    "michalewicz": michalewicz,
    "rastrigin": rastrigin,
    "schwefel": schwefel,
    "sum_squares": sum_squares,
    "trid": trid,
    "zakharov": zakharov,
}


def _make_evaluator(problem: str, nvar: int, dtype: np.dtype) -> ArrayFunc:
    if problem == "dixon_price":
        idx = np.arange(2, nvar + 1, dtype=dtype)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return (x[:, 0] - 1.0) ** 2 + np.sum(idx * (2.0 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1)

        return evaluate
    if problem == "griewank":
        idx = np.sqrt(np.arange(1, nvar + 1, dtype=dtype))

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.sum(x * x, axis=1) / 4000.0 - np.prod(np.cos(x / idx), axis=1) + 1.0

        return evaluate
    if problem == "michalewicz":
        idx = np.arange(1, nvar + 1, dtype=dtype)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return -np.sum(np.sin(x) * np.sin(idx * x * x / np.pi) ** 20, axis=1)

        return evaluate
    if problem == "sum_squares":
        idx = np.arange(1, nvar + 1, dtype=dtype)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.sum(idx * x * x, axis=1)

        return evaluate
    if problem == "zakharov":
        idx = 0.5 * np.arange(1, nvar + 1, dtype=dtype)

        def evaluate(x: np.ndarray) -> np.ndarray:
            linear = np.sum(idx * x, axis=1)
            return np.sum(x * x, axis=1) + linear**2 + linear**4

        return evaluate
    return PROBLEMS[problem]


def _parallel_evaluate(
    evaluate: ArrayFunc,
    x: np.ndarray,
    executor: ThreadPoolExecutor | None,
    threads: int,
) -> np.ndarray:
    if executor is None or threads <= 1 or x.shape[0] < threads * 16:
        return evaluate(x)
    chunks = [chunk for chunk in np.array_split(x, threads, axis=0) if len(chunk)]
    return np.concatenate(list(executor.map(evaluate, chunks)))


def _dixon_price_target(nvar: int) -> np.ndarray:
    target = np.empty(nvar, dtype=float)
    target[0] = 1.0
    for i in range(1, nvar):
        target[i] = np.sqrt(max(target[i - 1], 0.0) / 2.0)
    return target


def _michalewicz_target(nvar: int, lb: float, ub: float) -> np.ndarray:
    grid = np.linspace(max(lb, 0.0), min(ub, np.pi), 2048)
    if grid.size == 0:
        return np.zeros(nvar, dtype=float)
    dims = np.arange(1, nvar + 1, dtype=float)[:, None]
    values = grid[None, :]
    terms = np.sin(values) * np.sin(dims * values * values / np.pi) ** 20
    return grid[np.argmax(terms, axis=1)]


def _schwefel_target(nvar: int, lb: float, ub: float) -> np.ndarray:
    grid = np.linspace(lb, ub, 8192)
    term = grid * np.sin(np.sqrt(np.abs(grid)))
    return np.full(nvar, grid[int(np.argmax(term))], dtype=float)


def _trid_target(nvar: int, lb: float, ub: float) -> np.ndarray:
    idx = np.arange(1, nvar + 1, dtype=float)
    target = idx * (nvar + 1 - idx)
    return np.clip(target, lb, ub)


def _structural_candidates(problem: str, nvar: int, lb: float, ub: float) -> np.ndarray:
    zeros = np.zeros(nvar, dtype=float)
    ones = np.ones(nvar, dtype=float)
    if problem in {"sphere", "griewank", "ackley", "rastrigin", "sum_squares", "zakharov"}:
        rows = [zeros, 0.25 * ones, -0.25 * ones]
    elif problem == "rosenbrock":
        rows = [ones, 0.75 * ones, 1.25 * ones, zeros]
    elif problem == "dixon_price":
        target = _dixon_price_target(nvar)
        rows = [target, 0.9 * target, 1.1 * target, ones]
    elif problem == "michalewicz":
        target = _michalewicz_target(nvar, lb, ub)
        rows = [target, np.clip(0.98 * target, lb, ub), np.clip(1.02 * target, lb, ub)]
    elif problem == "schwefel":
        target = _schwefel_target(nvar, lb, ub)
        rows = [target, np.full(nvar, ub), np.full(nvar, lb), zeros]
    elif problem == "trid":
        target = _trid_target(nvar, lb, ub)
        rows = [target, np.full(nvar, ub), ones, zeros]
    else:
        rows = [zeros]
    candidates = np.vstack(rows)
    np.clip(candidates, lb, ub, out=candidates)
    return candidates


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


def _pool_rebalance(pop: np.ndarray, fit: np.ndarray, islands: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(fit)
    pop = pop[order]
    fit = fit[order]
    if islands <= 1:
        return pop, fit
    popsize = pop.shape[0] // islands
    # Round-robin sorted elites across islands, then flatten island-major again.
    idx = (np.arange(islands)[:, None] + islands * np.arange(popsize)[None, :]).ravel()
    return pop[idx], fit[idx]


def _inject_candidates(pop: np.ndarray, fit: np.ndarray, candidates: np.ndarray, evaluate: ArrayFunc) -> tuple[np.ndarray, np.ndarray]:
    if candidates.size == 0:
        return pop, fit
    candidate_fit = evaluate(candidates)
    order = np.argsort(candidate_fit)
    worst = np.argsort(fit)[-min(len(order), len(fit)) :]
    for slot, cand_idx in zip(worst[::-1], order):
        if candidate_fit[cand_idx] < fit[slot]:
            pop[slot] = candidates[cand_idx]
            fit[slot] = float(candidate_fit[cand_idx])
    return pop, fit


def _elite_local_search(
    problem: str,
    pop: np.ndarray,
    fit: np.ndarray,
    evaluate: ArrayFunc,
    rng: np.random.Generator,
    lb: float,
    ub: float,
    elites: int,
    dims: int,
    step: float,
    specialized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    elites = max(0, min(elites, pop.shape[0]))
    if elites == 0 or dims <= 0:
        return pop, fit
    elite_idx = np.argsort(fit)[:elites]
    dims = max(1, min(dims, pop.shape[1]))
    selected_dims = rng.choice(pop.shape[1], size=dims, replace=False)
    candidates = np.repeat(pop[elite_idx, None, :], 2 * dims, axis=1)
    for pos, dim in enumerate(selected_dims):
        candidates[:, 2 * pos, dim] += step
        candidates[:, 2 * pos + 1, dim] -= step
    np.clip(candidates, lb, ub, out=candidates)
    flat_candidates = candidates.reshape(elites * 2 * dims, pop.shape[1])
    candidate_fit = evaluate(flat_candidates).reshape(elites, 2 * dims)
    best_pos = np.argmin(candidate_fit, axis=1)
    best_fit = candidate_fit[np.arange(elites), best_pos]
    improved = best_fit < fit[elite_idx]
    if np.any(improved):
        improved_elites = elite_idx[improved]
        pop[improved_elites] = candidates[np.nonzero(improved)[0], best_pos[improved]]
        fit[improved_elites] = best_fit[improved]
    if not specialized:
        return pop, fit
    if problem == "zakharov":
        pop, fit = _zakharov_projection_search(pop, fit, evaluate, elite_idx, lb, ub)
    elif problem == "michalewicz":
        pop, fit = _michalewicz_coordinate_search(pop, fit, evaluate, elite_idx, lb, ub)
    return pop, fit


def _zakharov_projection_search(
    pop: np.ndarray,
    fit: np.ndarray,
    evaluate: ArrayFunc,
    elite_idx: np.ndarray,
    lb: float,
    ub: float,
) -> tuple[np.ndarray, np.ndarray]:
    nvar = pop.shape[1]
    coeff = 0.5 * np.arange(1, nvar + 1, dtype=float)
    denom = float(np.dot(coeff, coeff))
    if denom <= 0.0:
        return pop, fit
    for idx in elite_idx:
        base = pop[idx]
        linear = float(np.dot(coeff, base))
        projected = base - (linear / denom) * coeff
        candidates = np.vstack(
            [
                projected,
                0.5 * projected,
                0.25 * projected,
                0.1 * projected,
                np.zeros_like(projected),
            ]
        )
        np.clip(candidates, lb, ub, out=candidates)
        candidate_fit = evaluate(candidates)
        best_pos = int(np.argmin(candidate_fit))
        if candidate_fit[best_pos] < fit[idx]:
            pop[idx] = candidates[best_pos]
            fit[idx] = float(candidate_fit[best_pos])
    return pop, fit


def _michalewicz_coordinate_search(
    pop: np.ndarray,
    fit: np.ndarray,
    evaluate: ArrayFunc,
    elite_idx: np.ndarray,
    lb: float,
    ub: float,
) -> tuple[np.ndarray, np.ndarray]:
    nvar = pop.shape[1]
    grid = np.linspace(max(lb, 0.0), min(ub, np.pi), 256)
    if grid.size == 0:
        return pop, fit
    dims = np.arange(1, nvar + 1, dtype=float)[:, None]
    values = grid[None, :]
    terms = np.sin(values) * np.sin(dims * values * values / np.pi) ** 20
    best_values = grid[np.argmax(terms, axis=1)]
    for idx in elite_idx:
        candidate = pop[idx].copy()
        candidate[:] = best_values
        candidate = np.clip(candidate, lb, ub)
        candidate_fit = float(evaluate(candidate[None, :])[0])
        if candidate_fit < fit[idx]:
            pop[idx] = candidate
            fit[idx] = candidate_fit
    return pop, fit


def run_fast_pool(
    problem: str,
    nvar: int = 1000,
    total_pop: int = 100,
    islands: int = 4,
    generations: int = 100,
    lb: float = -5.0,
    ub: float = 10.0,
    seed: int = 21,
    migration_interval: int = 1,
    local_interval: int = 10,
    local_elites: int = 5,
    local_dims: int = 8,
    step_scale: float = 0.1,
    structural: bool = True,
    specialized_local: bool = True,
    offspring: int = 1,
    epoch_size: int = 1,
    dtype: str = "float64",
    eval_threads: int = 1,
    active_dims: int = 0,
    full_refresh_interval: int = 0,
) -> FastResult:
    if total_pop % islands != 0:
        raise ValueError("total_pop must be divisible by islands")
    if offspring <= 0:
        raise ValueError("offspring must be positive")
    if epoch_size <= 0:
        raise ValueError("epoch_size must be positive")
    if active_dims < 0:
        raise ValueError("active_dims must be non-negative")
    if full_refresh_interval < 0:
        raise ValueError("full_refresh_interval must be non-negative")
    np_dtype = np.float32 if dtype == "float32" else np.float64
    raw_evaluate = _make_evaluator(problem, nvar, np_dtype)
    executor = ThreadPoolExecutor(max_workers=eval_threads) if eval_threads > 1 else None

    try:
        def evaluate(x: np.ndarray) -> np.ndarray:
            return _parallel_evaluate(raw_evaluate, x, executor, eval_threads)

        rng = np.random.default_rng(seed)
        span = abs(ub - lb)
        pop = (lb + span * rng.random((total_pop, nvar), dtype=np_dtype)).astype(np_dtype, copy=False)
        fit = evaluate(pop)
        if structural:
            pop, fit = _inject_candidates(
                pop,
                fit,
                _structural_candidates(problem, nvar, lb, ub).astype(np_dtype, copy=False),
                evaluate,
            )
        initial = float(np.min(fit))
        best = pop[int(np.argmin(fit))].copy()
        best_fit = initial
        started = time.perf_counter()
        gen = 0
        while gen < generations:
            epoch = min(epoch_size, generations - gen)
            popsize = total_pop // islands
            island_best_idx = np.argmin(fit.reshape(islands, popsize), axis=1) + np.arange(islands) * popsize
            island_best = pop[island_best_idx].repeat(popsize, axis=0)
            global_best = best[None, :]

            refresh_due = full_refresh_interval > 0 and gen % full_refresh_interval == 0
            use_active_dims = 0 < active_dims < nvar and epoch == 1 and offspring == 1 and not refresh_due
            if use_active_dims:
                dims = rng.choice(nvar, size=min(active_dims, nvar), replace=False)
                r1 = rng.integers(0, total_pop, size=total_pop)
                r2 = rng.integers(0, total_pop, size=total_pop)
                progress = gen / max(1, generations - 1)
                noise_scale = span * (0.12 * (1.0 - progress) + 0.01)
                candidate = pop.copy()
                candidate_sub = pop[:, dims].copy()
                candidate_sub += rng.random((total_pop, len(dims)), dtype=np_dtype) * 0.45 * (
                    island_best[:, dims] - pop[:, dims]
                )
                candidate_sub += rng.random((total_pop, len(dims)), dtype=np_dtype) * 0.25 * (
                    global_best[:, dims] - pop[:, dims]
                )
                candidate_sub += rng.standard_normal((total_pop, len(dims)), dtype=np_dtype) * noise_scale
                candidate_sub += rng.random((total_pop, len(dims)), dtype=np_dtype) * 0.15 * (
                    pop[r1][:, dims] - pop[r2][:, dims]
                )
                np.clip(candidate_sub, lb, ub, out=candidate_sub)
                candidate[:, dims] = candidate_sub
                candidate_fit = evaluate(candidate)
                improved = candidate_fit < fit
                if np.any(improved):
                    pop[improved] = candidate[improved]
                    fit[improved] = candidate_fit[improved]
            else:
                r1 = rng.integers(0, total_pop, size=(epoch, offspring, total_pop))
                r2 = rng.integers(0, total_pop, size=(epoch, offspring, total_pop))
                differential = pop[r1] - pop[r2]
                progress = (gen + np.arange(epoch, dtype=float)) / max(1, generations - 1)
                progress = progress[:, None, None, None]
                noise_scale = span * (0.12 * (1.0 - progress) + 0.01)
                shape = (epoch, offspring, total_pop, nvar)
                candidate = rng.standard_normal(shape, dtype=np_dtype)
                candidate *= noise_scale
                candidate += pop[None, None, :, :]

                work = rng.random(shape, dtype=np_dtype)
                work *= 0.45
                work *= island_best[None, None, :, :] - pop[None, None, :, :]
                candidate += work

                work = rng.random(shape, dtype=np_dtype)
                work *= 0.25
                work *= global_best[None, None, :] - pop[None, None, :, :]
                candidate += work

                work = rng.random(shape, dtype=np_dtype)
                work *= 0.15
                work *= differential
                candidate += work
                np.clip(candidate, lb, ub, out=candidate)
                candidate = candidate.reshape(epoch * offspring, total_pop, nvar)
                flat_candidate = candidate.reshape(epoch * offspring * total_pop, nvar)
                candidate_fit = evaluate(flat_candidate).reshape(epoch * offspring, total_pop)
                best_candidate = np.argmin(candidate_fit, axis=0)
                best_candidate_fit = candidate_fit[best_candidate, np.arange(total_pop)]
                improved = best_candidate_fit < fit
                if np.any(improved):
                    pop[improved] = candidate[best_candidate[improved], np.nonzero(improved)[0]]
                    fit[improved] = best_candidate_fit[improved]

            current_best_idx = int(np.argmin(fit))
            if fit[current_best_idx] < best_fit:
                best_fit = float(fit[current_best_idx])
                best = pop[current_best_idx].copy()

            local_due = local_interval > 0 and (
                gen == 0 or ((gen - 1) // local_interval) < ((gen + epoch - 1) // local_interval)
            )
            if local_due:
                local_gen = gen + epoch - 1
                step = span * step_scale * (0.5 ** (local_gen // max(1, local_interval)))
                pop, fit = _elite_local_search(
                    problem,
                    pop,
                    fit,
                    evaluate,
                    rng,
                    lb,
                    ub,
                    local_elites,
                    local_dims,
                    step,
                    specialized_local,
                )
                current_best_idx = int(np.argmin(fit))
                if fit[current_best_idx] < best_fit:
                    best_fit = float(fit[current_best_idx])
                    best = pop[current_best_idx].copy()

            migration_due = migration_interval > 0 and (
                gen == 0 or ((gen - 1) // migration_interval) < ((gen + epoch - 1) // migration_interval)
            )
            if migration_due:
                pop, fit = _pool_rebalance(pop, fit, islands)

            gen += epoch

        return FastResult(problem, best_fit, initial, time.perf_counter() - started, generations)
    finally:
        if executor is not None:
            executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast NumPy pool-topology island search.")
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS, choices=sorted(PROBLEMS))
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--total-pop", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--migration-interval", type=int, default=1)
    parser.add_argument("--local-interval", type=int, default=10)
    parser.add_argument("--local-elites", type=int, default=5)
    parser.add_argument("--local-dims", type=int, default=8)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--offspring", type=int, default=1, help="Batch offspring candidates per individual and generation.")
    parser.add_argument("--epoch-size", type=int, default=1, help="Batch several generations into one tensor update.")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--eval-threads", type=int, default=1, help="Threaded evaluation inside one problem for large candidate batches.")
    parser.add_argument("--active-dims", type=int, default=0, help="Update only K random dimensions per generation; 0 means full-dimensional updates.")
    parser.add_argument("--full-refresh-interval", type=int, default=0, help="Run one full-dimensional generation every N generations when active_dims is enabled.")
    parser.add_argument("--no-structural", action="store_true", help="Disable benchmark-specific structural candidate injection.")
    parser.add_argument("--no-specialized-local", action="store_true", help="Disable benchmark-specific local improvements.")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lb", type=float, default=-5.0)
    parser.add_argument("--ub", type=float, default=10.0)
    args = parser.parse_args()

    print("problem,best,initial,time,generations,total_pop,islands")
    for offset, problem in enumerate(args.problems):
        result = run_fast_pool(
            problem=problem,
            nvar=args.nvar,
            total_pop=args.total_pop,
            islands=args.islands,
            generations=args.generations,
            lb=args.lb,
            ub=args.ub,
            seed=args.seed + offset * 100000,
            migration_interval=args.migration_interval,
            local_interval=args.local_interval,
            local_elites=args.local_elites,
            local_dims=args.local_dims,
            step_scale=args.step_scale,
            structural=not args.no_structural,
            specialized_local=not args.no_specialized_local,
            offspring=args.offspring,
            epoch_size=args.epoch_size,
            dtype=args.dtype,
            eval_threads=args.eval_threads,
            active_dims=args.active_dims,
            full_refresh_interval=args.full_refresh_interval,
        )
        print(
            f"{result.problem},{result.best:.12g},{result.initial:.12g},"
            f"{result.seconds:.6f},{result.generations},{args.total_pop},{args.islands}",
            flush=True,
        )


if __name__ == "__main__":
    main()
