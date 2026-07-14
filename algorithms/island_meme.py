from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from .abca_meme import MEMES, PROBLEMS
from .multimethod import MultiMet


@dataclass
class IslandState:
    island_id: int
    solver: MultiMet
    rng_state: Any
    meme_name: str
    generations_done: int = 0


@dataclass
class IslandRunResult:
    problem: str
    meme: str
    seed: int
    islands: int
    generations: int
    migrations: int
    initial: float
    best: float
    seconds: float
    backend: str
    island_bests: list[float]

    @property
    def improvement(self) -> float:
        if self.best <= 0:
            return float("inf")
        return self.initial / self.best


EpochTask = tuple[IslandState, int, int, float, int, int, int]
EliteRows = list[tuple[list[float], float]]


def _finish_generation(solver: MultiMet) -> None:
    solver.Evaluation(True, 0, solver.Popsize)
    solver.pop_update(0, solver.Popsize)
    solver.worst_and_best()
    solver.Elist()


def _selected_meme_indices(solver: MultiMet, meme_elites: int) -> list[int]:
    if meme_elites >= solver.Popsize:
        return list(range(solver.Popsize))
    count = max(1, min(meme_elites, solver.Popsize))
    return sorted(range(solver.Popsize), key=lambda idx: solver.pop_fit[idx])[:count]


def _apply_meme(solver: MultiMet, meme_name: str, generation: int, total_generations: int, scale: float, meme_elites: int) -> None:
    if meme_elites <= 0 or meme_name == "none":
        return
    if meme_elites >= solver.Popsize:
        MEMES[meme_name](solver, generation, total_generations, scale)
        return

    indices = _selected_meme_indices(solver, meme_elites)
    op_count = solver._local_search_count()
    if meme_name == "randperm":
        permu = list(range(op_count))
        random.shuffle(permu)
        for offset, idx in enumerate(indices):
            solver.meme_selection(idx, permu[offset % op_count], scale, 10)
    elif meme_name == "inheritance":
        for idx in indices:
            solver.meme_selection(idx, solver.Ind_meme[idx], scale, 10)
        for left, right in zip(indices[0::2], indices[1::2]):
            if solver.newpop_fit[left] <= solver.newpop_fit[right]:
                solver.Ind_meme[right] = solver.Ind_meme[left]
            else:
                solver.Ind_meme[left] = solver.Ind_meme[right]
    elif meme_name == "random_walk":
        step = generation % op_count
        for idx in indices:
            solver.meme_selection(idx, step, scale, 10)
            step = (step + (1 if solver.randval(0.0, 1.0) >= 0.5 else -1)) % op_count
    else:
        for idx in indices:
            solver.meme_selection(idx, random.randrange(op_count), scale, 10)


def _run_island_epoch(task: EpochTask) -> IslandState:
    state, epoch_generations, total_generations, scale, limit, meme_interval, meme_elites = task
    random.setstate(state.rng_state)
    solver = state.solver

    for offset in range(epoch_generations):
        generation = state.generations_done + offset
        solver.ABCA(limit, 0, solver.Popsize)
        if meme_interval > 0 and generation % meme_interval == 0:
            _apply_meme(solver, state.meme_name, generation, total_generations, scale, meme_elites)
        _finish_generation(solver)

    state.generations_done += epoch_generations
    state.rng_state = random.getstate()
    return state


def _run_epoch(tasks: list[EpochTask], workers: int) -> tuple[str, list[IslandState]]:
    if workers <= 1 or len(tasks) <= 1:
        return "serial", [_run_island_epoch(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return "process", list(executor.map(_run_island_epoch, tasks))
    except (OSError, PermissionError):
        return "serial_fallback", [_run_island_epoch(task) for task in tasks]


def _elite_rows(solver: MultiMet, count: int) -> EliteRows:
    count = max(1, min(count, solver.Popsize))
    order = sorted(range(solver.Popsize), key=lambda idx: solver.pop_fit[idx])
    return [(solver.pop[idx].copy(), solver.pop_fit[idx]) for idx in order[:count]]


def _inject_migrant(solver: MultiMet, values: list[float], fit: float) -> None:
    target = max(range(solver.Popsize), key=lambda idx: solver.pop_fit[idx])
    migrant = values.copy()
    solver.pop[target] = migrant.copy()
    solver.pop_fit[target] = fit
    solver.newpop[target] = migrant.copy()
    solver.newpop_fit[target] = fit
    solver.ibest[target] = migrant.copy()
    solver.ibest_fit[target] = fit
    if hasattr(solver, "trial"):
        solver.trial[target] = 0
    if fit < solver.gbest_fit:
        solver.gbest = migrant.copy()
        solver.gbest_fit = fit
    solver.worst_and_best()


def _migration_plan_from_rows(outbound: list[EliteRows], migrants: int, topology: str) -> tuple[list[EliteRows], int]:
    inbound: list[EliteRows] = [[] for _ in outbound]
    if len(outbound) <= 1 or migrants <= 0:
        return inbound, 0

    if topology == "ring":
        for source, rows in enumerate(outbound):
            target = (source + 1) % len(outbound)
            for values, fit in rows:
                inbound[target].append((values, fit))
        return inbound, sum(len(rows) for rows in inbound)

    if topology == "global":
        pooled: list[tuple[int, list[float], float]] = []
        for source, rows in enumerate(outbound):
            for values, fit in rows:
                pooled.append((source, values, fit))
        pooled.sort(key=lambda row: row[2])
        for target in range(len(outbound)):
            accepted = 0
            for source, values, fit in pooled:
                if source == target:
                    continue
                inbound[target].append((values, fit))
                accepted += 1
                if accepted >= migrants:
                    break
        return inbound, sum(len(rows) for rows in inbound)

    raise ValueError(f"unsupported topology: {topology}")


def _migrate(states: list[IslandState], migrants: int, topology: str) -> int:
    if topology == "pool":
        return _pool_rebalance(states)
    outbound = [_elite_rows(state.solver, migrants) for state in states]
    inbound, sent = _migration_plan_from_rows(outbound, migrants, topology)
    for target, rows in enumerate(inbound):
        for values, fit in rows:
            _inject_migrant(states[target].solver, values, fit)
    return sent


def _set_individual(solver: MultiMet, idx: int, values: list[float], fit: float) -> None:
    row = values.copy()
    solver.pop[idx] = row.copy()
    solver.pop_fit[idx] = fit
    solver.newpop[idx] = row.copy()
    solver.newpop_fit[idx] = fit
    solver.ibest[idx] = row.copy()
    solver.ibest_fit[idx] = fit
    if hasattr(solver, "trial"):
        solver.trial[idx] = 0


def _pool_rebalance(states: list[IslandState]) -> int:
    if len(states) <= 1:
        return 0
    pool: list[tuple[list[float], float]] = []
    for state in states:
        solver = state.solver
        pool.extend((solver.pop[idx].copy(), solver.pop_fit[idx]) for idx in range(solver.Popsize))
    pool.sort(key=lambda item: item[1])

    islands = len(states)
    popsize = states[0].solver.Popsize
    for rank, (values, fit) in enumerate(pool[: islands * popsize]):
        target = rank % islands
        slot = rank // islands
        _set_individual(states[target].solver, slot, values, fit)

    for state in states:
        solver = state.solver
        solver.worst_and_best()
        best_idx = solver.cur_best
        if solver.pop_fit[best_idx] < solver.gbest_fit:
            solver.gbest = solver.pop[best_idx].copy()
            solver.gbest_fit = solver.pop_fit[best_idx]
    return len(pool)


def _make_islands(
    problem_name: str,
    meme_names: list[str],
    seed: int,
    islands: int,
    popsize: int,
    nvar: int,
    lb: float,
    ub: float,
) -> list[IslandState]:
    evaluate = PROBLEMS[problem_name]
    states: list[IslandState] = []
    for island_id in range(islands):
        random.seed(seed + island_id * 100003)
        solver = MultiMet(popsize, nvar, lb, ub, evaluate)
        solver.Initial()
        states.append(
            IslandState(
                island_id=island_id,
                solver=solver,
                rng_state=random.getstate(),
                meme_name=meme_names[island_id % len(meme_names)],
            )
        )
    return states


@dataclass(frozen=True)
class PersistentIslandConfig:
    island_id: int
    problem_name: str
    meme_name: str
    seed: int
    nvar: int
    popsize: int
    lb: float
    ub: float
    limit: int
    scale: float
    meme_elites: int


def _persistent_worker(conn: Any, config: PersistentIslandConfig) -> None:
    try:
        random.seed(config.seed)
        solver = MultiMet(config.popsize, config.nvar, config.lb, config.ub, PROBLEMS[config.problem_name])
        solver.Initial()
        generations_done = 0
        conn.send(("ready", config.island_id, solver.gbest_fit))

        while True:
            command = conn.recv()
            name = command[0]
            if name == "run":
                epoch_generations, total_generations, meme_interval = command[1], command[2], command[3]
                for offset in range(epoch_generations):
                    generation = generations_done + offset
                    solver.ABCA(config.limit, 0, solver.Popsize)
                    if meme_interval > 0 and generation % meme_interval == 0:
                        _apply_meme(
                            solver,
                            config.meme_name,
                            generation,
                            total_generations,
                            config.scale,
                            config.meme_elites,
                        )
                    _finish_generation(solver)
                generations_done += epoch_generations
                conn.send(("done", config.island_id, generations_done, solver.gbest_fit))
            elif name == "export":
                migrants = command[1]
                conn.send(("elite", config.island_id, _elite_rows(solver, migrants), solver.gbest_fit))
            elif name == "inject":
                rows = command[1]
                for values, fit in rows:
                    _inject_migrant(solver, values, fit)
                conn.send(("injected", config.island_id, solver.gbest_fit))
            elif name == "stop":
                conn.send(("stopped", config.island_id, solver.gbest_fit))
                break
            else:
                raise ValueError(f"unsupported worker command: {name}")
    except BaseException as exc:
        conn.send(("error", config.island_id, repr(exc)))
    finally:
        conn.close()


def _recv_checked(conn: Any, expected: str) -> tuple[Any, ...]:
    message = conn.recv()
    if message[0] == "error":
        raise RuntimeError(f"island {message[1]} failed: {message[2]}")
    if message[0] != expected:
        raise RuntimeError(f"expected {expected}, got {message[0]}")
    return message


def _run_island_persistent(
    problem_name: str,
    meme_names: list[str],
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    islands: int,
    migration_interval: int,
    migrants: int,
    topology: str,
    workers: int,
    lb: float,
    ub: float,
    limit: int,
    scale: float,
    meme_interval: int,
    meme_elites: int,
) -> tuple[str, float, float, int, list[float], float]:
    context = mp.get_context("spawn")
    process_count = min(max(1, workers), islands)
    if process_count < islands:
        raise ValueError("persistent island backend requires workers >= islands")

    pipes = []
    procs = []
    started = time.perf_counter()
    try:
        for island_id in range(islands):
            parent_conn, child_conn = context.Pipe()
            config = PersistentIslandConfig(
                island_id=island_id,
                problem_name=problem_name,
                meme_name=meme_names[island_id % len(meme_names)],
                seed=seed + island_id * 100003,
                nvar=nvar,
                popsize=popsize,
                lb=lb,
                ub=ub,
                limit=limit,
                scale=scale,
                meme_elites=meme_elites,
            )
            proc = context.Process(target=_persistent_worker, args=(child_conn, config))
            proc.start()
            child_conn.close()
            pipes.append(parent_conn)
            procs.append(proc)

        initial = min(_recv_checked(conn, "ready")[2] for conn in pipes)
        generation = 0
        migrations = 0
        island_bests = [initial for _ in range(islands)]

        while generation < generations:
            epoch_generations = min(max(1, migration_interval), generations - generation)
            for conn in pipes:
                conn.send(("run", epoch_generations, generations, meme_interval))
            done = [_recv_checked(conn, "done") for conn in pipes]
            island_bests = [message[3] for message in sorted(done, key=lambda item: item[1])]
            generation += epoch_generations
            if generation >= generations:
                break

            for conn in pipes:
                conn.send(("export", migrants))
            exported = [_recv_checked(conn, "elite") for conn in pipes]
            outbound = [message[2] for message in sorted(exported, key=lambda item: item[1])]
            inbound, sent = _migration_plan_from_rows(outbound, migrants, topology)
            migrations += sent
            for conn, rows in zip(pipes, inbound):
                conn.send(("inject", rows))
            injected = [_recv_checked(conn, "injected") for conn in pipes]
            island_bests = [message[2] for message in sorted(injected, key=lambda item: item[1])]

        for conn in pipes:
            conn.send(("stop",))
        stopped = [_recv_checked(conn, "stopped") for conn in pipes]
        island_bests = [message[2] for message in sorted(stopped, key=lambda item: item[1])]
        return "persistent_process", initial, min(island_bests), migrations, island_bests, time.perf_counter() - started
    finally:
        for conn in pipes:
            conn.close()
        for proc in procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


def run_island_abca_meme(
    problem_name: str,
    meme_name: str,
    seed: int,
    nvar: int,
    popsize: int,
    generations: int,
    islands: int,
    migration_interval: int,
    migrants: int,
    topology: str,
    workers: int,
    lb: float,
    ub: float,
    limit: int,
    scale: float,
    meme_names: list[str] | None = None,
    persistent: bool = True,
    meme_interval: int = 1,
    meme_elites: int | None = None,
) -> IslandRunResult:
    selected_memes = meme_names or [meme_name]
    if not selected_memes:
        selected_memes = [meme_name]
    for selected in selected_memes:
        if selected not in MEMES:
            raise ValueError(f"unsupported meme: {selected}")
    meme_label = ",".join(selected_memes)
    local_meme_elites = popsize if meme_elites is None else max(0, min(meme_elites, popsize))

    if persistent and topology != "pool" and workers >= islands and islands > 1:
        try:
            backend, initial, best, migrations, island_bests, seconds = _run_island_persistent(
                problem_name,
                selected_memes,
                seed,
                nvar,
                popsize,
                generations,
                islands,
                migration_interval,
                migrants,
                topology,
                workers,
                lb,
                ub,
                limit,
                scale,
                meme_interval,
                local_meme_elites,
            )
            return IslandRunResult(
                problem=problem_name,
                meme=meme_label,
                seed=seed,
                islands=islands,
                generations=generations,
                migrations=migrations,
                initial=initial,
                best=best,
                seconds=seconds,
                backend=backend,
                island_bests=island_bests,
            )
        except (OSError, PermissionError):
            pass

    states = _make_islands(problem_name, selected_memes, seed, islands, popsize, nvar, lb, ub)
    initial = min(state.solver.gbest_fit for state in states)
    generation = 0
    migrations = 0
    backend = "serial"
    started = time.perf_counter()

    while generation < generations:
        epoch_generations = min(max(1, migration_interval), generations - generation)
        tasks = [
            (state, epoch_generations, generations, scale, limit, meme_interval, local_meme_elites)
            for state in states
        ]
        epoch_backend, states = _run_epoch(tasks, workers)
        backend = epoch_backend if backend == "serial" else backend
        generation += epoch_generations
        if generation < generations:
            migrations += _migrate(states, migrants, topology)

    island_bests = [state.solver.gbest_fit for state in states]
    return IslandRunResult(
        problem=problem_name,
        meme=meme_label,
        seed=seed,
        islands=islands,
        generations=generations,
        migrations=migrations,
        initial=initial,
        best=min(island_bests),
        seconds=time.perf_counter() - started,
        backend=backend,
        island_bests=island_bests,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve benchmark functions with an island-model ABCA + meme coevolution flow.")
    parser.add_argument("--problem", default="ackley", choices=sorted(PROBLEMS))
    parser.add_argument("--meme", default="randperm", choices=sorted(MEMES))
    parser.add_argument("--island-memes", nargs="+", choices=sorted(MEMES), help="Optional heterogeneous meme cycle across islands.")
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20, help="Population size per island.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--migration-interval", type=int, default=20)
    parser.add_argument("--meme-interval", type=int, default=10, help="Apply meme local search every N generations; 0 disables meme local search.")
    parser.add_argument("--meme-elites", type=int, default=8, help="Apply meme local search only to the best K individuals per island.")
    parser.add_argument("--migrants", type=int, default=1)
    parser.add_argument("--topology", choices=["ring", "global", "pool"], default="ring")
    parser.add_argument("--workers", type=int, default=0, help="Parallel worker count; 0 uses min(islands, CPU cores).")
    parser.add_argument("--no-persistent", action="store_true", help="Disable persistent island worker processes.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--lb", type=float, default=-5.0)
    parser.add_argument("--ub", type=float, default=10.0)
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    workers = min(args.islands, cpu_count) if args.workers == 0 else max(1, args.workers)
    result = run_island_abca_meme(
        problem_name=args.problem,
        meme_name=args.meme,
        seed=args.seed,
        nvar=args.nvar,
        popsize=args.popsize,
        generations=args.generations,
        islands=args.islands,
        migration_interval=args.migration_interval,
        migrants=args.migrants,
        topology=args.topology,
        workers=workers,
        lb=args.lb,
        ub=args.ub,
        limit=args.limit,
        scale=args.scale,
        meme_names=args.island_memes,
        persistent=not args.no_persistent,
        meme_interval=args.meme_interval,
        meme_elites=args.meme_elites,
    )

    print(
        f"island_backend={result.backend} workers={workers} topology={args.topology} "
        f"islands={args.islands} migrants={args.migrants} migration_interval={args.migration_interval} "
        f"meme_interval={args.meme_interval} meme_elites={args.meme_elites}"
    )
    print(
        f"[{result.problem}] ABCA + {result.meme} island_model nvar={args.nvar} "
        f"popsize_per_island={args.popsize} generations={result.generations}"
    )
    print(
        f"initial={result.initial:.6g} best={result.best:.6g} "
        f"improve={result.improvement:.3g} migrations={result.migrations} "
        f"time={result.seconds:.4f}s"
    )
    print("island_bests=" + " ".join(f"{value:.6g}" for value in result.island_bests))


if __name__ == "__main__":
    main()
