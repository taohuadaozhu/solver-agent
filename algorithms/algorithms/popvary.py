from __future__ import annotations

from collections.abc import Callable
import random

from .archive import radius_density_archive, truncation_archive
from .common import squared_distance


def select_best(
    candidates: list[list[float]],
    fits: list[float],
    size: int,
) -> tuple[list[list[float]], list[float]]:
    return select_population(candidates, fits, size, strategy="best")


def select_population(
    candidates: list[list[float]],
    fits: list[float],
    size: int,
    strategy: str = "best",
    nvar: int | None = None,
) -> tuple[list[list[float]], list[float]]:
    if size <= 0 or not candidates:
        return [], []
    strategy = strategy.lower()
    if strategy == "best":
        order = sorted(range(len(candidates)), key=lambda idx: fits[idx])[:size]
    elif strategy == "tournament":
        order = _tournament_order(fits, size)
    elif strategy == "roulette":
        order = _roulette_order(fits, size)
    elif strategy == "rank_diverse":
        order = _rank_diverse_order(candidates, fits, size, nvar)
    else:
        raise ValueError(f"unknown population selection strategy: {strategy}")
    return [candidates[idx].copy() for idx in order], [fits[idx] for idx in order]


def _tournament_order(fits: list[float], size: int) -> list[int]:
    selected: list[int] = []
    available = set(range(len(fits)))
    while available and len(selected) < size:
        contestants = random.sample(list(available), min(3, len(available)))
        winner = min(contestants, key=lambda idx: fits[idx])
        selected.append(winner)
        available.remove(winner)
    return selected


def _roulette_order(fits: list[float], size: int) -> list[int]:
    worst = max(fits)
    weights = [max(1e-12, worst - value + 1e-12) for value in fits]
    available = list(range(len(fits)))
    selected: list[int] = []
    while available and len(selected) < size:
        total = sum(weights[idx] for idx in available)
        cutoff = random.random() * total
        acc = 0.0
        picked = available[-1]
        for idx in available:
            acc += weights[idx]
            if acc >= cutoff:
                picked = idx
                break
        selected.append(picked)
        available.remove(picked)
    return selected


def _rank_diverse_order(
    candidates: list[list[float]],
    fits: list[float],
    size: int,
    nvar: int | None,
) -> list[int]:
    nvar = nvar or len(candidates[0])
    ranked = sorted(range(len(candidates)), key=lambda idx: fits[idx])
    selected = ranked[:1]
    pool = ranked[1 : max(2, min(len(ranked), 3 * size))]
    while pool and len(selected) < size:
        picked = max(
            pool,
            key=lambda idx: min(
                squared_distance(candidates[idx], candidates[j], nvar)
                for j in selected
            ),
        )
        selected.append(picked)
        pool.remove(picked)
    if len(selected) < size:
        selected.extend(idx for idx in ranked if idx not in selected)
    return selected[:size]


def fill_population(
    candidates: list[list[float]],
    size: int,
    nvar: int,
    lb: float,
    ub: float,
    randval: Callable[[float, float], float],
    strategy: str = "random",
) -> list[list[float]]:
    rows = [row[:nvar] for row in candidates[:size]]
    strategy = strategy.lower()
    while len(rows) < size:
        if strategy == "random" or not rows:
            rows.append([randval(lb, ub) for _ in range(nvar)])
        elif strategy == "cycle":
            source = candidates if candidates else rows
            rows.append(source[len(rows) % len(source)][:nvar].copy())
        elif strategy == "mutate_best":
            base = rows[0]
            scale = 0.05 * (ub - lb)
            rows.append([min(ub, max(lb, base[j] + randval(-scale, scale))) for j in range(nvar)])
        else:
            raise ValueError(f"unknown population fill strategy: {strategy}")
    return rows


def linear_population_size(initial: int, minimum: int, generation: int, max_generation: int) -> int:
    if max_generation <= 0:
        return max(minimum, initial)
    progress = min(1.0, max(0.0, generation / max_generation))
    return max(minimum, int(round(initial - progress * (initial - minimum))))


def random_subspace_groups(nvar: int, group_size: int) -> list[list[int]]:
    dims = list(range(nvar))
    random.shuffle(dims)
    group_size = max(1, min(nvar, group_size))
    return [dims[start : start + group_size] for start in range(0, nvar, group_size)]


def best_worst_pairs(fits: list[float], size: int) -> list[tuple[int, int]]:
    order = sorted(range(len(fits)), key=lambda idx: fits[idx])
    pairs = []
    for offset in range(min(size, len(order) // 2)):
        pairs.append((order[offset], order[-offset - 1]))
    return pairs


def elite_indices(fits: list[float], count: int) -> list[int]:
    count = max(1, min(len(fits), count))
    return sorted(range(len(fits)), key=lambda idx: fits[idx])[:count]


def adaptive_rmp(success: float, trials: float, low: float = 0.1, high: float = 0.9) -> float:
    rate = success / max(1.0, trials)
    return min(high, max(low, low + (high - low) * rate))


def evaluated_injection_popvary(
    population: list[list[float]],
    fit: list[float],
    evaluated: list[list[float]],
    evaluated_fit: list[float],
    size: int,
    strategy: str = "truncation",
) -> tuple[list[list[float]], list[float]]:
    if size <= 0:
        return [], []
    injected = [(row.copy(), value) for row, value in zip(evaluated, evaluated_fit)]
    injected_keys = {tuple(row) for row, _ in injected}
    old_rows: list[list[float]] = []
    old_fit: list[float] = []
    seen = set(injected_keys)
    for row, value in zip(population, fit):
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        old_rows.append(row.copy())
        old_fit.append(value)
    keep_size = max(0, size - len(injected))
    if strategy.lower() in {"radius_density", "density"}:
        kept_rows, kept_fit = radius_density_archive(old_rows, old_fit, keep_size)
    else:
        kept_rows, kept_fit = truncation_archive(old_rows, old_fit, keep_size)
    rows = kept_rows + [row for row, _ in injected]
    fits = kept_fit + [value for _, value in injected]
    order = sorted(range(len(rows)), key=lambda idx: fits[idx])[:size]
    return [rows[idx].copy() for idx in order], [fits[idx] for idx in order]


def subspace_population_maintenance(
    population: list[list[float]],
    fit: list[float],
    groups: list[list[int]],
    size: int,
    strategy: str = "rank_diverse",
) -> list[tuple[list[int], list[list[float]], list[float]]]:
    maintained: list[tuple[list[int], list[list[float]], list[float]]] = []
    for group in groups:
        if not group:
            continue
        projected = [[row[j] for j in group] for row in population]
        selected, selected_fit = select_population(projected, fit, size, strategy, len(group))
        maintained.append((group.copy(), selected, selected_fit))
    return maintained


def merge_subspace_population(
    subspaces: list[tuple[list[int], list[list[float]], list[float]]],
    nvar: int,
    size: int,
    fallback: list[list[float]] | None = None,
    lb: float | None = None,
    ub: float | None = None,
) -> list[list[float]]:
    rows = [[0.0 for _ in range(nvar)] for _ in range(size)]
    if fallback:
        for i in range(size):
            rows[i] = fallback[i % len(fallback)][:nvar]
    filled = [[False for _ in range(nvar)] for _ in range(size)]
    for group, subrows, _subfit in subspaces:
        if not subrows:
            continue
        for i in range(size):
            subrow = subrows[i % len(subrows)]
            for offset, dim in enumerate(group):
                if 0 <= dim < nvar and offset < len(subrow):
                    value = subrow[offset]
                    if lb is not None:
                        value = max(lb, value)
                    if ub is not None:
                        value = min(ub, value)
                    rows[i][dim] = value
                    filled[i][dim] = True
    if fallback:
        for i in range(size):
            for j in range(nvar):
                if not filled[i][j]:
                    rows[i][j] = fallback[i % len(fallback)][j]
    return rows
