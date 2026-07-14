from __future__ import annotations

import math
import random

from .common import squared_distance


def _ordered_archive(
    pop: list[list[float]],
    fit: list[float],
    order: list[int],
    size: int,
) -> tuple[list[list[float]], list[float]]:
    picked = order[: min(size, len(order))]
    return [pop[idx].copy() for idx in picked], [fit[idx] for idx in picked]


def best_archive(pop: list[list[float]], fit: list[float], size: int) -> tuple[list[list[float]], list[float]]:
    return build_archive(pop, fit, size, "best")


def build_archive(
    pop: list[list[float]],
    fit: list[float],
    size: int,
    strategy: str = "best",
) -> tuple[list[list[float]], list[float]]:
    if size <= 0:
        return [], []
    strategy = strategy.lower()
    order = sorted(range(len(pop)), key=lambda idx: fit[idx])
    if strategy == "best":
        return _ordered_archive(pop, fit, order, size)
    if strategy == "worst":
        return _ordered_archive(pop, fit, list(reversed(order)), size)
    if strategy == "random":
        shuffled = list(range(len(pop)))
        random.shuffle(shuffled)
        return _ordered_archive(pop, fit, shuffled, size)
    if strategy == "diverse":
        return diverse_archive(pop, fit, size)
    if strategy in {"truncation", "spea2"}:
        return truncation_archive(pop, fit, size)
    if strategy in {"radius_density", "density_radius", "density"}:
        return radius_density_archive(pop, fit, size)
    if strategy == "mixed":
        best_size = max(1, size // 2)
        archive, archive_fit = _ordered_archive(pop, fit, order, best_size)
        if len(archive) >= size:
            return archive, archive_fit
        diverse, diverse_fit = diverse_archive(pop, fit, size)
        seen = {tuple(row) for row in archive}
        for row, value in zip(diverse, diverse_fit):
            key = tuple(row)
            if key not in seen:
                archive.append(row)
                archive_fit.append(value)
                seen.add(key)
            if len(archive) >= size:
                break
        return archive, archive_fit
    raise ValueError(f"unknown archive strategy: {strategy}")


def diverse_archive(pop: list[list[float]], fit: list[float], size: int) -> tuple[list[list[float]], list[float]]:
    if size <= 0:
        return [], []
    nvar = len(pop[0]) if pop else 0
    remaining = set(range(len(pop)))
    first = min(remaining, key=lambda idx: fit[idx])
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < size:
        best_idx = max(
            remaining,
            key=lambda idx: (
                min(squared_distance(pop[idx], pop[picked], nvar) for picked in selected),
                -fit[idx] if math.isfinite(fit[idx]) else float("-inf"),
            ),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [pop[idx].copy() for idx in selected], [fit[idx] for idx in selected]


def _unique_rows(pop: list[list[float]], fit: list[float]) -> tuple[list[list[float]], list[float]]:
    best_by_key: dict[tuple[float, ...], float] = {}
    row_by_key: dict[tuple[float, ...], list[float]] = {}
    for row, value in zip(pop, fit):
        key = tuple(row)
        if key not in best_by_key or value < best_by_key[key]:
            best_by_key[key] = value
            row_by_key[key] = row.copy()
    rows = list(row_by_key.values())
    fits = [best_by_key[tuple(row)] for row in rows]
    return rows, fits


def truncation_archive(pop: list[list[float]], fit: list[float], size: int) -> tuple[list[list[float]], list[float]]:
    rows, fits = _unique_rows(pop, fit)
    if size <= 0 or not rows:
        return [], []
    if len(rows) <= size:
        return [row.copy() for row in rows], fits.copy()
    nvar = len(rows[0])
    ranked = sorted(range(len(rows)), key=lambda idx: fits[idx])
    pool_count = min(len(rows), max(size, 2 * size))
    selected = ranked[:pool_count]
    best = selected[0]
    while len(selected) > size:
        remove_idx = None
        remove_key: tuple[float, ...] | None = None
        for idx in selected:
            if idx == best:
                continue
            distances = sorted(squared_distance(rows[idx], rows[other], nvar) for other in selected if other != idx)
            key = tuple(distances) + (-fits[idx],)
            if remove_key is None or key < remove_key:
                remove_key = key
                remove_idx = idx
        if remove_idx is None:
            remove_idx = max(selected, key=lambda idx: fits[idx])
        selected.remove(remove_idx)
    selected.sort(key=lambda idx: fits[idx])
    return [rows[idx].copy() for idx in selected], [fits[idx] for idx in selected]


def radius_density_archive(
    pop: list[list[float]],
    fit: list[float],
    size: int,
) -> tuple[list[list[float]], list[float]]:
    rows, fits = _unique_rows(pop, fit)
    if size <= 0 or not rows:
        return [], []
    if len(rows) <= size:
        return [row.copy() for row in rows], fits.copy()
    nvar = len(rows[0])
    ranked = sorted(range(len(rows)), key=lambda idx: fits[idx])
    selected = ranked[: min(len(rows), max(size, 3 * size))]
    best = selected[0]
    while len(selected) > size:
        distance_rows = []
        for idx in selected:
            ds = [squared_distance(rows[idx], rows[other], nvar) ** 0.5 for other in selected if other != idx]
            ds.sort()
            distance_rows.append(ds)
        kth = min(nvar if nvar > 0 else 1, len(selected) - 1)
        kth_values = [ds[kth - 1] for ds in distance_rows if len(ds) >= kth]
        radius = sorted(kth_values)[len(kth_values) // 2] if kth_values else 1.0
        radius = max(radius, 1e-12)
        worst_idx = None
        worst_score = -1.0
        worst_fit = -math.inf
        for pos, idx in enumerate(selected):
            if idx == best:
                continue
            product = 1.0
            for dist in distance_rows[pos]:
                product *= min(dist / radius, 1.0)
            density = 1.0 - product
            if density > worst_score or (abs(density - worst_score) <= 1e-12 and fits[idx] > worst_fit):
                worst_score = density
                worst_fit = fits[idx]
                worst_idx = idx
        if worst_idx is None:
            worst_idx = max(selected, key=lambda idx: fits[idx])
        selected.remove(worst_idx)
    selected.sort(key=lambda idx: fits[idx])
    return [rows[idx].copy() for idx in selected], [fits[idx] for idx in selected]


def archive_value(archive: list[list[float]], bit: int) -> float | None:
    if not archive:
        return None
    return random.choice(archive)[bit]


def update_external_archive(
    archive: list[list[float]],
    archive_fit: list[float],
    candidates: list[list[float]],
    candidate_fit: list[float],
    size: int,
    strategy: str = "mixed",
) -> tuple[list[list[float]], list[float]]:
    rows = [row.copy() for row in archive] + [row.copy() for row in candidates]
    fits = archive_fit.copy() + candidate_fit.copy()
    if not rows or size <= 0:
        return [], []
    return build_archive(rows, fits, min(size, len(rows)), strategy)


def dual_archive(
    pop: list[list[float]],
    fit: list[float],
    convergence_size: int,
    diversity_size: int,
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    convergence, convergence_fit = build_archive(pop, fit, convergence_size, "truncation")
    diversity, diversity_fit = build_archive(pop, fit, diversity_size, "radius_density")
    if diversity_size <= 0:
        return convergence, convergence_fit, [], []
    seen = {tuple(row) for row in convergence}
    extra_rows = [row for row, value in zip(pop, fit) if tuple(row) not in seen]
    extra_fit = [value for row, value in zip(pop, fit) if tuple(row) not in seen]
    if len(extra_rows) >= diversity_size:
        diversity, diversity_fit = radius_density_archive(extra_rows, extra_fit, diversity_size)
    elif extra_rows:
        diversity, diversity_fit = radius_density_archive(extra_rows + pop, extra_fit + fit, diversity_size)
    return convergence, convergence_fit, diversity, diversity_fit


def update_sade_amss_archive(
    archive: list[list[float]],
    archive_fit: list[float],
    candidates: list[list[float]],
    candidate_fit: list[float],
    size: int,
) -> tuple[list[list[float]], list[float]]:
    return update_external_archive(archive, archive_fit, candidates, candidate_fit, size, "mixed")


def archive_sample_indices(archive: list[list[float]], count: int) -> list[int]:
    if count <= 0 or not archive:
        return []
    order = list(range(len(archive)))
    random.shuffle(order)
    return order[: min(count, len(order))]


def archive_sample(
    archive: list[list[float]],
    fallback: list[list[float]],
    nvar: int,
) -> list[float]:
    source = archive if archive else fallback
    if not source:
        return [0.0 for _ in range(nvar)]
    return random.choice(source)[:nvar]
