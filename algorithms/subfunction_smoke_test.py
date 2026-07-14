from __future__ import annotations

import math
import random

from .algorithms.archive import (
    archive_sample,
    archive_sample_indices,
    best_archive,
    build_archive,
    dual_archive,
    radius_density_archive,
    truncation_archive,
    update_external_archive,
)
from .algorithms.popvary import (
    adaptive_rmp,
    best_worst_pairs,
    elite_indices,
    evaluated_injection_popvary,
    fill_population,
    linear_population_size,
    merge_subspace_population,
    random_subspace_groups,
    select_population,
    subspace_population_maintenance,
)
from .algorithms.surrogate import (
    cubic_rbf_eval,
    cubic_rbf_fit,
    expected_improvement,
    kriging_fit,
    kriging_predict,
    lower_confidence_bound,
    pca_fit,
    pca_inverse,
    pca_transform,
    probability_improvement,
    quad_fit,
    quad_predict,
    solve_linear_system,
)
from .evaluate_methods import local_operators
from .multimethod import MultiMet
from .problems import Sphere


def _assert(condition: bool, name: str) -> None:
    if not condition:
        raise AssertionError(name)


def _finite_values(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _finite_rows(rows: list[list[float]]) -> bool:
    return all(_finite_values(row) for row in rows)


def _make_population() -> tuple[list[list[float]], list[float]]:
    rows = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    fits = [sum(value * value for value in row) for row in rows]
    fits[-1] = 0.25
    return rows, fits


def test_archive_helpers() -> None:
    rows, fits = _make_population()

    for name, func in (
        ("best_archive", best_archive),
        ("truncation_archive", truncation_archive),
        ("radius_density_archive", radius_density_archive),
    ):
        archive, archive_fit = func(rows, fits, 3)
        _assert(len(archive) == 3 and len(archive_fit) == 3, name)
        _assert(_finite_rows(archive) and _finite_values(archive_fit), name)

    for strategy in ("best", "worst", "random", "diverse", "truncation", "radius_density", "mixed"):
        archive, archive_fit = build_archive(rows, fits, 3, strategy)
        _assert(len(archive) == 3 and len(archive_fit) == 3, f"build_archive:{strategy}")

    convergence, convergence_fit, diversity, diversity_fit = dual_archive(rows, fits, 2, 2)
    _assert(len(convergence) == 2 and len(convergence_fit) == 2, "dual_archive:convergence")
    _assert(len(diversity) == 2 and len(diversity_fit) == 2, "dual_archive:diversity")

    updated, updated_fit = update_external_archive(rows[:2], fits[:2], rows[2:], fits[2:], 4)
    _assert(len(updated) == 4 and len(updated_fit) == 4, "update_external_archive")

    indices = archive_sample_indices(updated, 2)
    sample = archive_sample(updated, rows, 3)
    _assert(len(indices) == 2 and len(sample) == 3, "archive_sample")


def test_popvary_helpers() -> None:
    rows, fits = _make_population()

    selected, selected_fit = select_population(rows, fits, 3, "rank_diverse")
    _assert(len(selected) == 3 and len(selected_fit) == 3, "select_population")

    filled = fill_population([], 4, 3, -1.0, 1.0, random.uniform, "cycle")
    _assert(len(filled) == 4 and all(len(row) == 3 for row in filled), "fill_population:cycle_empty")
    _assert(_finite_rows(filled), "fill_population:finite")

    _assert(linear_population_size(10, 4, 5, 10) == 7, "linear_population_size")
    groups = random_subspace_groups(6, 2)
    covered = sorted(dim for group in groups for dim in group)
    _assert(covered == list(range(6)), "random_subspace_groups")
    _assert(len(best_worst_pairs(fits, 2)) == 2, "best_worst_pairs")
    _assert(elite_indices(fits, 2) == [0, 5], "elite_indices")
    _assert(0.1 <= adaptive_rmp(1.0, 2.0) <= 0.9, "adaptive_rmp")

    injected, injected_fit = evaluated_injection_popvary(rows, fits, [[0.0, 0.0, 0.0]], [0.0], 4)
    _assert(len(injected) == 4 and len(injected_fit) == 4, "evaluated_injection_popvary")
    _assert([0.0, 0.0, 0.0] in injected, "evaluated_injection_popvary:injected")

    population_groups = random_subspace_groups(3, 2)
    subspaces = subspace_population_maintenance(rows, fits, population_groups, 3)
    merged = merge_subspace_population(subspaces, 3, 3, fallback=[[0.0] * 3], lb=-1.0, ub=1.0)
    _assert(len(subspaces) == len(population_groups), "subspace_population_maintenance")
    _assert(len(merged) == 3 and all(len(row) == 3 for row in merged), "merge_subspace_population")


def test_surrogate_helpers() -> None:
    x = solve_linear_system([[2.0, 0.0], [0.0, 4.0]], [4.0, 8.0])
    _assert(x is not None and abs(x[0] - 2.0) < 1e-9 and abs(x[1] - 2.0) < 1e-9, "solve_linear_system")

    samples = [[-1.0], [0.0], [1.0], [2.0]]
    values = [row[0] * row[0] for row in samples]
    rbf = cubic_rbf_fit(samples, values)
    _assert(rbf is not None, "cubic_rbf_fit")
    lamb, gamma = rbf
    _assert(math.isfinite(cubic_rbf_eval([0.5], samples, lamb, gamma)), "cubic_rbf_eval")

    pca_rows = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    mean, basis, eigenvalues = pca_fit(pca_rows, 1)
    coords = pca_transform(pca_rows[2], mean, basis)
    inverse = pca_inverse(coords, mean, basis)
    _assert(len(mean) == 2 and len(basis) == 1 and len(eigenvalues) == 1, "pca_fit")
    _assert(len(inverse) == 2 and _finite_values(inverse), "pca_inverse")

    kriging_samples = [[-1.0, 0.0], [-0.5, 0.25], [0.0, 0.5], [0.5, 0.75], [1.0, 1.0]]
    kriging_values = [row[0] * row[0] + row[1] for row in kriging_samples]
    model = kriging_fit(kriging_samples, kriging_values, [1.0, 1.0])
    _assert(model is not None and model.ready, "kriging_fit")
    mean_value, mse = kriging_predict([0.0, 0.5], model)
    _assert(math.isfinite(mean_value) and math.isfinite(mse) and mse >= 0.0, "kriging_predict")

    ei = expected_improvement(0.5, 0.4, 0.01)
    lcb = lower_confidence_bound(0.4, 0.01, 2.0)
    pi = probability_improvement(0.5, 0.4, 0.01)
    _assert(math.isfinite(ei) and math.isfinite(lcb) and 0.0 <= pi <= 1.0, "acquisition")

    quad = quad_fit(kriging_samples, kriging_values)
    _assert(quad is not None and quad.ready, "quad_fit")
    _assert(math.isfinite(quad_predict([0.0, 0.5], quad)), "quad_predict")


def test_population_and_local_search_helpers() -> None:
    random.seed(20260624)
    solver = MultiMet(6, 4, -5.0, 5.0, Sphere)
    solver.Initial()
    solver.worst_and_best()
    solver.Elist()

    for name, operator in local_operators():
        random.seed(1000 + len(name))
        before = solver.gbest_fit
        operator(solver, 0, 1)
        solver.Evaluation(True, 0, solver.Popsize)
        solver.pop_update(0, solver.Popsize)
        solver.worst_and_best()
        solver.Elist()
        _assert(math.isfinite(solver.gbest_fit), f"local:{name}:finite")
        _assert(
            all(solver.Lbound <= value <= solver.Ubound for row in solver.pop + solver.newpop for value in row),
            f"local:{name}:bounds",
        )
        _assert(solver.gbest_fit <= before + 1e-9, f"local:{name}:state")


def main() -> None:
    random.seed(12345)
    test_archive_helpers()
    test_popvary_helpers()
    test_surrogate_helpers()
    test_population_and_local_search_helpers()
    print("OK,python_subfunctions")


if __name__ == "__main__":
    main()
