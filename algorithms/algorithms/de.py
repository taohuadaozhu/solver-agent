from __future__ import annotations

import random

from .archive import build_archive
from .surrogate import expected_improvement, kriging_fit, kriging_predict, quad_fit, quad_predict

class DEMixin:
    def differential_mutate(self, F: float, S: int, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            if S == 1:
                x0, x1 = random.sample(range(self.Popsize), 2)
                for j in range(self.Nvar):
                    value = self.gbest[j] + F * (self.ibest[x0][j] - self.ibest[x1][j])
                    self.newpop[i][j] = self._clip(value)
            elif S == 2:
                x0, x1, x2, x3, x4 = random.sample(range(self.Popsize), 5)
                for j in range(self.Nvar):
                    self.newpop[i][j] = min(
                        max(
                            self.ibest[x0][j]
                            + F * (self.ibest[x1][j] - self.ibest[x2][j])
                            + F * (self.ibest[x3][j] - self.ibest[x4][j]),
                            self.Lbound,
                        ),
                        self.Ubound,
                    )
            elif S == 3:
                x0, x1, x2, x3 = random.sample(range(self.Popsize), 4)
                for j in range(self.Nvar):
                    self.newpop[i][j] = min(
                        max(
                            self.gbest[j]
                            + F * (self.ibest[x0][j] - self.ibest[x1][j])
                            + F * (self.ibest[x2][j] - self.ibest[x3][j]),
                            self.Lbound,
                        ),
                        self.Ubound,
                    )
            elif S == 4:
                x0, x1 = random.sample(range(self.Popsize), 2)
                for j in range(self.Nvar):
                    self.newpop[i][j] = min(
                        max(
                            self.ibest[i][j]
                            + F * (self.gbest[j] - self.ibest[i][j])
                            + F * (self.ibest[x0][j] - self.ibest[x1][j]),
                            self.Lbound,
                        ),
                        self.Ubound,
                    )
            else:
                x0, x1, x2 = random.sample(range(self.Popsize), 3)
                for j in range(self.Nvar):
                    value = self.ibest[x0][j] + F * (
                        self.ibest[x1][j] - self.ibest[x2][j]
                    )
                    self.newpop[i][j] = self._clip(value)

    def differential_crossover(self, cr: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            d = random.randrange(self.Nvar)
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) > cr and j != d:
                    self.newpop[i][j] = self.pop[i][j]

    def DE(self, F: float, S: int, cr: float, p_start: int, p_end: int) -> None:
        self.differential_mutate(F, S, p_start, p_end)
        self.differential_crossover(cr, p_start, p_end)

    def _de_trial_row(self, index: int, F: float, S: int, cr: float) -> list[float]:
        if S == 1:
            x0, x1 = random.sample(range(self.Popsize), 2)
            mutant = [self.gbest[j] + F * (self.ibest[x0][j] - self.ibest[x1][j]) for j in range(self.Nvar)]
        elif S == 4:
            x0, x1 = random.sample(range(self.Popsize), 2)
            mutant = [self.ibest[index][j] + F * (self.gbest[j] - self.ibest[index][j]) + F * (self.ibest[x0][j] - self.ibest[x1][j]) for j in range(self.Nvar)]
        else:
            x0, x1, x2 = random.sample(range(self.Popsize), 3)
            mutant = [self.ibest[x0][j] + F * (self.ibest[x1][j] - self.ibest[x2][j]) for j in range(self.Nvar)]
        forced = random.randrange(self.Nvar)
        row = self.pop[index].copy()
        for j in range(self.Nvar):
            if j == forced or self.randval(0.0, 1.0) < cr:
                row[j] = self._clip(mutant[j])
        return row

    def DE_ARCHIVE_SURROGATE(
        self,
        F: float = 0.6,
        S: int = 1,
        cr: float = 0.8,
        p_start: int = 0,
        p_end: int | None = None,
        archive_strategy: str = "radius_density",
        surrogate_model: str = "kriging",
        pool_factor: int = 3,
        surrogate_dims: int = 12,
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        active = max(0, p_end - p_start)
        if active == 0:
            return
        known = [row.copy() for row in self.pop] + [row.copy() for row in self.ibest]
        known_fit = self.pop_fit.copy() + self.ibest_fit.copy()
        archive_size = min(len(known), max(self.Popsize, active * 2))
        archive, archive_fit = build_archive(known, known_fit, archive_size, archive_strategy)
        dims = list(range(self.Nvar))
        random.shuffle(dims)
        dims = dims[: max(1, min(self.Nvar, surrogate_dims))]
        archive_model = [[row[j] for j in dims] for row in archive]
        model_name = surrogate_model.lower()
        kriging = kriging_fit(archive_model, archive_fit) if model_name == "kriging" else None
        quad = quad_fit(archive_model, archive_fit) if model_name != "kriging" or kriging is None else None

        pool: list[list[float]] = []
        source_indices = list(range(p_start, p_end))
        for _ in range(max(1, pool_factor)):
            for idx in source_indices:
                pool.append(self._de_trial_row(idx, F, S, cr))
        scored: list[tuple[float, list[float]]] = []
        best = min(archive_fit) if archive_fit else min(self.pop_fit)
        for row in pool:
            model_row = [row[j] for j in dims]
            if kriging is not None:
                mean, mse = kriging_predict(model_row, kriging)
                score = -expected_improvement(best, mean, mse)
            elif quad is not None:
                score = quad_predict(model_row, quad)
            else:
                score = sum((row[j] - self.gbest[j]) ** 2 for j in range(self.Nvar))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0])
        for offset, i in enumerate(range(p_start, p_end)):
            row = scored[offset % len(scored)][1]
            self.newpop[i] = row.copy()
            self.newpop_fit[i] = self.EvaluFunc(row, self.Nvar)
