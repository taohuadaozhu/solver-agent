from __future__ import annotations

import random

from .popvary import select_best


class AutoVMixin:
    def _default_autov_weight(self) -> list[list[float]]:
        return [
            [0.08, 0.18, 0.00, 0.16],
            [0.04, 0.35, 0.20, 0.14],
            [0.02, 0.60, 0.50, 0.12],
            [0.12, 0.25, -0.20, 0.10],
            [0.16, 0.45, 0.75, 0.10],
            [0.01, 0.10, 1.00, 0.08],
            [0.20, 0.20, 0.50, 0.08],
            [0.03, 0.75, 0.25, 0.08],
            [0.10, 0.05, -0.60, 0.07],
            [0.06, 0.40, 0.90, 0.07],
        ]

    def _autov_pick_type(self, cumulative: list[float]) -> int:
        value = self.randval(0.0, 1.0)
        return next((idx for idx, cutoff in enumerate(cumulative) if value <= cutoff), len(cumulative) - 1)

    def _autov_tournament(self, count: int) -> list[int]:
        pool = []
        for _ in range(count):
            a = random.randrange(self.Popsize)
            b = random.randrange(self.Popsize)
            pool.append(a if self.pop_fit[a] < self.pop_fit[b] else b)
        return pool

    def AutoV(
        self,
        weight: list[list[float]] | None = None,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        weight = self._default_autov_weight() if weight is None else weight
        total = sum(max(1e-12, row[3]) for row in weight)
        cumulative = []
        acc = 0.0
        for row in weight:
            acc += max(1e-12, row[3]) / total
            cumulative.append(acc)
        mating = self._autov_tournament(max(2, 2 * (p_end - p_start)))
        offspring: list[list[float]] = []
        for k in range(p_end - p_start):
            p1 = self.pop[mating[k]]
            p2 = self.pop[mating[k + (p_end - p_start)]]
            child = p1.copy()
            for j in range(self.Nvar):
                typ = self._autov_pick_type(cumulative)
                w1, w2, w3, _ = weight[typ]
                r1 = random.gauss(0.0, 1.0)
                r2 = random.gauss(0.0, 1.0)
                child[j] = self._clip(
                    (self.Ubound - self.Lbound) * r1 * w1
                    + p2[j] * (r2 * w2 + w3)
                    + p1[j] * (1.0 - r2 * w2 - w3)
                )
            offspring.append(child)
        candidates = [self.pop[i].copy() for i in range(self.Popsize)] + offspring
        fits = self.pop_fit.copy() + [self.EvaluFunc(row, self.Nvar) for row in offspring]
        selected, selected_fit = select_best(candidates, fits, self.Popsize)
        for offset, i in enumerate(range(p_start, p_end)):
            self.newpop[i] = selected[offset].copy()
            self.newpop_fit[i] = selected_fit[offset]
