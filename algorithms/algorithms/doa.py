from __future__ import annotations

import math
import random

from .popvary import fill_population


class DOAMixin:
    def DOA(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        progress = min(1.0, max(0.0, Gen / max(1, MaxGen)))
        alpha = self.randval(0.0, 1.0) * (progress * progress - 2.0 * progress + 1.0)
        mean = [
            sum(self.pop[i][j] for i in range(self.Popsize)) / max(1, self.Popsize)
            for j in range(self.Nvar)
        ]
        elite = self.gbest
        levy_scale = (
            math.gamma(2.5) * math.sin(math.pi * 0.75)
            / math.gamma(1.25) / 1.5 / (2.0**0.25)
        ) ** (1.0 / 1.5)
        rows: list[list[float]] = []
        for i in range(p_start, p_end):
            base = self.pop[i].copy()
            if random.gauss(0.0, 1.0) < 1.5:
                theta = self.randval(-math.pi, math.pi)
                y = random.gauss(0.0, 1.0)
                iny = 0.0 if y <= 0 else math.exp(-0.5 * math.log(y) ** 2) / (y * math.sqrt(2.0 * math.pi))
                spiral = math.cos(theta) / math.exp(theta) * math.sin(theta) / math.exp(theta) * iny
                rise = [
                    base[j] + alpha * spiral * (self.randval(self.Lbound, self.Ubound) - base[j])
                    for j in range(self.Nvar)
                ]
            else:
                denom = MaxGen * MaxGen - 2.0 * MaxGen + 1.0
                factor = 1.0 - self.randval(0.0, 1.0) * (((Gen * Gen - 2.0 * Gen + 1.0) / max(1e-12, denom)) + 1.0)
                rise = [base[j] * factor for j in range(self.Nvar)]
            decline = []
            for j in range(self.Nvar):
                beta = random.gauss(0.0, 1.0)
                decline.append(rise[j] - alpha * beta * (mean[j] - alpha * beta * rise[j]))
            landed = []
            for j in range(self.Nvar):
                denom = abs(random.gauss(0.0, 1.0)) ** (1.0 / 1.5) + 1e-12
                levy = random.gauss(0.0, 1.0) * levy_scale / denom
                landed.append(elite[j] + levy * alpha * (elite[j] - decline[j] * 2.0 * progress))
            rows.append([self._clip(value) for value in landed])
        rows = fill_population(rows, p_end - p_start, self.Nvar, self.Lbound, self.Ubound, self.randval)
        for offset, i in enumerate(range(p_start, p_end)):
            self.newpop[i] = rows[offset]
