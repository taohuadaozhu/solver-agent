from __future__ import annotations

import random

from .popvary import best_worst_pairs


class SACOSOMixin:
    def SACOSO(self, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        pairs = best_worst_pairs(self.pop_fit, max(1, p_end - p_start))
        mean = [sum(self.pop[i][j] for i in range(self.Popsize)) / self.Popsize for j in range(self.Nvar)]
        for offset, i in enumerate(range(p_start, p_end)):
            winner, loser = pairs[offset % len(pairs)] if pairs else (self.cur_best, self.cur_worst)
            for j in range(self.Nvar):
                self.velocity[loser][j] = self.randval(0.0, 1.0) * self.velocity[loser][j]
                self.velocity[loser][j] += self.randval(0.0, 1.0) * (self.pop[winner][j] - self.pop[loser][j])
                self.velocity[loser][j] += 0.2 * self.randval(0.0, 1.0) * (mean[j] - self.pop[loser][j])
                self.newpop[i][j] = self._clip(self.pop[loser][j] + self.velocity[loser][j])
