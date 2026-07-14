from __future__ import annotations

import random


class CSOMixin:
    def CSO(self, phi: float = 0.1, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        indices = list(range(p_start, p_end))
        random.shuffle(indices)
        mean = [
            sum(self.pop[i][j] for i in range(self.Popsize)) / max(1, self.Popsize)
            for j in range(self.Nvar)
        ]
        for k in range(0, len(indices) - 1, 2):
            a = indices[k]
            b = indices[k + 1]
            loser, winner = (a, b) if self.pop_fit[a] > self.pop_fit[b] else (b, a)
            self.newpop[winner] = self.pop[winner].copy()
            self.newpop_fit[winner] = self.pop_fit[winner]
            for j in range(self.Nvar):
                r1 = self.randval(0.0, 1.0)
                r2 = self.randval(0.0, 1.0)
                r3 = self.randval(0.0, 1.0)
                self.velocity[loser][j] = (
                    r1 * self.velocity[loser][j]
                    + r2 * (self.pop[winner][j] - self.pop[loser][j])
                    + phi * r3 * (mean[j] - self.pop[loser][j])
                )
                self.newpop[loser][j] = self._clip(self.pop[loser][j] + self.velocity[loser][j])
        if len(indices) % 2:
            last = indices[-1]
            self.newpop[last] = self.pop[last].copy()
            self.newpop_fit[last] = self.pop_fit[last]
