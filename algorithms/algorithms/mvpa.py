from __future__ import annotations

import random

from .popvary import best_worst_pairs


class MVPAMixin:
    def MVPA(self, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        pairs = best_worst_pairs(self.pop_fit, max(1, (p_end - p_start) // 2))
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        for offset, i in enumerate(range(p_start, p_end)):
            winner, loser = pairs[offset % len(pairs)] if pairs else (ranked[0], ranked[-1])
            teammate = self.pop[random.choice(ranked[: max(2, self.Popsize // 2)])]
            for j in range(self.Nvar):
                value = self.pop[i][j] + self.randval(0.0, 1.0) * (self.pop[winner][j] - self.pop[loser][j])
                value += 0.25 * self.randval(0.0, 1.0) * (teammate[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(value)
