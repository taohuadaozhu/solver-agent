from __future__ import annotations

import random


class EGOMixin:
    def EGO(self, p_start: int = 0, p_end: int | None = None, infill: int = 3) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        elites = ranked[: max(1, min(self.Popsize, infill))]
        span = self.Ubound - self.Lbound
        for i in range(p_start, p_end):
            elite = self.pop[random.choice(elites)]
            scale = span * (0.25 if i % 2 else 0.08)
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < 0.75:
                    self.newpop[i][j] = self._clip(elite[j] + self.randnorm(0.0, scale))
                else:
                    self.newpop[i][j] = self.randval(self.Lbound, self.Ubound)
