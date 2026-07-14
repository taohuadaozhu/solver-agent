from __future__ import annotations

import random


class FROFIMixin:
    def FROFI(self, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        elite = self.pop[ranked[0]]
        median = self.pop[ranked[len(ranked) // 2]]
        for i in range(p_start, p_end):
            base = self.pop[i]
            for j in range(self.Nvar):
                robust_direction = elite[j] - median[j]
                if self.randval(0.0, 1.0) < 0.5:
                    value = base[j] + self.randval(0.0, 1.0) * robust_direction
                else:
                    value = elite[j] + self.randval(-0.5, 0.5) * (base[j] - median[j])
                self.newpop[i][j] = self._clip(value)
