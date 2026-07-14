from __future__ import annotations

import random


class SADESammonMixin:
    def SADE_Sammon(self, p_start: int = 0, p_end: int | None = None, sample_dims: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        dims = random.sample(range(self.Nvar), max(1, min(self.Nvar, sample_dims or min(32, self.Nvar))))
        f = 0.5 + 0.3 * self.randval(0.0, 1.0)
        cr = 0.8
        for i in range(p_start, p_end):
            r1, r2, r3 = random.sample(range(self.Popsize), 3)
            self.newpop[i] = self.pop[i].copy()
            forced = random.choice(dims)
            for j in dims:
                if j == forced or self.randval(0.0, 1.0) < cr:
                    value = self.pop[r1][j] + f * (self.pop[r2][j] - self.pop[r3][j])
                    self.newpop[i][j] = self._clip(value)
