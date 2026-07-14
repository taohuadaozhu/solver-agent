from __future__ import annotations

import random


class SADEATDSCMixin:
    def SADE_ATDSC(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        progress = Gen / max(1, MaxGen)
        cr = 0.9 - 0.5 * progress
        f = 0.4 + 0.5 * self.randval(0.0, 1.0)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        top = ranked[: max(2, self.Popsize // 5)]
        for i in range(p_start, p_end):
            pbest = self.pop[random.choice(top)]
            r1, r2 = random.sample(range(self.Popsize), 2)
            forced = random.randrange(self.Nvar)
            for j in range(self.Nvar):
                if j == forced or self.randval(0.0, 1.0) < cr:
                    value = self.pop[i][j] + f * (pbest[j] - self.pop[i][j]) + f * (self.pop[r1][j] - self.pop[r2][j])
                else:
                    value = self.pop[i][j]
                self.newpop[i][j] = self._clip(value)
