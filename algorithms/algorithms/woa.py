from __future__ import annotations

import math


class WOAMixin:
    def WOA(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        a = 2.0 - 2.0 * Gen / max(1, MaxGen)
        b = 1.0
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                p = self.randval(0.0, 1.0)
                if p < 0.5:
                    a_vec = 2.0 * a * self.randval(0.0, 1.0) - a
                    c_vec = 2.0 * self.randval(0.0, 1.0)
                    if abs(a_vec) < 1.0:
                        distance = abs(c_vec * self.gbest[j] - self.pop[i][j])
                        value = self.gbest[j] - a_vec * distance
                    else:
                        rand_idx = int(self.randval(0.0, self.Popsize))
                        rand_idx = min(self.Popsize - 1, rand_idx)
                        distance = abs(c_vec * self.pop[rand_idx][j] - self.pop[i][j])
                        value = self.pop[rand_idx][j] - a_vec * distance
                else:
                    distance = abs(self.gbest[j] - self.pop[i][j])
                    l = self.randval(-1.0, 1.0)
                    value = distance * math.exp(b * l) * math.cos(2.0 * math.pi * l) + self.gbest[j]
                self.newpop[i][j] = self._clip(value)
