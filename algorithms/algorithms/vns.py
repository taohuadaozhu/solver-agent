from __future__ import annotations

import random

class VNSMixin:
    def VNS(self, nst: int, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            bit = self.neigh[i]
            tmp = self.pop[i].copy()
            tmp[bit] += self.randval(0.0, 1.0) * (self.pop[random.randrange(self.Popsize)][bit] - self.pop[i][bit])
            row = self._fitness_row(tmp)
            self.localsearch(row, nst)
            if row[self.Nvar] < self.pop_fit[i]:
                self.newpop[i] = row[: self.Nvar]
                self.newpop_fit[i] = row[self.Nvar]
            else:
                self.neigh[i] = (self.neigh[i] + 1) % self.Nvar
