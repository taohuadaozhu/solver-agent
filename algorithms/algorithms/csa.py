from __future__ import annotations

import math
import random

from ..population import PI

class CSAMixin:
    def levy_cuckoo(self, p_start: int, p_end: int) -> None:
        beta = 3.0 / 2.0
        a = self.randval(0.0, 1.0)
        b = self.randval(1e-5, 1.0)
        sigma = (
            a
            * math.sin(PI * beta / 2.0)
            / (b * beta * math.pow(2.0, (beta - 1.0) / 2.0))
        ) ** (1.0 / beta)

        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                u = self.randval(0.0, 1.0) * sigma
                v = self.randval(0.0, 1.0)
                step = u / (math.pow(v, 1.0 / beta) + 1e-5)
                step = min(max(step, -1.0), 1.0)
                if self.randval(0.0, 1.0) < 0.3:
                    stepsize = 0.1 * step * (self.ibest[i][j] - self.pop[i][j])
                    self.newpop[i][j] = self.pop[i][j] + stepsize * self.randval(0.0, 1.0)
                    if self.newpop[i][j] < self.Lbound:
                        self.newpop[i][j] = self.Lbound
                    elif self.newpop[i][j] > self.Ubound:
                        self.newpop[i][j] = self.Ubound
                else:
                    self.newpop[i][j] = self.pop[i][j]

    def nest_discover(self, pa: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            r1 = random.randrange(self.Popsize)
            r2 = random.randrange(self.Popsize)
            if r1 == r2:
                continue
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < pa:
                    dis = self.newpop[r1][j] - self.newpop[r2][j]
                    self.newpop[i][j] += self.randval(0.0, 1.0) * dis
                    if self.newpop[i][j] < self.Lbound:
                        self.newpop[i][j] = self.Lbound
                    elif self.newpop[i][j] > self.Ubound:
                        self.newpop[i][j] = self.Ubound

    def CSA(self, pa: float, p_start: int, p_end: int) -> None:
        self.levy_cuckoo(p_start, p_end)
        self.nest_discover(pa, p_start, p_end)
