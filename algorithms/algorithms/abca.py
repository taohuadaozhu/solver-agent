from __future__ import annotations

import math
import random

class ABCAMixin:
    def EmployedBee(self, pn: int, tmp: list[float], pp: list[list[float]]) -> None:
        tmp[: self.Nvar] = self.pop[pn][: self.Nvar]
        para2change = random.randrange(self.Nvar)
        neighbor = random.randrange(self.Popsize)
        while neighbor == pn:
            neighbor = random.randrange(self.Popsize)
        tmp[para2change] += (pp[neighbor][para2change] - tmp[para2change]) * self.randval(-1.0, 1.0)
        tmp[para2change] = self._clip(tmp[para2change])

    def OnlookerBee(self, p_start: int, p_end: int) -> None:
        maxf = 0.0
        for i in range(p_start, p_end):
            exponent = self.newpop_fit[i] / 1000.0
            self.pr[i] = 1e10 if exponent > math.log(1e10) else math.exp(exponent)
            maxf = max(maxf, self.pr[i])
        for i in range(p_start, p_end):
            self.pr[i] = 0.9 * self.pr[i] / maxf + 0.1 if maxf != 0 else 1.0

    def ScoutBee(self, limit: int, p_start: int, p_end: int) -> None:
        maxindex = max(range(p_start, p_end), key=lambda idx: self.trial[idx])
        if self.trial[maxindex] > limit:
            self.newpop[maxindex] = [self.randval(self.Lbound, self.Ubound) for _ in range(self.Nvar)]
            self.newpop_fit[maxindex] = self.EvaluFunc(self.newpop[maxindex], self.Nvar)

    def ABCA(self, limit: int, p_start: int, p_end: int) -> None:
        tmp = [0.0 for _ in range(self.Nvar)]
        for i in range(self.Popsize):
            self.EmployedBee(i, tmp, self.pop)
            tmp_fit = self.EvaluFunc(tmp, self.Nvar)
            if tmp_fit < self.pop_fit[i]:
                self.newpop[i] = tmp.copy()
                self.newpop_fit[i] = tmp_fit
            else:
                self.newpop[i] = self.pop[i].copy()
                self.newpop_fit[i] = self.pop_fit[i]
                self.trial[i] += 1
        self.OnlookerBee(0, self.Popsize)
        t = 0
        i = p_start
        totaliter = 0
        while t < p_end and totaliter < 2 * self.Popsize:
            if self.randval(0.0, 1.0) < self.pr[i]:
                t += 1
                self.EmployedBee(i, tmp, self.newpop)
                tmp_fit = self.EvaluFunc(tmp, self.Nvar)
                if tmp_fit < self.newpop_fit[i]:
                    self.newpop[i] = tmp.copy()
                    self.newpop_fit[i] = tmp_fit
                else:
                    self.trial[i] += 1
            i += 1
            if i >= p_end:
                i = p_start
            totaliter += 1
        self.ScoutBee(limit, p_start, p_end)
