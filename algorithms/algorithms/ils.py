from __future__ import annotations

import random

class ILSMixin:
    def localsearch(self, pp: list[float], nst: int) -> None:
        if len(pp) <= self.Nvar:
            pp.append(self.EvaluFunc(pp, self.Nvar))
        permu = list(range(self.Nvar))
        for step in range(nst):
            if step % self.Nvar == 0:
                random.shuffle(permu)
            temp = pp.copy()
            bit = permu[step % self.Nvar]
            temp[bit] = self.randval(self.Lbound, self.Ubound)
            temp[self.Nvar] = self.EvaluFunc(temp, self.Nvar)
            if temp[self.Nvar] < pp[self.Nvar]:
                pp[: self.Nvar + 1] = temp[: self.Nvar + 1]

    def ILS(self, nst: int, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            tmp1 = []
            for j in range(self.Nvar):
                bit = random.randrange(self.Popsize)
                if self.randval(0.0, 1.0) < 0.5:
                    step = self.randval(0.0, 1.0) * (
                        self.pop[bit][j] - self.pop[i][j]
                    )
                    tmp1.append(self._wrap(self.pop[i][j] + step))
                else:
                    tmp1.append(self.pop[i][j])
            row = self._fitness_row(tmp1)
            self.localsearch(row, nst)
            if row[self.Nvar] < self.pop_fit[i]:
                self.newpop[i] = row[: self.Nvar]
                self.newpop_fit[i] = row[self.Nvar]
