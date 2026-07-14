from __future__ import annotations

import random

class COAMixin:
    def chaos(self, incpop: list[list[float]], chaos_n: int) -> None:
        x1 = [self.randval(0.0, 1.0)]
        x2 = [self.randval(0.0, 1.0)]
        for _ in range(1, chaos_n):
            x1.append(4.0 * x1[-1] * (1.0 - x1[-1]))
            x2.append(4.0 * x2[-1] * (1.0 - x2[-1]))
        for i in range(chaos_n):
            l = random.randrange(self.Popsize)
            point = int(x1[i] * self.Nvar) % self.Nvar
            incpop[l][: self.Nvar] = self.gbest.copy()
            incpop[l][point] = self.Lbound + x2[i] * (self.Ubound - self.Lbound)

    def COA(self, chaos_n: int, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            x1 = [self.randval(0.0, 1.0)]
            x2 = [self.randval(0.0, 1.0)]
            for _ in range(1, chaos_n):
                x1.append(4.0 * x1[-1] * (1.0 - x1[-1]))
                x2.append(4.0 * x2[-1] * (1.0 - x2[-1]))
            best_fit = float("inf")
            best_point = 0
            best_value = self.pop[i][0]
            for j in range(chaos_n):
                point = int(x1[j] * self.Nvar) % self.Nvar
                tmp = self.pop[i].copy()
                tmp[point] = self.Lbound + x2[j] * (self.Ubound - self.Lbound)
                fit = self.EvaluFunc(tmp, self.Nvar)
                if fit < best_fit:
                    best_fit = fit
                    best_point = point
                    best_value = tmp[point]
            if best_fit < self.newpop_fit[i]:
                self.newpop[i] = self.pop[i].copy()
                self.newpop[i][best_point] = best_value
                self.newpop_fit[i] = best_fit
