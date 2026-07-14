from __future__ import annotations

import math

from ..population import PI

class ACOMixin:
    def path_finding(self, epsl: float, p_start: int, p_end: int) -> None:
        psum = sum(row[self.Nvar + 1] for row in self.ant_tao)
        probs = [row[self.Nvar + 1] / psum if psum else 1.0 / self.Popsize for row in self.ant_tao]
        cprob = []
        acc = 0.0
        for prob in probs:
            acc += prob
            cprob.append(acc)
        for i in range(p_start, p_end):
            p = self.randval(0.0, 1.0)
            l = next((idx for idx, cutoff in enumerate(cprob) if p < cutoff), self.Popsize - 1)
            ssco = []
            for j in range(self.Nvar):
                spread = sum(
                    abs(self.ant_tao[k][j] - self.ant_tao[l][j])
                    for k in range(self.Popsize)
                )
                ssco.append(epsl * spread / max(1.0, self.Popsize - 1.0))
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < 0.15:
                    step = self.gauss() * math.sqrt(max(ssco[j], 0.0))
                    self.newpop[i][j] = self._clip(self.ant_tao[l][j] + step)
                else:
                    self.newpop[i][j] = self.pop[i][j]

    def phe_updating(self, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            if any(row[self.Nvar] == self.newpop_fit[i] for row in self.ant_tao):
                continue
            max_index = max(range(self.Popsize), key=lambda idx: self.ant_tao[idx][self.Nvar])
            if self.ibest_fit[i] < self.ant_tao[max_index][self.Nvar]:
                self.ant_tao[max_index][: self.Nvar] = self.newpop[i].copy()
                self.ant_tao[max_index][self.Nvar] = self.newpop_fit[i]
        self.heap_sort(self.ant_tao, self.Popsize, self.Nvar)
        sigma = 1e-4 * self.Popsize
        for i in range(self.Popsize):
            self.ant_tao[i][self.Nvar + 1] = math.exp(
                -(i**2.0) / (2.0 * sigma**2.0)
            ) / (sigma * math.sqrt(2.0 * PI))

    def ACO(self, epsl: float, p_start: int, p_end: int) -> None:
        self.path_finding(epsl, p_start, p_end)
        self.phe_updating(p_start, p_end)
