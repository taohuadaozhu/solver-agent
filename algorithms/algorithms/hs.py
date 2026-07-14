from __future__ import annotations

import random

class HSMixin:
    def newpop_worst_best(self, p_start: int, p_end: int) -> tuple[int, int]:
        best = min(range(p_start, p_end), key=lambda idx: self.newpop_fit[idx])
        worst = max(range(p_start, p_end), key=lambda idx: self.newpop_fit[idx])
        return worst, best

    def HS(self, srate: float, trate: float, bw: float, p_start: int, p_end: int) -> None:
        ww, _ = self.newpop_worst_best(p_start, p_end)
        for _ in range(p_start, p_end):
            tmp = []
            for _j in range(self.Nvar):
                if self.randval(0.0, 1.0) < srate:
                    tmp.append(self.newpop[random.randrange(self.Popsize)][_j])
                else:
                    tmp.append(self.randval(self.Lbound, self.Ubound))
            if self.randval(0.0, 1.0) < trate:
                tmp = [self._clip(x + self.randval(0.0, 1.0) * bw) for x in tmp]
            fit = self.EvaluFunc(tmp, self.Nvar)
            if fit < self.newpop_fit[ww]:
                self.newpop[ww] = tmp
                self.newpop_fit[ww] = fit
                ww, _ = self.newpop_worst_best(p_start, p_end)
