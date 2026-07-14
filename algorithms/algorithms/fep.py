from __future__ import annotations

import math
import random


class FEPMixin:
    def _ensure_fep_state(self) -> None:
        if getattr(self, "fep_sigma", None) and len(self.fep_sigma) == self.Popsize:
            return
        scale = 0.1 * (self.Ubound - self.Lbound)
        self.fep_sigma = [[scale for _ in range(self.Nvar)] for _ in range(self.Popsize)]

    def FEP(self, p_start: int = 0, p_end: int | None = None) -> None:
        self._ensure_fep_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        tau = 1.0 / math.sqrt(max(1, self.Nvar))
        tau_prime = 1.0 / math.sqrt(2.0 * max(1, self.Nvar))
        for i in range(p_start, p_end):
            common = random.gauss(0.0, 1.0)
            for j in range(self.Nvar):
                sigma = self.fep_sigma[i][j] * math.exp(tau_prime * common + tau * random.gauss(0.0, 1.0))
                sigma = min(max(sigma, 1e-12), self.Ubound - self.Lbound)
                self.fep_sigma[i][j] = sigma
                step = sigma * math.tan(math.pi * (self.randval(0.0, 1.0) - 0.5))
                self.newpop[i][j] = self._clip(self.pop[i][j] + step)
