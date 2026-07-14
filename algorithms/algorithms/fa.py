from __future__ import annotations

import math

class FAMixin:
    def FA(self, gama: float, alpha0: float, betamin: float, MGEN: int, p_start: int, p_end: int) -> None:
        if not hasattr(self, "alpha"):
            self.alpha = alpha0
        decay = (10.0 ** -4 / 0.9) ** (1.0 / MGEN) if MGEN else 1.0
        self.alpha = decay * (alpha0 if self.alpha == alpha0 else self.alpha)
        if self.alpha < 1e-5:
            self.alpha = 0.5
        order = sorted(range(self.Popsize), key=lambda idx: self.I[idx])
        self.I = [self.I[idx] for idx in order]
        self.Index = [self.Index[idx] for idx in order]
        for pos, idx in enumerate(self.Index):
            self.newpop[pos] = self.pop[idx].copy()
        scale = abs(self.Ubound - self.Lbound)
        for i in range(p_start, p_end):
            for j in range(self.Popsize):
                if i == j:
                    continue
                r = math.dist(self.newpop[i], self.newpop[j])
                if self.I[i] > self.I[j]:
                    beta = (1.0 - betamin) * math.exp(-gama * r * r) + betamin
                    for k in range(self.Nvar):
                        tmpf = self.alpha * (self.randval(0.0, 2.0) - 1.0) * scale
                        self.newpop[i][k] = self._clip(self.newpop[i][k] * (1.0 - beta) + self.pop[j][k] * beta + tmpf)
