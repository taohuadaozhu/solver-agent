from __future__ import annotations

import random


class SQPMixin:
    def SQP(self, p_start: int = 0, p_end: int | None = None, grad_dims: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        count = max(1, min(self.Nvar, grad_dims or min(12, self.Nvar)))
        step = max(1e-6, 1e-5 * (self.Ubound - self.Lbound))
        for i in range(p_start, p_end):
            x = self.pop[i].copy()
            dims = random.sample(range(self.Nvar), count)
            direction = [0.0 for _ in range(self.Nvar)]
            for bit in dims:
                plus = x.copy()
                minus = x.copy()
                plus[bit] = self._clip(plus[bit] + step)
                minus[bit] = self._clip(minus[bit] - step)
                den = plus[bit] - minus[bit]
                if abs(den) > 1e-12:
                    direction[bit] = -(self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / den
            norm = sum(abs(direction[j]) for j in dims) + 1e-12
            radius = 0.1 * (self.Ubound - self.Lbound)
            for j in range(self.Nvar):
                self.newpop[i][j] = x[j]
            for bit in dims:
                self.newpop[i][bit] = self._clip(x[bit] + radius * direction[bit] / norm)
