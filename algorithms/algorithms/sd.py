from __future__ import annotations

import random


class SDMixin:
    def SD(self, p_start: int = 0, p_end: int | None = None, grad_dims: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        count = max(1, min(self.Nvar, grad_dims or min(16, self.Nvar)))
        step = max(1e-6, 1e-5 * (self.Ubound - self.Lbound))
        lr = 0.05 * (self.Ubound - self.Lbound)
        for i in range(p_start, p_end):
            self.newpop[i] = self.pop[i].copy()
            for bit in random.sample(range(self.Nvar), count):
                plus = self.pop[i].copy()
                minus = self.pop[i].copy()
                plus[bit] = self._clip(plus[bit] + step)
                minus[bit] = self._clip(minus[bit] - step)
                den = plus[bit] - minus[bit]
                if abs(den) <= 1e-12:
                    continue
                grad = (self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / den
                self.newpop[i][bit] = self._clip(self.pop[i][bit] - lr * grad / (abs(grad) + 1e-8))
