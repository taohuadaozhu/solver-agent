from __future__ import annotations

import math
import random


class SAMixin:
    def SA(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        progress = min(1.0, max(0.0, Gen / max(1, MaxGen)))
        temp = max(1e-8, 0.1 * (0.99 ** Gen))
        sigma = 0.2 * (self.Ubound - self.Lbound) * (0.99 ** Gen)
        sigma = max(sigma, 1e-12 * max(1.0, self.Ubound - self.Lbound))
        for i in range(p_start, p_end):
            base = self.pop[i]
            base_fit = self.pop_fit[i]
            candidate = base.copy()
            changed = False
            mutation_rate = 0.5 * (1.0 - 0.5 * progress)
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < mutation_rate:
                    candidate[j] = self._clip(base[j] + self.randnorm(0.0, sigma))
                    changed = True
            if not changed and self.Nvar > 0:
                j = random.randrange(self.Nvar)
                candidate[j] = self._clip(base[j] + self.randnorm(0.0, sigma))
            cand_fit = self.EvaluFunc(candidate, self.Nvar)
            delta = cand_fit - base_fit
            denom = abs(base_fit) + 1e-6
            if delta <= 0.0 or self.randval(0.0, 1.0) < math.exp(-delta / denom / temp):
                self.newpop[i] = candidate
                self.newpop_fit[i] = cand_fit
            else:
                self.newpop[i] = base.copy()
                self.newpop_fit[i] = base_fit
