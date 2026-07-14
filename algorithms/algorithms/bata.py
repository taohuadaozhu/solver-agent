from __future__ import annotations

import math
import random

class BATAMixin:
    def BATA(self, gen: int, p_start: int, p_end: int) -> None:
        self.pop_heap_sort(self.Popsize)
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                self.BAT_v[i][j] += (self.pop[i][j] - self.pop[self.cur_best][j]) * self.randval(0.0, 100.0)
                self.newpop[i][j] = self._clip(self.pop[i][j] + self.BAT_v[i][j])
            if self.randval(0.0, 1.0) > self.BAT_r[i]:
                elite = random.randrange(max(1, self.Popsize // 2))
                for j in range(self.Nvar):
                    self.newpop[i][j] = self._clip(self.pop[elite][j] + self.randval(-1.0, 1.0) * self.BAT_A[i])
            self.newpop_fit[i] = self.EvaluFunc(self.newpop[i], self.Nvar)
            if self.randval(0.0, 1.0) < self.BAT_A[i] and self.newpop_fit[i] < self.pop_fit[self.cur_best]:
                self.BAT_A[i] *= 0.95
                self.BAT_r[i] *= 1.0 - math.exp(-0.95 * gen)
