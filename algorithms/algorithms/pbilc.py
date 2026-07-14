from __future__ import annotations

import math

class PBILCMixin:
    def PBILC(self, p_start: int, p_end: int, learn_rate: float) -> None:
        self.pop_heap_sort(self.Popsize)
        pre_center = self.PB_center.copy()
        pre_sigma = self.PB_sigma.copy()
        for i in range(self.Nvar):
            direction = self.pop[0][i] + self.pop[1][i] - self.pop[self.Popsize - 1][i]
            self.PB_center[i] = (1.0 - learn_rate) * pre_center[i] + learn_rate * direction
            elite_vals = [self.pop[j][i] for j in range(max(1, int(self.num_of_elite)))]
            average = sum(elite_vals) / len(elite_vals)
            sigma = math.sqrt(sum((x - average) ** 2 for x in elite_vals) / len(elite_vals))
            self.PB_sigma[i] = (1.0 - learn_rate) * pre_sigma[i] + learn_rate * sigma
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                self.newpop[i][j] = self._clip(self.PB_center[j] + self.PB_sigma[j] * self.gauss())
