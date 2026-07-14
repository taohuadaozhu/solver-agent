from __future__ import annotations

import random

from .popvary import random_subspace_groups


class SACCEAMIIMixin:
    def SACC_EAM_II(self, p_start: int = 0, p_end: int | None = None, group_size: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        groups = random_subspace_groups(self.Nvar, group_size or max(1, int(self.Nvar ** 0.5)))
        best = self.gbest
        for i in range(p_start, p_end):
            self.newpop[i] = self.pop[i].copy()
            group = groups[i % len(groups)]
            peer = self.pop[random.randrange(self.Popsize)]
            for j in group:
                value = self.pop[i][j] + self.randval(0.0, 1.0) * (best[j] - self.pop[i][j]) + self.randval(-0.5, 0.5) * (peer[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(value)
