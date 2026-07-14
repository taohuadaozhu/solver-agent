from __future__ import annotations

import random

from .popvary import random_subspace_groups


class L2SMEAMixin:
    def L2SMEA(self, p_start: int = 0, p_end: int | None = None, group_size: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        group_size = group_size or max(1, int(self.Nvar ** 0.5))
        groups = random_subspace_groups(self.Nvar, group_size)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        elite = self.pop[ranked[0]]
        for i in range(p_start, p_end):
            self.newpop[i] = self.pop[i].copy()
            group = groups[i % len(groups)]
            donor = self.pop[random.choice(ranked[: max(2, self.Popsize // 2)])]
            for j in group:
                value = self.pop[i][j] + self.randval(0.0, 1.0) * (elite[j] - self.pop[i][j]) + self.randval(-0.5, 0.5) * (donor[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(value)
