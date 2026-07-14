from __future__ import annotations

import random


class KMAMixin:
    def KMA(self, p_start: int = 0, p_end: int | None = None, clusters: int = 3) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        clusters = max(1, min(clusters, self.Popsize))
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        centers = [self.pop[idx] for idx in ranked[:clusters]]
        span = self.Ubound - self.Lbound
        for i in range(p_start, p_end):
            center = centers[i % clusters]
            peer = self.pop[random.choice(ranked[: max(clusters, self.Popsize // 2)])]
            for j in range(self.Nvar):
                value = center[j] + self.randval(0.0, 1.0) * (peer[j] - self.pop[i][j]) + self.randnorm(0.0, 0.03 * span)
                self.newpop[i][j] = self._clip(value)
