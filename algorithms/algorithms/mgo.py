from __future__ import annotations

import math
import random


class MGOMixin:
    def MGO(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        leader = self.pop[ranked[0]]
        progress = Gen / max(1, MaxGen)
        step = (1.0 - progress) * (self.Ubound - self.Lbound)
        for i in range(p_start, p_end):
            peer = self.pop[random.choice(ranked[: max(2, self.Popsize // 2)])]
            for j in range(self.Nvar):
                angle = self.randval(0.0, 2.0 * math.pi)
                value = self.pop[i][j] + math.cos(angle) * self.randval(0.0, 1.0) * (leader[j] - abs(self.pop[i][j]))
                value += 0.05 * step * math.sin(angle) + 0.3 * self.randval(0.0, 1.0) * (peer[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(value)
