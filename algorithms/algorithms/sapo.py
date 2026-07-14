from __future__ import annotations

import random


class SAPOMixin:
    def SAPO(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        elite = self.pop[ranked[0]]
        progress = Gen / max(1, MaxGen)
        for i in range(p_start, p_end):
            peer = self.pop[random.choice(ranked[: max(2, self.Popsize // 2)])]
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < 0.5:
                    value = elite[j] + (1.0 - progress) * self.randval(-1.0, 1.0) * abs(peer[j] - self.pop[i][j])
                else:
                    value = self.pop[i][j] + self.randval(0.0, 1.0) * (peer[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(value)
