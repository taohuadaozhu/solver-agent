from __future__ import annotations

import random


class SAMSOMixin:
    def SAMSO(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        leaders = ranked[: max(2, self.Popsize // 4)]
        inertia = 0.9 - 0.5 * Gen / max(1, MaxGen)
        for i in range(p_start, p_end):
            leader = self.pop[random.choice(leaders)]
            peer = self.pop[random.randrange(self.Popsize)]
            for j in range(self.Nvar):
                self.velocity[i][j] = inertia * self.velocity[i][j]
                self.velocity[i][j] += self.randval(0.0, 1.0) * (leader[j] - self.pop[i][j])
                self.velocity[i][j] += 0.5 * self.randval(0.0, 1.0) * (peer[j] - self.pop[i][j])
                self.newpop[i][j] = self._clip(self.pop[i][j] + self.velocity[i][j])
