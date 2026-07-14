from __future__ import annotations

import random

from .popvary import adaptive_rmp


class MFEAIIMixin:
    def _ensure_mfea_ii_state(self) -> None:
        if getattr(self, "mfea_ii_trials", None) is not None:
            return
        self.mfea_ii_trials = 0.0
        self.mfea_ii_success = 0.0

    def MFEA_II(self, p_start: int = 0, p_end: int | None = None, tasks: int = 2) -> None:
        self._ensure_mfea_ii_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        tasks = max(1, tasks)
        rmp = adaptive_rmp(self.mfea_ii_success, self.mfea_ii_trials, 0.2, 0.95)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        for i in range(p_start, p_end):
            skill = i % tasks
            pool = [idx for idx in ranked if idx % tasks == skill] or ranked
            p1 = self.pop[random.choice(pool)]
            p2 = self.pop[random.choice(ranked if self.randval(0.0, 1.0) < rmp else pool)]
            for j in range(self.Nvar):
                if self.randval(0.0, 1.0) < 0.5:
                    value = 0.5 * (p1[j] + p2[j]) + self.randnorm(0.0, 0.02 * (self.Ubound - self.Lbound))
                else:
                    value = p1[j]
                self.newpop[i][j] = self._clip(value)
            if self.randval(0.0, 1.0) < 0.2:
                j = min(self.Nvar - 1, int(self.randval(0.0, self.Nvar)))
                self.newpop[i][j] = self._clip(self.newpop[i][j] + self.randnorm(0.0, 0.1 * (self.Ubound - self.Lbound)))
        self.mfea_ii_trials += p_end - p_start
