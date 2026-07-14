from __future__ import annotations

import random

from .archive import archive_sample, update_external_archive


class SHADEMixin:
    def _ensure_shade_state(self) -> None:
        if getattr(self, "shade_mf", None) is not None:
            return
        self.shade_mf = [0.5 for _ in range(6)]
        self.shade_mcr = [0.8 for _ in range(6)]
        self.shade_pos = 0
        self.shade_archive = []
        self.shade_archive_fit = []

    def SHADE(self, p_start: int = 0, p_end: int | None = None) -> None:
        self._ensure_shade_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        self.shade_archive, self.shade_archive_fit = update_external_archive(
            self.shade_archive,
            self.shade_archive_fit,
            self.pop,
            self.pop_fit,
            self.Popsize,
            "mixed",
        )
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        pbest_pool = ranked[: max(2, self.Popsize // 5)]
        for i in range(p_start, p_end):
            mem = random.randrange(len(self.shade_mf))
            f = min(1.0, max(0.05, self.shade_mf[mem] + 0.1 * random.gauss(0.0, 1.0)))
            cr = min(1.0, max(0.0, self.shade_mcr[mem] + 0.1 * random.gauss(0.0, 1.0)))
            pbest = self.pop[random.choice(pbest_pool)]
            r1, r2 = random.sample(range(self.Popsize), 2)
            archive_row = archive_sample(self.shade_archive, self.pop, self.Nvar)
            forced = random.randrange(self.Nvar)
            for j in range(self.Nvar):
                if j == forced or self.randval(0.0, 1.0) < cr:
                    value = self.pop[i][j] + f * (pbest[j] - self.pop[i][j]) + f * (self.pop[r1][j] - archive_row[j])
                else:
                    value = self.pop[i][j]
                self.newpop[i][j] = self._clip(value)
