from __future__ import annotations

import random

from .archive import archive_sample, update_external_archive


class IMODEMixin:
    def _ensure_imode_state(self) -> None:
        if getattr(self, "imode_archive", None) is not None:
            return
        self.imode_archive = []
        self.imode_archive_fit = []

    def IMODE(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        self._ensure_imode_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        self.imode_archive, self.imode_archive_fit = update_external_archive(
            self.imode_archive,
            self.imode_archive_fit,
            self.pop,
            self.pop_fit,
            max(1, self.Popsize),
            "mixed",
        )
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        pbest_pool = ranked[: max(2, self.Popsize // 5)]
        progress = Gen / max(1, MaxGen)
        f = 0.5 + 0.3 * (1.0 - progress) * self.randval(0.0, 1.0)
        cr = 0.9 - 0.4 * progress
        for i in range(p_start, p_end):
            pbest = self.pop[random.choice(pbest_pool)]
            r1, r2 = random.sample(range(self.Popsize), 2)
            archive_row = archive_sample(self.imode_archive, self.pop, self.Nvar)
            forced = random.randrange(self.Nvar)
            for j in range(self.Nvar):
                if j == forced or self.randval(0.0, 1.0) < cr:
                    value = self.pop[i][j] + f * (pbest[j] - self.pop[i][j]) + f * (self.pop[r1][j] - archive_row[j])
                else:
                    value = self.pop[i][j]
                self.newpop[i][j] = self._clip(value)
