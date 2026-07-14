from __future__ import annotations

import random

from .archive import update_external_archive


class MiSACOMixin:
    def _ensure_misaco_state(self) -> None:
        if getattr(self, "misaco_archive", None) is not None:
            return
        self.misaco_archive = []
        self.misaco_archive_fit = []

    def MiSACO(self, p_start: int = 0, p_end: int | None = None) -> None:
        self._ensure_misaco_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        self.misaco_archive, self.misaco_archive_fit = update_external_archive(
            self.misaco_archive,
            self.misaco_archive_fit,
            self.pop,
            self.pop_fit,
            max(2, self.Popsize // 2),
            "best",
        )
        archive = self.misaco_archive or self.pop
        span = self.Ubound - self.Lbound
        for i in range(p_start, p_end):
            ant = random.choice(archive)
            peer = random.choice(archive)
            for j in range(self.Nvar):
                value = ant[j] + self.randval(-0.5, 0.5) * abs(ant[j] - peer[j]) + self.randnorm(0.0, 0.02 * span)
                self.newpop[i][j] = self._clip(value)
