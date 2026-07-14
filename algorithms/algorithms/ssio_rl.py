from __future__ import annotations

import random


class SSIORLMixin:
    def _ensure_ssio_state(self) -> None:
        if getattr(self, "ssio_q", None) is not None:
            return
        self.ssio_q = [0.0, 0.0, 0.0]

    def SSIO_RL(self, p_start: int = 0, p_end: int | None = None) -> None:
        self._ensure_ssio_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        elite = self.pop[ranked[0]]
        action = max(range(3), key=lambda idx: self.ssio_q[idx]) if self.randval(0.0, 1.0) > 0.2 else random.randrange(3)
        span = self.Ubound - self.Lbound
        for i in range(p_start, p_end):
            peer = self.pop[random.choice(ranked[: max(2, self.Popsize // 2)])]
            for j in range(self.Nvar):
                if action == 0:
                    value = self.pop[i][j] + self.randval(0.0, 1.0) * (elite[j] - self.pop[i][j])
                elif action == 1:
                    value = self.pop[i][j] + self.randval(-1.0, 1.0) * (peer[j] - self.pop[i][j])
                else:
                    value = elite[j] + self.randnorm(0.0, 0.05 * span)
                self.newpop[i][j] = self._clip(value)
        self.ssio_q[action] = 0.95 * self.ssio_q[action] + 0.05
