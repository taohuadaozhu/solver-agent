from __future__ import annotations

import math


class AESSPSOMixin:
    def _aesspso_weight(self, beta_values: list[float], gamma_values: list[float]) -> float:
        beta = sum(beta_values) / max(1, len(beta_values))
        gamma = sum(gamma_values) / max(1, len(gamma_values))
        pressure = abs(beta) + abs(gamma)
        return max(-10.0, min(10.0, 1.0 / (1.0 + math.exp(pressure - 4.0))))

    def AESSPSO(
        self,
        Beta: float = 2.05,
        Gamma: float = 2.05,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        beta_values = [Beta * self.randval(0.0, 1.0) for _ in range(p_start, p_end)]
        gamma_values = [Gamma * self.randval(0.0, 1.0) for _ in range(p_start, p_end)]
        weight = self._aesspso_weight(beta_values, gamma_values)
        for offset, i in enumerate(range(p_start, p_end)):
            r1 = beta_values[offset]
            r2 = gamma_values[offset]
            for j in range(self.Nvar):
                velocity = (
                    weight * self.velocity[i][j]
                    + r1 * (self.ibest[i][j] - self.newpop[i][j])
                    + r2 * (self.gbest[j] - self.newpop[i][j])
                )
                self.velocity[i][j] = velocity
                self.newpop[i][j] = self._clip(self.newpop[i][j] + velocity)
