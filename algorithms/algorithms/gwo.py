from __future__ import annotations


class GWOMixin:
    def GWO(self, Gen: int, MaxGen: int, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        leaders = [self.pop[idx] for idx in ranked[:3]]
        while len(leaders) < 3:
            leaders.append(leaders[0])
        a = 2.0 - 2.0 * Gen / max(1, MaxGen)
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                value = 0.0
                for leader in leaders:
                    a1 = 2.0 * a * self.randval(0.0, 1.0) - a
                    c1 = 2.0 * self.randval(0.0, 1.0)
                    distance = abs(c1 * leader[j] - self.pop[i][j])
                    value += leader[j] - a1 * distance
                self.newpop[i][j] = self._clip(value / 3.0)
