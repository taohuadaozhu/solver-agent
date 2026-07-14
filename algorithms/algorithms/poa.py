from __future__ import annotations

class POAMixin:
    def plant_growth(self, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            r = self.randval(0.0, 1.0)
            if r < 0.4:
                for j in range(self.Nvar):
                    if self.randval(0.0, 1.0) < 0.5:
                        step = (self.gbest[j] - self.newpop[i][j]) * self.randval(0.0, 1.0)
                        self.newpop[i][j] = self._wrap(self.newpop[i][j] + step)
            if 0.4 < r < 0.6:
                for j in range(self.Nvar):
                    if self.randval(0.0, 1.0) < 0.5:
                        step = (self.pop[self.cur_best][j] - self.newpop[i][j]) * self.randval(0.0, 1.0)
                        self.newpop[i][j] = self._wrap(step)
            elif 0.6 < r < 0.8:
                for j in range(self.Nvar):
                    if self.randval(0.0, 1.0) < 0.5:
                        self.newpop[i][j] = self._wrap((self.ibest[i][j] - self.newpop[i][j]) * self.randval(0.0, 1.0))
            else:
                for j in range(self.Nvar):
                    if self.randval(0.0, 1.0) < 0.2:
                        self.newpop[i][j] = self.randval(self.Lbound, self.Ubound)

    def POA(self, p_start: int, p_end: int) -> None:
        self.plant_growth(p_start, p_end)
