from __future__ import annotations

class BAMixin:
    def BA(self, p_start: int, p_end: int) -> None:
        self.pop_heap_sort(self.Popsize)
        for i in range(self.ne):
            self.NeighborFlowerPatch(self.nre, i)
        for i in range(self.ne, self.nb):
            self.NeighborFlowerPatch(self.nrb, i)
        self.mutate(1.0, self.nb, self.Popsize)

    def NeighborFlowerPatch(self, nr: int, point: int) -> None:
        min_value = self.pop_fit[point]
        best = None
        for _ in range(nr):
            scale = self.ngh[point] * (self.Ubound - self.Lbound)
            cand = [
                self._clip(self.pop[point][j] + self.randval(-1.0, 1.0) * scale)
                for j in range(self.Nvar)
            ]
            fit = self.EvaluFunc(cand, self.Nvar)
            if fit < min_value:
                min_value = fit
                best = cand
        if best is None:
            self.ngh[point] *= self.ngh_decay
            self.ngh_decay_count[point] += 1
            if self.ngh_decay_count[point] < self.stlim:
                self.newpop[point] = self.pop[point].copy()
                self.newpop_fit[point] = self.pop_fit[point]
            else:
                self.newpop[point] = [
                    self.randval(self.Lbound, self.Ubound)
                    for _ in range(self.Nvar)
                ]
                self.newpop_fit[point] = self.EvaluFunc(self.newpop[point], self.Nvar)
                self.ngh[point] = self.ngh_origin
                self.ngh_decay_count[point] = 0
        else:
            self.newpop[point] = best
            self.newpop_fit[point] = min_value
