from __future__ import annotations

import random

class GRASPMixin:
    def GRASP(self, alfa: float, nst: int, p_start: int, p_end: int) -> None:
        threshold = self.pop_fit[self.cur_best] + alfa * (
            self.pop_fit[self.cur_worst] - self.pop_fit[self.cur_best]
        )
        self.pop_heap_sort(self.Popsize)
        rcl = [
            self.pop[i] + [self.pop_fit[i]]
            for i in range(self.Popsize)
            if self.pop_fit[i] <= threshold
        ] or [self.pop[0] + [self.pop_fit[0]]]
        for i in range(p_start, p_end):
            row = random.choice(rcl).copy()
            self.localsearch(row, nst)
            if row[self.Nvar] < self.pop_fit[i]:
                self.newpop[i] = row[: self.Nvar]
                self.newpop_fit[i] = row[self.Nvar]
