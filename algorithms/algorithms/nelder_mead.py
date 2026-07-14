from __future__ import annotations


class NelderMeadMixin:
    def Nelder_Mead(self, p_start: int = 0, p_end: int | None = None, stepn: int = 4) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        for i in range(p_start, p_end):
            self.newpop_simplex(i, stepn, 4, 0.08)
