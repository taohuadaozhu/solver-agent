from __future__ import annotations


class GPSOMixin:
    def GPSO(self, w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445, max_ve: float = 0.1, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        self.PSO(w, c1, c2, max_ve, max(0, p_start), p_end)
