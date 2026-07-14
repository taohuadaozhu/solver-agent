from __future__ import annotations

import random


class FRCGMixin:
    def _ensure_frcg_state(self) -> None:
        if getattr(self, "frcg_direction", None) and len(self.frcg_direction) == self.Popsize:
            return
        self.frcg_direction = self.CreateMatrix(self.Popsize, self.Nvar)
        self.frcg_grad = self.CreateMatrix(self.Popsize, self.Nvar)

    def _sparse_gradient(self, x: list[float], dims: list[int], step: float) -> list[float]:
        grad = [0.0 for _ in range(self.Nvar)]
        for bit in dims:
            plus = x.copy()
            minus = x.copy()
            plus[bit] = self._clip(x[bit] + step)
            minus[bit] = self._clip(x[bit] - step)
            den = plus[bit] - minus[bit]
            if abs(den) > 1e-12:
                grad[bit] = (self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / den
        return grad

    def FRCG(self, p_start: int = 0, p_end: int | None = None, grad_dims: int | None = None) -> None:
        self._ensure_frcg_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        count = self.Nvar if grad_dims is None else max(1, min(self.Nvar, grad_dims))
        step = max(1e-6, 1e-5 * (self.Ubound - self.Lbound))
        for i in range(p_start, p_end):
            dims = list(range(self.Nvar)) if count >= self.Nvar else random.sample(range(self.Nvar), count)
            grad = self._sparse_gradient(self.pop[i], dims, step)
            old_grad = self.frcg_grad[i]
            numerator = sum(grad[j] * (grad[j] - old_grad[j]) for j in dims)
            denominator = sum(old_grad[j] * old_grad[j] for j in dims) + 1e-12
            beta = max(0.0, numerator / denominator)
            direction = self.frcg_direction[i]
            alpha = 0.05 * (self.Ubound - self.Lbound)
            for j in range(self.Nvar):
                if j in dims:
                    direction[j] = -grad[j] + beta * direction[j]
                    self.newpop[i][j] = self._clip(self.pop[i][j] + alpha * direction[j])
                else:
                    self.newpop[i][j] = self.pop[i][j]
            self.frcg_grad[i] = grad
