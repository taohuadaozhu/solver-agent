from __future__ import annotations

import math
import random


class AdamMixin:
    def _ensure_adam_state(self) -> None:
        if getattr(self, "adam_m", None) and len(self.adam_m) == self.Popsize and len(self.adam_m[0]) == self.Nvar:
            return
        self.adam_m = self.CreateMatrix(self.Popsize, self.Nvar)
        self.adam_v = self.CreateMatrix(self.Popsize, self.Nvar)
        self.adam_t = [0 for _ in range(self.Popsize)]

    def _finite_gradient(self, x: list[float], dims: list[int], step: float) -> list[float]:
        grad = [0.0 for _ in range(self.Nvar)]
        for bit in dims:
            plus = x.copy()
            minus = x.copy()
            plus[bit] = self._clip(plus[bit] + step)
            minus[bit] = self._clip(minus[bit] - step)
            den = plus[bit] - minus[bit]
            if abs(den) <= 1e-12:
                continue
            grad[bit] = (self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / den
        return grad

    def Adam(
        self,
        alpha: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        p_start: int = 0,
        p_end: int | None = None,
        grad_dims: int | None = None,
        fd_step: float = 1e-5,
    ) -> None:
        self._ensure_adam_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        dim_count = self.Nvar if grad_dims is None else max(1, min(self.Nvar, grad_dims))
        for i in range(p_start, p_end):
            dims = list(range(self.Nvar)) if dim_count >= self.Nvar else random.sample(range(self.Nvar), dim_count)
            grad = self._finite_gradient(self.newpop[i], dims, fd_step)
            self.adam_t[i] += 1
            t = self.adam_t[i]
            for j in dims:
                self.adam_m[i][j] = beta1 * self.adam_m[i][j] + (1.0 - beta1) * grad[j]
                self.adam_v[i][j] = beta2 * self.adam_v[i][j] + (1.0 - beta2) * grad[j] * grad[j]
                mhat = self.adam_m[i][j] / (1.0 - beta1**t)
                vhat = self.adam_v[i][j] / (1.0 - beta2**t)
                self.newpop[i][j] = self._clip(self.newpop[i][j] - alpha * mhat / (math.sqrt(vhat) + 1e-8))
