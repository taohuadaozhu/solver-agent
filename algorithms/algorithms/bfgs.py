from __future__ import annotations

import random


class BFGSMixin:
    def _ensure_bfgs_state(self) -> None:
        if getattr(self, "bfgs_diag", None) and len(self.bfgs_diag) == self.Popsize and len(self.bfgs_diag[0]) == self.Nvar:
            return
        self.bfgs_diag = [[1.0 for _ in range(self.Nvar)] for _ in range(self.Popsize)]
        self.bfgs_grad = self.CreateMatrix(self.Popsize, self.Nvar)

    def _bfgs_gradient(self, x: list[float], dims: list[int], step: float) -> list[float]:
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

    def BFGS(
        self,
        beta: float = 0.6,
        sigma: float = 0.4,
        p_start: int = 0,
        p_end: int | None = None,
        grad_dims: int | None = None,
        fd_step: float = 1e-5,
    ) -> None:
        self._ensure_bfgs_state()
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        dim_count = self.Nvar if grad_dims is None else max(1, min(self.Nvar, grad_dims))
        for i in range(p_start, p_end):
            dims = list(range(self.Nvar)) if dim_count >= self.Nvar else random.sample(range(self.Nvar), dim_count)
            x = self.newpop[i].copy()
            fit = self.EvaluFunc(x, self.Nvar)
            grad = self._bfgs_gradient(x, dims, fd_step)
            direction = [0.0 for _ in range(self.Nvar)]
            for bit in dims:
                direction[bit] = -self.bfgs_diag[i][bit] * grad[bit]
            accepted = x
            accepted_fit = fit
            step = 1.0
            gd = sum(grad[bit] * direction[bit] for bit in dims)
            for _ in range(21):
                trial = x.copy()
                for bit in dims:
                    trial[bit] = self._clip(x[bit] + step * direction[bit])
                trial_fit = self.EvaluFunc(trial, self.Nvar)
                if trial_fit <= fit + sigma * step * gd:
                    accepted = trial
                    accepted_fit = trial_fit
                    break
                step *= beta
            next_grad = self._bfgs_gradient(accepted, dims, fd_step)
            for bit in dims:
                sk = accepted[bit] - x[bit]
                yk = next_grad[bit] - grad[bit]
                if yk * sk > 1e-12:
                    self.bfgs_diag[i][bit] = max(1e-8, min(1e8, sk / yk))
                self.bfgs_grad[i][bit] = next_grad[bit]
            self.newpop[i] = accepted
            self.newpop_fit[i] = accepted_fit
