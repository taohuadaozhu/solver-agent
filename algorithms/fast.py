from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .multimethod import MultiMet, RunResult
from .population import PI
from .problems import Sphere


class NumpyMultiMet(MultiMet):
    """Vectorized CSA path for Sphere-compatible runs."""

    def __init__(
        self,
        psize: int = 20,
        nn: int = 1000,
        lb: float = -5.0,
        ub: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(psize, nn, lb, ub, Sphere)
        self.rng = np.random.default_rng(seed)

    def _randval_matrix(self, rows: int, cols: int, low: float, high: float) -> np.ndarray:
        return (self.rng.integers(0, 1000, size=(rows, cols)) / 1000.0) * (high - low) + low

    def Initial(self) -> None:
        self.pop = self._randval_matrix(self.Popsize, self.Nvar, self.Lbound, self.Ubound)
        self.newpop = self.pop.copy()
        self.velocity = self._randval_matrix(self.Popsize, self.Nvar, self.Lbound, self.Ubound)
        self.ibest = self.pop.copy()
        self.pop_fit = np.sum(self.pop * self.pop, axis=1)
        self.newpop_fit = self.pop_fit.copy()
        self.ibest_fit = self.pop_fit.copy()
        self.worst_and_best()
        self.gbest = self.pop[self.cur_best].copy()
        self.gbest_fit = float(self.pop_fit[self.cur_best])
        self.CRfit()
        self.CRold = self.CRnew

    def worst_and_best(self) -> None:
        self.cur_best = int(np.argmin(self.pop_fit))
        self.cur_worst = int(np.argmax(self.pop_fit))

    def average_fit(self) -> float:
        return float(np.mean(self.pop_fit))

    def Elist(self) -> None:
        if self.pop_fit[self.cur_best] < self.gbest_fit:
            self.gbest = self.pop[self.cur_best].copy()
            self.gbest_fit = float(self.pop_fit[self.cur_best])
        else:
            self.pop[self.cur_worst] = self.gbest.copy()
            self.pop_fit[self.cur_worst] = self.gbest_fit

    def Evaluation(self, s: bool, p_start: int, p_end: int) -> None:
        if not s:
            self.pop_fit[p_start:p_end] = np.sum(self.pop[p_start:p_end] ** 2, axis=1)
        else:
            self.newpop_fit[p_start:p_end] = np.sum(self.newpop[p_start:p_end] ** 2, axis=1)

    def pop_update(self, p_start: int, p_end: int) -> None:
        self.pop[p_start:p_end] = self.newpop[p_start:p_end]
        self.pop_fit[p_start:p_end] = self.newpop_fit[p_start:p_end]
        better = self.newpop_fit[p_start:p_end] < self.ibest_fit[p_start:p_end]
        idx = np.arange(p_start, p_end)[better]
        self.ibest[idx] = self.newpop[idx]
        self.ibest_fit[idx] = self.newpop_fit[idx]

    def levy_cuckoo(self, p_start: int, p_end: int) -> None:
        beta = 3.0 / 2.0
        a = self.rng.integers(0, 1000) / 1000.0
        b = (self.rng.integers(0, 1000) / 1000.0) * (1.0 - 1e-5) + 1e-5
        sigma = (
            a
            * np.sin(PI * beta / 2.0)
            / (b * beta * np.power(2.0, (beta - 1.0) / 2.0))
        ) ** (1.0 / beta)

        rows = p_end - p_start
        u = (self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0) * sigma
        v = self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0
        step = np.clip(u / (np.power(v, 1.0 / beta) + 1e-5), -1.0, 1.0)
        mask = self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0 < 0.3
        rand = self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0
        base = self.pop[p_start:p_end]
        stepsize = 0.1 * step * (self.ibest[p_start:p_end] - base)
        candidate = base + stepsize * rand
        self.newpop[p_start:p_end] = np.where(mask, candidate, base)
        np.clip(self.newpop[p_start:p_end], self.Lbound, self.Ubound, out=self.newpop[p_start:p_end])

    def nest_discover(self, pa: float, p_start: int, p_end: int) -> None:
        rows = p_end - p_start
        r1 = self.rng.integers(0, self.Popsize, size=rows)
        r2 = self.rng.integers(0, self.Popsize, size=rows)
        valid = (r1 != r2)[:, None]
        mask = (self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0 < pa) & valid
        dis = self.newpop[r1] - self.newpop[r2]
        rand = self.rng.integers(0, 1000, size=(rows, self.Nvar)) / 1000.0
        self.newpop[p_start:p_end] += np.where(mask, rand * dis, 0.0)
        np.clip(self.newpop[p_start:p_end], self.Lbound, self.Ubound, out=self.newpop[p_start:p_end])


def run_csa_fast(
    nvar: int = 1000,
    popsize: int = 20,
    maxgen: Optional[int] = None,
    threshold: float = 1e-6,
    lb: float = -5.0,
    ub: float = 10.0,
    seed: Optional[int] = None,
) -> RunResult:
    maxgen = maxgen if maxgen is not None else 100
    solver = NumpyMultiMet(popsize, nvar, lb, ub, seed=seed)
    solver.Initial()

    generation = 0
    gen_count = 0
    best = solver.gbest_fit
    started = time.perf_counter()
    while generation < maxgen and solver.gbest_fit > threshold and gen_count < maxgen / 10:
        solver.CSA(0.3, 0, popsize)
        solver.Evaluation(True, 0, popsize)
        solver.pop_update(0, popsize)
        solver.worst_and_best()
        solver.Elist()

        if solver.gbest_fit < best:
            gen_count = 0
        else:
            gen_count += 1
        generation += 1
    return RunResult(solver.gbest_fit, generation, time.perf_counter() - started)
