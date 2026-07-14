from __future__ import annotations

import random
import math

from .archive import archive_value, build_archive
from .popvary import fill_population, select_population


class ECPOMixin:
    def _ecpo_pop_factor(self, strategy: int, n_ecpi: int) -> int:
        pairs = n_ecpi * (n_ecpi - 1) // 2
        if strategy == 1:
            return max(1, 2 * pairs)
        if strategy == 3:
            return max(1, 2 * pairs + n_ecpi)
        return max(1, n_ecpi)

    def _ecpo_candidates(
        self,
        strategy: int,
        n_ecpi: int,
        archive: list[list[float]],
    ) -> list[list[float]]:
        n_ecpi = max(2, min(n_ecpi, self.Popsize))
        pop_factor = self._ecpo_pop_factor(strategy, n_ecpi)
        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        ecp = [self.pop[idx] for idx in ranked]
        rows: list[list[float]] = []
        rounds = math.ceil(self.Popsize / pop_factor)
        for _ in range(rounds):
            force = random.gauss(0.7, 0.2)
            selected = sorted(random.sample(range(self.Popsize), n_ecpi))
            if strategy == 1:
                for ii in range(n_ecpi):
                    for jj in range(n_ecpi):
                        if ii == jj:
                            continue
                        sign = 1.0 if jj < ii else -1.0
                        rows.append([
                            ecp[selected[ii]][d]
                            + force * (ecp[0][d] - ecp[selected[ii]][d])
                            + sign * force * (ecp[selected[jj]][d] - ecp[selected[ii]][d])
                            for d in range(self.Nvar)
                        ])
            elif strategy == 3:
                for ii in range(n_ecpi):
                    combined = [
                        ecp[selected[ii]][d] + force * (ecp[0][d] - ecp[selected[ii]][d])
                        for d in range(self.Nvar)
                    ]
                    for jj in range(n_ecpi):
                        if ii == jj:
                            continue
                        sign = 1.0 if jj < ii else -1.0
                        rows.append([
                            combined[d] + sign * force * (ecp[selected[jj]][d] - ecp[selected[ii]][d])
                            for d in range(self.Nvar)
                        ])
                        combined = [
                            combined[d] + sign * force * (ecp[selected[jj]][d] - ecp[selected[ii]][d])
                            for d in range(self.Nvar)
                        ]
                    rows.append(combined)
            else:
                for ii in range(n_ecpi):
                    row = ecp[selected[ii]].copy()
                    for jj in range(n_ecpi):
                        if ii == jj:
                            continue
                        sign = 1.0 if jj < ii else -1.0
                        row = [
                            row[d] + sign * force * (ecp[selected[jj]][d] - ecp[selected[ii]][d])
                            for d in range(self.Nvar)
                        ]
                    rows.append(row)
        for row in rows:
            for bit in range(self.Nvar):
                row[bit] = self._clip(row[bit])
                if self.randval(0.0, 1.0) < 0.2:
                    value = archive_value(archive, bit)
                    if value is not None:
                        row[bit] = value
        return rows

    def ECPO(
        self,
        Strategy: int = 2,
        nECPI: int = 3,
        naECP: int | None = None,
        p_start: int = 0,
        p_end: int | None = None,
        archive_strategy: str = "best",
        pop_strategy: str = "best",
        fill_strategy: str = "random",
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        archive_size = max(1, naECP if naECP is not None else round(self.Popsize / 3))
        archive, archive_fit = build_archive(self.pop, self.pop_fit, archive_size, archive_strategy)
        candidates = archive + self._ecpo_candidates(Strategy, nECPI, archive)
        fits = archive_fit + [self.EvaluFunc(row, self.Nvar) for row in candidates[len(archive):]]
        selected, selected_fit = select_population(candidates, fits, self.Popsize, pop_strategy, self.Nvar)
        selected = fill_population(selected, self.Popsize, self.Nvar, self.Lbound, self.Ubound, self.randval, fill_strategy)
        for offset, i in enumerate(range(p_start, p_end)):
            self.newpop[i] = selected[offset].copy()
            self.newpop_fit[i] = selected_fit[offset] if offset < len(selected_fit) else self.EvaluFunc(self.newpop[i], self.Nvar)
