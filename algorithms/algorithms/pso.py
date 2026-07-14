from __future__ import annotations

import math
import random

class PSOMixin:
    def _sample_dimensions_by_magnitude(self, values: list[float], budget: int) -> list[int]:
        budget = max(1, min(self.Nvar, budget))
        if budget >= self.Nvar:
            return list(range(self.Nvar))
        ranked = sorted(range(self.Nvar), key=lambda idx: abs(values[idx]), reverse=True)
        dims = ranked[: max(1, budget // 2)]
        seen = set(dims)
        while len(dims) < budget:
            idx = random.randrange(self.Nvar)
            if idx not in seen:
                dims.append(idx)
                seen.add(idx)
        return dims

    def SparseSubgradient(
        self,
        theta: list[float],
        c_step: float,
        dims: list[int],
        q: int = 1,
    ) -> list[float]:
        c_step = c_step if abs(c_step) > 1e-12 else 1e-12
        q = max(1, q)
        result = [0.0 for _ in range(self.Nvar)]
        for i in dims:
            total = 0.0
            for _ in range(q):
                delta = max(1e-3, self.randval(0.0, 1.0))
                plus = theta[: self.Nvar]
                minus = theta[: self.Nvar]
                plus[i] = self._clip(theta[i] + c_step * delta)
                minus[i] = self._clip(theta[i] - c_step * delta)
                total += (self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / (2.0 * c_step * delta + 0.001)
            result[i] = total / q
        return result

    def PSO(self, w: float, c1: float, c2: float, max_ve: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                r1 = self.randval(0.0, 1.0)
                r2 = self.randval(0.0, 1.0)
                v = (
                    w * self.velocity[i][j]
                    + c1 * r1 * (self.ibest[i][j] - self.newpop[i][j])
                    + c2 * r2 * (self.gbest[j] - self.newpop[i][j])
                )
                self.velocity[i][j] = min(max(v, -max_ve), max_ve)
                self.newpop[i][j] += self.velocity[i][j]
                if self.newpop[i][j] > self.Ubound:
                    self.newpop[i][j] = self.Lbound
                elif self.newpop[i][j] < self.Lbound:
                    self.newpop[i][j] = self.Ubound

    def CPSO(self, w: float, c1: float, c2: float, max_ve: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            for j in range(self.Nvar):
                r1 = self.randval(0.0, 1.0)
                r2 = self.randval(0.0, 1.0)
                v = 0.729 * (
                    w * self.velocity[i][j]
                    + c1 * r1 * (self.ibest[i][j] - self.newpop[i][j])
                    + c2 * r2 * (self.gbest[j] - self.newpop[i][j])
                )
                self.velocity[i][j] = min(max(v, -max_ve), max_ve)
                self.newpop[i][j] = self._wrap(self.newpop[i][j] + self.velocity[i][j])

    def Subgradient(self, theta: list[float], q: int, c_step: float, subgrad: list[float] | None = None) -> list[float]:
        q = max(1, q)
        c_step = c_step if abs(c_step) > 1e-12 else 1e-12
        result = subgrad if subgrad is not None else [0.0 for _ in range(self.Nvar)]
        for i in range(self.Nvar):
            total = 0.0
            for _ in range(q):
                delta = self.randval(0.0, 1.0)
                plus = theta[: self.Nvar]
                minus = theta[: self.Nvar]
                plus[i] = self._clip(theta[i] + c_step * delta)
                minus[i] = self._clip(theta[i] - c_step * delta)
                total += (self.EvaluFunc(plus, self.Nvar) - self.EvaluFunc(minus, self.Nvar)) / (
                    2.0 * c_step * delta + 0.001
                )
            result[i] = total / q
        return result

    def SPSO(
        self,
        w: float,
        c1: float,
        c2: float,
        max_ve: float,
        Gen: int,
        MaxGen: int,
        p_start: int,
        p_end: int,
    ) -> None:
        for i in range(p_start, p_end):
            velnorm = 0.0
            for j in range(self.Nvar):
                r1 = self.randval(0.0, 1.0)
                r2 = self.randval(0.0, 1.0)
                v = (
                    w * self.velocity[i][j]
                    + c1 * r1 * (self.ibest[i][j] - self.newpop[i][j])
                    + c2 * r2 * (self.gbest[j] - self.newpop[i][j])
                )
                self.velocity[i][j] = min(max(v, -max_ve), max_ve)
                velnorm += self.velocity[i][j] ** 2
            velnorm = math.sqrt(velnorm)
            grad_dims = self._sample_dimensions_by_magnitude(
                self.velocity[i],
                max(4, min(16, int(math.sqrt(self.Nvar)) + 2)),
            )
            subgrad = self.SparseSubgradient(self.newpop[i], velnorm, grad_dims)
            subgnorm = math.sqrt(sum(v * v for v in subgrad))
            for j in range(self.Nvar):
                self.newpop[i][j] = self._clip(
                    self.newpop[i][j] - (velnorm / (subgnorm + 1e-5)) * subgrad[j]
                )

    def Cauchy_mutation(self, pp: list[float], Gen: int, MaxGen: int) -> None:
        bit = random.randrange(self.Nvar)
        exponent = 1.0 - float(Gen) / MaxGen if MaxGen else 1.0
        if self.randval(0.0, 1.0) < 0.5:
            pp[bit] += (self.Ubound - pp[bit]) * (1.0 - self.randval(0.0, 1.0) ** exponent)
        else:
            pp[bit] += (pp[bit] - self.Lbound) * (1.0 - self.randval(0.0, 1.0) ** exponent)
        pp[bit] = self._clip(pp[bit])

    def CMPSO(
        self,
        w: float,
        c1: float,
        c2: float,
        max_ve: float,
        Gen: int,
        MaxGen: int,
        p_start: int,
        p_end: int,
    ) -> None:
        self.PSO(w, c1, c2, max_ve, p_start, p_end)

    def APSO_1(
        self,
        c1: float,
        c2: float,
        max_ve: float,
        Gen: int,
        MaxGen: int,
        p_start: int,
        p_end: int,
    ) -> None:
        w = 0.9 - 0.5 * Gen / MaxGen if MaxGen else 0.9
        self.PSO(w, c1, c2, max_ve, p_start, p_end)

    def APSO_2(self, c1: float, c2: float, max_ve: float, p_start: int, p_end: int) -> None:
        self.PSO(0.5 + self.randval(0.0, 1.0) / 2.0, c1, c2, max_ve, p_start, p_end)

    def APSO_3(self, max_ve: float, p_start: int, p_end: int) -> None:
        dg = sum(math.dist(self.newpop[i], self.gbest) for i in range(self.Popsize))
        dg /= max(1, self.Popsize - 1)
        d = []
        for i in range(self.Popsize):
            distance = sum(
                math.dist(self.newpop[i], self.newpop[j])
                for j in range(self.Popsize)
            )
            d.append(distance / max(1, self.Popsize - 1))
        dmin, dmax = min(d), max(d)
        f = 0.5 if dmax == dmin else (dg - dmin) / (dmax - dmin)
        s1 = 0 if f <= 0.4 else (5 * f - 2 if f <= 0.6 else (1 if f <= 0.7 else (-10 * f + 8 if f <= 0.8 else 0)))
        s2 = 0 if f <= 0.2 else (10 * f - 2 if f <= 0.3 else (1 if f <= 0.4 else (-5 * f + 3 if f <= 0.6 else 0)))
        s3 = 1 if f <= 0.1 else (-5 * f + 1.5 if f <= 0.3 else 0)
        s4 = 0 if f <= 0.7 else (5 * f - 3.5 if f <= 0.9 else 1)
        if s3 > s2:
            self.ac1 += 0.1
            self.ac2 -= 0.1
        elif s2 > s1:
            self.ac1 += 0.05
            self.ac2 -= 0.05
        elif s1 > s4:
            self.ac1 += 0.05
            self.ac2 += 0.05
        else:
            self.ac1 -= 0.1
            self.ac2 += 0.1
        if self.ac1 + self.ac2 > 4.0:
            total = self.ac1 + self.ac2
            self.ac1 = 4.0 * self.ac1 / total
            self.ac2 = 4.0 * self.ac2 / total
        if self.ac1 < 0:
            self.ac1 = 2.0
        if self.ac2 < 0:
            self.ac2 = 2.0
        w = 1.0 / (1.0 + 1.5 * math.exp(-2.6 * f))
        self.PSO(w, self.ac1, self.ac2, max_ve, p_start, p_end)

    def APSO_4(self, max_ve: float, Gen: int, MaxGen: int, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            fdist = math.dist(self.newpop[i], self.gbest)
            gw = gc1 = gc2 = 0.0
            for j in range(self.Nvar):
                delta = self.newpop[i][j] - self.gbest[j]
                gw += delta * self.velocity[i][j]
                gc1 += delta * self.randval(0.0, 1.0) * (self.ibest[i][j] - self.newpop[i][j])
                gc2 += delta * self.randval(0.0, 1.0) * (self.gbest[j] - self.newpop[i][j])
            gw *= 2.0
            gc1 *= 2.0
            gc2 *= 2.0
            alp = fdist / (gw * gw + gc1 * gc1 + gc2 * gc2 + 1.0)
            self.AW[i] = min(max(self.AW[i] - alp * gw, 0.4), 0.9)
            self.AC1[i] = min(max(self.AC1[i] - alp * gc1, 0.5), 2.5)
            self.AC2[i] = min(max(self.AC2[i] - alp * gc2, 0.5), 2.5)
            r1 = self.randval(0.0, 1.0)
            r2 = self.randval(0.0, 1.0)
            for j in range(self.Nvar):
                v = (
                    self.AW[i] * self.velocity[i][j]
                    + self.AC1[i] * r1 * (self.ibest[i][j] - self.newpop[i][j])
                    + self.AC2[i] * r2 * (self.gbest[j] - self.newpop[i][j])
                )
                self.velocity[i][j] = min(max(v, -max_ve), max_ve)
                self.newpop[i][j] = self._wrap(self.newpop[i][j] + self.velocity[i][j])

    def APSO_5(self, max_ve: float, Gen: int, MaxGen: int, p_start: int, p_end: int) -> None:
        self.APSO_4(max_ve, Gen, MaxGen, p_start, p_end)
        for i in range(p_start, p_end):
            velnorm = math.sqrt(sum(v * v for v in self.velocity[i]))
            grad_dims = self._sample_dimensions_by_magnitude(
                self.velocity[i],
                max(4, min(16, int(math.sqrt(self.Nvar)) + 2)),
            )
            subgrad = self.SparseSubgradient(self.newpop[i], velnorm / 100.0, grad_dims)
            subgnorm = math.sqrt(sum(v * v for v in subgrad))
            for j in range(self.Nvar):
                self.newpop[i][j] = self._wrap(
                    self.newpop[i][j]
                    - (velnorm / (100.0 * subgnorm + 0.001)) * subgrad[j]
                )

    def Orthogonal_P(self, P0: list[float], w: float, c: float, ppn: int) -> None:
        if self.OArow > 32:
            self.LightOrthogonal_P(P0, w, c, ppn)
            return
        candidates = []
        for row in range(self.OArow):
            cand = []
            for j in range(self.Nvar):
                r = self.randval(0.0, 1.0)
                target = self.ibest[ppn][j] if self.OAValue(row, j) == 0 else self.gbest[j]
                cand.append(
                    self._wrap(
                        self.newpop[ppn][j]
                        + w * self.velocity[ppn][j]
                        + c * r * (target - self.newpop[ppn][j])
                    )
                )
            candidates.append(self._fitness_row(cand))
        best = min(range(len(candidates)), key=lambda idx: candidates[idx][self.Nvar])
        for i in range(self.Nvar):
            la1 = sum(
                row[self.Nvar]
                for idx, row in enumerate(candidates)
                if self.OAValue(idx, i) == 0
            ) / 2.0
            la2 = sum(
                row[self.Nvar]
                for idx, row in enumerate(candidates)
                if self.OAValue(idx, i) != 0
            ) / 2.0
            P0[i] = self.ibest[ppn][i] if la1 < la2 else self.gbest[i]
        trial = [
            self._wrap(
                self.newpop[ppn][j]
                + w * self.velocity[ppn][j]
                + c * self.randval(0.0, 1.0) * (P0[j] - self.newpop[ppn][j])
            )
            for j in range(self.Nvar)
        ]
        if candidates[best][self.Nvar] < self.EvaluFunc(trial, self.Nvar):
            P0[:] = candidates[best][: self.Nvar]

    def LightOrthogonal_P(self, P0: list[float], w: float, c: float, ppn: int) -> None:
        row_budget = max(8, min(24, 2 * int(math.log2(max(2, self.Nvar))) + 2))
        if row_budget >= self.OArow:
            rows = list(range(self.OArow))
        else:
            rows = [0, self.OArow - 1]
            seen = set(rows)
            while len(rows) < row_budget:
                idx = random.randrange(self.OArow)
                if idx not in seen:
                    rows.append(idx)
                    seen.add(idx)
        candidates = []
        for row in rows:
            cand = []
            for j in range(self.Nvar):
                r = self.randval(0.0, 1.0)
                target = self.ibest[ppn][j] if self.OAValue(row, j) == 0 else self.gbest[j]
                cand.append(
                    self._wrap(
                        self.newpop[ppn][j]
                        + w * self.velocity[ppn][j]
                        + c * r * (target - self.newpop[ppn][j])
                    )
                )
            candidates.append(self._fitness_row(cand))
        best = min(range(len(candidates)), key=lambda idx: candidates[idx][self.Nvar])
        for i in range(self.Nvar):
            left_sum = right_sum = 0.0
            left_count = right_count = 0
            for idx, row in enumerate(rows):
                if self.OAValue(row, i) == 0:
                    left_sum += candidates[idx][self.Nvar]
                    left_count += 1
                else:
                    right_sum += candidates[idx][self.Nvar]
                    right_count += 1
            left_avg = left_sum / max(1, left_count)
            right_avg = right_sum / max(1, right_count)
            P0[i] = self.ibest[ppn][i] if left_avg < right_avg else self.gbest[i]
        trial = [
            self._wrap(
                self.newpop[ppn][j]
                + w * self.velocity[ppn][j]
                + c * self.randval(0.0, 1.0) * (P0[j] - self.newpop[ppn][j])
            )
            for j in range(self.Nvar)
        ]
        if candidates[best][self.Nvar] < self.EvaluFunc(trial, self.Nvar):
            P0[:] = candidates[best][: self.Nvar]

    def OLPSO(self, w: float, c: float, max_ve: float, p_start: int, p_end: int) -> None:
        P0 = [0.0 for _ in range(self.Nvar)]
        for i in range(p_start, p_end):
            self.Orthogonal_P(P0, w, c, i)
            for j in range(self.Nvar):
                v = w * self.velocity[i][j] + c * self.randval(0.0, 1.0) * (P0[j] - self.newpop[i][j])
                self.velocity[i][j] = min(max(v, -max_ve), max_ve)
                self.newpop[i][j] = self._wrap(self.newpop[i][j] + self.velocity[i][j])
