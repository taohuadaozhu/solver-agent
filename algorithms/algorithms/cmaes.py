from __future__ import annotations

import math
import random

class CMAESMixin:
    def pop_heap_adjust(self, s: int, length: int) -> None:
        self.pop_heap_sort(length)

    def pop_heap_sort(self, length: int) -> None:
        order = sorted(range(length), key=lambda idx: self.pop_fit[idx])
        self.pop[:length] = [self.pop[idx] for idx in order]
        self.pop_fit[:length] = [self.pop_fit[idx] for idx in order]
        if hasattr(self, "ngh"):
            self.ngh[:length] = [self.ngh[idx] for idx in order]

    def newpop_heap_adjust(self, s: int, length: int) -> None:
        self.newpop_heap_sort(length)

    def newpop_heap_sort(self, length: int) -> None:
        order = sorted(range(length), key=lambda idx: self.newpop_fit[idx])
        self.newpop[:length] = [self.newpop[idx] for idx in order]
        self.newpop_fit[:length] = [self.newpop_fit[idx] for idx in order]

    def pop_niche(self, mdis: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            for j in range(p_start + 1, p_end):
                if math.dist(self.pop[i], self.pop[j]) < mdis:
                    if self.pop_fit[i] > self.pop_fit[j]:
                        self.pop_fit[i] = abs(self.pop_fit[i] * 100.0)
                    else:
                        self.pop_fit[j] = abs(self.pop_fit[i] * 100.0)

    def newpop_localstep(self, step: float, nst: int, p_start: int, p_end: int) -> None:
        fave = sum(self.newpop_fit[p_start:p_end]) / max(1, self.Popsize)
        for i in range(p_start, p_end):
            if self.randval(0.0, 1.0) < 1.0 / (1.0 + math.exp(min(700.0, self.newpop_fit[i] - fave))):
                temp = self.newpop[i].copy()
                bit = random.randrange(self.Nvar)
                for j in range(nst):
                    temp[bit] -= j * step
                    if temp[bit] < self.Lbound:
                        temp[bit] = self.randval(self.Lbound, self.Ubound)
                    fit = self.EvaluFunc(temp, self.Nvar)
                    if fit < self.newpop_fit[i]:
                        self.newpop[i][bit] = temp[bit]
                        self.newpop_fit[i] = fit
                        break

    def pop_random_variance(self, p_start: int, p_end: int) -> float:
        bit = random.randrange(self.Nvar)
        values = [self.pop[i][bit] for i in range(p_start, p_end)]
        ave = sum(values) / max(1, self.Popsize)
        return math.sqrt(sum((x - ave) ** 2 for x in values)) / max(1, self.Popsize)

    def pop_random_entropy(self, subn: int, p_start: int, p_end: int) -> float:
        subn = max(1, int(subn))
        bit = random.randrange(self.Nvar)
        pp = [0.0 for _ in range(subn)]
        span = self.Ubound - self.Lbound if self.Ubound != self.Lbound else 1.0
        for i in range(p_start, p_end):
            reg = int(subn * (self.pop[i][bit] - self.Lbound) / span)
            pp[min(max(reg, 0), subn - 1)] += 1.0
        pp = [x / subn for x in pp]
        return -sum(p * math.log(p) for p in pp if p != 0.0)

    def CMAES(self, firstrun: int, p_start: int, p_end: int) -> None:
        if not getattr(self, "CMAisdone", False):
            self.CMAES_parametersetting()
            self.CMAES_initial()
            self.CMAisdone = True
        self.CMAES_updateDistribution(firstrun)
        self.CMAES_sampleGenerate(firstrun, p_start, p_end)

    def CMAES_parametersetting(self) -> None:
        self.xstart = [0.5 * (self.Lbound + self.Ubound) for _ in range(self.Nvar)]
        self.stddev = [0.3 * (self.Ubound - self.Lbound) for _ in range(self.Nvar)]
        self.lambda_ = 4 + math.floor(3 * math.log(self.Nvar))
        self.lambda_size = self.lambda_
        self.mu = self.lambda_ // 2
        self.weights = [math.log(self.mu + 1.0) - math.log(i + 1.0) for i in range(self.mu)]
        s1 = sum(self.weights)
        s2 = sum(w * w for w in self.weights)
        self.mueff = s1 * s1 / s2 if s2 else 1.0
        self.weights = [w / s1 for w in self.weights]
        self.cs = (self.mueff + 2.0) / (self.Nvar + self.mueff + 3.0)
        self.ccumcov = (4.0 + self.mueff / self.Nvar) / (self.Nvar + 4.0 + 2.0 * self.mueff / self.Nvar)
        self.mucov = self.mueff
        t1 = 2.0 / ((self.Nvar + 1.4142) ** 2)
        t2 = min((2.0 * self.mueff - 1.0) / ((self.Nvar + 2.0) ** 2 + self.mueff), 1.0)
        self.ccov = (1.0 / self.mucov) * t1 + (1.0 - 1.0 / self.mucov) * t2
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (self.Nvar + 1.0)) - 1.0) + self.cs

    def CMAES_initial(self) -> None:
        trace = sum(s * s for s in self.stddev)
        self.sigma = math.sqrt(trace / self.Nvar)
        self.chiN = math.sqrt(self.Nvar * (1.0 - 1.0 / (4.0 * self.Nvar) + 1.0 / (21.0 * self.Nvar * self.Nvar)))
        self.pcc = [0.0 for _ in range(self.Nvar)]
        self.ps = [0.0 for _ in range(self.Nvar)]
        self.tempRandom = [0.0 for _ in range(self.Nvar + 1)]
        self.BDz = [0.0 for _ in range(self.Nvar)]
        self.xmean = self.xstart.copy()
        self.xold = self.xstart.copy()
        self.rgD = [s * math.sqrt(self.Nvar / trace) for s in self.stddev]
        self.C = [[0.0 for _ in range(self.Nvar)] for _ in range(self.Nvar)]
        self.B = [[0.0 for _ in range(self.Nvar)] for _ in range(self.Nvar)]
        for i in range(self.Nvar):
            self.B[i][i] = 1.0
            self.C[i][i] = self.rgD[i] * self.rgD[i]
        self.index = list(range(self.lambda_))

    def CMAES_sampleGenerate(self, firstrun: int, p_start: int, p_end: int) -> None:
        if firstrun == 0:
            self.rgD = [math.sqrt(max(self.C[i][i], 0.0)) for i in range(self.Nvar)]
        else:
            self.updateEigensystem()
        for iPop in range(p_start, p_end):
            if firstrun == 0:
                for j in range(self.Nvar):
                    self.newpop[iPop][j] = self._clip(
                        self.xmean[j] + self.sigma * self.rgD[j] * self.gauss()
                    )
            else:
                temp = [self.rgD[j] * self.gauss() for j in range(self.Nvar)]
                for i in range(self.Nvar):
                    total = sum(self.B[i][j] * temp[j] for j in range(self.Nvar))
                    self.newpop[iPop][i] = self._clip(self.xmean[i] + self.sigma * total)

    def CMAES_updateDistribution(self, gen: int) -> list[float]:
        if gen <= 0:
            return self.xmean
        sample_count = min(self.lambda_, len(self.pop_fit))
        self.index = self.sortIndex(self.pop_fit, list(range(sample_count)), sample_count)
        if (
            len(self.index) > self.lambda_ // 2
            and self.pop_fit[self.index[0]] == self.pop_fit[self.index[self.lambda_ // 2]]
        ):
            self.sigma *= math.exp(0.2 + self.cs / self.damps)
        sqrtmueffdivsigma = math.sqrt(self.mueff) / self.sigma
        for i in range(self.Nvar):
            self.xold[i] = self.xmean[i]
            self.xmean[i] = sum(
                self.weights[k] * self.pop[self.index[k]][i]
                for k in range(min(self.mu, len(self.index)))
            )
            self.BDz[i] = sqrtmueffdivsigma * (self.xmean[i] - self.xold[i])
        for i in range(self.Nvar):
            total = sum(self.B[j][i] * self.BDz[j] for j in range(self.Nvar))
            self.tempRandom[i] = total / (self.rgD[i] if self.rgD[i] != 0 else 1e-12)
        sqrt_factor = math.sqrt(self.cs * (2.0 - self.cs))
        for i in range(self.Nvar):
            total = sum(self.B[i][j] * self.tempRandom[j] for j in range(self.Nvar))
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + sqrt_factor * total
        psxps = sum(x * x for x in self.ps)
        denom = math.sqrt(max(1e-12, 1.0 - (1.0 - self.cs) ** (2.0 * gen)))
        hsig = int(math.sqrt(psxps) / denom / self.chiN < 1.4 + 2.0 / (self.Nvar + 1.0))
        hsig_factor = hsig * math.sqrt(self.ccumcov * (2.0 - self.ccumcov))
        for i in range(self.Nvar):
            self.pcc[i] = (1.0 - self.ccumcov) * self.pcc[i] + hsig_factor * self.BDz[i]
        self.adaptC2(hsig, gen)
        self.sigma *= math.exp(((math.sqrt(psxps) / self.chiN) - 1.0) * self.cs / self.damps)
        return self.xmean

    def sortIndex(
        self,
        rgFunVal: list[float],
        iindex: list[int] | None = None,
        n: int | None = None,
    ) -> list[int]:
        n = len(rgFunVal) if n is None else n
        return sorted(range(n), key=lambda idx: rgFunVal[idx])

    def adaptC2(self, hsig: int, gen: int) -> None:
        if self.ccov == 0.0:
            return
        mucovinv = 1.0 / self.mucov
        common = self.ccov * ((self.Nvar + 1.5) / 3.0 if gen == 0 else 1.0)
        ccov1 = min(common * mucovinv, 1.0)
        ccovmu = min(common * (1.0 - mucovinv), 1.0 - ccov1)
        sigmasquare = self.sigma * self.sigma
        long_factor = (1.0 - hsig) * self.ccumcov * (2.0 - self.ccumcov)
        for i in range(self.Nvar):
            for j in range(i + 1):
                cij = (1.0 - ccov1 - ccovmu) * self.C[i][j]
                cij += ccov1 * (
                    self.pcc[i] * self.pcc[j] + long_factor * self.C[i][j]
                )
                for k in range(min(self.mu, len(self.index))):
                    row = self.pop[self.index[k]]
                    cij += (
                        ccovmu
                        * self.weights[k]
                        * (row[i] - self.xold[i])
                        * (row[j] - self.xold[j])
                        / sigmasquare
                    )
                self.C[i][j] = self.C[j][i] = cij

    def updateEigensystem(self) -> None:
        try:
            import numpy as np

            vals, vecs = np.linalg.eigh(np.array(self.C, dtype=float))
            vals = np.maximum(vals, 1e-30)
            self.rgD = [math.sqrt(float(v)) for v in vals]
            self.B = vecs.tolist()
        except Exception:
            self.rgD = [math.sqrt(max(self.C[i][i], 1e-30)) for i in range(self.Nvar)]
            self.B = [[1.0 if i == j else 0.0 for j in range(self.Nvar)] for i in range(self.Nvar)]

    def eigen(self, diag: list[float], Q: list[list[float]], rgtmp: list[float]) -> None:
        self.updateEigensystem()
        diag[: self.Nvar] = [d * d for d in self.rgD]
        for i in range(self.Nvar):
            Q[i][: self.Nvar] = self.B[i][: self.Nvar]

    def ql(self, d: list[float], e: list[float], V: list[list[float]]) -> None:
        self.eigen(d, V, e)

    def householder(self, V: list[list[float]], d: list[float], e: list[float]) -> None:
        for i in range(self.Nvar):
            d[i] = V[i][i]
            e[i] = 0.0
