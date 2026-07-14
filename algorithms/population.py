import math
import random
import time
from typing import Generic, TypeVar

T = TypeVar("T", int, float)
PI = 3.1415926


class Population(Generic[T]):
    def __init__(self, psize: int, nn: int, lb: T, ub: T) -> None:
        self.Popsize = psize
        self.Nvar = nn
        self.Lbound = lb
        self.Ubound = ub

        self.pop = self.CreateMatrix(psize, nn)
        self.newpop = self.CreateMatrix(psize, nn)
        self.gbest = [self.randval(lb, ub) for _ in range(nn)]
        self.pop_fit = [0.0 for _ in range(psize)]
        self.newpop_fit = [0.0 for _ in range(psize)]
        self.gbest_fit = 0.0

        self.cur_best = 0
        self.cur_worst = 0
        self.CRold = 0.0
        self.CRnew = 0.0

        self.stored = False
        self.hold = 0.0
        self.t = int(100 * time.time() + time.process_time())
        self.seed = abs(self.t) or 1
        self.aktseed = self.seed
        self.rgrand = [1 for _ in range(32)]
        for i in range(39, -1, -1):
            tmp = self.aktseed // 127773
            self.aktseed = 16807 * (self.aktseed - tmp * 127773) - 2836 * tmp
            if self.aktseed < 0:
                self.aktseed += 2147483647
            if i < 32:
                self.rgrand[i] = self.aktseed
        self.aktrand = self.rgrand[0]

        for i in range(psize):
            for j in range(nn):
                self.pop[i][j] = self.randval(lb, ub)
                self.newpop[i][j] = self.randval(lb, ub)

    def CreateMatrix(self, nRow: int, nCol: int) -> list[list[float]]:
        return [[0.0 for _ in range(nCol)] for _ in range(nRow)]

    def randval(self, low: float, high: float) -> float:
        # Matches the C++ helper's thousand-level granularity.
        return (random.randrange(1000) / 1000.0) * (high - low) + low

    def heap_sort(self, num: list[list[float]], length: int, cbit: int) -> None:
        num[:length] = sorted(num[:length], key=lambda row: row[cbit])

    def swap(self, x: T, y: T) -> tuple[T, T]:
        return y, x

    def worst_and_best(self) -> None:
        self.cur_best = 0
        self.cur_worst = 0
        for i in range(self.Popsize):
            if self.pop_fit[i] < self.pop_fit[self.cur_best]:
                self.cur_best = i
            elif self.pop_fit[i] > self.pop_fit[self.cur_worst]:
                self.cur_worst = i

    def Elist(self) -> None:
        if self.pop_fit[self.cur_best] < self.gbest_fit:
            self.gbest = self.pop[self.cur_best].copy()
            self.gbest_fit = self.pop_fit[self.cur_best]
        else:
            self.pop[self.cur_worst] = self.gbest.copy()
            self.pop_fit[self.cur_worst] = self.gbest_fit

    def average_fit(self) -> float:
        return sum(self.pop_fit) / self.Popsize

    def CRfit(self) -> None:
        self.CRold = self.CRnew
        if self.pop_fit[self.cur_worst] == 0:
            self.CRnew = 1.0
            return
        ave = self.average_fit() / self.pop_fit[self.cur_worst]
        best = self.pop_fit[self.cur_best] / self.pop_fit[self.cur_worst]
        self.CRnew = 1.0 if best == 1 else (1.0 - ave) / (1.0 - best)

    def square(self, d: float) -> float:
        return d * d

    def uniform(self) -> float:
        tmp = self.aktseed // 127773
        self.aktseed = 16807 * (self.aktseed - tmp * 127773) - 2836 * tmp
        if self.aktseed < 0:
            self.aktseed += 2147483647
        tmp = self.aktrand // 67108865
        self.aktrand = self.rgrand[tmp]
        self.rgrand[tmp] = self.aktseed
        return self.aktrand / 2.147483647e9

    def gauss(self) -> float:
        if self.stored:
            self.stored = False
            return self.hold
        self.stored = True
        while True:
            x1 = 2.0 * self.uniform() - 1.0
            x2 = 2.0 * self.uniform() - 1.0
            rquad = x1 * x1 + x2 * x2
            if 0.0 < rquad < 1.0:
                break
        fac = math.sqrt((-2.0) * math.log(rquad) / rquad)
        self.hold = fac * x1
        return fac * x2

    def myhypot(self, a: float, b: float) -> float:
        return math.hypot(a, b)
