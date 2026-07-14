from __future__ import annotations

import math
import random

class GAMixin:
    def select(self, p_start: int, p_end: int) -> None:
        scores = [1000.0 / (fit if fit != 0 else 1e-300) for fit in self.pop_fit]
        total = sum(scores)
        if total == 0:
            probs = [1.0 / self.Popsize for _ in range(self.Popsize)]
        else:
            probs = [score / total for score in scores]
        cfitness = []
        acc = 0.0
        for prob in probs:
            acc += prob
            cfitness.append(acc)
        for i in range(p_start, p_end):
            p = self.randval(0.0, 1.0)
            chosen = 0
            for j, cutoff in enumerate(cfitness):
                if p < cutoff:
                    chosen = j
                    break
            self.newpop[i] = self.pop[chosen].copy()

    def crossover(self, pc: float, p_start: int, p_end: int) -> None:
        for mem in range(p_start, p_end):
            pos = random.randrange(self.Popsize)
            while pos == mem:
                pos = random.randrange(self.Popsize)
            if self.randval(0.0, 1.0) < pc:
                self.xover(pos, mem)

    def xover(self, one: int, two: int) -> None:
        point = 1 if self.Nvar == 2 else random.randrange(1, self.Nvar)
        for i in range(point):
            r = self.randval(0.0, 1.0)
            temp1 = self.newpop[one][i] * r + (1.0 - r) * self.newpop[two][i]
            self.newpop[one][i] = self._clip(temp1)

    def mutate(self, pm: float, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            if self.randval(0.0, 1.0) < pm:
                self.newpop[i][random.randrange(self.Nvar)] = self.randval(self.Lbound, self.Ubound)

    def GA(self, pc: float, pm: float, p_start: int, p_end: int) -> None:
        self.select(p_start, p_end)
        self.crossover(pc, p_start, p_end)
        self.mutate(pm, p_start, p_end)

    def NGA(self, pc: float, pm: float, mdis: float, p_start: int, p_end: int) -> None:
        self.pop_niche(mdis, p_start, p_end)
        self.GA(pc, pm, p_start, p_end)

    def LGA(self, pc: float, pm: float, step: float, nst: int, p_start: int, p_end: int) -> None:
        self.GA(pc, pm, p_start, p_end)
        self.Evaluation(True, p_start, p_end)
        for i in range(p_start, p_end):
            row = self._fitness_row(self.newpop[i])
            self.localsearch(row, nst)
            self.newpop[i] = row[: self.Nvar]
            self.newpop_fit[i] = row[self.Nvar]

    def VAGA(self, p_start: int, p_end: int) -> None:
        vari = self.pop_random_variance(0, self.Popsize)
        p = math.exp(-vari)
        self.select(p_start, p_end)
        self.crossover(1.0 - p, p_start, p_end)
        self.mutate(p, p_start, p_end)

    def EAGA(self, p_start: int, p_end: int) -> None:
        entr = self.pop_random_entropy(max(1, int((self.Ubound - self.Lbound) * 0.1 / 2)), 0, self.Popsize)
        p = math.exp(-entr)
        self.select(p_start, p_end)
        self.crossover(1.0 - p, p_start, p_end)
        self.mutate(p, p_start, p_end)
