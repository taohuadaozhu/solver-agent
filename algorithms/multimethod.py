from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

from .population import PI, Population
from .problems import Sphere
from .algorithms import (
    ADEMixin,
    AESSPSOMixin,
    AdamMixin,
    AutoVMixin,
    BFGSMixin,
    CSOMixin,
    DEMixin,
    DOAMixin,
    CSAMixin,
    ECPOMixin,
    GAMixin,
    PSOMixin,
    HSMixin,
    ACOMixin,
    COAMixin,
    ABCAMixin,
    POAMixin,
    ILSMixin,
    VNSMixin,
    GRASPMixin,
    PBILCMixin,
    BATAMixin,
    FAMixin,
    CMAESMixin,
    BAMixin,
    EGOMixin,
    FEPMixin,
    FRCGMixin,
    FROFIMixin,
    GPSOMixin,
    GWOMixin,
    IMODEMixin,
    KMAMixin,
    L2SMEAMixin,
    MFEAMixin,
    MFEAIIMixin,
    MGOMixin,
    MVPAMixin,
    MiSACOMixin,
    NelderMeadMixin,
    OFAMixin,
    RMSPropMixin,
    SACCEAMIIMixin,
    SACOSOMixin,
    SAMixin,
    SADEAMSSMixin,
    SADEATDSCMixin,
    SADESammonMixin,
    SAMSOMixin,
    SAPOMixin,
    SDMixin,
    SHADEMixin,
    SQPMixin,
    SSIORLMixin,
    WOAMixin,
    LocalSearchMixin,
)

FitnessFunc = Callable[[list[float], int], float]

@dataclass(frozen=True)
class RunResult:
    best: float
    generations: int
    seconds: float

class MultiMet(
    ADEMixin,
    AESSPSOMixin,
    AdamMixin,
    AutoVMixin,
    BFGSMixin,
    CSOMixin,
    DEMixin,
    DOAMixin,
    CSAMixin,
    ECPOMixin,
    GAMixin,
    PSOMixin,
    HSMixin,
    ACOMixin,
    COAMixin,
    ABCAMixin,
    POAMixin,
    ILSMixin,
    VNSMixin,
    GRASPMixin,
    PBILCMixin,
    BATAMixin,
    FAMixin,
    CMAESMixin,
    BAMixin,
    EGOMixin,
    FEPMixin,
    FRCGMixin,
    FROFIMixin,
    GPSOMixin,
    GWOMixin,
    IMODEMixin,
    KMAMixin,
    L2SMEAMixin,
    MFEAMixin,
    MFEAIIMixin,
    MGOMixin,
    MVPAMixin,
    MiSACOMixin,
    NelderMeadMixin,
    OFAMixin,
    RMSPropMixin,
    SACCEAMIIMixin,
    SACOSOMixin,
    SAMixin,
    SADEAMSSMixin,
    SADEATDSCMixin,
    SADESammonMixin,
    SAMSOMixin,
    SAPOMixin,
    SDMixin,
    SHADEMixin,
    SQPMixin,
    SSIORLMixin,
    WOAMixin,
    LocalSearchMixin,
    Population[float],
):
    LOCAL_SEARCH_COUNT = 20

    def __init__(
        self,
        psize: int = 10,
        nn: int = 1000,
        lb: float = 0.0,
        ub: float = 10.0,
        evaluate: FitnessFunc = Sphere,
    ) -> None:
        super().__init__(psize, nn, lb, ub)
        self.EvaluFunc = evaluate

        self.ibest = self.CreateMatrix(self.Popsize, self.Nvar)
        self.velocity = self.CreateMatrix(self.Popsize, self.Nvar)
        self.ibest_fit = [0.0 for _ in range(self.Popsize)]
        self.AC1 = [2.0 for _ in range(self.Popsize)]
        self.AC2 = [2.0 for _ in range(self.Popsize)]
        self.AW = [0.85 for _ in range(self.Popsize)]
        self.OArow = self.Nvar + 1
        self.OA: list[list[int]] = []

        self.ant_tao = self.CreateMatrix(self.Popsize, self.Nvar + 2)
        self.trial = [0 for _ in range(self.Popsize)]
        self.pr = [0.0 for _ in range(self.Popsize)]
        self.neigh = [0 for _ in range(self.Popsize)]
        self.ngh = [0.0 for _ in range(self.Popsize)]
        self.ngh_decay_count = [0 for _ in range(self.Popsize)]
        self.Ind_meme = [0 for _ in range(self.Popsize)]
        self.SubDecBase = self.CreateMatrix(self.Popsize, self.Nvar + 2)
        self.ade_memory_size = 8
        self.ade_seri_size = 25
        self.ade_max_diff_terms = 4
        self.ade_generation = 0
        self.ade_stagnant = 0
        self.ade_active_size = self.Popsize
        self.ade_memory_f1 = [0.45 for _ in range(self.ade_memory_size)]
        self.ade_memory_f2 = [0.25 for _ in range(self.ade_memory_size)]
        self.ade_memory_cr = [0.85 for _ in range(self.ade_memory_size)]
        self.ade_memory_pos = 0
        self.ade_success_history = [0.0 for _ in range(self.Popsize)]
        self.ade_path_center = [0.0 for _ in range(self.Nvar)]
        self.ade_recent_path = [0.0 for _ in range(self.Nvar)]
        self.ade_path_ready = False
        self.ade_policy = self.default_seri_policy()
        self.meme_q: list[list[float]] = []
        self.meme_q_trials: list[list[int]] = []
        self.meme_trials = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]
        self.meme_rewards = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]
        self.meme_successes = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]
        self.sade_amss_archive: list[list[float]] = []
        self.sade_amss_archive_fit: list[float] = []

    def randnorm(self, miu: float, score: float) -> float:
        return miu + score * math.sqrt(abs(-2.0 * math.log(random.random()))) * math.cos(
            2.0 * PI * random.random()
        )

    def pop_update(self, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            self.pop[i] = self.newpop[i].copy()
            self.pop_fit[i] = self.newpop_fit[i]
            if self.newpop_fit[i] < self.ibest_fit[i]:
                self.ibest[i] = self.newpop[i].copy()
                self.ibest_fit[i] = self.newpop_fit[i]

    def pop_better_update(self, p_start: int, p_end: int) -> None:
        for i in range(p_start, p_end):
            if self.newpop_fit[i] < self.pop_fit[i]:
                self.pop[i] = self.newpop[i].copy()
                self.pop_fit[i] = self.newpop_fit[i]
            if self.newpop_fit[i] < self.ibest_fit[i]:
                self.ibest[i] = self.newpop[i].copy()
                self.ibest_fit[i] = self.newpop_fit[i]

    def CreateOA(self) -> None:
        if self.OArow <= 1:
            return
        if self.OArow * self.Nvar > 2_000_000:
            self.OA = []
            return
        self.OA = [[0 for _ in range(self.Nvar)] for _ in range(self.OArow)]
        u = int(math.log(self.OArow) / math.log(2.0))
        for i in range(self.OArow):
            for j in range(u):
                b = int(math.pow(2.0, j)) - 1
                if b >= self.Nvar:
                    continue
                tmp = math.floor(i / math.pow(2.0, u - j - 1))
                self.OA[i][b] = int(tmp) % 2

        for i in range(self.OArow):
            for j in range(u):
                b = int(math.pow(2.0, j)) - 1
                for s in range(b):
                    target = b + s + 1
                    if target < self.Nvar:
                        self.OA[i][target] = (self.OA[i][s] + self.OA[i][b]) % 2

    def OAValue(self, row: int, col: int) -> int:
        if self.OA:
            return self.OA[row][col]
        u = int(math.log(self.OArow) / math.log(2.0))
        marker = col + 1
        if marker >= 2**u:
            return 0
        value = 0
        for j in range(u):
            if marker & (1 << j):
                tmp = math.floor(row / math.pow(2.0, u - j - 1))
                value ^= int(tmp) % 2
        return value

    def Initial(self) -> None:
        for i in range(self.Popsize):
            for j in range(self.Nvar):
                value = self.randval(self.Lbound, self.Ubound)
                self.pop[i][j] = value
                self.newpop[i][j] = value
                self.velocity[i][j] = self.randval(self.Lbound, self.Ubound)

        for i in range(self.Popsize):
            self.ibest[i] = self.pop[i].copy()
            self.pop_fit[i] = self.EvaluFunc(self.pop[i], self.Nvar)
            self.newpop_fit[i] = self.pop_fit[i]
            self.ibest_fit[i] = self.pop_fit[i]

        self.worst_and_best()
        self.gbest = self.pop[self.cur_best].copy()
        self.gbest_fit = self.pop_fit[self.cur_best]
        self.CRfit()
        self.CRold = self.CRnew

        self.ac1 = 2.0
        self.ac2 = 2.0
        for i in range(self.Popsize):
            self.AC1[i] = 2.0
            self.AC2[i] = 2.0
            self.AW[i] = 0.85
        self.CreateOA()

        for i in range(self.Popsize):
            for j in range(self.Nvar):
                self.ant_tao[i][j] = self.randval(self.Lbound, self.Ubound)
            self.ant_tao[i][self.Nvar] = self.EvaluFunc(self.ant_tao[i], self.Nvar)
        self.heap_sort(self.ant_tao, self.Popsize, self.Nvar)
        for i in range(self.Popsize):
            sigma = 1e-4 * self.Popsize
            self.ant_tao[i][self.Nvar + 1] = math.exp(-(i**2.0) / (2.0 * sigma**2.0)) / (
                sigma * math.sqrt(2.0 * PI)
            )

        for i in range(self.Popsize):
            self.trial[i] = 0
            self.neigh[i] = random.randrange(self.Nvar)
            self.Ind_meme[i] = random.randrange(self._local_search_count())
        self.ade_generation = 0
        self.ade_stagnant = 0
        self.ade_active_size = self.Popsize
        self.ade_memory_f1 = [0.45 for _ in range(self.ade_memory_size)]
        self.ade_memory_f2 = [0.25 for _ in range(self.ade_memory_size)]
        self.ade_memory_cr = [0.85 for _ in range(self.ade_memory_size)]
        self.ade_memory_pos = 0
        self.ade_success_history = [0.0 for _ in range(self.Popsize)]
        self.ade_path_center = [0.0 for _ in range(self.Nvar)]
        self.ade_recent_path = [0.0 for _ in range(self.Nvar)]
        self.ade_path_ready = False
        self.ade_policy = self.default_seri_policy()
        self.meme_q = []
        self.meme_q_trials = []
        self.meme_trials = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]
        self.meme_rewards = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]
        self.meme_successes = [0.0 for _ in range(self.LOCAL_SEARCH_COUNT)]

        self.num_of_elite = min(2, self.Popsize // 2)
        self.PB_center = self.pop[0].copy()
        self.PB_sigma = [abs(self.Ubound - self.Lbound) / 2.0 for _ in range(self.Nvar)]
        self.BAT_r = [self.randval(0.0, 1.0) for _ in range(self.Popsize)]
        self.BAT_A = [1.0 for _ in range(self.Popsize)]
        self.BAT_v = self.CreateMatrix(self.Popsize, self.Nvar)
        self.I = self.pop_fit.copy()
        self.Index = list(range(self.Popsize))
        self.ne = self.Popsize // 5
        self.nb = self.Popsize // 5 * 2
        self.nre = 20
        self.nrb = 10
        self.stlim = 10
        self.ngh_decay = 0.8
        self.ngh_origin = (self.Ubound - self.Lbound) * 0.1
        self.ngh = [self.ngh_origin for _ in range(self.Popsize)]
        self.ngh_decay_count = [0 for _ in range(self.Popsize)]

    def Evaluation(self, s: bool, p_start: int, p_end: int) -> None:
        target = self.newpop_fit if s else self.pop_fit
        source = self.newpop if s else self.pop
        for i in range(p_start, p_end):
            target[i] = self.EvaluFunc(source[i], self.Nvar)

    def _clip(self, value: float) -> float:
        return min(max(value, self.Lbound), self.Ubound)

    def _wrap(self, value: float) -> float:
        if value > self.Ubound:
            return self.Lbound
        if value < self.Lbound:
            return self.Ubound
        return value

    def _fitness_row(self, row: list[float]) -> list[float]:
        result = row[: self.Nvar]
        result.append(self.EvaluFunc(result, self.Nvar))
        return result

def run_csa(
    nvar: int = 1000,
    popsize: int = 20,
    maxgen: int | None = None,
    threshold: float = 1e-6,
    lb: float = -5.0,
    ub: float = 10.0,
    seed: int | None = None,
    verbose: bool = False,
) -> RunResult:
    if seed is not None:
        random.seed(seed)
    maxgen = maxgen if maxgen is not None else 100
    solver = MultiMet(popsize, nvar, lb, ub, Sphere)
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
        if verbose:
            print(solver.gbest_fit)
    return RunResult(solver.gbest_fit, generation, time.perf_counter() - started)

def main() -> None:
    parser = argparse.ArgumentParser(description="Python port of the active MultiAlg CSA run.")
    parser.add_argument("--nvar", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--maxgen", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_csa(
        nvar=args.nvar,
        popsize=args.popsize,
        maxgen=args.maxgen,
        threshold=args.threshold,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"The best solution = {result.best:.12g}")
    print(f"Generations = {result.generations}")
    print(f"Time = {result.seconds:.6f} s")

if __name__ == "__main__":
    main()
