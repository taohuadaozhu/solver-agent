from __future__ import annotations

import math
import random


class ADEMixin:
    ADE_BASE_COUNT = 8
    ADE_DIFF_MODE_COUNT = 9
    ADE_MUTATION_COUNT = 7
    HYPER_STATE_COUNT = 4

    def seri_diff_left_index(self, term: int) -> int:
        return 9 + term * 3

    def seri_diff_right_index(self, term: int) -> int:
        return 10 + term * 3

    def seri_diff_weight_index(self, term: int) -> int:
        return 11 + term * 3

    def search_state(self, progress: float, stagnant_generations: int) -> int:
        threshold = 80 if progress < 0.30 else 50
        if stagnant_generations >= threshold:
            return 3
        if progress < 0.30:
            return 0
        if progress < 0.75:
            return 1
        return 2

    def default_seri_policy(self) -> dict[str, list[list[float]]]:
        return {
            "base": [
                [0.10, 0.55, 0.35, 0.12, 1.05, 0.42, 0.85, 0.30],
                [0.10, 0.55, 0.38, 0.10, 1.10, 0.38, 0.95, 0.34],
                [0.08, 0.45, 0.45, 0.08, 1.15, 0.30, 1.00, 0.38],
                [0.14, 0.50, 0.34, 0.16, 1.00, 0.48, 0.82, 0.42],
            ],
            "mutation": [
                [1.05, 0.42, 0.18, 0.55, 0.12, 0.18, 0.52],
                [1.12, 0.36, 0.14, 0.50, 0.10, 0.12, 0.62],
                [1.20, 0.28, 0.10, 0.46, 0.08, 0.08, 0.56],
                [0.92, 0.46, 0.24, 0.56, 0.16, 0.18, 0.74],
            ],
            "diff": [
                [1.00, 0.35, 0.12, 0.06],
                [1.06, 0.28, 0.10, 0.05],
                [1.12, 0.22, 0.08, 0.04],
                [0.86, 0.42, 0.16, 0.07],
            ],
            "diff_mode": [
                [0.95, 0.42, 0.38, 0.70, 0.85, 0.26, 0.45, 0.20, 0.62],
                [0.90, 0.34, 0.44, 0.82, 0.95, 0.20, 0.50, 0.16, 0.72],
                [0.82, 0.24, 0.52, 0.90, 1.05, 0.14, 0.58, 0.12, 0.66],
                [0.92, 0.46, 0.36, 0.74, 0.86, 0.30, 0.46, 0.24, 0.86],
            ],
        }

    def choose_policy_action(
        self,
        policy: list[float],
        count: int,
        epsilon: float = 0.0,
        squared_weight: bool = False,
    ) -> int:
        if count <= 1:
            return 0
        if self.randval(0.0, 1.0) < epsilon:
            return random.randrange(count)
        weights = [max(1e-6, policy[i]) for i in range(count)]
        if squared_weight:
            weights = [value * value for value in weights]
        total = sum(weights)
        pick = self.randval(0.0, total)
        acc = 0.0
        for idx, weight in enumerate(weights):
            acc += weight
            if pick <= acc:
                return idx
        return count - 1

    def sample_left_source(self, active_limit: int) -> int:
        r = self.randval(0.0, 1.0)
        if r < 0.45:
            return self.Popsize
        if r < 0.65:
            return self.Popsize + 2
        if r < 0.82:
            return random.randrange(active_limit)
        if r < 0.92:
            return self.Popsize + 1
        return random.randrange(active_limit)

    def sample_right_source(self, active_limit: int) -> int:
        r = self.randval(0.0, 1.0)
        if r < 0.62:
            return random.randrange(active_limit)
        if r < 0.78:
            return self.Popsize + 2
        if r < 0.90:
            return self.Popsize + 1
        return self.Popsize

    def normalize_seri(self, seri: list[float], progress: float) -> None:
        active = max(1, self.ade_active_size)
        seri[0] = int(max(0, min(self.ADE_BASE_COUNT - 1, int(seri[0]))))
        seri[1] = int(max(0, min(active - 1, int(seri[1]))))
        seri[2] = int(max(1, min(self.ade_max_diff_terms, int(seri[2]))))
        seri[3] = max(0.05, min(1.0, seri[3]))
        seri[4] = int(max(0, min(self.ADE_MUTATION_COUNT - 1, int(seri[4]))))
        seri[5] = max(0.0, min(0.16, seri[5]))
        seri[6] = max(1e-6, min(max(0.0015, 0.05 * (1.0 - progress) + 0.0015), seri[6]))
        seri[7] = max(0.0, min(0.38, seri[7]))
        seri[8] = int(max(0, min(self.ADE_DIFF_MODE_COUNT - 1, int(seri[8]))))
        for term in range(self.ade_max_diff_terms):
            left = self.seri_diff_left_index(term)
            right = self.seri_diff_right_index(term)
            weight = self.seri_diff_weight_index(term)
            seri[left] = int(max(0, min(self.Popsize + 2, int(seri[left]))))
            seri[right] = int(max(0, min(self.Popsize + 2, int(seri[right]))))
            if int(seri[left]) == int(seri[right]):
                seri[right] = (int(seri[right]) + 1) % active
            seri[weight] = max(0.03, min(max(0.08, 0.55 - 0.25 * progress), seri[weight]))
        seri[21] = max(0.0, min(0.24, seri[21]))
        seri[22] = int(max(0, min(3, int(seri[22]))))
        seri[23] = max(1e-6, min(max(0.001, 0.06 * (1.0 - progress) + 0.001), seri[23]))
        seri[24] = max(0.0, min(0.25, seri[24]))

    def sample_seri(
        self,
        seri: list[float] | None = None,
        mf1: list[float] | None = None,
        mf2: list[float] | None = None,
        mcr: list[float] | None = None,
        progress: float = 0.0,
        hyper_state: int | None = None,
    ) -> list[float]:
        seri = [0.0 for _ in range(self.ade_seri_size)] if seri is None else seri
        mf1 = self.ade_memory_f1 if mf1 is None else mf1
        mf2 = self.ade_memory_f2 if mf2 is None else mf2
        mcr = self.ade_memory_cr if mcr is None else mcr
        state = self.search_state(progress, self.ade_stagnant) if hyper_state is None else hyper_state
        state = max(0, min(self.HYPER_STATE_COUNT - 1, state))
        explore = 1.0 - progress
        policy = self.ade_policy
        m = random.randrange(self.ade_memory_size)
        seri[0] = self.choose_policy_action(policy["base"][state], self.ADE_BASE_COUNT, 0.12 * explore, progress > 0.55)
        seri[1] = random.randrange(max(1, self.ade_active_size))
        seri[2] = self.choose_policy_action(policy["diff"][state], self.ade_max_diff_terms, 0.10 * explore, True) + 1
        seri[3] = mcr[m] + self.randnorm(0.0, 0.22 * explore + 0.03)
        seri[4] = self.choose_policy_action(policy["mutation"][state], self.ADE_MUTATION_COUNT, 0.12 * explore, True)
        seri[5] = 0.003 + self.randval(0.0, 0.04 + 0.08 * explore)
        seri[6] = 0.001 + self.randval(0.0, 0.035 * explore + 0.012)
        seri[7] = self.randval(0.02, 0.25 + 0.12 * explore)
        seri[8] = self.choose_policy_action(policy["diff_mode"][state], self.ADE_DIFF_MODE_COUNT, 0.12 * explore, True)
        active = max(1, self.ade_active_size)
        for term in range(self.ade_max_diff_terms):
            seri[self.seri_diff_left_index(term)] = self.sample_left_source(active)
            seri[self.seri_diff_right_index(term)] = self.sample_right_source(active)
            memory = mf1 if term == 0 else mf2
            seri[self.seri_diff_weight_index(term)] = memory[m] + self.randnorm(0.0, 0.18 * explore + 0.03)
        seri[21] = (0.006 + 0.05 * explore) if self.ade_path_ready else 0.0
        seri[22] = random.randrange(4)
        seri[23] = 0.001 + self.randval(0.0, 0.014 * explore + 0.006)
        seri[24] = 0.12 if self.gbest_fit <= 100.0 else 0.22
        self.normalize_seri(seri, progress)
        return seri

    def reset_seri_pool(
        self,
        seri_pool: list[list[float]],
        mf1: list[float] | None = None,
        mf2: list[float] | None = None,
        mcr: list[float] | None = None,
        progress: float = 0.0,
        hyper_state: int | None = None,
    ) -> None:
        for i in range(len(seri_pool)):
            seri_pool[i] = self.sample_seri(seri_pool[i], mf1, mf2, mcr, progress, hyper_state)

    def _ade_source(self, source: int, index: int) -> list[float]:
        active = max(1, self.ade_active_size)
        if source == self.Popsize:
            return self.gbest
        if source == self.Popsize + 1:
            return self.ibest[index]
        if source == self.Popsize + 2:
            return self.pop[index]
        return self.ibest[source % active]

    def _ade_base(self, seri: list[float], index: int, elite_mean: list[float]) -> list[float]:
        mode = int(seri[0])
        base_index = int(seri[1]) % max(1, self.ade_active_size)
        if mode == 1:
            return self.ibest[index].copy()
        if mode == 2:
            return self.gbest.copy()
        if mode == 3:
            return self.ibest[random.randrange(max(1, self.ade_active_size))].copy()
        if mode == 4:
            f = seri[self.seri_diff_weight_index(0)]
            return [self.pop[index][j] + f * (self.gbest[j] - self.pop[index][j]) for j in range(self.Nvar)]
        if mode == 5:
            return self.ibest[base_index].copy()
        if mode == 6:
            pbest = self.ibest[random.randrange(max(1, self.ade_active_size))]
            f = seri[self.seri_diff_weight_index(0)]
            return [self.pop[index][j] + f * (pbest[j] - self.pop[index][j]) for j in range(self.Nvar)]
        if mode == 7 and elite_mean:
            return elite_mean.copy()
        return self.pop[index].copy()

    def build_ade_candidate(
        self,
        seri: list[float],
        index: int,
        elite_indices: list[int] | None = None,
        elite_mean: list[float] | None = None,
    ) -> None:
        active = max(1, self.ade_active_size)
        elite_indices = elite_indices or [self.cur_best]
        elite_mean = elite_mean or []
        trial = self._ade_base(seri, index, elite_mean)
        mode = int(seri[8])
        diff_count = int(seri[2])
        scale = 1.0 / math.sqrt(max(1, diff_count))
        for term in range(diff_count):
            left = self._ade_source(int(seri[self.seri_diff_left_index(term)]), index)
            right = self._ade_source(int(seri[self.seri_diff_right_index(term)]), index)
            weight = seri[self.seri_diff_weight_index(term)] * scale
            for j in range(self.Nvar):
                trial[j] += weight * (left[j] - right[j])
        if mode in (2, 3):
            for j in range(self.Nvar):
                trial[j] += 0.5 * seri[self.seri_diff_weight_index(0)] * (self.gbest[j] - self.pop[index][j])
        elif mode == 4:
            pbest = self.ibest[random.choice(elite_indices)]
            for j in range(self.Nvar):
                trial[j] += seri[self.seri_diff_weight_index(0)] * (pbest[j] - self.pop[index][j])
        elif mode == 7:
            bound_sum = self.Lbound + self.Ubound
            for j in range(self.Nvar):
                trial[j] += 0.5 * seri[self.seri_diff_weight_index(0)] * (bound_sum - self.pop[index][j] - trial[j])
        elif mode == 8:
            neighbor = self.ibest[random.randrange(active)]
            phi = self.randval(-1.0, 1.0)
            for j in range(self.Nvar):
                trial[j] += phi * seri[self.seri_diff_weight_index(0)] * (neighbor[j] - self.pop[index][j])
        mutation_type = int(seri[4])
        mutation_scale = seri[6] * abs(self.Ubound - self.Lbound)
        forced = random.randrange(self.Nvar)
        cr = seri[3]
        for j in range(self.Nvar):
            value = trial[j] if j == forced or self.randval(0.0, 1.0) <= cr else self.pop[index][j]
            if mutation_type == 1 and self.randval(0.0, 1.0) < seri[5]:
                value += self.randnorm(0.0, mutation_scale)
            elif mutation_type == 2 and self.randval(0.0, 1.0) < seri[5]:
                value += mutation_scale * max(-4.0, min(4.0, math.tan(math.pi * (self.randval(0.0, 1.0) - 0.5))))
            elif mutation_type == 3:
                value += seri[7] * (self.gbest[j] - value)
            elif mutation_type == 4 and self.randval(0.0, 1.0) < seri[5]:
                value = self.randval(self.Lbound, self.Ubound)
            elif mutation_type == 5 and self.randval(0.0, 1.0) < seri[5]:
                value = 0.5 * value + 0.5 * (self.Lbound + self.Ubound - value)
            elif mutation_type == 6 and self.randval(0.0, 1.0) < seri[5]:
                value += seri[7] * self.randval(-1.0, 1.0) * (self.ibest[random.randrange(active)][j] - value)
            if self.ade_path_ready and seri[21] > 0.0:
                value += seri[21] * seri[23] * self.ade_path_center[j]
            self.newpop[index][j] = self._clip(value)

    def collect_ade_trial_stats(self, seri_pool, sflag, stats=None, p_start=0, p_end=None):
        return {
            "success_count": sum(1 for flag in sflag[p_start:p_end] if flag),
            "total_improvement": sum(max(0.0, value) for value in self.ade_success_history),
        }

    def update_ade_memory(self, mf1=None, mf2=None, mcr=None, memory_pos=None, stats=None) -> None:
        if not any(value > 0.0 for value in self.ade_success_history):
            return
        pos = self.ade_memory_pos
        self.ade_memory_f1[pos] = max(0.05, min(0.8, self.ade_memory_f1[pos] * 0.95 + 0.05 * 0.5))
        self.ade_memory_f2[pos] = max(0.03, min(0.6, self.ade_memory_f2[pos] * 0.95 + 0.05 * 0.25))
        self.ade_memory_cr[pos] = max(0.05, min(1.0, self.ade_memory_cr[pos] * 0.95 + 0.05 * 0.9))
        self.ade_memory_pos = (pos + 1) % self.ade_memory_size

    def update_seri_policy(self, hyper_state: int, stats=None) -> None:
        return None

    def share_seri_policy(self, source_state: int, rate: float) -> None:
        source_state = max(0, min(self.HYPER_STATE_COUNT - 1, source_state))
        for key, table in self.ade_policy.items():
            for state, row in enumerate(table):
                if state == source_state:
                    continue
                for i in range(len(row)):
                    row[i] += rate * (table[source_state][i] - row[i])

    def ade_active_population_size(self) -> int:
        return max(1, min(self.Popsize, self.ade_active_size))

    def update_shade_population_size(self, progress: float, stagnant_generations: int, stats=None) -> None:
        floor = max(4, self.Popsize // 3)
        target = int(round(self.Popsize - (self.Popsize - floor) * min(1.0, max(0.0, progress))))
        if stagnant_generations > 30:
            target = min(self.Popsize, target + max(1, self.Popsize // 5))
        self.ade_active_size = max(floor, min(self.Popsize, target))

    def adapt_successful_seri(self, seri, mf1=None, mf2=None, mcr=None, progress: float = 0.0) -> None:
        self.normalize_seri(seri, progress)

    def compact_active_population(self) -> None:
        order = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        for slot, idx in enumerate(order):
            if slot >= idx:
                continue
            self.pop[slot], self.pop[idx] = self.pop[idx], self.pop[slot]
            self.pop_fit[slot], self.pop_fit[idx] = self.pop_fit[idx], self.pop_fit[slot]

    def allocate_algorithm_state(self) -> None:
        self.ade_policy = self.default_seri_policy()

    def ADE(self, seri_pool=None, p_start: int = 0, p_end: int | None = None) -> None:
        p_end = self.Popsize if p_end is None else min(p_end, self.Popsize)
        p_start = max(0, p_start)
        if p_start >= p_end:
            return
        progress = min(1.0, self.ade_generation / 100.0)
        hyper_state = self.search_state(progress, self.ade_stagnant)
        if seri_pool is None:
            seri_pool = [[0.0 for _ in range(self.ade_seri_size)] for _ in range(self.Popsize)]
        if len(seri_pool) < self.Popsize:
            seri_pool.extend([[0.0 for _ in range(self.ade_seri_size)] for _ in range(self.Popsize - len(seri_pool))])
        self.reset_seri_pool(seri_pool, progress=progress, hyper_state=hyper_state)
        elite_count = max(1, min(self.Popsize, int(math.sqrt(self.Popsize)) + 1))
        elite_indices = sorted(range(self.Popsize), key=lambda idx: self.ibest_fit[idx])[:elite_count]
        elite_mean = [
            sum(self.ibest[idx][j] for idx in elite_indices) / elite_count
            for j in range(self.Nvar)
        ]
        old_best = self.gbest_fit
        for i in range(p_start, p_end):
            self.build_ade_candidate(seri_pool[i], i, elite_indices, elite_mean)
        self.ade_generation += 1
        self.ade_stagnant = self.ade_stagnant + 1 if self.gbest_fit >= old_best else 0
