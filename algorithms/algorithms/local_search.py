from __future__ import annotations

import math
import random

class LocalSearchMixin:
    def newpop_bit_climbing(self, popi: int, L: int, scale: float) -> None:
        permu = list(range(self.Nvar))
        random.shuffle(permu)
        for j in range(min(L, self.Nvar)):
            temp = self.newpop[popi].copy()
            bit = permu[j]
            temp[bit] = self._clip(temp[bit] + scale * self.randval(self.Lbound, self.Ubound))
            fit = self.EvaluFunc(temp, self.Nvar)
            if fit < self.newpop_fit[popi]:
                self.newpop[popi] = temp
                self.newpop_fit[popi] = fit

    def _candidate_row(self, values: list[float]) -> list[float]:
        candidate = [self._clip(values[j]) for j in range(self.Nvar)]
        candidate.append(self.EvaluFunc(candidate, self.Nvar))
        return candidate

    def _random_dims(self, count: int) -> list[int]:
        count = max(1, min(count, self.Nvar))
        if count >= self.Nvar:
            return list(range(self.Nvar))
        return random.sample(range(self.Nvar), count)

    def _local_search_count(self) -> int:
        return self.LOCAL_SEARCH_COUNT

    def _line_search_row(
        self,
        row: list[float],
        direction: list[float],
        step: float,
        budget: int,
    ) -> list[float]:
        best = row
        alpha = max(abs(step), 1e-12)
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-15:
            return best
        unit = [value / norm for value in direction]
        for _ in range(max(1, budget)):
            improved = False
            for sign in (1.0, -1.0):
                candidate = self._candidate_row([best[j] + sign * alpha * unit[j] for j in range(self.Nvar)])
                if candidate[self.Nvar] < best[self.Nvar]:
                    best = candidate
                    improved = True
            alpha *= 1.15 if improved else 0.5
            if alpha < 1e-12:
                break
        return best

    def newpop_simplex(self, popi: int, L: int, stepn: int, scale: float) -> None:
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = self._random_dims(max(2, stepn))
        simplex = [self._candidate_row(self.newpop[popi])]
        for j in dims:
            cand = self.newpop[popi].copy()
            cand[j] += step
            simplex.append(self._candidate_row(cand))
        for _ in range(max(1, L)):
            simplex.sort(key=lambda row: row[self.Nvar])
            best = simplex[0]
            worst = simplex[-1]
            centroid = [sum(row[j] for row in simplex[:-1]) / max(1, len(simplex) - 1) for j in range(self.Nvar)]
            reflected = self._candidate_row([centroid[j] + (centroid[j] - worst[j]) for j in range(self.Nvar)])
            if reflected[self.Nvar] < best[self.Nvar]:
                expanded = self._candidate_row(
                    [
                        centroid[j] + 2.0 * (reflected[j] - centroid[j])
                        for j in range(self.Nvar)
                    ]
                )
                simplex[-1] = expanded if expanded[self.Nvar] < reflected[self.Nvar] else reflected
            elif reflected[self.Nvar] < simplex[-2][self.Nvar]:
                simplex[-1] = reflected
            else:
                contracted = self._candidate_row(
                    [
                        centroid[j] + 0.5 * (worst[j] - centroid[j])
                        for j in range(self.Nvar)
                    ]
                )
                if contracted[self.Nvar] < worst[self.Nvar]:
                    simplex[-1] = contracted
                else:
                    simplex = [best] + [
                        self._candidate_row([best[j] + 0.5 * (row[j] - best[j]) for j in range(self.Nvar)])
                        for row in simplex[1:]
                    ]
        best = min(simplex, key=lambda row: row[self.Nvar])
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_box_complex(self, popi: int, L: int, stepN: int, scale: float) -> None:
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = self._random_dims(max(2, stepN))
        complex_size = max(2, min(max(stepN, 2), 2 * len(dims)))
        candidates = [self._candidate_row(self.newpop[popi])]
        for k in range(1, complex_size):
            cand = self.newpop[popi].copy()
            bit = dims[(k - 1) % len(dims)]
            cand[bit] += step * (1.0 if k <= self.Nvar else self.randval(-1.0, 1.0))
            if k > self.Nvar:
                cand = [x + self.randnorm(0.0, 1.0) * step * 0.25 for x in cand]
            candidates.append(self._candidate_row(cand))
        for _ in range(max(1, L)):
            candidates.sort(key=lambda row: row[self.Nvar])
            best = candidates[0]
            centroid = [sum(row[j] for row in candidates[:-1]) / max(1, len(candidates) - 1) for j in range(self.Nvar)]
            worst = candidates[-1]
            replacement = self._candidate_row([centroid[j] + 1.3 * (centroid[j] - worst[j]) for j in range(self.Nvar)])
            contraction = 0
            while replacement[self.Nvar] >= worst[self.Nvar] and contraction < 3:
                replacement = self._candidate_row([0.5 * (replacement[j] + best[j]) for j in range(self.Nvar)])
                contraction += 1
            if replacement[self.Nvar] < worst[self.Nvar]:
                candidates[-1] = replacement
            else:
                candidates[-1] = self._candidate_row([best[j] + 0.5 * (worst[j] - best[j]) for j in range(self.Nvar)])
        best = min(candidates, key=lambda row: row[self.Nvar])
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_powell(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = self._random_dims(max(1, stepn))
        directions = [[1.0 if i == j else 0.0 for i in range(self.Nvar)] for j in dims]
        cycles = min(max(1, L), 4)
        line_budget = min(max(1, stepn // 2), 2)
        for _ in range(cycles):
            start = current
            improved = False
            for direction in directions:
                candidate = self._line_search_row(current, direction, step, line_budget)
                if candidate[self.Nvar] < current[self.Nvar]:
                    current = candidate
                    improved = True
            displacement = [current[j] - start[j] for j in range(self.Nvar)]
            candidate = self._line_search_row(current, displacement, step, line_budget)
            if candidate[self.Nvar] < current[self.Nvar]:
                current = candidate
                directions = directions[1:] + [displacement]
                improved = True
            if not improved:
                step *= 0.5
            else:
                step *= 1.05
            if step < 1e-10:
                break
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_newton(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-8)
        fd_step = max(step * 0.1, 1e-6)
        coord_budget = min(self.Nvar, max(1, stepn))
        for _ in range(max(1, L)):
            direction = [0.0 for _ in range(self.Nvar)]
            f0 = current[self.Nvar]
            for bit in self._random_dims(coord_budget):
                plus = current[: self.Nvar]
                minus = current[: self.Nvar]
                plus[bit] += fd_step
                minus[bit] -= fd_step
                fp = self._candidate_row(plus)[self.Nvar]
                fm = self._candidate_row(minus)[self.Nvar]
                grad = (fp - fm) / (2.0 * fd_step)
                hdiag = (fp - 2.0 * f0 + fm) / (fd_step * fd_step)
                if hdiag > 1e-12:
                    delta = -grad / hdiag
                else:
                    delta = -math.copysign(step, grad) if grad != 0.0 else 0.0
                direction[bit] = max(-step, min(step, delta))
            candidate = self._line_search_row(current, direction, 1.0, max(1, min(3, stepn)))
            if candidate[self.Nvar] < current[self.Nvar]:
                current = candidate
                step *= 1.1
                fd_step = max(fd_step * 0.9, 1e-7)
            else:
                step *= 0.5
                fd_step *= 0.5
            if step < 1e-10:
                break
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_mcts(self, popi: int, L: int, stepn: int, scale: float) -> None:
        root = self._candidate_row(self.newpop[popi])
        best = root
        current = root
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = self._random_dims(max(2, stepn))
        actions: list[tuple[int, float, float]] = []
        for dim in dims:
            actions.append((dim, 1.0, step))
            actions.append((dim, -1.0, step))
            actions.append((dim, 1.0, step * 0.5))
            actions.append((dim, -1.0, step * 0.5))
        visits = [0 for _ in actions]
        rewards = [0.0 for _ in actions]
        total_visits = 0
        iterations = max(1, L) * max(1, stepn)
        for _ in range(iterations):
            total_visits += 1
            unexplored = [idx for idx, seen in enumerate(visits) if seen == 0]
            if unexplored:
                action_idx = random.choice(unexplored)
            else:
                log_total = math.log(total_visits + 1.0)
                action_idx = max(
                    range(len(actions)),
                    key=lambda idx: rewards[idx] / visits[idx] + 1.4 * math.sqrt(log_total / visits[idx]),
                )
            dim, sign, radius = actions[action_idx]
            rollout = current[: self.Nvar]
            rollout[dim] += sign * radius
            for rd in self._random_dims(max(1, min(3, stepn))):
                rollout[rd] += self.randval(-radius, radius) * 0.5
            candidate = self._candidate_row(rollout)
            reward = max(0.0, current[self.Nvar] - candidate[self.Nvar])
            visits[action_idx] += 1
            rewards[action_idx] += reward
            if candidate[self.Nvar] < best[self.Nvar]:
                best = candidate
            if candidate[self.Nvar] < current[self.Nvar] or self.randval(0.0, 1.0) < 0.05:
                current = candidate
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_simulated_annealing(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        best = current
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        temperature = max(step, 1e-9)
        iterations = max(1, L) * max(1, stepn)
        for _ in range(iterations):
            candidate_values = current[: self.Nvar]
            for bit in self._random_dims(max(1, min(stepn, self.Nvar))):
                candidate_values[bit] += self.randnorm(0.0, step)
            candidate = self._candidate_row(candidate_values)
            delta = candidate[self.Nvar] - current[self.Nvar]
            if delta < 0.0 or self.randval(0.0, 1.0) < math.exp(-min(700.0, delta / max(temperature, 1e-12))):
                current = candidate
                if current[self.Nvar] < best[self.Nvar]:
                    best = current
            temperature *= 0.9
            step *= 0.97
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_tabu_search(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        best = current
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        tenure = max(2, min(self.Nvar, stepn + 2))
        tabu: list[int] = []
        for _ in range(max(1, L)):
            trial_best = None
            for bit in self._random_dims(max(1, min(stepn * 2, self.Nvar))):
                if bit in tabu and current[self.Nvar] >= best[self.Nvar]:
                    continue
                for sign in (1.0, -1.0):
                    values = current[: self.Nvar]
                    values[bit] += sign * step
                    candidate = self._candidate_row(values)
                    if trial_best is None or candidate[self.Nvar] < trial_best[self.Nvar]:
                        trial_best = candidate
                        trial_dim = bit
            if trial_best is None:
                step *= 0.5
                continue
            current = trial_best
            tabu.append(trial_dim)
            if len(tabu) > tenure:
                tabu.pop(0)
            if current[self.Nvar] < best[self.Nvar]:
                best = current
                step *= 1.05
            else:
                step *= 0.7
            if step < 1e-10:
                break
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_pattern_search(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        coord_budget = max(1, min(stepn, self.Nvar))
        for _ in range(max(1, L)):
            start = current
            improved = False
            for bit in self._random_dims(coord_budget):
                best_neighbor = current
                for sign in (1.0, -1.0):
                    values = current[: self.Nvar]
                    values[bit] += sign * step
                    candidate = self._candidate_row(values)
                    if candidate[self.Nvar] < best_neighbor[self.Nvar]:
                        best_neighbor = candidate
                if best_neighbor[self.Nvar] < current[self.Nvar]:
                    current = best_neighbor
                    improved = True
            if improved:
                pattern = [current[j] + (current[j] - start[j]) for j in range(self.Nvar)]
                candidate = self._candidate_row(pattern)
                if candidate[self.Nvar] < current[self.Nvar]:
                    current = candidate
                step *= 1.15
            else:
                step *= 0.5
            if step < 1e-10:
                break
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_threshold_accepting(self, popi: int, L: int, stepn: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        best = current
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        threshold = max(abs(current[self.Nvar]) * 0.01, step)
        iterations = max(1, L) * max(1, stepn)
        for _ in range(iterations):
            values = current[: self.Nvar]
            for bit in self._random_dims(max(1, min(stepn, self.Nvar))):
                values[bit] += self.randval(-step, step)
            candidate = self._candidate_row(values)
            if candidate[self.Nvar] <= current[self.Nvar] + threshold:
                current = candidate
                if candidate[self.Nvar] < best[self.Nvar]:
                    best = candidate
            threshold *= 0.9
            step *= 0.95
        self.newpop[popi] = best[: self.Nvar]
        self.newpop_fit[popi] = best[self.Nvar]

    def newpop_elite_contraction(self, popi: int, L: int, scale: float) -> None:
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        current = self._candidate_row(self.newpop[popi])
        dims = max(1, min(self.Nvar, int(math.sqrt(self.Nvar)) + 1))
        for it in range(max(1, L)):
            to_global = self.randval(0.08, 0.45)
            to_personal = self.randval(0.04, 0.25)
            values = [
                current[j]
                + to_global * (self.gbest[j] - current[j])
                + to_personal * (self.ibest[popi][j] - current[j])
                for j in range(self.Nvar)
            ]
            jitter = step * (1.0 - 0.75 * it / max(1, L))
            for bit in self._random_dims(dims):
                values[bit] += self.randnorm(0.0, jitter)
            candidate = self._candidate_row(values)
            if candidate[self.Nvar] < current[self.Nvar]:
                current = candidate
                step *= 0.92
            else:
                step *= 0.65
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_coordinate_descent(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        for _ in range(max(1, L)):
            bit = random.randrange(self.Nvar)
            for direction in (1.0, -1.0):
                values = current[: self.Nvar]
                values[bit] += direction * step
                candidate = self._candidate_row(values)
                if candidate[self.Nvar] < current[self.Nvar]:
                    current = candidate
            step = max(step * 0.98, 1e-12)
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_mcts_local_search(self, popi: int, L: int, scale: float) -> None:
        self.newpop_mcts(popi, L, max(2, int(math.sqrt(self.Nvar))), scale)

    def newpop_elite_line_search(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        for it in range(max(1, L)):
            target = self.gbest if it % 2 == 0 else self.ibest[popi]
            shrink = 1.0 / (1.0 + it)
            for alpha in (0.12, 0.25, 0.50, 0.85, -0.20):
                values = [
                    current[j] + alpha * shrink * (target[j] - current[j])
                    for j in range(self.Nvar)
                ]
                if alpha < 0:
                    for bit in self._random_dims(max(1, int(math.sqrt(self.Nvar)))):
                        values[bit] += self.randnorm(0.0, step * 0.10 * shrink)
                candidate = self._candidate_row(values)
                if candidate[self.Nvar] < current[self.Nvar]:
                    current = candidate
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_multi_scale_gaussian(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        sigma = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = max(1, min(self.Nvar, 2 * int(math.sqrt(self.Nvar)) + 1))
        for _ in range(max(1, L)):
            guide = self.gbest if self.randval(0.0, 1.0) < 0.5 else self.ibest[popi]
            mix = self.randval(0.0, 0.25)
            values = [current[j] + mix * (guide[j] - current[j]) for j in range(self.Nvar)]
            for bit in self._random_dims(dims):
                values[bit] += self.randnorm(0.0, sigma)
            candidate = self._candidate_row(values)
            if candidate[self.Nvar] < current[self.Nvar]:
                current = candidate
                sigma *= 1.03
            else:
                sigma *= 0.60
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_random_subspace_pattern(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = self._random_dims(max(2, int(math.sqrt(self.Nvar))))
        for _ in range(max(1, L)):
            improved = False
            random.shuffle(dims)
            for bit in dims:
                for direction in (1.0, -1.0):
                    values = current[: self.Nvar]
                    values[bit] += direction * step
                    candidate = self._candidate_row(values)
                    if candidate[self.Nvar] < current[self.Nvar]:
                        current = candidate
                        improved = True
            step *= 1.05 if improved else 0.5
            if step < 1e-12:
                break
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_newton_subspace(self, popi: int, L: int, scale: float) -> None:
        self.newpop_newton(popi, L, max(2, int(math.sqrt(self.Nvar))), scale)

    def newpop_gradient_backtracking(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-8)
        fd = max(step * 0.1, 1e-6)
        dims = self._random_dims(max(2, int(math.sqrt(self.Nvar))))
        for _ in range(max(1, L)):
            direction = [0.0 for _ in range(self.Nvar)]
            f0 = current[self.Nvar]
            for bit in dims:
                plus = current[: self.Nvar]
                minus = current[: self.Nvar]
                plus[bit] += fd
                minus[bit] -= fd
                fp = self._candidate_row(plus)[self.Nvar]
                fm = self._candidate_row(minus)[self.Nvar]
                direction[bit] = -(fp - fm) / (2.0 * fd)
            norm = math.sqrt(sum(value * value for value in direction))
            if norm <= 1e-15:
                break
            alpha = step
            while alpha > 1e-12:
                values = [current[j] + alpha * direction[j] / norm for j in range(self.Nvar)]
                candidate = self._candidate_row(values)
                if candidate[self.Nvar] < f0:
                    current = candidate
                    break
                alpha *= 0.5
            step *= 0.9
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_cauchy_basin_hop(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        step = max(abs(self.Ubound - self.Lbound) * scale, 1e-12)
        dims = max(1, min(self.Nvar, int(math.sqrt(self.Nvar)) + 1))
        temperature = max(1e-9, abs(current[self.Nvar]) * 0.01)
        for _ in range(max(1, L)):
            values = current[: self.Nvar]
            for bit in self._random_dims(dims):
                cauchy = math.tan(math.pi * (self.randval(0.0, 1.0) - 0.5))
                values[bit] += step * max(-4.0, min(4.0, cauchy))
            candidate = self._candidate_row(values)
            delta = candidate[self.Nvar] - current[self.Nvar]
            if delta < 0.0 or self.randval(0.0, 1.0) < math.exp(-delta / temperature):
                current = candidate
            temperature *= 0.85
            step *= 0.92
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def newpop_opposition_elite_blend(self, popi: int, L: int, scale: float) -> None:
        current = self._candidate_row(self.newpop[popi])
        bound_sum = self.Lbound + self.Ubound
        for _ in range(max(1, L)):
            alpha = self.randval(0.15, 0.75)
            beta = self.randval(0.05, 0.35)
            values = []
            for j in range(self.Nvar):
                opposition = bound_sum - current[j]
                elite = self.gbest[j] if self.randval(0.0, 1.0) < 0.5 else self.ibest[popi][j]
                values.append((1.0 - alpha - beta) * current[j] + alpha * elite + beta * opposition)
            candidate = self._candidate_row(values)
            if candidate[self.Nvar] < current[self.Nvar]:
                current = candidate
        self.newpop[popi] = current[: self.Nvar]
        self.newpop_fit[popi] = current[self.Nvar]

    def meme_selection(self, popi: int, X: int, scale: float, Iter: int) -> None:
        op = X % self._local_search_count()
        if op == 0:
            self.newpop_bit_climbing(popi, Iter, scale)
        elif op == 1:
            self.newpop_simplex(popi, Iter, 10, scale)
        elif op == 2:
            self.newpop_box_complex(popi, Iter, 10, scale)
        elif op == 3:
            self.newpop_powell(popi, Iter, 10, scale)
        elif op == 4:
            self.newpop_newton(popi, Iter, 10, scale)
        elif op == 5:
            self.newpop_mcts(popi, Iter, 10, scale)
        elif op == 6:
            self.newpop_simulated_annealing(popi, Iter, 10, scale)
        elif op == 7:
            self.newpop_tabu_search(popi, Iter, 10, scale)
        elif op == 8:
            self.newpop_pattern_search(popi, Iter, 10, scale)
        elif op == 9:
            self.newpop_threshold_accepting(popi, Iter, 10, scale)
        elif op == 10:
            self.newpop_elite_contraction(popi, Iter, scale)
        elif op == 11:
            self.newpop_coordinate_descent(popi, Iter * 2, scale)
        elif op == 12:
            self.newpop_mcts_local_search(popi, Iter * 2, scale)
        elif op == 13:
            self.newpop_elite_line_search(popi, Iter, scale)
        elif op == 14:
            self.newpop_multi_scale_gaussian(popi, Iter * 2, scale)
        elif op == 15:
            self.newpop_random_subspace_pattern(popi, Iter, scale)
        elif op == 16:
            self.newpop_newton_subspace(popi, Iter, scale)
        elif op == 17:
            self.newpop_gradient_backtracking(popi, Iter, scale)
        elif op == 18:
            self.newpop_cauchy_basin_hop(popi, Iter, scale)
        else:
            self.newpop_opposition_elite_blend(popi, Iter, scale)

    def _ensure_meme_q(self) -> None:
        states = 4
        actions = self._local_search_count()
        if len(self.meme_q) == states and all(len(row) == actions for row in self.meme_q):
            return
        self.meme_q = [[0.0 for _ in range(actions)] for _ in range(states)]
        self.meme_q_trials = [[0 for _ in range(actions)] for _ in range(states)]

    def _ensure_meme_stats(self) -> None:
        actions = self._local_search_count()
        if len(self.meme_trials) == actions and len(self.meme_rewards) == actions:
            if not hasattr(self, "meme_successes") or len(self.meme_successes) != actions:
                self.meme_successes = [0.0 for _ in range(actions)]
            return
        self.meme_trials = [0.0 for _ in range(actions)]
        self.meme_rewards = [0.0 for _ in range(actions)]
        self.meme_successes = [0.0 for _ in range(actions)]

    def _meme_range(self, p_start: int = 0, p_end: int | None = None) -> tuple[int, int]:
        start = max(0, p_start)
        end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        return start, max(start, end)

    def _meme_reward(self, oldfit: float, newfit: float) -> float:
        if newfit >= oldfit:
            return 0.0
        return (oldfit - newfit) / (abs(oldfit) + 1e-12)

    def _ranked_meme_actions(self, op_count: int) -> list[int]:
        self._ensure_meme_stats()
        ranked = list(range(op_count))
        ranked.sort(
            key=lambda idx: (
                self.meme_rewards[idx] / (self.meme_trials[idx] + 1.0),
                self.meme_successes[idx],
            ),
            reverse=True,
        )
        return ranked

    def update_local_guides(self, index: int) -> None:
        if 0 <= index < self.Popsize:
            self.SubDecBase[index][: self.Nvar] = self.newpop[index].copy()
            self.SubDecBase[index][self.Nvar] = self.newpop_fit[index]

    def apply_meme_action(self, index: int, action: int, scale: float, iter_count: int) -> float:
        oldfit = self.newpop_fit[index]
        self.meme_selection(index, action, scale, iter_count)
        self.update_local_guides(index)
        return oldfit

    def meme_q_learning(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        hyper_state: int | None = None,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        self._ensure_meme_q()
        progress = Gen / max(1, MaxG)
        state = self.search_state(progress, 0) if hyper_state is None else hyper_state
        state = max(0, min(len(self.meme_q) - 1, state))
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        epsilon = max(0.05, 0.35 * (1.0 - progress))
        alpha = 0.25
        gamma = 0.85
        actions = self._local_search_count()
        for i in range(max(0, p_start), p_end):
            if self.randval(0.0, 1.0) < epsilon:
                action = random.randrange(actions)
            else:
                action = max(range(actions), key=lambda idx: self.meme_q[state][idx])
            oldfit = self.apply_meme_action(i, action, scale, max(2, min(10, MaxG)))
            denom = abs(oldfit) + 1e-12
            reward = max(0.0, (oldfit - self.newpop_fit[i]) / denom)
            future = max(self.meme_q[state])
            self.meme_q[state][action] = (
                (1.0 - alpha) * self.meme_q[state][action]
                + alpha * (reward + gamma * future)
            )
            self.meme_q_trials[state][action] += 1

    def meme_q_learning_range(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        p_start: int,
        p_end: int,
    ) -> None:
        self.meme_q_learning(Gen, MaxG, scale, p_start=p_start, p_end=p_end)

    def meme_q_learning_generated_eo(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        hyper_state: int | None = None,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        self._ensure_meme_q()
        progress = Gen / max(1, MaxG)
        state = self.search_state(progress, 0) if hyper_state is None else hyper_state
        state = max(0, min(len(self.meme_q) - 1, state))
        p_start, p_end = self._meme_range(p_start, p_end)
        op_count = min(self._local_search_count(), self.Popsize)
        if op_count <= 0 or p_start >= p_end:
            return
        warmup_gen = min(max(MaxG // 25, op_count), 50)
        epsilon = max(0.03, 0.08 * (1.0 - progress) + 0.03)
        alpha = 0.16
        gamma = 0.10
        for i in range(p_start, p_end):
            if Gen < warmup_gen:
                action = (i + Gen) % op_count
            elif self.randval(0.0, 1.0) < epsilon:
                action = random.randrange(op_count)
            else:
                action = max(range(op_count), key=lambda idx: self.meme_q[state][idx])
            self.Ind_meme[i] = action
            oldfit = self.apply_meme_action(i, action, scale, 3)
            reward = self._meme_reward(oldfit, self.newpop_fit[i])
            reward = max(-0.002, min(1.0, reward if reward > 0.0 else -0.002))
            future = max(self.meme_q[state][:op_count])
            self.meme_q[state][action] += alpha * (
                reward + gamma * future - self.meme_q[state][action]
            )
            self.meme_q[state][action] = max(0.0, min(2.0, self.meme_q[state][action]))
            self.meme_q_trials[state][action] += 1
        best_index = min(range(p_start, p_end), key=lambda idx: self.newpop_fit[idx])
        elite_pool = self._ranked_meme_actions(op_count)[:3]
        for action in (5, 7, 9, 10, 4):
            if action < op_count and action not in elite_pool:
                elite_pool.append(action)
            if len(elite_pool) >= 5:
                break
        for k, action in enumerate(elite_pool):
            op_scale = scale * (0.55 - 0.05 * min(k, 4))
            oldfit = self.apply_meme_action(best_index, action, op_scale, 4 if k == 0 else 3)
            reward = self._meme_reward(oldfit, self.newpop_fit[best_index])
            self.meme_q[state][action] += alpha * (
                reward + gamma * max(self.meme_q[state][:op_count]) - self.meme_q[state][action]
            )
            self.meme_q_trials[state][action] += 1

    def meme_random_walk(self, scale: float) -> None:
        step = 0
        count = 0
        for i in range(self.Popsize):
            self.meme_selection(i, step, scale, 10)
            rr = self.randval(0.0, 1.0)
            if count < 5:
                step += -1 if rr < 0.5 else 1
            elif rr < 0.2:
                step -= 1
            elif rr < 0.4:
                step += 1
            elif rr < 0.55:
                step -= 2
            elif rr < 0.7:
                step += 2
            elif rr < 0.85:
                step -= 3
            else:
                step += 3
            step = 0 if step < 0 else step % self._local_search_count()
            count += 1

    def meme_simple_random(self, scale: float) -> None:
        op_count = self._local_search_count()
        for i in range(self.Popsize):
            self.meme_selection(i, random.randrange(op_count), scale, 10)

    def meme_randperm(self, scale: float) -> None:
        op_count = self._local_search_count()
        permu = list(range(op_count))
        random.shuffle(permu)
        count = 0
        for i in range(self.Popsize):
            self.meme_selection(i, permu[count], scale, 10)
            count += 1
            if count >= op_count:
                count = 0
                random.shuffle(permu)

    def meme_inheritance(self, scale: float) -> None:
        for i in range(self.Popsize):
            self.meme_selection(i, self.Ind_meme[i], scale, 10)
        for i in range(0, self.Popsize - 1, 2):
            if self.newpop_fit[i] == self.newpop_fit[i + 1]:
                if self.randval(0.0, 1.0) < 0.5:
                    self.Ind_meme[i] = self.Ind_meme[i + 1]
                else:
                    self.Ind_meme[i + 1] = self.Ind_meme[i]
            elif self.newpop_fit[i] < self.newpop_fit[i + 1]:
                self.Ind_meme[i + 1] = self.Ind_meme[i]
            else:
                self.Ind_meme[i] = self.Ind_meme[i + 1]

    def meme_subprob_decomposition(self, Gen: int, MaxG: int, kk: int, scale: float) -> None:
        kk = max(1, min(kk, self.Popsize))
        op_count = self._local_search_count()
        if Gen < MaxG:
            for i in range(self.Popsize):
                rr = random.randrange(op_count)
                oldfit = self.newpop_fit[i]
                self.meme_selection(i, rr, scale, 10)
                denom = self.newpop_fit[i] if self.newpop_fit[i] != 0 else 1e-300
                reward = (self.gbest_fit / denom) * (oldfit - self.newpop_fit[i]) / 100.0
                target = i if Gen == 0 else min(range(self.Popsize), key=lambda idx: self.SubDecBase[idx][self.Nvar])
                if Gen == 0 or reward > self.SubDecBase[target][self.Nvar]:
                    self.SubDecBase[target][: self.Nvar] = self.newpop[i].copy()
                    self.SubDecBase[target][self.Nvar] = reward
                    self.SubDecBase[target][self.Nvar + 1] = rr
            self.heap_sort(self.SubDecBase, self.Popsize, self.Nvar)
        else:
            for i in range(self.Popsize):
                rows = []
                for j in range(self.Popsize):
                    rows.append(
                        [
                            math.dist(self.newpop[i], self.SubDecBase[j][: self.Nvar]),
                            self.SubDecBase[j][self.Nvar],
                            self.SubDecBase[j][self.Nvar + 1],
                        ]
                    )
                rows.sort(key=lambda row: row[0])
                rows[:kk] = sorted(rows[:kk], key=lambda row: row[1])
                self.Ind_meme[i] = int(rows[kk - 1][2]) % op_count
                oldfit = self.newpop_fit[i]
                self.meme_selection(i, self.Ind_meme[i], scale, 10)
                denom = self.newpop_fit[i] if self.newpop_fit[i] != 0 else 1e-300
                reward = (self.gbest_fit / denom) * (oldfit - self.newpop_fit[i]) / 100.0
                worst = min(range(self.Popsize), key=lambda idx: self.SubDecBase[idx][self.Nvar])
                if reward > self.SubDecBase[worst][self.Nvar]:
                    self.SubDecBase[worst][: self.Nvar] = self.newpop[i].copy()
                    self.SubDecBase[worst][self.Nvar] = reward
                    self.SubDecBase[worst][self.Nvar + 1] = self.Ind_meme[i]
                    self.heap_sort(self.SubDecBase, self.Popsize, self.Nvar)

    def meme_biasd_roulette(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        op_count = self._local_search_count()
        self._ensure_meme_stats()
        p_start = max(0, p_start)
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        if Gen < MaxG:
            for i in range(p_start, p_end):
                rr = random.randrange(op_count)
                oldfit = self.newpop_fit[i]
                self.meme_selection(i, rr, scale, 10)
                denom = self.newpop_fit[i] if self.newpop_fit[i] != 0 else 1e-300
                reward = (self.gbest_fit / denom) * (oldfit - self.newpop_fit[i]) / 100.0
                self.meme_trials[rr] += 1.0
                self.meme_rewards[rr] += reward
            return
        scores = [self.meme_rewards[i] / (self.meme_trials[i] + 1.0) for i in range(op_count)]
        total = sum(scores)
        if total == 0:
            probs = [1.0 / op_count for _ in range(op_count)]
        else:
            probs = [score / total for score in scores]
        cprob = []
        acc = 0.0
        for prob in probs:
            acc += prob
            cprob.append(acc)
        for i in range(p_start, p_end):
            p = self.randval(0.0, 1.0)
            rr = next((idx for idx, cutoff in enumerate(cprob) if p < cutoff), op_count - 1)
            self.Ind_meme[i] = rr
            oldfit = self.newpop_fit[i]
            self.meme_selection(i, rr, scale, 10)
            denom = self.newpop_fit[i] if self.newpop_fit[i] != 0 else 1e-300
            reward = (self.gbest_fit / denom) * (oldfit - self.newpop_fit[i]) / 100.0
            self.meme_trials[rr] += 1.0
            self.meme_rewards[rr] += reward

    def meme_biasd_roulette_range(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        p_start: int,
        p_end: int,
    ) -> None:
        self.meme_biasd_roulette(Gen, MaxG, scale, p_start, p_end)

    def meme_biasd_roulette_generated_eo(
        self,
        Gen: int,
        MaxG: int,
        scale: float,
        p_start: int = 0,
        p_end: int | None = None,
    ) -> None:
        op_count = min(self._local_search_count(), self.Popsize)
        if op_count <= 0:
            return
        self._ensure_meme_stats()
        p_start, p_end = self._meme_range(p_start, p_end)
        if p_start >= p_end:
            return
        if Gen == 0:
            for i in range(op_count):
                self.meme_trials[i] = 0.0
                self.meme_rewards[i] = 0.0
                self.meme_successes[i] = 0.0
        else:
            for i in range(op_count):
                self.meme_trials[i] *= 0.96
                self.meme_successes[i] *= 0.94
                self.meme_rewards[i] *= 0.90
        base_reward = 1e-6
        scores = []
        for i in range(op_count):
            avg_reward = self.meme_rewards[i] / (self.meme_trials[i] + 1.0)
            success_rate = self.meme_successes[i] / (self.meme_trials[i] + 1.0)
            scores.append(max(base_reward, base_reward + avg_reward * (0.30 + success_rate)))
        total = sum(scores)
        exploitation_rate = 0.88
        probs = [
            exploitation_rate * score / total + (1.0 - exploitation_rate) / op_count
            for score in scores
        ]
        cumulative = []
        acc = 0.0
        for prob in probs:
            acc += prob
            cumulative.append(acc)

        def choose_action() -> int:
            if self.randval(0.0, 1.0) < 0.12:
                return random.randrange(op_count)
            pick = self.randval(0.0, 1.0)
            return next((idx for idx, cutoff in enumerate(cumulative) if pick <= cutoff), op_count - 1)

        warmup_gen = max(MaxG, op_count)
        for i in range(p_start, p_end):
            action = (i + Gen) % op_count if Gen < warmup_gen else choose_action()
            self.Ind_meme[i] = action
            oldfit = self.apply_meme_action(i, action, scale, 3)
            reward = self._meme_reward(oldfit, self.newpop_fit[i])
            self.meme_trials[action] += 1.0
            self.meme_rewards[action] += reward
            if reward > 0.0:
                self.meme_successes[action] += 1.0
        best_index = min(range(p_start, p_end), key=lambda idx: self.newpop_fit[idx])
        elite_pool = self._ranked_meme_actions(op_count)[:3]
        for action in (4, 5, 6, 7, 9, 10, 11):
            if action < op_count and action not in elite_pool:
                elite_pool.append(action)
            if len(elite_pool) >= 5:
                break
        for k, action in enumerate(elite_pool):
            op_scale = scale * (0.52 - 0.04 * min(k, 4))
            oldfit = self.apply_meme_action(best_index, action, op_scale, 4 if k == 0 else 3)
            reward = self._meme_reward(oldfit, self.newpop_fit[best_index])
            self.meme_trials[action] += 1.0
            self.meme_rewards[action] += reward
            if reward > 0.0:
                self.meme_successes[action] += 1.0
