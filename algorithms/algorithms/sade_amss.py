from __future__ import annotations

import math
import random

from .archive import archive_sample_indices, update_sade_amss_archive
from .surrogate import cubic_rbf_eval, cubic_rbf_fit, pca_fit, pca_inverse, pca_transform


class SADEAMSSMixin:
    def SADE_AMSS(
        self,
        Gen: int,
        MaxGen: int,
        p_start: int = 0,
        p_end: int | None = None,
        K: int = 20,
        maxd: int = 100,
        Gm: int = 5,
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        active = max(0, p_end - p_start)
        if active == 0:
            return

        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        best = self.pop[ranked[0]]
        trial_rows = [self.pop[i].copy() for i in range(p_start, p_end)]
        trial_fit = [self.pop_fit[i] for i in range(p_start, p_end)]
        subspaces = max(1, min(K, self.Nvar, active))
        group_cap = max(1, min(maxd, self.Nvar))
        archive_count = min(self.Popsize, max(active, 2 * group_cap + 2))

        for _ in range(subspaces):
            d = random.randint(1, group_cap)
            dims = list(range(self.Nvar))
            random.shuffle(dims)
            dims = dims[:d]
            train_idx = ranked[:archive_count]
            samples = [[self.pop[idx][j] for j in dims] for idx in train_idx]
            values = [self.pop_fit[idx] for idx in train_idx]
            model = cubic_rbf_fit(samples, values) if len(samples) >= d + 2 else None

            sub = [[row[j] for j in dims] for row in trial_rows]
            sub_scores = []
            for row, fit in zip(sub, trial_fit):
                if model is None:
                    dist = sum((row[k] - best[dims[k]]) ** 2 for k in range(d))
                    sub_scores.append(fit + 1e-6 * dist)
                else:
                    sub_scores.append(cubic_rbf_eval(row, samples, model[0], model[1]))

            for _g in range(max(1, Gm)):
                candidates = []
                scores = []
                for i, row in enumerate(sub):
                    if len(sub) >= 3:
                        choices = [idx for idx in range(len(sub)) if idx != i]
                        r1, r2 = random.sample(choices, 2)
                    else:
                        r1 = r2 = i
                    forced = random.randrange(d)
                    cand = row.copy()
                    for k in range(d):
                        if k == forced or self.randval(0.0, 1.0) < 0.95:
                            value = best[dims[k]] + 0.8 * (sub[r1][k] - sub[r2][k])
                            cand[k] = self._clip(value)
                    candidates.append(cand)
                    if model is None:
                        dist = sum((cand[k] - best[dims[k]]) ** 2 for k in range(d))
                        scores.append(trial_fit[i] + 1e-6 * dist)
                    else:
                        scores.append(cubic_rbf_eval(cand, samples, model[0], model[1]))
                combined = [(sub_scores[i], sub[i]) for i in range(len(sub))]
                combined.extend((scores[i], candidates[i]) for i in range(len(candidates)))
                combined.sort(key=lambda item: item[0])
                sub = [row[:] for _, row in combined[:active]]
                sub_scores = [score for score, _ in combined[:active]]

            for i, row in enumerate(sub):
                for k, dim in enumerate(dims):
                    trial_rows[i][dim] = self._clip(row[k])

        for offset, i in enumerate(range(p_start, p_end)):
            self.newpop[i] = trial_rows[offset]
            self.newpop_fit[i] = self.EvaluFunc(self.newpop[i], self.Nvar)

    def _ensure_sade_amss_orig_archive(self, size: int) -> None:
        if not hasattr(self, "sade_amss_archive") or len(getattr(self, "sade_amss_archive", [])) == 0:
            self.sade_amss_archive = [row.copy() for row in self.pop]
            self.sade_amss_archive_fit = self.pop_fit.copy()
        while len(self.sade_amss_archive) < size:
            row = [self.randval(self.Lbound, self.Ubound) for _ in range(self.Nvar)]
            self.sade_amss_archive.append(row)
            self.sade_amss_archive_fit.append(self.EvaluFunc(row, self.Nvar))
        self.sade_amss_archive, self.sade_amss_archive_fit = update_sade_amss_archive(
            self.sade_amss_archive,
            self.sade_amss_archive_fit,
            self.pop,
            self.pop_fit,
            size,
        )

    def SADE_AMSS_ORIG(
        self,
        Gen: int,
        MaxGen: int,
        p_start: int = 0,
        p_end: int | None = None,
        K: int = 20,
        maxd: int = 100,
        Gm: int = 5,
        archive_size: int | None = None,
    ) -> None:
        p_end = self.Popsize if p_end is None else min(self.Popsize, p_end)
        p_start = max(0, p_start)
        active = max(0, p_end - p_start)
        if active == 0 or self.Nvar <= 0:
            return

        archive_size = archive_size or max(self.Popsize, min(200, 4 * self.Popsize))
        self._ensure_sade_amss_orig_archive(archive_size)
        archive = self.sade_amss_archive
        archive_fit = self.sade_amss_archive_fit

        ranked = sorted(range(self.Popsize), key=lambda idx: self.pop_fit[idx])
        best = self.pop[ranked[0]]
        trial_rows = [self.pop[i].copy() for i in range(p_start, p_end)]
        trial_fit = [self.pop_fit[i] for i in range(p_start, p_end)]

        use_pca = Gen > max(1, MaxGen // 5) and len(archive) >= max(3, self.Nvar // 2)
        if use_pca:
            mean, basis, _eig = pca_fit(archive, self.Nvar)
            archive_space = [pca_transform(row, mean, basis) for row in archive]
            trial_space = [pca_transform(row, mean, basis) for row in trial_rows]
            best_space = pca_transform(best, mean, basis)
        else:
            mean, basis = [], []
            archive_space = [row.copy() for row in archive]
            trial_space = [row.copy() for row in trial_rows]
            best_space = best.copy()

        subspaces = max(1, min(K, self.Nvar, active))
        group_cap = max(1, min(maxd, self.Nvar))
        train_count = min(len(archive), max(active, 2 * group_cap + 2))

        for _ in range(subspaces):
            d = random.randint(1, group_cap)
            if use_pca:
                top = max(d, min(self.Nvar, group_cap))
                dims = list(range(top))
                random.shuffle(dims)
                dims = dims[:d]
            else:
                dims = list(range(self.Nvar))
                random.shuffle(dims)
                dims = dims[:d]

            train_idx = archive_sample_indices(archive, train_count)
            if len(train_idx) < train_count:
                by_fit = sorted(range(len(archive)), key=lambda idx: archive_fit[idx])
                seen = set(train_idx)
                train_idx += [idx for idx in by_fit if idx not in seen][: train_count - len(train_idx)]
            samples = [[archive_space[idx][j] for j in dims] for idx in train_idx]
            values = [archive_fit[idx] for idx in train_idx]
            model = cubic_rbf_fit(samples, values) if len(samples) >= d + 2 else None

            sub = [[row[j] for j in dims] for row in trial_space]
            sub_scores = []
            for row, fit in zip(sub, trial_fit):
                if model is None:
                    dist = sum((row[k] - best_space[dims[k]]) ** 2 for k in range(d))
                    sub_scores.append(fit + 1e-6 * dist)
                else:
                    sub_scores.append(cubic_rbf_eval(row, samples, model[0], model[1]))

            for _g in range(max(1, Gm)):
                candidates = []
                scores = []
                for i, row in enumerate(sub):
                    if len(sub) >= 3:
                        choices = [idx for idx in range(len(sub)) if idx != i]
                        r1, r2 = random.sample(choices, 2)
                    else:
                        r1 = r2 = i
                    forced = random.randrange(d)
                    cand = row.copy()
                    for k in range(d):
                        if k == forced or self.randval(0.0, 1.0) < 0.9:
                            scale = 0.5 + 0.5 * (1.0 - min(1.0, Gen / max(1, MaxGen)))
                            cand[k] = best_space[dims[k]] + scale * (sub[r1][k] - sub[r2][k])
                    candidates.append(cand)
                    if model is None:
                        dist = sum((cand[k] - best_space[dims[k]]) ** 2 for k in range(d))
                        scores.append(trial_fit[i] + 1e-6 * dist)
                    else:
                        scores.append(cubic_rbf_eval(cand, samples, model[0], model[1]))
                combined = [(sub_scores[i], sub[i]) for i in range(len(sub))]
                combined.extend((scores[i], candidates[i]) for i in range(len(candidates)))
                combined.sort(key=lambda item: item[0])
                sub = [row[:] for _, row in combined[:active]]
                sub_scores = [score for score, _ in combined[:active]]

            for i, row in enumerate(sub):
                for k, dim in enumerate(dims):
                    trial_space[i][dim] = row[k]

        for offset, i in enumerate(range(p_start, p_end)):
            row = pca_inverse(trial_space[offset], mean, basis) if use_pca else trial_space[offset]
            self.newpop[i] = [self._clip(value) for value in row]
            self.newpop_fit[i] = self.EvaluFunc(self.newpop[i], self.Nvar)

        candidates = [self.newpop[i].copy() for i in range(p_start, p_end)]
        candidate_fit = [self.newpop_fit[i] for i in range(p_start, p_end)]
        self.sade_amss_archive, self.sade_amss_archive_fit = update_sade_amss_archive(
            self.sade_amss_archive,
            self.sade_amss_archive_fit,
            candidates,
            candidate_fit,
            archive_size,
        )
