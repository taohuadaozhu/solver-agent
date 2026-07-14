from __future__ import annotations

from dataclasses import dataclass
import math


def solve_linear_system(a: list[list[float]], b: list[float], eps: float = 1e-12) -> list[float] | None:
    n = len(b)
    if n == 0 or len(a) != n or any(len(row) != n for row in a):
        return None
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < eps:
            return None
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        div = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def cubic_rbf_fit(samples: list[list[float]], values: list[float]) -> tuple[list[float], list[float]] | None:
    m = len(samples)
    if m == 0 or m != len(values):
        return None
    d = len(samples[0])
    if d == 0 or any(len(row) != d for row in samples):
        return None
    size = m + d + 1
    a = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]
    for i in range(m):
        rhs[i] = values[i]
        for j in range(m):
            dist2 = 0.0
            for k in range(d):
                diff = samples[i][k] - samples[j][k]
                dist2 += diff * diff
            a[i][j] = dist2 ** 1.5
        for k in range(d):
            a[i][m + k] = samples[i][k]
            a[m + k][i] = samples[i][k]
        a[i][m + d] = 1.0
        a[m + d][i] = 1.0
    for i in range(size):
        a[i][i] += 1e-10
    params = solve_linear_system(a, rhs)
    if params is None:
        return None
    return params[:m], params[m:]


def cubic_rbf_eval(x: list[float], samples: list[list[float]], lamb: list[float], gamma: list[float]) -> float:
    value = gamma[-1] if gamma else 0.0
    for j, xj in enumerate(x):
        if j < len(gamma) - 1:
            value += gamma[j] * xj
    for row, coef in zip(samples, lamb):
        dist2 = 0.0
        for a, b in zip(x, row):
            diff = a - b
            dist2 += diff * diff
        value += coef * (dist2 ** 1.5)
    return value if math.isfinite(value) else float("inf")


def pca_fit(rows: list[list[float]], max_components: int | None = None) -> tuple[list[float], list[list[float]], list[float]]:
    if not rows:
        return [], [], []
    n = len(rows)
    d = len(rows[0])
    components = d if max_components is None else max(1, min(max_components, d))
    mean = [sum(row[j] for row in rows) / n for j in range(d)]
    if n < 2 or d == 0:
        return mean, [[1.0 if i == j else 0.0 for j in range(d)] for i in range(components)], [0.0] * components

    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    for row in rows:
        centered = [row[j] - mean[j] for j in range(d)]
        for i in range(d):
            ci = centered[i]
            for j in range(i, d):
                cov[i][j] += ci * centered[j]
    scale = 1.0 / max(1, n - 1)
    for i in range(d):
        for j in range(i, d):
            cov[i][j] *= scale
            cov[j][i] = cov[i][j]

    eigvec = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    max_iter = max(16, 80 * d * d)
    for _ in range(max_iter):
        p, q = 0, 1 if d > 1 else 0
        best = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                value = abs(cov[i][j])
                if value > best:
                    best, p, q = value, i, j
        if best < 1e-12:
            break
        app, aqq, apq = cov[p][p], cov[q][q], cov[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        for k in range(d):
            if k != p and k != q:
                akp, akq = cov[k][p], cov[k][q]
                cov[k][p] = cov[p][k] = c * akp - s * akq
                cov[k][q] = cov[q][k] = s * akp + c * akq
        cov[p][p] = app - t * apq
        cov[q][q] = aqq + t * apq
        cov[p][q] = cov[q][p] = 0.0
        for k in range(d):
            vkp, vkq = eigvec[k][p], eigvec[k][q]
            eigvec[k][p] = c * vkp - s * vkq
            eigvec[k][q] = s * vkp + c * vkq

    order = sorted(range(d), key=lambda idx: cov[idx][idx], reverse=True)
    basis = [[eigvec[row][idx] for row in range(d)] for idx in order[:components]]
    values = [max(0.0, cov[idx][idx]) for idx in order[:components]]
    return mean, basis, values


def pca_transform(row: list[float], mean: list[float], basis: list[list[float]]) -> list[float]:
    return [sum((row[j] - mean[j]) * comp[j] for j in range(len(mean))) for comp in basis]


def pca_inverse(coords: list[float], mean: list[float], basis: list[list[float]]) -> list[float]:
    row = mean.copy()
    for value, comp in zip(coords, basis):
        for j in range(len(row)):
            row[j] += value * comp[j]
    return row


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class KrigingModel:
    samples: list[list[float]]
    theta: list[float]
    beta: float
    gamma: list[float]
    sigma2: float
    one_r_inv_one: float
    y_mean: float
    y_std: float
    ready: bool = False


def kriging_fit(
    samples: list[list[float]],
    values: list[float],
    theta: list[float] | None = None,
    nugget: float = 1e-8,
) -> KrigingModel | None:
    m = len(samples)
    if m == 0 or m != len(values):
        return None
    d = len(samples[0])
    if d == 0 or any(len(row) != d for row in samples):
        return None
    theta = theta[:] if theta is not None else [1.0 for _ in range(d)]
    if len(theta) != d:
        theta = [theta[0] if theta else 1.0 for _ in range(d)]
    y_mean = sum(values) / m
    var = sum((value - y_mean) ** 2 for value in values) / max(1, m - 1)
    y_std = math.sqrt(var) if var > 1e-24 else 1.0
    y = [(value - y_mean) / y_std for value in values]
    r = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            dist = sum(theta[k] * (samples[i][k] - samples[j][k]) ** 2 for k in range(d))
            r[i][j] = math.exp(-dist)
        r[i][i] += nugget
    ones = [1.0 for _ in range(m)]
    r_inv_y = solve_linear_system(r, y)
    r_inv_one = solve_linear_system(r, ones)
    if r_inv_y is None or r_inv_one is None:
        return None
    den = sum(r_inv_one)
    if abs(den) < 1e-12:
        return None
    beta = sum(r_inv_y) / den
    residual = [y[i] - beta for i in range(m)]
    gamma = solve_linear_system(r, residual)
    if gamma is None:
        return None
    sigma2 = max(1e-16, sum(residual[i] * gamma[i] for i in range(m)) / max(1, m))
    return KrigingModel([row.copy() for row in samples], theta, beta, gamma, sigma2, den, y_mean, y_std, True)


def kriging_predict(x: list[float], model: KrigingModel) -> tuple[float, float]:
    if not model.ready:
        return 0.0, float("inf")
    r = []
    for row in model.samples:
        dist = sum(model.theta[k] * (x[k] - row[k]) ** 2 for k in range(min(len(x), len(row), len(model.theta))))
        r.append(math.exp(-dist))
    scaled_mean = model.beta + sum(g * ri for g, ri in zip(model.gamma, r))
    # This lightweight DACE variant omits the full R^-1 r term in MSE and keeps a conservative distance-aware variance.
    nearest = max(r) if r else 0.0
    scaled_mse = max(1e-16, model.sigma2 * (1.0 - nearest * nearest + 1.0 / max(model.one_r_inv_one, 1e-12)))
    return model.y_mean + model.y_std * scaled_mean, (model.y_std * model.y_std) * scaled_mse


def expected_improvement(best: float, mean: float, mse: float) -> float:
    sigma = math.sqrt(max(0.0, mse))
    if sigma <= 1e-14:
        return max(0.0, best - mean)
    z = (best - mean) / sigma
    return max(0.0, (best - mean) * _normal_cdf(z) + sigma * _normal_pdf(z))


def lower_confidence_bound(mean: float, mse: float, kappa: float = 2.0) -> float:
    return mean - kappa * math.sqrt(max(0.0, mse))


def probability_improvement(best: float, mean: float, mse: float, xi: float = 0.0) -> float:
    sigma = math.sqrt(max(0.0, mse))
    if sigma <= 1e-14:
        return 1.0 if mean < best - xi else 0.0
    return _normal_cdf((best - xi - mean) / sigma)


@dataclass
class QuadModel:
    kind: str
    beta: list[float]
    dim: int
    mse: float
    bic: float
    ready: bool = False


def _quad_features(x: list[float], kind: str) -> list[float]:
    row = [1.0] + x[:]
    if kind in {"purequadratic", "quadratic"}:
        row.extend(value * value for value in x)
    if kind == "quadratic":
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                row.append(x[i] * x[j])
    return row


def _least_squares(features: list[list[float]], values: list[float], ridge: float = 1e-8) -> list[float] | None:
    if not features:
        return None
    p = len(features[0])
    ata = [[0.0 for _ in range(p)] for _ in range(p)]
    aty = [0.0 for _ in range(p)]
    for row, value in zip(features, values):
        for i in range(p):
            aty[i] += row[i] * value
            for j in range(i, p):
                ata[i][j] += row[i] * row[j]
    for i in range(p):
        for j in range(i, p):
            ata[j][i] = ata[i][j]
        ata[i][i] += ridge
    return solve_linear_system(ata, aty)


def quad_fit(samples: list[list[float]], values: list[float]) -> QuadModel | None:
    if not samples or len(samples) != len(values):
        return None
    d = len(samples[0])
    best: QuadModel | None = None
    for kind in ("linear", "purequadratic", "quadratic"):
        features = [_quad_features(row, kind) for row in samples]
        beta = _least_squares(features, values)
        if beta is None:
            continue
        preds = [sum(b * x for b, x in zip(beta, row)) for row in features]
        mse = sum((value - pred) ** 2 for value, pred in zip(values, preds)) / max(1, len(values))
        mse = max(mse, 1e-24)
        bic = len(beta) * math.log(max(2, len(values))) / max(1, len(values)) + 2.0 * math.log(mse)
        model = QuadModel(kind, beta, d, mse, bic, True)
        if best is None or model.bic < best.bic:
            best = model
    return best


def quad_predict(x: list[float], model: QuadModel) -> float:
    if not model.ready:
        return float("inf")
    features = _quad_features(x[: model.dim], model.kind)
    return sum(b * value for b, value in zip(model.beta, features))
