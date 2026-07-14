import math
import random
from typing import Optional, Sequence

PI = 3.1415926


def randnorm(miu: float, score: float) -> float:
    return miu + score * math.sqrt(-2.0 * math.log(random.random())) * math.cos(
        2.0 * PI * random.random()
    )


def Sphere(var: Sequence[float], num: Optional[int] = None) -> float:
    values = var if num is None else var[:num]
    return sum(x * x for x in values)


def Rosenbrock(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return sum(
        100.0 * (var[i + 1] - var[i] * var[i]) ** 2 + (var[i] - 1.0) ** 2
        for i in range(n - 1)
    )


def Dixon_Price(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    f = (var[0] - 1.0) ** 2
    for i in range(1, n):
        f += (i + 1) * (2.0 * var[i] * var[i] - var[i - 1]) ** 2
    return f


def Griewank(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    f = sum(var[i] * var[i] for i in range(n)) / 4000.0
    product = math.cos(var[0])
    for i in range(1, n):
        product *= math.cos(var[i] / math.sqrt(i + 1.0))
    return f - product + 1.0


def Ackley(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    temp1 = sum(var[i] * var[i] for i in range(n))
    temp2 = sum(math.cos(2.0 * PI * var[i]) for i in range(n))
    return -20.0 * math.exp(-0.2 * math.sqrt(temp1 / n)) - math.exp(temp2 / n) + 20.0 + math.e


def Michalewicz(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return -sum(
        math.sin(var[i]) * math.sin((i + 1) * var[i] * var[i] / PI) ** 20
        for i in range(n)
    )


def Rastrigin(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return sum(var[i] * var[i] - 10.0 * math.cos(2.0 * PI * var[i]) + 10.0 for i in range(n))


def Schwefel(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return 418.9829 * n - sum(var[i] * math.sin(math.sqrt(abs(var[i]))) for i in range(n))


def Sum_Squares(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return sum((i + 1) * var[i] * var[i] for i in range(n))


def Trid(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return sum((var[i] - 1.0) ** 2 for i in range(n)) - sum(var[i] * var[i - 1] for i in range(1, n))


def Zakharov(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    temp1 = sum(var[i] * var[i] for i in range(n))
    temp2 = sum(0.5 * (i + 1) * var[i] for i in range(n))
    return temp1 + temp2**2 + temp2**4


def Weierstrass(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    f = 0.0
    for i in range(n):
        f += sum((0.5**j) * math.cos(2.0 * PI * (3.0**j) * (var[i] + 0.5)) for j in range(21))
    tmp2 = sum((0.5**j) * math.cos(2.0 * PI * (3.0**j) * 0.5) for j in range(21))
    return f - n * tmp2


def High_Conditioned_Elliptic(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return sum((10.0 ** (6.0 * (i - 1) / (n - 1))) * var[i] * var[i] for i in range(n))


def Schwefel_P222(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    product = 1.0
    total = 0.0
    for i in range(n):
        value = abs(var[i])
        total += value
        product *= value
    return total + product


def Schwefel_P221(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return max(abs(var[i]) for i in range(n))


def Tablet(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    return 1000.0 * var[0] * 1000.0 * var[0] + sum(var[i] * var[i] for i in range(1, n))


def Salomon(var: Sequence[float], num: Optional[int] = None) -> float:
    n = len(var) if num is None else num
    root = math.sqrt(sum(var[i] * var[i] for i in range(n)))
    return 1.0 - math.cos(2.0 * PI * root) + 0.1 * root
