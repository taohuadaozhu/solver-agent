from __future__ import annotations


def squared_distance(left: list[float], right: list[float], nvar: int) -> float:
    return sum((left[j] - right[j]) ** 2 for j in range(nvar))
