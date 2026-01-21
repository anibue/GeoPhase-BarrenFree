"""Math helpers for simple quantum-state manipulations."""
from __future__ import annotations

import cmath
from typing import Iterable, Sequence


def inner_product(left: Sequence[complex], right: Sequence[complex]) -> complex:
    """Compute the inner product <left|right>."""
    if len(left) != len(right):
        raise ValueError("State vectors must have the same dimension.")
    return sum(l.conjugate() * r for l, r in zip(left, right))


def norm(state: Sequence[complex]) -> float:
    """Compute the Euclidean norm of a state vector."""
    return cmath.sqrt(sum(abs(amplitude) ** 2 for amplitude in state)).real


def normalize_state(state: Sequence[complex]) -> Sequence[complex]:
    """Normalize a state vector."""
    length = norm(state)
    if length == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return [amplitude / length for amplitude in state]


def phase_distance(phase_a: float, phase_b: float) -> float:
    """Compute wrapped distance between two phases."""
    delta = phase_a - phase_b
    while delta <= -cmath.pi:
        delta += 2 * cmath.pi
    while delta > cmath.pi:
        delta -= 2 * cmath.pi
    return abs(delta)
