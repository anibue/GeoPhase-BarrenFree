"""Noise channel utilities."""
from __future__ import annotations

from typing import Sequence, List

from geophase.math_utils import normalize_state

Matrix = List[List[complex]]


def depolarizing_channel(state: Sequence[complex], probability: float) -> Matrix:
    """Apply a depolarizing channel to a pure state.

    Returns the density matrix after mixing with maximally mixed state.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be within [0, 1].")
    normalized = normalize_state(state)
    dimension = len(normalized)
    density = [[0j for _ in range(dimension)] for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            density[i][j] = normalized[i] * normalized[j].conjugate()
    mixed_weight = probability / dimension
    for i in range(dimension):
        for j in range(dimension):
            density[i][j] = (1 - probability) * density[i][j]
            if i == j:
                density[i][j] += mixed_weight
    return density


def amplitude_damping_channel(state: Sequence[complex], gamma: float) -> Matrix:
    """Apply amplitude damping for a single qubit-like state.

    The model reduces excited-state population by gamma.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be within [0, 1].")
    normalized = normalize_state(state)
    if len(normalized) != 2:
        raise ValueError("amplitude_damping_channel expects a 2D state.")
    ground, excited = normalized
    damped_excited = (1 - gamma) ** 0.5 * excited
    leaked = (gamma**0.5) * excited
    density = [
        [ground * ground.conjugate() + leaked * leaked.conjugate(), ground * damped_excited.conjugate()],
        [damped_excited * ground.conjugate(), damped_excited * damped_excited.conjugate()],
    ]
    return density
