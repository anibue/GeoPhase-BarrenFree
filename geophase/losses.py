"""Loss functions for geometric-phase optimization."""
from __future__ import annotations

from geophase.math_utils import phase_distance


def geometric_phase_loss(estimated_phase: float, target_phase: float) -> float:
    """Squared wrapped distance between estimated and target phases."""
    distance = phase_distance(estimated_phase, target_phase)
    return distance**2


__all__ = ["geometric_phase_loss", "phase_distance"]
