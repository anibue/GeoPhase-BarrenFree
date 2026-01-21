"""Circuit abstractions for geometric-phase evaluation.

These are lightweight stand-ins for MindQuantum circuits. They model a
parameterized path of quantum states and provide geometric-phase estimates
via overlaps between consecutive states.
"""
from __future__ import annotations

from dataclasses import dataclass
import cmath
from typing import Iterable, List, Sequence

from geophase.math_utils import inner_product, normalize_state


@dataclass(frozen=True)
class CircuitPath:
    """A closed path of quantum states used to estimate geometric phase."""

    states: Sequence[Sequence[complex]]

    def geometric_phase(self) -> float:
        """Estimate the geometric phase from successive state overlaps.

        The phase is computed as the argument of the product of overlaps
        between normalized states along the path, including the closure term.
        """
        if len(self.states) < 2:
            raise ValueError("CircuitPath requires at least two states.")

        normalized = [normalize_state(state) for state in self.states]
        overlap_product = 1 + 0j
        for left, right in zip(normalized, normalized[1:]):
            overlap_product *= inner_product(left, right)
        overlap_product *= inner_product(normalized[-1], normalized[0])
        return cmath.phase(overlap_product)


@dataclass
class ParameterizedCircuit:
    """Simple parameterized circuit placeholder."""

    parameters: List[float]

    def update(self, deltas: Iterable[float]) -> None:
        """Apply parameter deltas in-place."""
        self.parameters = [p + dp for p, dp in zip(self.parameters, deltas)]

    def build_path(self) -> CircuitPath:
        """Construct a toy circuit path based on current parameters."""
        states = []
        for theta in self.parameters:
            state = [cmath.cos(theta), cmath.sin(theta)]
            states.append(state)
        return CircuitPath(states=states)
