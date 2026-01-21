"""Training utilities for toy geometric-phase optimization."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, List, Tuple

from geophase.circuits import ParameterizedCircuit
from geophase.losses import geometric_phase_loss


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a simple training loop."""

    learning_rate: float = 0.1
    steps: int = 100
    seed: int = 7


class Trainer:
    """Simple trainer that optimizes circuit parameters."""

    def __init__(self, circuit: ParameterizedCircuit, config: TrainingConfig) -> None:
        self.circuit = circuit
        self.config = config
        self.rng = random.Random(config.seed)

    def step(self, target_phase: float) -> Tuple[float, float]:
        """Perform a single optimization step.

        Returns the loss before and after the update.
        """
        current_phase = self.circuit.build_path().geometric_phase()
        current_loss = geometric_phase_loss(current_phase, target_phase)
        gradients = self._finite_difference(target_phase)
        updates = [-self.config.learning_rate * g for g in gradients]
        self.circuit.update(updates)
        new_phase = self.circuit.build_path().geometric_phase()
        new_loss = geometric_phase_loss(new_phase, target_phase)
        return current_loss, new_loss

    def train(self, target_phase: float) -> List[float]:
        """Run the training loop and return loss history."""
        losses = []
        for _ in range(self.config.steps):
            loss_before, loss_after = self.step(target_phase)
            losses.append(loss_after)
            if loss_after > loss_before:
                self._jitter_parameters()
        return losses

    def _finite_difference(self, target_phase: float) -> List[float]:
        """Estimate gradients via finite differences."""
        epsilon = 1e-3
        base_phase = self.circuit.build_path().geometric_phase()
        base_loss = geometric_phase_loss(base_phase, target_phase)
        gradients = []
        for idx, original in enumerate(self.circuit.parameters):
            self.circuit.parameters[idx] = original + epsilon
            phase_plus = self.circuit.build_path().geometric_phase()
            loss_plus = geometric_phase_loss(phase_plus, target_phase)
            gradients.append((loss_plus - base_loss) / epsilon)
            self.circuit.parameters[idx] = original
        return gradients

    def _jitter_parameters(self) -> None:
        """Apply a small random perturbation to escape plateaus."""
        jitter = [self.rng.uniform(-0.01, 0.01) for _ in self.circuit.parameters]
        self.circuit.update(jitter)
