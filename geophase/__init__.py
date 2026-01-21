"""Geometric-phase-inspired utilities for noise-robust VQC research."""

from geophase.circuits import CircuitPath, ParameterizedCircuit
from geophase.losses import geometric_phase_loss, phase_distance
from geophase.noise import depolarizing_channel, amplitude_damping_channel
from geophase.train import Trainer, TrainingConfig

__all__ = [
    "CircuitPath",
    "ParameterizedCircuit",
    "geometric_phase_loss",
    "phase_distance",
    "depolarizing_channel",
    "amplitude_damping_channel",
    "Trainer",
    "TrainingConfig",
]
