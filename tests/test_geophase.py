import math
import unittest

from geophase.circuits import CircuitPath, ParameterizedCircuit
from geophase.losses import geometric_phase_loss
from geophase.noise import amplitude_damping_channel, depolarizing_channel
from geophase.train import Trainer, TrainingConfig


class CircuitPathTests(unittest.TestCase):
    def test_geometric_phase_closed_loop(self):
        states = [
            [1 + 0j, 0j],
            [0j, 1 + 0j],
            [1 + 0j, 0j],
        ]
        path = CircuitPath(states=states)
        phase = path.geometric_phase()
        self.assertTrue(-math.pi <= phase <= math.pi)


class LossTests(unittest.TestCase):
    def test_geometric_phase_loss_zero(self):
        self.assertEqual(geometric_phase_loss(0.5, 0.5), 0.0)


class NoiseTests(unittest.TestCase):
    def test_depolarizing_channel_mixes_state(self):
        density = depolarizing_channel([1 + 0j, 0j], probability=1.0)
        self.assertAlmostEqual(density[0][0].real, 0.5, places=6)
        self.assertAlmostEqual(density[1][1].real, 0.5, places=6)

    def test_amplitude_damping_channel_reduces_excited(self):
        density = amplitude_damping_channel([0j, 1 + 0j], gamma=0.5)
        self.assertLess(density[1][1].real, 1.0)


class TrainerTests(unittest.TestCase):
    def test_trainer_runs(self):
        circuit = ParameterizedCircuit(parameters=[0.1, 0.2, 0.3])
        trainer = Trainer(circuit, TrainingConfig(steps=5, learning_rate=0.05, seed=1))
        losses = trainer.train(target_phase=0.0)
        self.assertEqual(len(losses), 5)


if __name__ == "__main__":
    unittest.main()
