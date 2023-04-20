"""Driven Systems."""
import numpy as np

from ._base import _BaseSimFlowDriven


class SimplestDrivenChaotic(_BaseSimFlowDriven):
    def __init__(self, omega: float = 1.88, dt: float = 0.1, solver="rk4") -> None:
        super().__init__(dt, solver)
        self.omega = omega

    def flow(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[1], -(x[0] ** 3) + np.sin(self.omega * x[2]), 1])

    def get_default_starting_pnt(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])
