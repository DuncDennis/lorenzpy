"""Driven Systems."""
import numpy as np

from .base import _BaseSimFlowDriven


class SimplestDrivenChaotic(_BaseSimFlowDriven):
    """Simulate the Simplest Driven Chaotic system from Sprott.

    Taken from (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series
    analysis. Vol. 69. Oxford: Oxford university press, 2003.)
    """

    def __init__(self, omega: float = 1.88, dt: float = 0.1, solver="rk4") -> None:
        """Initialize the SimplestDrivenChaotic simulation object."""
        super().__init__(dt, solver)
        self.omega = omega

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow ."""
        return np.array([x[1], -(x[0] ** 3) + np.sin(self.omega * x[2]), 1])

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point."""
        return np.array([0.0, 0.0, 0.0])
