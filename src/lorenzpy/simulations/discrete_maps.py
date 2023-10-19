"""Discrete maps."""
from __future__ import annotations

import numpy as np

from .base import _BaseSimIterate


class Logistic(_BaseSimIterate):
    """Simulation class for the Logistic map."""

    def __init__(self, r: float = 4.0) -> None:
        """Initialize the Logistic Map simulation object.

        Args:
            r: r parameter of the logistic map.
        """
        self.r = r

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Iterate the logistic map one step."""
        return np.array(
            [
                self.r * x[0] * (1 - x[0]),
            ]
        )

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of the Logistic map."""
        return np.array([0.1])


class Henon(_BaseSimIterate):
    """Simulate the 2-dimensional dissipative map: Henon map."""

    def __init__(self, a: float = 1.4, b: float = 0.3) -> None:
        """Initialize the Logistic Map simulation object.

        Args:
            a: a parameter of the Henon map.
            b: b parameter of the Henon map.
        """
        self.a = a
        self.b = b

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Iterate the Henon map one step."""
        return np.array([1 - self.a * x[0] ** 2 + self.b * x[1], x[0]])

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of the Henon map."""
        return np.array([0.0, 0.9])
