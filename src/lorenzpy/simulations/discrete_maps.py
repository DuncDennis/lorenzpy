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
