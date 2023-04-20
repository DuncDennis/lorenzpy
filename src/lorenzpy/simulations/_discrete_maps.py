"""Discrete maps."""
import numpy as np

from ._base import _BaseSimIterate


class Logistic(_BaseSimIterate):
    def __init__(self, r: float = 4.0) -> None:
        self.r = r

    def iterate(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.r * x[0] * (1 - x[0]),
            ]
        )

    def get_default_starting_pnt(self) -> np.ndarray:
        return np.array([0.1])
