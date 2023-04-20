"""Autonomous flows."""
import numpy as np

from ._base import _BaseSimFlow


class Lorenz63(_BaseSimFlow):
    """Simulation class for the Lorenz63 system.

    This function is able to simulate the chaotic dynamical system originally
    introduced by Lorenz.

    Attributes:
        sigma: Sigma parameter.
        rho: rho parameter.
    """

    def __init__(self, sigma=10.0, rho=28.0, beta=8 / 3, dt=0.1, solver="rk4"):
        """Initialize the Lorenz63.

        Args:
            sigma: a.
            rho: b.
            beta: c.
            dt: d.
            solver: e.
        """
        super().__init__(dt, solver)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def flow(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.sigma * (x[1] - x[0]),
                x[0] * (self.rho - x[2]) - x[1],
                x[0] * x[1] - self.beta * x[2],
            ]
        )

    def get_default_starting_pnt(self) -> np.ndarray:
        return np.array([0.0, -0.01, 9.0])


class Lorenz96(_BaseSimFlow):
    """Simulate the n-dimensional Lorenz 96 model."""

    def __init__(
        self, sys_dim: int = 30, force: float = 8.0, dt: float = 0.05, solver="rk4"
    ) -> None:
        super().__init__(dt=dt, solver=solver)
        self.sys_dim = sys_dim
        self.force = force

    def get_default_starting_pnt(self) -> np.ndarray:
        return np.sin(np.arange(self.sys_dim))

    def flow(self, x: np.ndarray) -> np.ndarray:
        system_dimension = x.shape[0]
        derivative = np.zeros(system_dimension)
        # Periodic Boundary Conditions for the 3 edge cases i=1,2,system_dimension
        derivative[0] = (x[1] - x[system_dimension - 2]) * x[system_dimension - 1] - x[
            0
        ]
        derivative[1] = (x[2] - x[system_dimension - 1]) * x[0] - x[1]
        derivative[system_dimension - 1] = (x[0] - x[system_dimension - 3]) * x[
            system_dimension - 2
        ] - x[system_dimension - 1]

        # TODO: Rewrite using numpy vectorization to make faster
        for i in range(2, system_dimension - 1):
            derivative[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

        derivative = derivative + self.force
        return derivative
