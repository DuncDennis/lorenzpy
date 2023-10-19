"""Autonomous flows."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import _BaseSimFlow
from .solvers import create_scipy_ivp_solver


class Lorenz63(_BaseSimFlow):
    """Simulation class for the Lorenz63 system.

    This function is able to simulate the chaotic dynamical system originally
    introduced by Lorenz.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        dt: float = 0.1,
        solver: str | str | Callable[[Callable, float, np.ndarray], np.ndarray] = "rk4",
    ):
        """Initialize the Lorenz63 simulation object.

        Args:
            sigma: Sigma parameter of Lorenz63 equation.
            rho: Rho parameter of Lorenz63 equation.
            beta: beta parameter of Lorenz63 equation.
            dt: Time step to simulate.
            solver: The solver.
        """
        super().__init__(dt, solver)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow of Lorenz63 equation."""
        return np.array(
            [
                self.sigma * (x[1] - x[0]),
                x[0] * (self.rho - x[2]) - x[1],
                x[0] * x[1] - self.beta * x[2],
            ]
        )

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of Lorenz63 system."""
        return np.array([0.0, -0.01, 9.0])


class Roessler(_BaseSimFlow):
    """Simulation class for the Roessler system."""

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        dt: float = 0.1,
        solver: str | str | Callable[[Callable, float, np.ndarray], np.ndarray] = "rk4",
    ):
        """Initialize the Roessler simulation object.

        Args:
            a: a parameter of Roessler equation.
            b: b parameter of Roessler equation.
            c: c parameter of Roessler equation.
            dt: Time step to simulate.
            solver: The solver.
        """
        super().__init__(dt, solver)
        self.a = a
        self.b = b
        self.c = c

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow of Roessler equation."""
        return np.array(
            [-x[1] - x[2], x[0] + self.a * x[1], self.b + x[2] * (x[0] - self.c)]
        )

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of Roessler system."""
        return np.array([-9.0, 0.0, 0.0])


class DoublePendulum(_BaseSimFlow):
    """Simulation class for the dimensionless double pendulum with m1 = m2 and l1=l2.

    The state space is given by [angle1, angle2, angular_vel, angular_vel2].
    """

    def __init__(
        self,
        dt: float = 0.1,
        solver: str
        | str
        | Callable[[Callable, float, np.ndarray], np.ndarray] = create_scipy_ivp_solver(
            "DOP853"
        ),
    ):
        """Initialize the Doueble Pendulum simulation object.

        Args:
            dt: Time step to simulate.
            solver: The solver. Default is DOP853 scipy solver here.
        """
        super().__init__(dt, solver)

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow of double pendulum."""
        angle1, angle2, angle1_dot, angle2_dot = x[0], x[1], x[2], x[3]

        delta_angle = angle1 - angle2

        angle1_dotdot = (
            9 * np.cos(delta_angle) * np.sin(delta_angle) * angle1_dot**2
            + 6 * np.sin(delta_angle) * angle2_dot**2
            + 18 * np.sin(angle1)
            - 9 * np.cos(delta_angle) * np.sin(angle2)
        ) / (9 * np.cos(delta_angle) ** 2 - 16)

        angle2_dotdot = (
            24 * np.sin(delta_angle) * angle1_dot**2
            + 9 * np.cos(delta_angle) * np.sin(delta_angle) * angle2_dot**2
            + 27 * np.cos(delta_angle) * np.sin(angle1)
            - 24 * np.sin(angle2)
        ) / (16 - 9 * np.cos(delta_angle) ** 2)

        return np.array([angle1_dot, angle2_dot, angle1_dotdot, angle2_dotdot])

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of Double Pendulum."""
        return np.array([0.6, 2.04, 0, 0])


class Lorenz96(_BaseSimFlow):
    """Simulate the n-dimensional Lorenz 96 model."""

    def __init__(
        self,
        sys_dim: int = 30,
        force: float = 8.0,
        dt: float = 0.05,
        solver: str | str | Callable[[Callable, float, np.ndarray], np.ndarray] = "rk4",
    ) -> None:
        """Initialize the Lorenz96 simulation object.

        Args:
            sys_dim: The dimension of the Lorenz96 system.
            force: The force value.
            dt: Time step to simulate.
            solver: The solver.
        """
        super().__init__(dt=dt, solver=solver)
        self.sys_dim = sys_dim
        self.force = force

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow of Lorenz96 equation."""
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

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return default starting point of Lorenz96 system."""
        return np.sin(np.arange(self.sys_dim))
