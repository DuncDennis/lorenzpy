"""Simulate various continuous and discrete chaotic dynamical system.

Every dynamical system is represented as a class.

The available classes are:
- Lorenz63
- MackeyGlass

The system's parameters are introduced in the class's constructor.

For example when creating a system object of the Lorenz63, the Lorenz parameters,
sigma, rho, beta, and the timestep dt are parsed as:

sys_obj = Lorenz63(sigma=10, rho=10, beta=5, dt=1)

Each sys_obj contains a "simulate" function.
To simulate 1000 time-steps of the Lorenz63 system call:

sys_obj.simulate(1000).

The general syntax to create a trajectory of a System is given as:

trajectory = <SystemClass>(<parameters>=<default>).
simulate(time_steps, starting_point=<default>)

Examples:
    >>> import lorenzpy.simulations as sims
    >>> data = sims.Lorenz63().simulate(1000)
    >>> data.shape
    (1000, 3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


def _runge_kutta(
    f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray
) -> np.ndarray:
    """Simulate one step for ODEs of the form dx/dt = f(x(t)) using Runge-Kutta.

    Args:
        f: function used to calculate the time derivative at point x.
        dt: time step size.
        x: d-dim position at time t.

    Returns:
       d-dim position at time t+dt.

    """
    k1: np.ndarray = dt * f(x)
    k2: np.ndarray = dt * f(x + k1 / 2)
    k3: np.ndarray = dt * f(x + k2 / 2)
    k4: np.ndarray = dt * f(x + k3)
    next_step: np.ndarray = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_step


def _timestep_iterator(
    f: Callable[[np.ndarray], np.ndarray], time_steps: int, starting_point: np.ndarray
) -> np.ndarray:
    """Iterate a function f: x(i+1) = f(x(i)) multiple times to obtain a full traj.

    Args:
        f: The iterator function x(i+1) = f(x(i)).
        time_steps: The number of time_steps of the output trajectory.
                    The starting_point is included as the 0th element in the trajectory.
        starting_point: Starting point of the trajectory.

    Returns:
        trajectory: system-state at every simulated timestep.

    """
    starting_point = np.array(starting_point)
    traj_size = (time_steps, starting_point.shape[0])
    traj = np.zeros(traj_size)
    traj[0, :] = starting_point
    for t in range(1, traj_size[0]):
        traj[t] = f(traj[t - 1])
    return traj


class _SimBase(ABC):
    """A base class for all the simulation classes."""

    default_starting_point: np.ndarray
    sys_dim: int

    @abstractmethod
    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Generate the next time step when the previous one is given."""

    def simulate(
        self, time_steps: int, starting_point: np.ndarray | None = None
    ) -> np.ndarray:
        """Simulate the trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory shape (sys_dim,).
                            If None, take the default starting point.

        Returns:
            Trajectory of shape (t, sys_dim).

        """
        if starting_point is None:
            starting_point = self.default_starting_point
        else:
            if starting_point.size != self.sys_dim:
                raise ValueError(
                    "Provided starting_point has the wrong dimension. "
                    f"{self.sys_dim} was expected and {starting_point.size} "
                    "was given"
                )
        return _timestep_iterator(self.iterate, time_steps, starting_point)


class _SimBaseRungeKutta(_SimBase):
    dt: float

    @abstractmethod
    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return the flow of the continuous dynamical system."""

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculate the next timestep x(t+dt) with given x(t) using runge kutta.

        Args:
            x: the previous point x(t).

        Returns:
            : x(t+dt) corresponding to input x(t).

        """
        return _runge_kutta(self.flow, self.dt, x)


class Lorenz63(_SimBaseRungeKutta):
    """Simulate the 3-dimensional autonomous flow: Lorenz-63 attractor.

    Literature values (Sprott, Julien Clinton, and Julien C. Sprott. Chaos and
    time-series analysis. Vol. 69. Oxford: Oxford university press, 2003.) for default
    parameters and starting_point:
    - Lyapunov exponents: (0.9059, 0.0, -14.5723)
    - Kaplan-Yorke dimension: 2.06215
    - Correlation dimension: 2.068 +- 0.086
    They refer to:
    - Parameters: {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
    - Starting point: [0.0, -0.01, 9.0]
    """

    default_parameters = {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3, "dt": 0.05}
    default_starting_point = np.array([0.0, -0.01, 9.0])
    sys_dim = 3

    def __init__(
        self,
        sigma: float | None = None,
        rho: float | None = None,
        beta: float | None = None,
        dt: float | None = None,
    ) -> None:
        """Define the system parameters.

        Args:
            sigma: 'sigma' parameter in the Lorenz 63 equations.
            rho: 'rho' parameter in the Lorenz 63 equations.
            beta: 'beta' parameter in the Lorenz 63 equations.
            dt: Size of time steps.
        """
        self.sigma = self.default_parameters["sigma"] if sigma is None else sigma
        self.rho = self.default_parameters["rho"] if rho is None else rho
        self.beta = self.default_parameters["beta"] if beta is None else beta
        self.dt = self.default_parameters["dt"] if dt is None else dt

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Calculate (dx/dt, dy/dt, dz/dt) with given (x,y,z) for RK4.

        Args:
            x: (x,y,z) coordinates. Needs to have shape (3,).

        Returns:
            : (dx/dt, dy/dt, dz/dt) corresponding to input x.

        """
        return np.array(
            [
                self.sigma * (x[1] - x[0]),
                x[0] * (self.rho - x[2]) - x[1],
                x[0] * x[1] - self.beta * x[2],
            ]
        )


class MackeyGlass:
    """Simulate the Mackey-Glass delay differential system.

    TODO: Add literature values for Lyapunov etc.
    TODO: Hint the differences between this class and the other Sim classes (delay).
    TODO: Check if the structure is really good?
    TODO: Add Proper Tests.
    TODO: Decide whether to use the simple forward-euler or RK4-style update.

    Note: As the Mackey-Glass system is a delay-differential equation, the class does
    not contain a simple iterate function.
    """

    default_parameters = {"a": 0.2, "b": 0.1, "c": 10, "tau": 23.0, "dt": 0.1}
    default_starting_point = np.array([0.9])
    sys_dim = 1

    def __init__(
        self,
        a: float | None = None,
        b: float | None = None,
        c: int | None = None,
        tau: float | None = None,
        dt: float | None = None,
    ) -> None:
        """Define the Mackey-Glass system parameters.

        Args:
            a: "a" parameter of the Mackey-Glass equation.
            b: "b" parameter of the Mackey-Glass equation.
            c: "c" parameter of the Mackey-Glass equation.
            tau: Time delay of Mackey-Glass equation.
            dt: Size of time steps.
        """
        self.a = self.default_parameters["a"] if a is None else a
        self.b = self.default_parameters["b"] if b is None else b
        self.c = self.default_parameters["c"] if c is None else c
        self.tau = self.default_parameters["tau"] if tau is None else tau
        self.dt = self.default_parameters["dt"] if dt is None else dt

        # The number of time steps between t=0 and t=-tau:
        self.history_steps = int(self.tau / self.dt)

    def flow_mg(self, x: np.ndarray, x_past: np.ndarray) -> np.ndarray:
        """Calculate the flow of the Mackey-Glass equation.

        Args:
            x: The immediate value of the system. Needs to have shape (1,).
            x_past: The delayed value of the system. Needs to have shape (1,).

        Returns:
            : The flow corresponding to x and x_past.
        """
        return np.array(
            [self.a * x_past[0] / (1 + x_past[0] ** self.c) - self.b * x[0]]
        )

    def iterate_mg(self, x: np.ndarray, x_past: np.ndarray) -> np.ndarray:
        """Calculate the next time step in the Mackey-Glass equation.

        Args:
            x: The immediate value of the system. Needs to have shape (1,).
            x_past: The delayed value of the system. Needs to have shape (1,).

        Returns:
            : The next value given the immediate and delayed values.
        """
        return x + self.dt * self.flow_mg(x, x_past)
        # f = lambda x_use: self.flow_mg(x_use, x_past)
        # return _runge_kutta(f, self.dt, x)

    def simulate(
        self, time_steps: int, starting_point: np.ndarray | None = None
    ) -> np.ndarray:
        """Simulate the Mackey-Glass trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory shape (sys_dim,).
                            If None, take the default starting point.

        Returns:
            Trajectory of shape (t, sys_dim).

        """
        if starting_point is None:
            starting_point = self.default_starting_point
        else:
            if starting_point.size != self.sys_dim:
                raise ValueError(
                    "Provided starting_point has the wrong dimension. "
                    f"{self.sys_dim} was expected and {starting_point.size} "
                    "was given"
                )

        traj_w_hist = np.zeros((self.history_steps + time_steps, self.sys_dim))
        traj_w_hist[: self.history_steps, :] = np.repeat(
            starting_point, self.history_steps
        )[:, np.newaxis]
        traj_w_hist[self.history_steps, :] = starting_point

        for t in range(1, time_steps):
            t_shifted = t + self.history_steps
            traj_w_hist[t_shifted] = self.iterate_mg(
                traj_w_hist[t_shifted - 1],
                traj_w_hist[t_shifted - 1 - self.history_steps],
            )

        return traj_w_hist[self.history_steps :, :]
