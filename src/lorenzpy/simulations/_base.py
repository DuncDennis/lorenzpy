"""Base code for simulations."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


def _forward_euler(
    f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray
) -> np.ndarray:
    """Simulate one step for ODEs of the form dx/dt = f(x(t)) using the forward euler.

    Args:
        f: function used to calculate the time derivative at point x.
        dt: time step size.
        x: d-dim position at time t.

    Returns:
       d-dim position at time t+dt.

    """
    return x + dt * f(x)


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
    """Iterate an iterator-function f: x(i+1) = f(x(i)) multiple times."""
    starting_point = np.array(starting_point)
    traj_size = (time_steps, starting_point.shape[0])
    traj = np.zeros(traj_size)
    traj[0, :] = starting_point
    for t in range(1, traj_size[0]):
        traj[t] = f(traj[t - 1])
    return traj


class _BaseSim(ABC):
    """Base class for all simulation classes."""

    @classmethod
    def _get_init_params(cls) -> list[inspect.Parameter]:
        """Get the init parameter names and default values."""
        init = cls.__init__

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)

        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]
        return parameters

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        """Get the default parameters."""
        parameters = cls._get_init_params()
        out = dict()
        for param in parameters:
            out[param.name] = param.default
        return out

    def get_params(self) -> dict[str, Any]:
        """Get the current parameters."""
        out = dict()
        parameters = self._get_init_params()
        for param in parameters:
            key = param.name
            value = getattr(self, key)
            out[key] = value
        return out

    @abstractmethod
    def simulate(
        self,
        time_steps: int,
        starting_point: np.ndarray | None = None,
        transient: int = 0,
    ) -> np.ndarray:
        """Simulate the trajectory of the Dynamical System.

        Args:
            time_steps: Number of time steps in returned trajectory.
            starting_point: The starting point of the trajectory of shape (sim_dim,).
                            If None is provided, the default value will be used.
                            If transient is 0, the first point of the returned
                            trajectory will be the starting_point.
            transient: Optional number of transient points to discard before tracking
                       the returned trajectory.

        Returns:
            The trajectory of shape (time_steps, sim_dim).
        """

    @abstractmethod
    def get_default_starting_pnt(self) -> np.ndarray:
        """Get the default starting point to be used in the simulate function."""


class _BaseSimIterate(_BaseSim):
    """Base class for Simulation classes using an iterate function."""

    @abstractmethod
    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Iterate the system from one time step to the next."""

    def simulate(
        self,
        time_steps: int,
        starting_point: np.ndarray | None = None,
        transient: int = 0,
    ) -> np.ndarray:
        """Simulate the trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory shape (sys_dim,).
                            If None, take the default starting point.
            transient: TBD

        Returns:
            Trajectory of shape (t, sys_dim).

        """
        if starting_point is None:
            starting_point = self.get_default_starting_pnt()

        return _timestep_iterator(self.iterate, time_steps + transient, starting_point)[
            transient:, :
        ]


class _BaseSimFlow(_BaseSimIterate):
    """Base class for continuous-time Simulation classes defined by a flow."""

    @abstractmethod
    def __init__(
        self,
        dt: float,
        solver: str | Callable[[Callable, float, np.ndarray], np.ndarray],
    ):
        """Initialize the time step dt and the solver."""
        self.dt = dt
        self.solver = solver

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
        if isinstance(self.solver, str):
            if self.solver == "rk4":
                x_next = _runge_kutta(self.flow, self.dt, x)
            elif self.solver == "forward_euler":
                x_next = _forward_euler(self.flow, self.dt, x)
            else:
                raise ValueError(f"Unsupported solver: {self.solver}")
        else:
            x_next = self.solver(self.flow, self.dt, x)

        return x_next


class _BaseSimFlowDriven(_BaseSimFlow, ABC):
    """Base class for driven continuous time dynamical systems."""

    def simulate(
        self,
        time_steps: int,
        starting_point: np.ndarray | None = None,
        transient: int = 0,
        time_included: bool = False,
    ) -> np.ndarray:
        if starting_point is not None:
            if not time_included:
                starting_point = np.append(starting_point, 0)
        traj = super().simulate(
            time_steps, starting_point=starting_point, transient=transient
        )
        if not time_included:
            return traj[:, :-1]
        else:
            return traj
