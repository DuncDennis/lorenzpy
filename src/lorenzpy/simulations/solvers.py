"""The solvers used to solve the flow equation."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def forward_euler(
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


def runge_kutta_4(
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


def create_scipy_ivp_solver(
    method: str = "RK45", **additional_solve_ivp_args
) -> Callable[[Callable, float, np.ndarray], np.ndarray]:
    """Create a scipy solver for initializing flow systems.

    This function creates a scipy solver that can be used to initialize flow simulation
    classes. It wraps the scipy.integrate.solve_ivp function.

    Note: The scipy solvers often internally integrate more than 1 time step in the
        range 0 to dt.

    Args:
        method (str): The integration method to use, e.g., 'RK45', 'RK23', 'DOP853',
            'Radau', 'BDF', or 'LSODA'. See the documentation for
            scipy.integrate.solve_ivp for more information.
        **additional_solve_ivp_args: Additional arguments passed to
            scipy.integrate.solve_ivp as `solve_ivp(fun, t_span, y0, method=method,
            **additional_solve_ivp_args)`. For example you can pass the relative and
            absolute tolerances as rtol=x and atol=x.

    Returns:
        Callable[[Callable[[np.ndarray], np.ndarray], float, np.ndarray], np.ndarray]:
        A solver function that takes three arguments:
        1. f (Callable[[np.ndarray], np.ndarray]): The flow function.
        2. dt (float): The time step for the integration.
        3. x (np.ndarray): The initial state.

        The solver returns the integrated state at the end of the time step.
    """

    def solver(
        f: Callable[[np.ndarray], np.ndarray], dt: float, x: np.ndarray
    ) -> np.ndarray:
        """Solves the flow function for a time step using scipy.integrate.solve_ivp.

        Args:
            f (Callable[[np.ndarray], np.ndarray]): The flow function.
            dt (float): The time step for the integration.
            x (np.ndarray): The initial state.

        Returns:
            np.ndarray: The integrated state at the end of the time step.
        """

        def scipy_func_form(t, y):
            """Ignores the time argument for the flow f."""
            return f(y)

        out = solve_ivp(
            scipy_func_form, (0, dt), x, method=method, **additional_solve_ivp_args
        )
        return out.y[:, -1]

    return solver


def timestep_iterator(
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
