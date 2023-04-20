"""Dynamical systems implemented with other algorithms."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._base import _BaseSim, _BaseSimIterate, _forward_euler, _runge_kutta


class KuramotoSivashinsky(_BaseSimIterate):
    """Simulate the n-dimensional Kuramoto-Sivashinsky PDE.

    Note: dimension must be an even number.

    PDE: y_t = -y*y_x - (1+eps)*y_xx - y_xxxx.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Literature values (doi:10.1017/S1446181119000105) for Lyapunov Exponents:
    - lyapunov exponents: (0.080, 0.056, 0.014, 0.003, -0.003 ...)
    They refer to:
    - Parameters: {"sys_length": 36.0, "eps": 0.0}
    """

    def __init__(
        self,
        sys_dim: int = 50,
        sys_length: float = 36.0,
        eps: float = 0.0,
        dt: float = 0.1,
    ) -> None:
        self.sys_dim = sys_dim
        self.sys_length = sys_length
        self.eps = eps
        self.dt = dt

        self._prepare()

    def get_default_starting_pnt(self) -> np.ndarray:
        x = (
            self.sys_length
            * np.transpose(np.conj(np.arange(1, self.sys_dim + 1)))
            / self.sys_dim
        )
        return np.array(
            np.cos(2 * np.pi * x / self.sys_length)
            * (1 + np.sin(2 * np.pi * x / self.sys_length))
        )

    def _prepare(self) -> None:
        """Calculate internal attributes.

        TODO: make auxiliary variables protected.
        """
        k = (
            np.transpose(
                np.conj(
                    np.concatenate(
                        (
                            np.arange(0, self.sys_dim / 2),
                            np.array([0]),
                            np.arange(-self.sys_dim / 2 + 1, 0),
                        )
                    )
                )
            )
            * 2
            * np.pi
            / self.sys_length
        )

        L = (1 + self.eps) * k**2 - k**4

        self.E = np.exp(self.dt * L)
        self.E_2 = np.exp(self.dt * L / 2)
        M = 64
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = self.dt * np.transpose(np.repeat([L], M, axis=0)) + np.repeat(
            [r], self.sys_dim, axis=0
        )
        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = self.dt * np.real(
            np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
        )
        self.f2 = self.dt * np.real(
            np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
        )
        self.f3 = self.dt * np.real(
            np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
        )

        self.g = -0.5j * k

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Calculate next timestep x(t+1) with given x(t).

        Args:
            x: (x_0(i),x_1(i),..) coordinates. Needs to have shape (self.sys_dim,).

        Returns:
            : (x_0(i+1),x_1(i+1),..) corresponding to input x.

        """
        v = np.fft.fft(x)
        Nv = self.g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = self.E_2 * v + self.Q * Nv
        Na = self.g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = self.E_2 * v + self.Q * Na
        Nb = self.g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = self.E_2 * a + self.Q * (2 * Nb - Nv)
        Nc = self.g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
        v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        return np.real(np.fft.ifft(v))


class MackeyGlass(_BaseSim):
    """Simulate the Mackey-Glass delay differential system.

    TODO: Add literature values for Lyapunov etc.
    TODO: Hint the differences between this class and the other Sim classes (delay).
    TODO: Check if the structure is really good?
    TODO: Add Proper Tests.
    TODO: Decide whether to use the simple forward-euler or RK4-style update.

    Note: As the Mackey-Glass system is a delay-differential equation, the class does
    not contain a simple iterate function.
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.1,
        c: int = 10,
        tau: float = 23.0,
        dt: float = 0.1,
        solver: str | Callable = "rk4",
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.tau = tau
        self.dt = dt
        self.solver = solver

        # The number of time steps between t=0 and t=-tau:
        self.history_steps = int(self.tau / self.dt)

    def get_default_starting_pnt(self) -> np.ndarray:
        return np.array([0.9])

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

        def flow_like(x_use: np.ndarray) -> np.ndarray:
            return self.flow_mg(x_use, x_past)

        if isinstance(self.solver, str):
            if self.solver == "rk4":
                x_next = _runge_kutta(flow_like, self.dt, x)
            elif self.solver == "forward_euler":
                x_next = _forward_euler(flow_like, self.dt, x)
            else:
                raise ValueError(f"Unsupported solver: {self.solver}")
        else:
            x_next = self.solver(flow_like, self.dt, x)

        return x_next

    def simulate(
        self,
        time_steps: int,
        starting_point: np.ndarray | None = None,
        transient: int = 0,
    ) -> np.ndarray:
        """Simulate the Mackey-Glass trajectory.

        Args:
            time_steps: Number of time steps t to simulate.
            starting_point: Starting point of the trajectory shape (sys_dim,).
                            If None, take the default starting point.
            transient: Washout before storing the trajectory.

        Returns:
            Trajectory of shape (t, sys_dim).
        """
        if starting_point is None:
            starting_point = self.get_default_starting_pnt()

        if starting_point.size == self.history_steps + 1:
            initial_history = starting_point
        elif starting_point.size == 1:
            initial_history = np.repeat(starting_point, self.history_steps)
        else:
            raise ValueError("Wrong size of starting point.")

        traj_w_hist = np.zeros((self.history_steps + time_steps, 1))
        traj_w_hist[: self.history_steps, :] = initial_history[:, np.newaxis]
        traj_w_hist[self.history_steps, :] = starting_point

        for t in range(1, time_steps + transient):
            t_shifted = t + self.history_steps
            traj_w_hist[t_shifted] = self.iterate_mg(
                traj_w_hist[t_shifted - 1],
                traj_w_hist[t_shifted - 1 - self.history_steps],
            )

        return traj_w_hist[self.history_steps + transient :, :]
