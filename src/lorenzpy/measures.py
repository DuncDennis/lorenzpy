"""Measures for chaotic dynamical systems."""
from __future__ import annotations

from typing import Callable

import numpy as np


def largest_lyapunov_exponent(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    steps: int = int(1e3),
    part_time_steps: int = 15,
    steps_skip: int = 50,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis.
    Vol. 69. Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step:
                       x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        steps: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        steps_skip: Number of renormalization steps to perform, before tracking the log
                    divergence. Avoid transients by using steps_skip.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be
                np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy
                            array of the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence
        (np.ndarray), else just the float value, which is the last value in the
        convergence.
    """
    x_dim = starting_point.size

    if initial_pert_direction is None:
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation = initial_pert_direction * (
        deviation_scale / np.linalg.norm(initial_pert_direction)
    )

    log_divergence = np.zeros(steps)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(steps + steps_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= steps_skip:
            log_divergence[i_n - steps_skip] = np.log(norm_dx / deviation_scale)

    if return_convergence:
        return np.array(
            np.cumsum(log_divergence) / (np.arange(1, steps + 1) * dt * part_time_steps)
        )
    else:
        return float(np.average(log_divergence) / (dt * part_time_steps))


def lyapunov_exponent_spectrum(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    steps: int = int(1e3),
    part_time_steps: int = 15,
    steps_skip: int = 50,
    dt: float = 1.0,
    m: int | None = None,
    initial_pert_directions: np.ndarray | None = None,
    return_convergence: bool = False,
) -> np.ndarray:
    """Calculate the spectrum of m largest lyapunov exponent given an iterator function.

    A mixture of:
    - The algorithm for the largest lyapunov exponent: Sprott, Julien Clinton, and
        Julien C. Sprott. Chaos and time-series analysis. Vol. 69. Oxford: Oxford
        university press, 2003.
    - The algorithm for the spectrum given in 1902.09651 "LYAPUNOV EXPONENTS of the
        KURAMOTO-SIVASHINSKY PDE".

    Args:
        iterator_func: Function to iterate the system to the next time step:
                        x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        steps: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        steps_skip: Number of renormalization steps to perform, before tracking the log
                    divergence.
                Avoid transients by using steps_skip.
        dt: Size of time step.
        m: Number of Lyapunov exponents to compute. If None: take all (m = x_dim).
        initial_pert_directions:
            - If np.ndarray: The direction of the initial perturbations of shape
                            (m, x_dim). Will be qr decomposed
                             and the q matrix will be used.
            - If None: The direction of the initial perturbation is assumed to be
                        np.eye(x_dim)[:m, :].
        return_convergence: If True, return the convergence of the largest LE; a
                            numpy array of the shape (N, m).

    Returns:
        The Lyapunov exponent spectrum of largest m values. If return_convergence is
        True: The convergence (2D N x m np.ndarray), else a (1D m-size np.ndarray),
        which holds the last values in the convergence.
    """
    x_dim = starting_point.size

    if m is None:
        m = x_dim

    if initial_pert_directions is None:
        initial_perturbations = (
            np.eye(x_dim)[:m, :] * deviation_scale
        )  # each vector is of length deviation_scale
    else:
        q, _ = np.linalg.qr(initial_pert_directions)
        initial_perturbations = q * deviation_scale

    log_divergence_spec = np.zeros((steps, m))

    x = starting_point
    x_pert = starting_point + initial_perturbations

    for i_n in range(steps + steps_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            for i_m in range(m):
                x_pert[i_m, :] = iterator_func(x_pert[i_m, :])
        dx = x_pert - x
        q, r = np.linalg.qr(dx.T, mode="reduced")
        x_pert = x + q.T * deviation_scale

        if i_n >= steps_skip:
            log_divergence_spec[i_n - steps_skip, :] = np.log(
                np.abs(np.diag(r)) / deviation_scale
            )

    if return_convergence:
        return np.array(
            np.cumsum(log_divergence_spec, axis=0)
            / (
                np.repeat(np.arange(1, steps + 1)[:, np.newaxis], m, axis=1)
                * dt
                * part_time_steps
            )
        )
    else:
        return np.average(log_divergence_spec, axis=0) / (dt * part_time_steps)
