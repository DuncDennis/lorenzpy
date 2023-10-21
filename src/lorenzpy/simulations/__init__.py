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

from . import solvers
from .autonomous_flows import (
    Chen,
    ChuaCircuit,
    ComplexButterfly,
    DoublePendulum,
    DoubleScroll,
    Halvorsen,
    Lorenz63,
    Lorenz96,
    Roessler,
    Rucklidge,
    Thomas,
    WindmiAttractor,
)
from .discrete_maps import Henon, Logistic
from .driven_systems import SimplestDrivenChaotic
from .others import KuramotoSivashinsky, MackeyGlass

__all__ = [
    "Lorenz63",
    "Roessler",
    "ComplexButterfly",
    "Chen",
    "ChuaCircuit",
    "Thomas",
    "WindmiAttractor",
    "Rucklidge",
    "Halvorsen",
    "DoubleScroll",
    "DoublePendulum",
    "Lorenz96",
    "Logistic",
    "Henon",
    "SimplestDrivenChaotic",
    "KuramotoSivashinsky",
    "MackeyGlass",
]
