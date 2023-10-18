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


TODO <below>
    - Probably for each concrete simulation class + public methods. Compare with sklearn
    - Find out which functionality is missing. E.g. Raising error when wrong values are
      parsed.
    - Check where to add proper tests and how to add them efficiently. Fixtures?
      Parametrization?
    - Implement all the other dynamical systems.
    - Check if the names of files and functions make sense?
    - Add functionality to add your own dynamical system? As my base-classes are
      protected this is maybe not so easy? -> Make ABC public?
    - Think about adding NARMA? Maybe I need a random number generator framework.
    - Check if I can further reduce code duplication. Maybe regarding solvers.
    - Check for proper doc-generation. It seems that the methods of inhereted members
      is not implemented yet. See:
      https://github.com/mkdocstrings/mkdocstrings/issues/78
"""

from .autonomous_flows import Lorenz63, Lorenz96
from .discrete_maps import Logistic
from .driven_systems import SimplestDrivenChaotic
from .others import KuramotoSivashinsky, MackeyGlass

__all__ = [
    "Lorenz63",
    "Lorenz96",
    "Logistic",
    "SimplestDrivenChaotic",
    "KuramotoSivashinsky",
    "MackeyGlass",
]
