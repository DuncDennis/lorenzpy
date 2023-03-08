"""Testing simulations."""
from lorenzpy import simulations


def test_lorenz63_simulation():
    """Testing lorenz63 simulation."""
    shape = simulations.Lorenz63().simulate(2).shape
    assert shape == (2, 3)
