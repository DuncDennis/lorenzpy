from lorenzpy import simulations


def test_lorenz63_simulation():
    shape = simulations.Lorenz63().simulate(2).shape
    assert shape == (2, 3)
