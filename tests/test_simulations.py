"""Testing simulations."""
import numpy as np
import pytest

from lorenzpy import simulations


class DemoSim(simulations._SimBase):
    """A simple simulation class subclassing simulations._SimBase."""

    sys_dim = 3

    def __init__(self):
        """Set the sys_dim to 3."""
        self.sys_dim = 3

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Just return the same value again."""
        return x


def test_lorenz63_simulation_shape():
    """Testing lorenz63 simulation."""
    shape = simulations.Lorenz63().simulate(2).shape
    assert shape == (2, 3)


def test_simulate_trajectory_lorenz63_single_step():
    """Testing lorenz63 single step simulation."""
    simulation_time_steps = 2
    starting_point = np.array([-14.03020521, -20.88693127, 25.53545])

    sim_data = simulations.Lorenz63(dt=2e-2).simulate(
        time_steps=simulation_time_steps, starting_point=starting_point
    )

    exp_sim_data = np.array(
        [
            [-14.03020521, -20.88693127, 25.53545],
            [-15.257976883416845, -20.510306180264724, 30.15606333510718],
        ]
    )

    np.testing.assert_almost_equal(sim_data, exp_sim_data, decimal=14)


def test_simulation_with_wrong_dim_of_starting_point():
    """Testing that the ValueError is raised from wrong starting point in simulate."""
    test_sim = DemoSim()
    wrong_starting_point = np.array([1, 1])

    with pytest.raises(ValueError):
        test_sim.simulate(time_steps=10, starting_point=wrong_starting_point)


def test_mackey_glass_simulation_shape():
    """Testing that the MackeyGlass simulation outputs the correct shape."""
    shape = simulations.MackeyGlass().simulate(2).shape
    assert shape == (2, 1)
