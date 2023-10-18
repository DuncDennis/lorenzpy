"""Testing simulations."""
import numpy as np
import pytest

from lorenzpy import simulations  # type: ignore
from lorenzpy.simulations.base import _BaseSimFlow, _BaseSimIterate  # type: ignore


@pytest.fixture
def all_sim_classes():
    """Fixture that returns all simulation classes."""
    list_of_sim_classes = []
    for sim_class_str in simulations.__all__:
        sim_class = getattr(simulations, sim_class_str)
        list_of_sim_classes.append(sim_class)
    return list_of_sim_classes


@pytest.fixture
def all_flow_sim_classes(all_sim_classes):
    """Fixture that returns all simulation classes with the flow method."""
    all_flow_sim_classes = []
    for sim_class in all_sim_classes:
        flow_func = getattr(sim_class, "flow", None)
        if callable(flow_func):
            all_flow_sim_classes.append(sim_class)
    return all_flow_sim_classes


@pytest.fixture(params=["rk4", "forward_euler"])
def supported_flow_solver(request):
    """Return the supported flow solvers."""
    return request.param


class DemoSim(_BaseSimIterate):
    """A simple simulation class subclassing simulations._SimBase."""

    sys_dim = 3

    def __init__(self):
        """Set the sys_dim to 3."""
        self.sys_dim = 3

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """Just return the same value again."""
        return x

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return some default starting point."""
        return np.array([0, 0, 0])


class DemoFlowSim(_BaseSimFlow):
    """A simple simulation class subclassing simulations._BaseSimFlow."""

    def __init__(self, dt=0.1, solver="rk4"):
        """Just use the necessary arguments of the _BaseSimFlow init."""
        super().__init__(dt, solver)

    def flow(self, x: np.ndarray) -> np.ndarray:
        """Return some arbitrary 3-dim flow."""
        return np.array([x[0] ** 2, x[1] ** 3, x[2] ** 4])

    def get_default_starting_pnt(self) -> np.ndarray:
        """Return some arbitrary 3-dim starting point."""
        return np.array([1, 2, 3])


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


def test_mackey_glass_simulation_shape():
    """Testing that the MackeyGlass simulation outputs the correct shape."""
    shape = simulations.MackeyGlass().simulate(2).shape
    assert shape == (2, 1)


def test_all_sim_classes_default_simulation(all_sim_classes):
    """Simulate every sim class with default settings for two time steps."""
    for sim_class in all_sim_classes:
        sim_obj = sim_class()
        data = sim_obj.simulate(2)
        assert data.shape[0] == 2


def test_supported_flow_solvers(all_flow_sim_classes, supported_flow_solver):
    """For all flow sim classes, simulate the system for each supported solver."""
    for sim_class in all_flow_sim_classes:
        sim_obj = sim_class(solver=supported_flow_solver)
        data = sim_obj.simulate(2)
        assert data.shape[0] == 2


def test_get_default_params(all_sim_classes):
    """Get the default parameters of each sim class."""
    for sim_class in all_sim_classes:
        default_params = sim_class.get_default_params()
        assert isinstance(default_params, dict)


def test_get_params(all_sim_classes):
    """Get the chosen parameters of each sim_obj."""
    for sim_class in all_sim_classes:
        sim_obj = sim_class()
        params = sim_obj.get_params()
        assert isinstance(params, dict)


def test_unsupported_solver():
    """Test the error raising of an unsupported solver."""
    sim_obj = DemoFlowSim(solver="WRONG_SOLVER")
    with pytest.raises(ValueError):
        sim_obj.simulate(2)


def test_custom_solver_as_forward_euler():
    """Test a custom solver, which behaves like forward_euler."""

    def custom_solver(flow, dt, x):
        # is forward euler
        return dt * flow(x) + x

    data_custom = DemoFlowSim(solver=custom_solver).simulate(2)

    data_forward_euler = DemoFlowSim(solver="forward_euler").simulate(2)

    assert (data_custom == data_forward_euler).all()


def test_simplest_driven_chaotic_time_vs_no_time():
    """Test simplest driven chaotic system as an example of a driven system."""
    sim_obj = simulations.SimplestDrivenChaotic()
    data_w_time = sim_obj.simulate(2, time_included=True)

    assert data_w_time.shape == (2, 3)


def test_simplest_driven_chaotic_custom_starting_point_no_time():
    """Test the case where a custom sp is provided with no time included."""
    sim_obj = simulations.SimplestDrivenChaotic()
    custom_starting_pnt = np.ones(2)
    data_w_time = sim_obj.simulate(
        2, starting_point=custom_starting_pnt, time_included=False
    )

    assert data_w_time.shape == (2, 2)
