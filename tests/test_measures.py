"""Testing simulations."""

import numpy as np
import pytest

from lorenzpy import measures, simulations


@pytest.fixture
def get_demo_exponential_decay_system():
    """Fixture to get a demo iterator func and starting point.

    Returns the iterator_func, starting_point, dt and a for exponential decay.
    """
    dt = 0.1
    a = -0.1

    def iterator_func(x):
        return x + dt * a * x

    starting_point = np.array(1)

    return iterator_func, starting_point, dt, a


def test_largest_lyapunov_one_d_linear_time_dependent(
    get_demo_exponential_decay_system,
):
    """Testing measures.largest_lyapunov_exponent."""
    iterator_func, starting_point, dt, a = get_demo_exponential_decay_system

    return_convergence = False
    deviation_scale = 1e-10
    steps = 10
    steps_skip = 1
    part_time_steps = 5

    actual = measures.largest_lyapunov_exponent(
        iterator_func,
        starting_point,
        return_convergence=return_convergence,
        deviation_scale=deviation_scale,
        steps=steps,
        steps_skip=steps_skip,
        part_time_steps=part_time_steps,
        dt=dt,
    )
    desired = np.array(a)
    np.testing.assert_almost_equal(actual, desired, 2)


def test_largest_lyapunov_one_d_linear_time_dependent_return_conv(
    get_demo_exponential_decay_system,
):
    """Testing measures.largest_lyapunov_exponent with convergence."""
    iterator_func, starting_point, dt, a = get_demo_exponential_decay_system

    return_convergence = True
    deviation_scale = 1e-10
    steps = 10
    steps_skip = 1
    part_time_steps = 5

    actual = measures.largest_lyapunov_exponent(
        iterator_func,
        starting_point,
        return_convergence=return_convergence,
        deviation_scale=deviation_scale,
        steps=steps,
        steps_skip=steps_skip,
        part_time_steps=part_time_steps,
        dt=dt,
    )
    desired = np.array(
        [
            a,
        ]
        * steps
    )
    np.testing.assert_almost_equal(actual, desired, 2)


def test_lyapunov_spectrum_lorenz63():
    """Testing lyapunov_exponent_spectrum with initial_pert_directions."""
    dt = 0.05
    lorenz63_obj = simulations.Lorenz63(dt=dt)
    iterator_func = lorenz63_obj.iterate
    starting_point = lorenz63_obj.get_default_starting_pnt()

    m = 3
    deviation_scale = 1e-10
    steps = 1000
    part_time_steps = 15
    steps_skip = 50
    initial_pert_directions = None
    return_convergence = False

    lyap_spec_conv = measures.lyapunov_exponent_spectrum(
        iterator_func=iterator_func,
        starting_point=starting_point,
        deviation_scale=deviation_scale,
        steps=steps,
        part_time_steps=part_time_steps,
        steps_skip=steps_skip,
        dt=dt,
        m=m,
        initial_pert_directions=initial_pert_directions,
        return_convergence=return_convergence,
    )

    expected_lyapunovs = np.array([9.05e-01, 1.70e-03, -1.44e01])
    np.testing.assert_almost_equal(lyap_spec_conv, expected_lyapunovs, decimal=1)


def test_lyapunov_spectrum_vs_largest_le_lorenz63():
    """Testing lyapunov_exponent_spectrum with initial_pert_directions."""
    dt = 0.05
    lorenz63_obj = simulations.Lorenz63(dt=dt)
    iterator_func = lorenz63_obj.iterate
    starting_point = lorenz63_obj.get_default_starting_pnt()

    m = 1
    deviation_scale = 1e-10
    steps = 500
    part_time_steps = 10
    steps_skip = 50
    return_convergence = True

    lyap_spec_conv = measures.lyapunov_exponent_spectrum(
        iterator_func=iterator_func,
        starting_point=starting_point,
        deviation_scale=deviation_scale,
        steps=steps,
        part_time_steps=part_time_steps,
        steps_skip=steps_skip,
        dt=dt,
        m=m,
        initial_pert_directions=None,
        return_convergence=return_convergence,
    )

    lyap_largest_conv = measures.largest_lyapunov_exponent(
        iterator_func=iterator_func,
        starting_point=starting_point,
        deviation_scale=deviation_scale,
        steps=steps,
        part_time_steps=part_time_steps,
        steps_skip=steps_skip,
        dt=dt,
        initial_pert_direction=None,
        return_convergence=return_convergence,
    )

    np.testing.assert_almost_equal(lyap_spec_conv, lyap_largest_conv[:, np.newaxis], 3)


def test_lyapunov_spectrum_custom_pert(get_demo_exponential_decay_system):
    """Test if it works if custom initial_pert_directions is given."""
    iterator_func, starting_point, dt, a = get_demo_exponential_decay_system

    lle = measures.lyapunov_exponent_spectrum(
        iterator_func=iterator_func,
        starting_point=starting_point,
        dt=dt,
        steps=10,
        part_time_steps=1,
        steps_skip=0,
        initial_pert_directions=np.array([[1]]),
    )
    np.testing.assert_almost_equal(lle, a, decimal=2)
