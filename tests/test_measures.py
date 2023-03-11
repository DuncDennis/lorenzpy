"""Testing simulations."""
import numpy as np

from lorenzpy import measures


def test_largest_lyapunov_one_d_linear_time_dependent():
    """Testing measures.largest_lyapunov_exponent."""
    dt = 0.1
    a = -0.1

    def iterator_func(x):
        return x + dt * a * x

    return_convergence = False
    starting_point = np.array(1)
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


def test_largest_lyapunov_one_d_linear_time_dependent_return_conv():
    """Testing measures.largest_lyapunov_exponent with convergence."""
    dt = 0.1
    a = -0.1

    def iterator_func(x):
        return x + dt * a * x

    return_convergence = True
    starting_point = np.array(1)
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
