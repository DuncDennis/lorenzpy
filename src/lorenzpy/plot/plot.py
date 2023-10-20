"""Plot the data of dynamical systems."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def create_3d_line_plot(data: np.ndarray, ax: "Axes3D" = None, **kwargs) -> "Axes3D":
    """Create a three-dimensional line plot of data.

    Args:
        data (np.ndarray): A NumPy array containing 3D data points with shape (n, 3).
        ax (Axes3D, optional): A Matplotlib 3D axis to use for plotting.
        If not provided, a new 3D plot will be created.
        **kwargs: Additional keyword arguments to pass to ax.plot.

    Returns:
        Axes3D: The Matplotlib 3D axis used for the plot.

    Example:
        >>> data = np.random.rand(100, 3)  # Replace this with your own 3D data
        >>> create_3d_line_plot(data, color='b', linestyle='--')
        >>> plt.show()
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    ax.plot(x, y, z, **kwargs)  # Use plot for a line plot with kwargs

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    return ax
