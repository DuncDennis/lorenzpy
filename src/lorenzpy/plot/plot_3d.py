"""Module to produce three-dimensional plots of time series."""
import numpy as np
import plotly.graph_objects as go

DEFAULT_FIGSIZE = (650, 500)


def plot_3d(time_series: np.ndarray) -> None:
    """Plot simple three-dim time series."""
    fig = go.Figure()
    fig.update_layout(height=DEFAULT_FIGSIZE[1], width=DEFAULT_FIGSIZE[0])
    fig.add_trace(
        go.Scatter3d(
            x=time_series[:, 0], y=time_series[:, 1], z=time_series[:, 2], mode="lines"
        )
    )
    fig.show()
