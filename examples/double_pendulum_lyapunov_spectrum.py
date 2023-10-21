"""Lyapunov Spectrum of single system."""

import matplotlib.pyplot as plt
import numpy as np

from lorenzpy import measures as meas
from lorenzpy import simulations as sims

sys_obj = sims.DoublePendulum(dt=0.1)
dt = sys_obj.dt

# Calculate exponents:
m = 4
deviation_scale = 1e-10
steps = 1000
part_time_steps = 15
steps_skip = 0

iterator_func = sys_obj.iterate
starting_point = sys_obj.get_default_starting_pnt()

le_spectrum = meas.lyapunov_exponent_spectrum(
    iterator_func=iterator_func,
    starting_point=starting_point,
    deviation_scale=deviation_scale,
    steps=steps,
    part_time_steps=part_time_steps,
    steps_skip=steps_skip,
    dt=dt,
    m=m,
    initial_pert_directions=None,
    return_convergence=True,
)

fig, ax = plt.subplots(
    1, 1, figsize=(6, 6), layout="constrained", sharex=True, sharey=False
)

# x and y titles:
fig.supxlabel("Number of renormalization steps")
fig.supylabel("Lyapunov exponent convergence")

x = np.arange(1, steps + 1)
ax.plot(
    x,
    le_spectrum,
    linewidth=1,
)
ax.grid(True)

final_les = np.round(le_spectrum[-1, :], 4).tolist()
final_les = [str(x) for x in final_les]
le_string = "\n".join(final_les)
le_string = "Final LEs: \n" + le_string
x_position = 0.1  # X-coordinate of the upper-left corner for each subplot
y_position = 0.5
ax.text(
    x_position,
    y_position,
    le_string,
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    verticalalignment="center",
    horizontalalignment="left",
    transform=ax.transAxes,
)

plt.show()
