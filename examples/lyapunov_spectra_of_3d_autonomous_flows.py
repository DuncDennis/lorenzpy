"""Calculate and plot the Lyapunov Spectra of all three-dimensional chaotic flows."""

import matplotlib.pyplot as plt
import numpy as np

from lorenzpy import measures as meas
from lorenzpy import simulations as sims

systems = [
    "Lorenz63",
    "Chen",
    "ChuaCircuit",
    "ComplexButterfly",
    "DoubleScroll",
    "Halvorsen",
    "Roessler",
    "Rucklidge",
    "Thomas",
    "WindmiAttractor",
]

# Calculate exponents:
m = 3
deviation_scale = 1e-10
steps = 1000
part_time_steps = 15
steps_skip = 50

solver = "rk4"
# solver = sims.solvers.create_scipy_ivp_solver(method="RK45")

lyap_dict = {}
for i_sys, system in enumerate(systems):
    print(system)
    sys_obj = getattr(sims, system)(solver=solver)
    iterator_func = sys_obj.iterate
    starting_point = sys_obj.get_default_starting_pnt()
    dt = sys_obj.dt

    lyap_dict[system] = meas.lyapunov_exponent_spectrum(
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

fig, axs = plt.subplots(
    2, 5, figsize=(15, 8), layout="constrained", sharex=True, sharey=False
)

# x and y titles:
fig.supxlabel("Number of renormalization steps")
fig.supylabel("Lyapunov exponent convergence")

axs = axs.flatten()
x = np.arange(1, steps + 1)
for i_ax, ax in enumerate(axs):
    system = systems[i_ax]
    le_spectrum = lyap_dict[system]
    ax.title.set_text(system)
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
