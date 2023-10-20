"""Generate 3 x 3 subplots with rotating attractors."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

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
    # "DoublePendulum",
    # "Lorenz96",
    # "Logistic",
    # "Henon",
    # "SimplestDrivenChaotic",
    # "KuramotoSivashinsky",
    # "MackeyGlass",
]

# create data:
N = 1000
data_dict = {}
for i_sys, system in enumerate(systems):
    sys_class = sim_class = getattr(sims, system)
    data = sys_class().simulate(N)
    data_dict[system] = data

# Plot:
plt.style.use("dark_background")
fig, axs = plt.subplots(2, 5, figsize=(10, 3), subplot_kw=dict(projection="3d"))
# plt.subplots_adjust(wspace=5)
plt.subplots_adjust(hspace=0.5)
# fig.tight_layout()
axs = axs.flatten()
for i_ax, ax in enumerate(axs):
    system = systems[i_ax]
    data = data_dict[system]
    ax.title.set_text(system)
    ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=0.2, color="white")
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_zticks([])  # Remove y-axis ticks
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axis("off")


def update_graph(num):
    """Rotate camera angle."""
    azim = num
    roll = 0
    elev = 0
    for i_ax, ax in enumerate(axs):
        ax.view_init(elev, azim, roll)


# Create frames:

frames_dir = "frames"
Path(frames_dir).mkdir(exist_ok=True)
previous_files = [f"{frames_dir}/" + x for x in os.listdir(frames_dir)]
for prev_file in previous_files:
    os.remove(prev_file)

for num in range(0, 360):
    update_graph(num)
    plt.savefig(f"{frames_dir}/frame_{str(num).zfill(3)}.png", transparent=True, dpi=50)

# gif:
image_filenames = [
    f"{frames_dir}/" + x
    for x in os.listdir("frames")
    if x.endswith(".png") and x.startswith("frame_")
]

images = [Image.open(filename) for filename in image_filenames]

# Create a GIF
images[0].save(
    "attractor_animation.gif",
    format="GIF",
    save_all=True,
    append_images=images[1:],
    duration=20,
    loop=0,
    transparency=0,
    disposal=2,
)

# Cleanup:
previous_files = [f"{frames_dir}/" + x for x in os.listdir(frames_dir)]
for prev_file in previous_files:
    os.remove(prev_file)
os.rmdir(frames_dir)
