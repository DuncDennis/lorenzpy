# LorenzPy

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![codecov](https://codecov.io/gh/DuncDennis/lorenzpy/branch/main/graph/badge.svg?token=ATWAEQHBYB)](https://codecov.io/gh/DuncDennis/lorenzpy)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ⚙️ Installation

To install only the core functionality:
```bash
$ pip install lorenzpy
```

To install with the additional plotting functionality.
This also installs `plotly`.
```bash
$ pip install lorenzpy[plot]
```

## ▶️ Usage

LorenzPy can be used to simulate and measure chaotic dynamical systems.
The following example shows how to simulate the famous
[Lorenz63 system](https://de.wikipedia.org/wiki/Lorenz-Attraktor), and measure its
largest [Lyapunov exponent](https://en.wikipedia.org/wiki/Lyapunov_exponent) from the
Lorenz63 iterator:

````python
import lorenzpy as lpy

# Initialize the Lorenz63 simulation object with a RK4 time step of dt=0.05
l63_obj = lpy.simulations.Lorenz63(dt=0.05)

# Simulate 5000 steps of the Lorenz63 system:
data = l63_obj.simulate(5000)    # -> data.shape = (5000,3)

# Calculate the largest Lyapunov exponent from the l63_obj iterator:
iterator = l63_obj.iterate
lle = lpy.measures.largest_lyapunov_exponent(
    iterator_func=iterator,
    starting_point=l63_obj.get_default_starting_pnt(),
    dt=l63_obj.dt
)
# -> lle = 0.905144329...
````

The calculated largest Lyapunov exponent of *0.9051...* is very close to the literature
value of *0.9056*[^SprottChaos].

## 💫 Supported systems


| Name                                  | Type                        | System Dimension |
|:--------------------------------------|-----------------------------|:-----------------|
| `Lorenz63`                            | autonomous dissipative flow | 3                |
| `Roessler`                            | autonomous dissipative flow | 3                |
| `ComplexButterfly`                    | autonomous dissipative flow | 3                |
| `Chen`                                | autonomous dissipative flow | 3                |
| `ChuaCircuit`                         | autonomous dissipative flow | 3                |
| `Thomas`                              | autonomous dissipative flow | 3                |
| `WindmiAttractor`                     | autonomous dissipative flow | 3                |
| `Rucklidge`                     | autonomous dissipative flow | 3                |
| `Halvorsen`                     | autonomous dissipative flow | 3                |
| `DoubleScroll`                     | autonomous dissipative flow | 3                |
| `Lorenz96`                            | autonomous dissipative flow | variable         |
| `DoublePendulum`                      | conservative flow           | 4                |
| `Logistic`                            | noninvertible map           | 1                |
| `Henon`                               | dissipative map             | 2                |
| `SimplestDrivenChaoticFlow`           | conservative flow           | 2 space + 1 time |
| `KuramotoSivashinsky`                 | PDE                         | variable         |
| `MackeyGlass`                         | delay differential equation | variable         |
## 📗 Documentation

- The main documentation can be found here: https://duncdennis.github.io/lorenzpy/
    - ⚠️: The documentation is not in a useful state.
##  ⚠️ Further notes
- So far the usefulness of this package is very limited.
The authors main purpose to creating this package was to learn the full workflow to
develop a Python package.
More information about the development process can be found in [CONTRIBUTING.md](CONTRIBUTING.md).
- The plotting functionality, which can be installed with ``pip install lorenzpy[plot]`` is not tested so far.
- See [Pynamical](https://github.com/gboeing/pynamical) for a similar package

[^SprottChaos]:
    Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.
