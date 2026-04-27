# Heat Equation Solver

This example implements a parallel heat equation solver using the CCCL Runtime.

## Heat Equation Solver

The heat equation is a partial differential equation that describes the variation of temperature in a given region over time

$$ \frac{du}{dt} = \alpha \cdot \nabla^2 \cdot u $$

where $u(x, y, z, t)$ represents temperature variation over space at a given time, and $\alpha$ is a thermal diffusivity constant.

We limit ourselves to two dimensions (plane) and discretize the equation onto a grid.  Then the Laplacian can be expressed as finite differences as

$$ \nabla^2 u(i, j) = \frac{u(i - 1, j) - 2 \cdot u(i, j) + u(i + 1, j)}{(\Delta x)^2} + \frac{u(i, j - 1) - 2 \cdot u(i, j)  + u(i, j + 1)}{(\Delta y)^2} $$

Where $\Delta x$ and $\Delta y$ are the grid spacing of the temperature grid $u(i, j)$. We can study the development of the temperature grid with explicit time evolution
over time steps $\Delta t$:

$$ u^{m + 1}(i, j) = u^m(i, j) + \Delta t \cdot \alpha \cdot \nabla^2 \cdot u^m(i, j)$$

## Running The Example

Compiling the example should result `${build_dir}/heat_diffusion` executable, which can be invoked as:
```sh
$ ./heat_diffusion [nt=5000]
```

The binary runs `nt` steps of the simulation and outputs a [PGM](https://en.wikipedia.org/wiki/PGM) image of the temperature after each 500 steps of the simulation.
