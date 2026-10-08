# Finitewave-Tools

Tools for preparing cardiac meshes, visualizing simulation results, building
animations, and analyzing cardiac electrophysiology data. Finitewave-Tools
complements [Finitewave](https://github.com/finitewave/Finitewave).

The package includes:

- PyVista grids for structured tissue meshes, element meshes, and surfaces.
- 2D and 3D frame rendering with MP4 and GIF export.
- Linear solvers, geodesic coordinates, and rule-based fiber generation.
- Point sampling and conduction velocity calculation utilities.

## Installation

Python **3.11 or newer** is required. Clone the repository and create an isolated
environment:

```bash
git clone https://github.com/finitewave/Finitewave-Tools.git
cd Finitewave-Tools
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Installation includes NumPy, SciPy, PyAMG, Matplotlib, tqdm, natsort, PyVista,
VTK, PyAV, ImageIO, and scikit-image. PyAV handles MP4 encoding; ImageIO handles
GIF export. Dependency minimum versions are defined in
[pyproject.toml](pyproject.toml).

## Quick start

Render a 2D scalar field as an RGB array:

```python
import numpy as np
from finitewave_tools.animation import Frame2DRenderer

mesh = np.ones((64, 64), dtype=np.uint8)
field = np.linspace(0, 1, mesh.size).reshape(mesh.shape)

renderer = Frame2DRenderer()
renderer.set_frames(frames=[field])
renderer.configure_grid(mesh, upscale_factor=2)
rgb = renderer.render_frame(0, clim=(0, 1), cmap="viridis")
print(rgb.shape)  # (128, 128, 3)
renderer.finalize()
```

## Tutorials

[Browse the documentation](https://finitewave.github.io/Finitewave-Tools/) for
worked examples covering meshes, fibers, animations, electrophysiology,
electrograms, and neural networks. The notebooks and accompanying data live in
[Tutorials/](Tutorials/).

To run them locally from the activated environment:

```bash
python -m pip install jupyterlab
cd Tutorials
jupyter lab
```

Run notebook cells in order with `Tutorials/` as the working directory so data
paths resolve correctly. Simulation notebooks also require Finitewave; consult
each notebook for additional packages and setup instructions.

## Build the documentation

To work on the documentation without installing the package's runtime dependencies:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve -f docs/mkdocs.yml
```

Build with `python -m mkdocs build --strict -f docs/mkdocs.yml`. See
[docs/pages/contributing.md](docs/pages/contributing.md) for authoring and publishing.
