# Finitewave-Tools tutorials

Learn to prepare cardiac meshes, visualize scalar fields, and explore simulation
results with worked Jupyter notebooks. Each tutorial combines explanation, Python
code, and saved figures or animations.

## Choose a tutorial

| Tutorial | What you will learn |
| --- | --- |
| [Plotting and animation](<Plot and Animation.ipynb>) | Convert tissue and element meshes into PyVista grids, assign scalar values, and export animations. |
| [Left ventricle fibers](<Left Ventricle Fibers.ipynb>) | Generate myocardial fiber directions using Laplace fields and local anatomical axes. |
| [Alternans and spiral breakup](<Alternans and Spiral Breakup.ipynb>) | Explore action potential restitution, alternans, and the role of sodium recovery in spiral-wave stability. |
| [Atrial tachycardia ablation](<Atrial Tachycardia Ablation.ipynb>) | Explore wave propagation and the effect of ablation lines in an atrial model. |
| [Lead field electrograms](<Lead Field Electrograms.ipynb>) | Construct electrode patches, solve for lead-field weights, and calculate a bipolar electrogram during cardiac excitation. |
| [LAT-based electrograms](<LAT based Electrograms.ipynb>) | Calculate electrograms from local activation times, an action potential template, and electrode lead fields. |
| [LAT interpolation with PINN](<LAT Interpolation with PINN.ipynb>) | Reconstruct activation times and conduction speed from sparse LAT samples using surface eigenfunctions and an eikonal constraint. |
| [Fiber estimation with PINN](<Fiber Estimation with PINN.ipynb>) | Infer fiber orientation and longitudinal and transverse conduction speeds from multiple activation maps using an anisotropic eikonal constraint. |

Start with **Plotting and animation** to become familiar with the visualization
objects used throughout the examples.

## Run the notebooks locally

Clone the repository so the notebooks and their companion data stay together:

```bash
git clone https://github.com/finitewave/Finitewave-Tools.git
cd Finitewave-Tools
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -e .
python -m pip install jupyterlab av imageio
cd Tutorials
jupyter lab
```

Run notebook cells in order. Their relative paths assume `Tutorials/` is the
working directory. Some examples also require Finitewave or other packages;
follow the imports and setup instructions in the selected notebook.

The **Download notebook** link provides the notebook alone. Obtain its companion
`data/` files from the repository as well. The website displays saved outputs;
it does not run Python or simulations in your browser.

## Planned tutorials

Spiral Wave Core 2D is a placeholder and will be included when its notebook is ready.
