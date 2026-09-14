# Finitewave-Tools tutorials

Learn to prepare cardiac meshes, visualize scalar fields, and explore simulation
results with worked Jupyter notebooks. Each tutorial combines explanation, Python
code, and saved figures or animations.

## Choose a tutorial

| Tutorial | What you will learn |
| --- | --- |
| [Plotting and animation](<Plot and Animation.ipynb>) | Convert tissue and element meshes into PyVista grids, assign scalar values, and export animations. |
| [Left ventricle fibers](<Left Ventricle Fibers.ipynb>) | Generate myocardial fiber directions using Laplace fields and local anatomical axes. |
| [Atrial tachycardia ablation](<Atrial Tachycardia Ablation.ipynb>) | Explore wave propagation and the effect of ablation lines in an atrial model. |

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

Spiral Wave Core 2D and Lead Field Electrograms are placeholders and will be
included when their notebooks are ready.
