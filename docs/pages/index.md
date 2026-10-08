# Getting started

Explore cardiac modeling with Finitewave-Tools through four tutorial groups: mesh
preparation and visualization, electrophysiology, electrograms, and neural networks.
Each notebook brings together a short explanation, runnable code, and saved results.

Browse the tutorials online, or run them locally to explore your own experiments.

## Explore the tutorials

<div class="tutorial-groups" markdown>

<div class="tutorial-group" markdown>

### Mesh

Prepare tissue geometry and bring simulation results into view.

- [Plotting and animation](<Plot and Animation.ipynb>)
- [Left ventricle fibers](<Left Ventricle Fibers.ipynb>)

</div>

<div class="tutorial-group" markdown>

### Electrophysiology

Explore how cardiac waves form, evolve, and respond to interventions.

- [Spiral wave termination](<Spiral Wave Termination.ipynb>)
- [Alternans and spiral breakup](<Alternans and Spiral Breakup.ipynb>)
- [Atrial tachycardia ablation](<Atrial Tachycardia Ablation.ipynb>)
- [Phase mapping](<Phase Mapping.ipynb>)

</div>

<div class="tutorial-group" markdown>

### Electrograms

Connect tissue activation to the signals recorded by electrodes.

- [Lead field electrograms](<Lead Field Electrograms.ipynb>)
- [LAT-based electrograms](<LAT based Electrograms.ipynb>)

</div>

<div class="tutorial-group" markdown>

### Neural Networks

Use physics-informed learning to recover information from sparse observations.

- [Conduction velocity estimation with PINN](<Conduction Velocity Estimation with PINN.ipynb>)
- [Fiber estimation with PINN](<Fiber Estimation with PINN.ipynb>)

</div>

</div>

New to Finitewave-Tools? Begin with [Plotting and animation](<Plot and Animation.ipynb>)
to learn how to view meshes and animate simulation results, then choose a group
that matches your interests.

## Run the notebooks locally

Clone the repository to keep the notebooks and their accompanying data together,
then create a Python environment:

```bash
git clone https://github.com/finitewave/Finitewave-Tools.git
cd Finitewave-Tools
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then install Finitewave-Tools and JupyterLab:

```bash
python -m pip install -e .
python -m pip install jupyterlab
cd Tutorials
jupyter lab
```

Open a notebook in JupyterLab and run its cells in order, using `Tutorials/` as
the working directory so relative data paths resolve correctly. Simulation
examples also require Finitewave; check the selected notebook for additional
packages and setup instructions.

On the website, figures and animations show saved results. To change parameters
or run a simulation, use a local notebook. If you use **Download notebook**, also
obtain any accompanying data files from the repository.

## Planned tutorials

A tutorial on tracking spiral-wave cores in 2D is planned and will appear here
when it is ready.
