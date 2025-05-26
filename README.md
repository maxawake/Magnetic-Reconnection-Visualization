# Magnetic-Reconnection-Visualization

This repositories hosts the code used in [paper link] about the "Local Extraction of Magnetic Reconnection in Three-Dimensional Plasma Simulations". The core of this code is the python package `mrvis` located in `src/mrvis`. It contains the implementations of the Paraview filter plugins and magnetic reconnection models, as well as some helper functions. 

The tools provided here can be used to localize singular field line magnetic reconnection in 3D plasma simulations requiring only the magnetic field data. The regions can be found as topological features of the vector field using (pseudo-) bifurcation lines.  

## Installation

To use the python package `mrvis` as a standalone tool, create a new venv with `uv`
```bash
uv venv
```
and install the dependencies with
```bash
uv pip install -r pyproject.toml
```

## Paraview Plugins

Most of the algorithms used in the paper are implemented as Paraview filter plugins. To use them, open Paraview and go to **Tools > Manage Plugins... > Load new** and select the `src/mrvis/plugin_loader.py`. This should load all plugins at once. They can then be found in Paraview using the "MRVIS" keyword.

Some of the VTK filters used in this project are not exposed in paraview, such as the `vtkParallelVectors` or `vtkVectorFieldTopology` filters. They can, however, be easily used in paraview using XML filters. Examples are given in `src/paraview-plugins`.

## Usage

The typical workflow to extract singular field lines using the parallel vectors operator is similar to the following:
1. Load simulation data or generate analytical model
2. Compute the convective acceleration 
3. Tetrahedralize
4. Evaluate the parallel vectors operator
5. Filter the solution using line tangent
6. Filter for bifurcation line criterion
7.1 Additional: Find point of largest hyperbolicity for pseudo-bifurcation lines
7.2 Do streamline integration from seed points

See the `notebooks` folder with an example reconnection model and analytic derivation, as well as loading scripts for different data formats.