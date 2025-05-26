# Magnetic-Reconnection-Visualization

This repositories hosts the code used in [paper link] about the "Local Extraction of Magnetic Reconnection in Three-Dimensional Plasma Simulations". The core of the repository is the python package `mrvis` located in `src/mrvis`. It contains the implementations of the Paraview filter plugins and magnetic reconnection models, as well as some helper functions. 

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

Most of the algorithms used in the paper are implemented as Paraview plugins. To use them, open Paraview and go to **Tools > Manage Plugins... > Load new** and select the `src/mrvis/plugin_loader.py`. This should load all plugins at once. They can then be found using the "MRVIS" keyword.

## Usage

The typical workflow to extract singular field lines using the parallel vectors operator is similar to the following:
1. Compute the convective acceleration 
2. Tetrahedralize
3. Evaluate the parallel vectors operator
4. Filter the solution using line tangent
5. Filter for bifurcation line criterion
5.1 Additional: Find point of largest hyperbolicity for pseudo-bifurcation lines
5.2 Do streamline integration from seed points

See the `notebooks` folder with an example reconnection model and analytic derivation, as well as loading scripts for different data formats.