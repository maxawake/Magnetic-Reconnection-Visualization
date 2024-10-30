# Magnetic-Reconnection-Visualization

This package provides algorithms to extract and visualize magnetic reconnection in 3D plasma simulations using bifurcation lines and the parallel vectors operator.

## Installation

To use the python package make a new python environment with python version `3.10`, e.g. with conda

```bash
conda create -n mrvis python=3.10
conda activate mrvis
```

Then install the needed python packages with pip

```bash
pip install -r requirements.txt
```

To use the package in a new file import add it to the path, e.g.
 
```python
import sys
sys.path.insert(0, "/home/max/Github/Magnetic-Reconnection-Visualization/src")
```

## Usage

See the `notebooks` folder with an example reconnection model and analytic derivation, as well as loading scripts for different data formats.