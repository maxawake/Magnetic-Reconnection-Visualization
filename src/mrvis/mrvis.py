# all_plugins.py
import os
from paraview.simple import LoadPlugin

plugin_list = [
    "mrvisExtractByType.py",
    "mrvisParallelVectorsVTK.py",
    "mrvisSaveStuff.py",
    "mrvisConvectiveAcceleration.py",
    "mrvisOutlineAxes.py",
    "mrvisPerpSlice.py",
    "mrvisShearLayer.py",
    "mrvisCoordinateAxes.py",
    "mrvisParallelVectorsPython.py",
    "mrvisPseudoBifurcationLine.py",
    "mrvisStreamTube.py",
    "mrvisEigenvalues.py",
]

plugin_dir = os.path.dirname(__file__)
for name in plugin_list:
    LoadPlugin(os.path.join(plugin_dir, "filter", name), remote=True)
