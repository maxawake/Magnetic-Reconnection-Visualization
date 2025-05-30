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
    "mrvisReconnectionRate.py",
]

model_list = ["mrvis3DMagneticReconnection.py", "mrvisCriticalPoint.py", "mrvisSingularFieldLine.py"]

plugin_dir = os.path.dirname(__file__)
for name in plugin_list:
    LoadPlugin(os.path.join(plugin_dir, "plugins", name), remote=True)
for name in model_list:
    LoadPlugin(os.path.join(plugin_dir, "models", name), remote=True)
