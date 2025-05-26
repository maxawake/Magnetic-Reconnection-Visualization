import os
import sys

import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smproperty, smproxy
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkImageData

plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mrvis.filter.decorators import smproperty_inputarray


@smproxy.filter(label="MRVIS Jacobian Eigen System")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class mrvisJacobianEigenSystem(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData")
        self._array_field = None
        self._array_name = "Gradient"

    @smproperty_inputarray("Data", idx=0, input_name="Input", attribute_type="Vectors")
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 0))

        output.ShallowCopy(input.VTKObject)

        dimensions = list(input.VTKObject.GetDimensions())
        spacing = list(input.VTKObject.GetSpacing())

        array = input.PointData[self._array_name]

        components = 1 if len(array.shape) == 1 else array.shape[1]
        data = np.copy(array)

        n_points = input.GetNumberOfPoints()

        # Create numpy containers for output arrays
        eigenvalues = np.empty((n_points, 3))
        eigenvectors = np.empty((n_points, 9))  # Flattened 3x3 matrix

        for i in range(n_points):
            J = np.array(data[i]).reshape((3, 3))
            eigvals, eigvecs = np.linalg.eig(J)
            eigenvalues[i, :] = eigvals.real
            eigenvectors[i, :] = eigvecs.real.T.flatten()  # column-major order

        # Create VTK arrays and assign to output
        evals_vtk = vtkDoubleArray()
        evals_vtk.SetName("Eigenvalues")
        evals_vtk.SetNumberOfComponents(3)
        evals_vtk.SetNumberOfTuples(n_points)
        for i in range(n_points):
            evals_vtk.SetTuple(i, eigenvalues[i])

        evecs_vtk = vtkDoubleArray()
        evecs_vtk.SetName("Eigenvectors")
        evecs_vtk.SetNumberOfComponents(9)
        evecs_vtk.SetNumberOfTuples(n_points)
        for i in range(n_points):
            evecs_vtk.SetTuple(i, eigenvectors[i])

        output.ShallowCopy(input.VTKObject)
        output.PointData.AddArray(evals_vtk)
        output.PointData.AddArray(evecs_vtk)

        return 1
