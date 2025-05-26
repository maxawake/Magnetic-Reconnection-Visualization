import os
import sys

from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase, smdomain, smproperty, smproxy
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkFiltersFlowPaths import vtkParallelVectors

plugin_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src_path = os.path.join(plugin_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mrvis.plugins.decorators import smproperty_inputarray


@smproxy.filter(label="MRVIS Parallel Vectors VTK")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"])
class mrvisParallelVectorsVTK(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType="vtkPolyData")
        self._array_field = [0] * 2
        self._array_name = [None] * 2
        self._array_modified = [False] * 2

    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        if self._array_name[idx] != name or self._array_field[idx] != field:
            self._array_modified[idx] = True
        self._array_field[idx] = field
        self._array_name[idx] = name
        self.Modified()

    @smproperty_inputarray("u", attribute_type="Vectors", none_string="None", idx=0, command="SetInputArrayToProcess")
    def SetInputArrayToProcess1():
        pass

    @smproperty_inputarray("w", attribute_type="Vectors", none_string="None", idx=1, command="SetInputArrayToProcess")
    def SetInputArrayToProcess2():
        pass

    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkUnstructuredGrid.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        pv = vtkParallelVectors()
        pv.SetInputData(input.VTKObject)
        pv.SetFirstVectorFieldName(self._array_name[0])
        pv.SetSecondVectorFieldName(self._array_name[1])
        pv.Update()

        output.ShallowCopy(pv.GetOutput())
        return 1
