a__all__ = ['prtlParallelVectorsVTK']
from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkImageData, vtkPolyData
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersFlowPaths import vtkParallelVectors

@smproxy.filter(label="PRTL Parallel Vectors VTK")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlParallelVectorsVTK(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')
        self._array_field = [0] * 2
        self._array_name = [None] * 2
        self._array_modified = [False] * 2


    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        if self._array_name[idx] != name or self._array_field[idx] != field:
            self._array_modified[idx] = True
        self._array_field[idx] = field
        self._array_name[idx] = name        
        self.Modified()

    @smproperty_inputarray('u', attribute_type='Vectors', none_string='None', idx=0, command='SetInputArrayToProcess')
    def SetInputArrayToProcess1(): pass

    @smproperty_inputarray('w', attribute_type='Vectors', none_string='None', idx=1, command='SetInputArrayToProcess')
    def SetInputArrayToProcess2(): pass


    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        pv = vtkParallelVectors()
        pv.SetInputData(input.VTKObject)
        pv.SetFirstVectorFieldName(self._array_name[0])
        pv.SetSecondVectorFieldName(self._array_name[1])
        pv.Update()

        output.ShallowCopy(pv.GetOutput())
        return 1





