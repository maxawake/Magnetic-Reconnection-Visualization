a__all__ = ['prtlLIC']
from pyprtl.util.vtkAlgorithm import *
from pyprtl.util.logging import Log
import pyprtl.util.arrays as util
from vtkmodules.vtkCommonCore import VTK_FLOAT, VTK_DOUBLE
from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkPolyData
from vtkmodules.numpy_interface import dataset_adapter as dsa
import vtk
import numpy as np

@smproxy.filter(label="PRTL LIC")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet'])
class prtlLICr(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        VTKPythonAlgorithmBase.__init__(self,
                                        nInputPorts=1, nOutputPorts=1,
                                        inputType='vtkPolyData', outputType='vtkPolyData')


    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkPolyData.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        #filter = vtkLineInegralConvolution()
        painter = vtk.vtkLineIntegralConvolution2D()

        output.ShallowCopy(input.VTKObject)

        print(dir(painter))
        output.ShallowCopy(input.VTKObject)

        #dimensions = list(input.VTKObject.GetDimensions())
        #spacing = list(input.VTKObject.GetSpacing())

        array = input.PointData[0]
        
        #components = 1 if len(array.shape) == 1 else array.shape[1]
        data = np.copy(array)

        print(data)
        return 1
