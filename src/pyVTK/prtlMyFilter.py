a__all__ = ['prtlMyFilter']
from pyprtl.util.vtkAlgorithm import *
from pyprtl.util.logging import Log
import pyprtl.util.arrays as util
from vtkmodules.vtkCommonCore import VTK_FLOAT, VTK_DOUBLE
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject, vtkPolyData
from vtkmodules.numpy_interface import dataset_adapter as dsa
import numpy as np
from vtk import vtkSphereSource
from vespa import vtkCGALPMP

@smproxy.filter(label="PRTL MyFilter")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet'])
class prtlMyFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        VTKPythonAlgorithmBase.__init__(self,
                                        nInputPorts=1, nOutputPorts=1,
                                        inputType='vtkImageData', outputType='vtkPolyData')


    def RequestData(self, request, inInfo, outInfo):
        inp = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))

        

        pd = vtkSphereSource()

        rm = vtkCGALPMP.vtkCGALIsotropicRemesher()
        rm.SetInputConnection(pd.GetOutputPort())
        rm.SetTargetLength(0.01)
        rm.SetProtectAngle(90)
        rm.Update()

        output.ShallowCopy(rm.GetOutput())

        return 1
