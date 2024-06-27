__all__ = ['prtlDirectionalDerivative']

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
import math
import numpy as np
from scipy import ndimage
from findiff import Gradient

def get_gradient(scalar, linx, liny, linz):
    dx = ndimage.sobel(scalar, 0)  # x derivative
    dy = ndimage.sobel(scalar, 1)  # y derivative
    dz = ndimage.sobel(scalar, 2)  # z derivative
    return dx, dy, dz


def get_jacobian(vec, linx, liny, linz):
    grad = Gradient(h=[linx, liny, linz], acc=6)
    dudx, dudy, dudz = grad(vec[...,0])
    dvdx, dvdy, dvdz = grad(vec[...,1])
    dwdx, dwdy, dwdz = grad(vec[...,2])
    return np.array([[dudx, dudy, dudz], [dvdx, dvdy, dvdz], [dwdx, dwdy, dwdz]])

@smproxy.filter(label="PRTL Directional Derivative")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlDirectionalDerivative(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = 0
        self._array_name = None
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1, outputType='vtkImageData')

    @smproperty_inputarray('Vectors', attribute_type='Vectors')
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field = field
        self._array_name = name
        self.Modified()

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = vtkDataSet.GetData(inInfo[0], 0)
        if not inp:
            return 0
        for i in range(self.GetNumberOfOutputPorts()):
            output = vtkDataSet.GetData(outInfo, i)
            if not output or not output.IsA(inp.GetClassName()):
                outInfo.GetInformationObject(i).Set(
                    vtkDataObject.DATA_OBJECT(), inp.NewInstance())
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        if not in_info.Has(executive.WHOLE_EXTENT()):
            return 1

        extent = list(in_info.Get(executive.WHOLE_EXTENT()))
        dims = [extent[2 * i + 1] - extent[2 * i] + 1 for i in range(len(extent) // 2)]
        
        out_info = outInfo.GetInformationObject(0)
        out_info.Set(executive.WHOLE_EXTENT(), extent, 6)
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        in_info = inInfo[0].GetInformationObject(0)
        in_info.Set(executive.UPDATE_EXTENT(), in_info.Get(executive.WHOLE_EXTENT()), 6)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        input = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 0))

        output.ShallowCopy(input.VTKObject)

        dimensions = list(input.VTKObject.GetDimensions())
        spacing = list(input.VTKObject.GetSpacing())

        array = input.PointData[self._array_name]
        
        components = 1 if len(array.shape) == 1 else array.shape[1]
        data = np.copy(array.reshape(dimensions + [components], order='F'))
        
        # First version
        """dudx, dudy, dudz = np.gradient(data[...,0], spacing[0], spacing[1], spacing[2])
        dvdx, dvdy, dvdz = np.gradient(data[...,1], spacing[0], spacing[1], spacing[2])
        dwdx, dwdy, dwdz = np.gradient(data[...,2], spacing[0], spacing[1], spacing[2])

        

        Jacobian = np.array([[dudx, dudy, dudz],
                             [dvdx, dvdy, dvdz],
                             [dwdx, dwdy, dwdz]])"""
        
        # Get jacobian with higher order derivative
        Jacobian = get_jacobian(data, spacing[0], spacing[1], spacing[2])

        dirDiv = Jacobian.transpose(2,3,4,1,0).reshape(-1,3,3)@data.reshape(-1,3)[...,np.newaxis]
        dirDiv = dirDiv.reshape(*data.shape)

        dirDiv = dirDiv.reshape((-1, dirDiv.shape[-1]), order='F')
        output.PointData.append(dirDiv, "Directional Derivative")

        return 1





