a__all__ = ['prtlParallelVectorsPython']
from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
import math
import numpy as np


def get_parallel_vector_operator(vec1, vec2):
    #nx, ny, nz = vec1.shape[0], vec1.shape[1], vec1.shape[2]
    #sizex, sizey, sizez = vec1.shape[1], vec1.shape[2], vec1.shape[3]

    cross_product = np.cross(vec1.reshape(-1, vec1.shape[-1]), vec2.reshape(-1, vec1.shape[-1]))
    #parallel = np.linalg.norm(cross_product, axis=1).reshape(sizex, sizey, sizez)

    return cross_product.reshape(*vec1.shape)

def get_angle(vec1, vec2, idx):
    nx, ny, nz = vec1.shape[0], vec1.shape[1], vec1.shape[2]
    #sizex, sizey, sizez = vec1.shape[1], vec1.shape[2], vec1.shape[3]

    """cross_product = np.cross(vec1.reshape(-1, vec1.shape[-1]), vec2.reshape(-1, vec1.shape[-1]))
    #parallel = np.linalg.norm(cross_product, axis=1).reshape(sizex, sizey, sizez)

    return cross_product.reshape(*vec1.shape)"""
    vec1 = vec1.reshape(-1, vec1.shape[-1])
    vec2 = vec2.reshape(-1, vec2.shape[-1])

    crossp = np.cross(vec1, vec2)
    dotp = np.sum(vec1 * vec2, axis=1)

    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)

    return np.arcsin(crossp[:,idx].reshape(nx, ny , nz)/(mag_vec1*mag_vec2))#.reshape(nx, ny , nz)


@smproxy.filter(label="PRTL Parallel Vectors Python")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlParallelVectorsPython(VTKPythonAlgorithmBase):
    def __init__(self):
        self._array_field = [0] * 2
        self._array_name = ['None'] * 2
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=3, outputType='vtkImageData')
    
    def SetInputArrayToProcess(self, idx, port, connection, field, name):
        self._array_field[idx] = field
        self._array_name[idx] = name
        self.Modified()

    @smproperty_inputarray('u', none_string='None', idx=0, command='SetInputArrayToProcess')
    def SetInputArrayToProcess1(): pass

    @smproperty_inputarray('w', none_string='None', idx=1, command='SetInputArrayToProcess')
    def SetInputArrayToProcess2(): pass

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
        output1 = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 1))
        output2 = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 2))

        output.ShallowCopy(input.VTKObject)
        output1.ShallowCopy(input.VTKObject)
        output2.ShallowCopy(input.VTKObject)

        dimensions = list(input.VTKObject.GetDimensions())
        spacing = list(input.VTKObject.GetSpacing())

        array0= input.PointData[self._array_name[0]]
        array1 = input.PointData[self._array_name[1]]
        
        components = 1 if len(array0.shape) == 1 else array0.shape[1]
        data0 = np.copy(array0.reshape(dimensions + [components], order='F'))

        components = 1 if len(array1.shape) == 1 else array1.shape[1]
        data1 = np.copy(array1.reshape(dimensions + [components], order='F'))

        # result = get_parallel_vector_operator(data0, data1)
        # result = result.reshape((-1, result.shape[-1]), order='F')
        # output.PointData.append(result, "Parallel Vectors")

        result = get_angle(data0, data1, 0)
        result = result.reshape(-1, order='F')
        output.PointData.append(result, "Angles")

        result1 = get_angle(data0, data1, 1)
        result1 = result1.reshape(-1, order='F')
        output1.PointData.append(result1, "Angles1")

        result2 = get_angle(data0, data1, 2)
        result2 = result2.reshape(-1, order='F')
        output2.PointData.append(result2, "Angles2")

        return 1
