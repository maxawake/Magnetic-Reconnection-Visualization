__all__ = ['prtlShearLayer']

from pyprtl.util.vtkAlgorithm import *
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataSet, vtkDataObject
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
import math
import numpy as np

@smproxy.filter(label="PRTL Shear Layer")
@smhint_menu('prtl')
@smproperty.input(name='Input', port_index=0)
@smdomain.datatype(dataTypes=['vtkDataSet']) 
class prtlShearLayer(VTKPythonAlgorithmBase):
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

        dudx, dudy, dudz = np.gradient(data[...,0], spacing[0], spacing[1], spacing[2])
        dvdx, dvdy, dvdz = np.gradient(data[...,1], spacing[0], spacing[1], spacing[2])
        dwdx, dwdy, dwdz = np.gradient(data[...,2], spacing[0], spacing[1], spacing[2])

        Jacobian = np.array([[dudx, dudy, dudz],
                             [dvdx, dvdy, dvdz],
                             [dwdx, dwdy, dwdz]])
        
        Jac = Jacobian.reshape(Jacobian.shape[0], Jacobian.shape[1], -1).transpose(2, 0, 1)

        S = (Jac + Jac.transpose(0,2,1)) / 2
        eigenvals = np.linalg.eigvals(S)
        result = -(eigenvals[:,0]*eigenvals[:,1] + eigenvals[:,0]*eigenvals[:,2] + eigenvals[:,1]*eigenvals[:,2])
        # result = np.sqrt(
        #     (
        #         (eigenvals[:,0] - eigenvals[:,1]) ** 2
        #         + (eigenvals[:,0] - eigenvals[:,2]) ** 2
        #         + (eigenvals[:,1] - eigenvals[:,2]) ** 2
        #     )
        #     / 6
        # )
        
        result_reshaped = result.reshape(data.shape[0], data.shape[1], data.shape[2])

        result = result_reshaped.reshape(-1, order='F')
        output.PointData.append(result, "Shear Layer")

        return 1





